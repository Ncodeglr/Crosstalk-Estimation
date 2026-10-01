#!/usr/bin/env python3
"""
Mesure le crosstalk aux réflecteurs en coin, avant et après correction.

C'est le protocole de validation de Babu et al. (2022, §4.6 et figure 9), et la
seule mesure du projet qui s'appuie sur une **vérité terrain**. Un trièdre a
pour matrice de diffusion `S = sqrt(sigma) I` : ses canaux croisés sont nuls par
construction, donc tout ce qu'on y mesure est du crosstalk. Rien n'y est jugé
par l'estimateur lui-même, contrairement au résiduel de `run_baselines.py`.

Trois jeux de données se rencontrent ici :

- le catalogue JPL `data/corner_reflectors/rosamond_<date>.csv`, une ligne par
  réflecteur avec latitude, longitude, azimut et longueur de côté ;
- le fichier `.llh`, qui donne latitude et longitude de **chaque pixel** de la
  grille 2x8 et sert à inverser les coordonnées géographiques en coordonnées
  radar ;
- les `.slc` eux-mêmes, lus par `numpy.memmap` autour de chaque réflecteur.
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.crosstalk.estimators import create_from_config  # noqa: E402
from src.crosstalk.estimators.base import CrosstalkEstimator  # noqa: E402
from src.crosstalk.exceptions import CrosstalkError  # noqa: E402
from src.crosstalk.metrics import corner_reflector_crosstalk  # noqa: E402
from src.crosstalk.parameters import CrosstalkParameters  # noqa: E402
from src.crosstalk.physics import POL_ORDER, inverse_distortion_matrix  # noqa: E402

logger = logging.getLogger("measure_reflectors")

#: Écart d'azimut toléré entre un réflecteur et le cap de l'avion. Le réseau de
#: Rosamond contient deux jeux orientés en sens opposés, un par direction de
#: vol : sans ce filtre, la moitié des réflecteurs ne renverrait que du bruit.
TOLERANCE_AZIMUT_DEG: float = 45.0

#: Demi-côté de la fenêtre de recherche du pic, en pixels 1x1. L'inversion par
#: le `.llh` tombe à 2-5 pixels près ; 60 laisse une marge confortable.
RECHERCHE_PIC: int = 60

#: Demi-largeur de la bande de distance sur laquelle `theta` est estimé, en
#: pixels. C'est la « stripe » de Fore et al. (2015, §V-B) : « a stripe of pixels
#: that lie within ten samples of the current range bin ».
DEMI_BANDE: int = 10

#: Hauteur des tranches d'azimut traitées d'un coup. La bande fait toute la
#: hauteur de la fauchée, soit 61 349 lignes : la découper borne la mémoire à
#: environ 50 Mo par tranche, au lieu de plusieurs gigaoctets d'un coup.
PAS_AZIMUT: int = 8192

#: Un trièdre a `|S_VV / S_HH| = 1` et une phase co-polaire nulle. Au-delà de
#: ces bornes, la cible n'est pas un trièdre — réflecteur masqué, déplacé, ou
#: cible parasite tombée dans la fenêtre de recherche.
RATIO_COPOL: tuple[float, float] = (0.7, 1.4)
PHASE_COPOL_DEG: float = 30.0


@dataclass(frozen=True, slots=True)
class Reflecteur:
    """Une ligne du catalogue JPL."""

    ident: str
    latitude: float
    longitude: float
    azimut: float
    cote: float


# ------------------------------------------------------------------- lectures


def lire_catalogue(chemin: Path) -> list[Reflecteur]:
    """Lit le `.csv` de JPL : identifiant, latitude, longitude, azimut, côté.

    Les sept colonnes sont celles que décrit Babu et al. (2022) dans sa
    *Data Availability Statement*. Le référentiel choisi au téléchargement —
    `instantaneous-crust`, `mean-crust` ou `tide-free` — n'est pas consigné
    dans l'en-tête, et il est sans conséquence : les trois diffèrent d'au plus
    30 cm, soit un dixième de pixel en distance.
    """
    reflecteurs = []
    with chemin.open(encoding="utf-8") as fichier:
        for ligne in csv.reader(fichier):
            if ligne and ligne[0].strip().isdigit():
                reflecteurs.append(
                    Reflecteur(
                        ident=ligne[0].strip(),
                        latitude=float(ligne[1]),
                        longitude=float(ligne[2]),
                        azimut=float(ligne[4]),
                        cote=float(ligne[6]),
                    )
                )
    return reflecteurs


def lire_annotation(chemin: Path) -> dict[str, float]:
    """Extrait du `.ann` les dimensions et le cap, nécessaires à la géométrie."""
    cles = {
        "lignes": r"slc_1_1x1 Rows\s+\(pixels\)\s+=\s+(\d+)",
        "colonnes": r"slc_1_1x1 Columns\s+\(pixels\)\s+=\s+(\d+)",
        "lignes_llh": r"llh_1_2x8 Rows\s+\(pixels\)\s+=\s+(\d+)",
        "colonnes_llh": r"llh_1_2x8 Columns\s+\(pixels\)\s+=\s+(\d+)",
        "cap": r"Peg Heading\s+\(deg\)\s+=\s+(-?[\d.]+)",
    }
    texte = chemin.read_text(encoding="latin-1")
    valeurs = {}
    for nom, motif in cles.items():
        trouve = re.search(motif, texte)
        if trouve is None:
            raise ValueError(f"clé introuvable dans {chemin.name} : {nom}")
        valeurs[nom] = float(trouve.group(1))
    return valeurs


# ------------------------------------------------------------- géolocalisation


def oriente_vers_le_radar(
    reflecteur: Reflecteur, cap_deg: float, tolerance: float = TOLERANCE_AZIMUT_DEG
) -> bool:
    """Le réflecteur fait-il face à l'avion ?

    Le réseau de Rosamond comporte deux jeux orientés à 180° l'un de l'autre,
    un par direction de vol. Sur l'acquisition du 11 février, les réflecteurs
    à 170-175° renvoient une amplitude de 0,1 à 1,5 quand ceux à 349-355°,
    alignés sur le cap de -10,03°, renvoient 23 à 83.
    """
    ecart = (reflecteur.azimut - cap_deg) % 360.0
    return min(ecart, 360.0 - ecart) <= tolerance


def position_radar(
    reflecteur: Reflecteur, latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[int, int] | None:
    """Coordonnées 1x1 approchées, par plus proche voisin dans la grille `.llh`.

    Le `.llh` porte la position géographique de chaque pixel de la grille 2x8 ;
    on y cherche le pixel le plus proche, puis on multiplie par le nombre de
    vues — 8 en azimut, 2 en distance — pour revenir en pleine résolution.

    Returns:
        Le couple `(ligne, colonne)`, ou `None` si le réflecteur tombe hors de
        la fauchée.
    """
    distance2 = (latitudes - reflecteur.latitude) ** 2 + (
        longitudes - reflecteur.longitude
    ) ** 2
    indice = np.unravel_index(np.argmin(distance2), distance2.shape)
    if np.sqrt(distance2[indice]) > 1e-3:  # ~100 m : hors fauchée
        return None
    return int(indice[0]) * 8, int(indice[1]) * 2


def affine_sur_le_pic(
    canaux: dict[str, np.memmap], ligne: int, colonne: int, demi: int = RECHERCHE_PIC
) -> tuple[int, int]:
    """Cherche le maximum de `|HH|` autour de la position approchée."""
    r0, c0 = max(0, ligne - demi), max(0, colonne - demi)
    bloc = np.abs(np.asarray(canaux["HH"][r0 : ligne + demi, c0 : colonne + demi]))
    decalage = np.unravel_index(np.argmax(bloc), bloc.shape)
    return r0 + int(decalage[0]), c0 + int(decalage[1])


def est_un_triedre(hh: complex, vv: complex) -> bool:
    """Contrôle de signature : `|VV/HH| ~ 1` et phase co-polaire ~ 0.

    Un trièdre répond `S = sqrt(sigma) I` (Fore et al., 2015, équation 5). Une
    cible qui s'en écarte n'est pas celle qu'on croit — réflecteur masqué ou
    déplacé, ou structure parasite tombée dans la fenêtre de recherche. Sur
    l'acquisition du 11 février, les cibles brillantes non réflectrices
    présentent `|VV/HH| ~ 0,2` et des phases de 150°.
    """
    if hh == 0:
        return False
    ratio = abs(vv) / abs(hh)
    phase = abs(np.degrees(np.angle(hh * np.conj(vv))))
    return RATIO_COPOL[0] < ratio < RATIO_COPOL[1] and phase < PHASE_COPOL_DEG


# ------------------------------------------------------------------- mesure


def theta_de_la_bande(
    canaux: dict[str, np.memmap],
    colonne: int,
    nombre_de_lignes: int,
    estimateur: CrosstalkEstimator,
) -> CrosstalkParameters | None:
    """Estime `theta` sur la bande de distance, et **jamais** sur le réflecteur seul.

    C'est le point qui rend la mesure non circulaire. Le lac sec de Rosamond
    rétrodiffuse si peu qu'un réflecteur porte 23 à 31 % de la puissance d'un
    patch 256x256 — dans un **seul pixel**. Estimer `theta` sur ce patch revient
    donc à l'ajuster sur le réflecteur, puis à s'en servir pour le corriger : la
    fuite tombe de 18 dB sans que rien ne l'ait justifié.

    Deux précautions lèvent cette circularité :

    - la **bande** de Fore et al. (2015, §V-B) : toute la hauteur de la fauchée
      plutôt qu'un patch, ce qui dilue le réflecteur dans 61 349 lignes ;
    - la **moyenne des `theta`** plutôt que de la covariance, qui donne le même
      poids à chaque fenêtre 5x5. Une covariance moyennée laisse au contraire un
      pixel brillant écraser l'estimation. C'est une justification de fond du
      protocole de Babu et al. (2022, §3.3.1) que ni lui ni Fore n'énoncent, et
      elle n'apparaît que sur une scène portant des cibles ponctuelles.

    La bande est traitée par tranches d'azimut pour borner la mémoire, et les
    `theta` des tranches sont moyennés. Les tranches étant de même hauteur, cela
    équivaut à la moyenne globale, à la dernière tranche près.

    Returns:
        Le `theta` de la bande, ou `None` si aucune tranche n'est exploitable.
    """
    c0, c1 = colonne - DEMI_BANDE, colonne + DEMI_BANDE + 1
    jeux: list[np.ndarray] = []
    for a0 in range(0, nombre_de_lignes, PAS_AZIMUT):
        a1 = min(a0 + PAS_AZIMUT, nombre_de_lignes)
        tranche = np.stack([np.asarray(canaux[pol][a0:a1, c0:c1]) for pol in POL_ORDER])
        try:
            jeux.append(estimateur.estimate(tranche).to_vector())
        except CrosstalkError as erreur:
            logger.debug("tranche %d-%d, colonne %d : %s", a0, a1, colonne, erreur)
    if not jeux:
        return None
    return CrosstalkParameters.from_vector(np.mean(jeux, axis=0))


def mesure_un_reflecteur(
    canaux: dict[str, np.memmap],
    ligne: int,
    colonne: int,
    estimateur: CrosstalkEstimator,
    nombre_de_lignes: int,
) -> dict[str, float] | None:
    """Fuite co-pol vers cross-pol au réflecteur, avant et après correction.

    Returns:
        Les deux mesures et leur différence, ou `None` si la signature n'est pas
        celle d'un trièdre ou si l'estimation échoue.
    """
    valeurs = np.array([canaux[pol][ligne, colonne] for pol in POL_ORDER])
    if not est_un_triedre(valeurs[0], valeurs[3]):
        return None
    avant = corner_reflector_crosstalk(*valeurs)

    theta = theta_de_la_bande(canaux, colonne, nombre_de_lignes, estimateur)
    if theta is None:
        return None
    try:
        sigma = inverse_distortion_matrix(theta)
    except CrosstalkError as erreur:
        logger.debug("réflecteur (%d, %d) : %s", ligne, colonne, erreur)
        return None

    apres = corner_reflector_crosstalk(*(sigma @ valeurs))
    if not (np.isfinite(avant) and np.isfinite(apres)):
        return None
    return {"avant_db": avant, "apres_db": apres, "reduction_db": avant - apres}


def mesure_une_acquisition(
    prefixe: str,
    catalogue: Path,
    racine_donnees: Path,
    estimateur: CrosstalkEstimator,
) -> list[dict[str, float | str]]:
    """Parcourt les réflecteurs orientés vers le radar pour une acquisition."""
    annotation = lire_annotation(racine_donnees / f"{prefixe}_L090HH_04_BC.ann")
    formes = (int(annotation["lignes"]), int(annotation["colonnes"]))
    canaux = {
        pol: np.memmap(
            racine_donnees / f"{prefixe}_L090{pol}_04_BC_s1_1x1.slc",
            dtype=np.complex64, mode="r", shape=formes,
        )
        for pol in POL_ORDER
    }
    llh = np.memmap(
        next(racine_donnees.glob("*.llh")), dtype=np.float32, mode="r",
        shape=(int(annotation["lignes_llh"]), int(annotation["colonnes_llh"]), 3),
    )
    latitudes, longitudes = np.asarray(llh[:, :, 0]), np.asarray(llh[:, :, 1])

    lignes: list[dict[str, float | str]] = []
    for reflecteur in lire_catalogue(catalogue):
        if not oriente_vers_le_radar(reflecteur, annotation["cap"]):
            continue
        approche = position_radar(reflecteur, latitudes, longitudes)
        if approche is None:
            continue
        ligne, colonne = affine_sur_le_pic(canaux, *approche)
        mesure = mesure_un_reflecteur(canaux, ligne, colonne, estimateur, formes[0])
        if mesure is None:
            logger.info("réflecteur %s écarté (signature ou estimation)", reflecteur.ident)
            continue
        lignes.append({"ident": reflecteur.ident, "cote": reflecteur.cote,
                       "ligne": ligne, "colonne": colonne, **mesure})
    return lignes


# ------------------------------------------------------------------- sorties


def resume(lignes: list[dict[str, float | str]], titre: str) -> str:
    """Tableau par réflecteur, à la manière de la figure 9 de Babu et al."""
    entete = (f"{'CR':>4} {'côté m':>7} {'ligne':>7} {'colonne':>8} "
              f"{'avant dB':>10} {'après dB':>10} {'réduction dB':>13}")
    sorties = [titre, entete, "-" * len(entete)]
    for ligne in sorted(lignes, key=lambda l: str(l["ident"])):
        sorties.append(
            f"{ligne['ident']:>4} {ligne['cote']:>7.2f} {ligne['ligne']:>7} "
            f"{ligne['colonne']:>8} {ligne['avant_db']:>10.2f} "
            f"{ligne['apres_db']:>10.2f} {ligne['reduction_db']:>13.2f}"
        )
    if lignes:
        avant = np.median([l["avant_db"] for l in lignes])
        apres = np.median([l["apres_db"] for l in lignes])
        sorties.append("-" * len(entete))
        sorties.append(f"{'médiane':>4} {'':>7} {'':>7} {'':>8} "
                       f"{avant:>10.2f} {apres:>10.2f} {avant - apres:>13.2f}"
                       f"   ({len(lignes)} réflecteurs)")
    return "\n".join(sorties)


# ----------------------------------------------------------------------- CLI


def main() -> int:
    """Mesure les cinq acquisitions et affiche un tableau par date."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    # Une bande de 8192 lignes porte 139 196 fenêtres 5x5 ; en écarter 150 pour
    # mauvais conditionnement est le régime normal, et `estimate` le signale à
    # chaque tranche. Soit 75 réflecteurs x 8 tranches de messages identiques,
    # qui noieraient les tableaux. On les laisse en DEBUG.
    logging.getLogger("src.crosstalk.estimators.quegan").setLevel(logging.ERROR)
    donnees = RACINE / "data" / "Rosamd_35012_04"
    catalogues = RACINE / "data" / "corner_reflectors"
    # `(5, 5)` et non `full` : voir `theta_de_la_bande`. Sur une scène à cibles
    # ponctuelles, moyenner la covariance laisse le réflecteur écraser `theta`.
    estimateur = create_from_config({"name": "quegan", "params": {"window": [5, 5]}})

    dates = {
        "Rosamd_35012_16002_001_160211": "rosamond_2016-02-11.csv",
        "Rosamd_35012_16052_001_160520": "rosamond_2016-05-20.csv",
        "Rosamd_35012_16053_001_160610": "rosamond_2016-06-10.csv",
        "Rosamd_35012_16060_009_160816": "rosamond_2016-08-16.csv",
        "Rosamd_35012_16074_005_160920": "rosamond_2016-09-20.csv",
    }
    toutes: list[dict[str, float | str]] = []
    for prefixe, catalogue in dates.items():
        lignes = mesure_une_acquisition(
            prefixe, catalogues / catalogue, donnees, estimateur
        )
        print("\n" + resume(lignes, f"=== {prefixe} ==="))
        toutes.extend(lignes)

    if toutes:
        avant = np.median([l["avant_db"] for l in toutes])
        apres = np.median([l["apres_db"] for l in toutes])
        print(f"\n=== Les cinq acquisitions, {len(toutes)} mesures ===")
        print(f"  fuite médiane avant correction : {avant:.2f} dB")
        print(f"  fuite médiane après correction : {apres:.2f} dB")
        print(f"  réduction                      : {avant - apres:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
