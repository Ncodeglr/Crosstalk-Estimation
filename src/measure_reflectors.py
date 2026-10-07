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

Avec `--desequilibre`, le déséquilibre de canaux et les biais de phase sont
estimés puis corrigés **avant** le crosstalk, dans l'ordre de Babu et al.
(2022, §3.2 puis §3.3) : `f` et `phi_t + phi_r` sur les pics des trièdres,
`g` et `phi_t - phi_r` sur toute l'image. La colonne « avant » reste mesurée
sur les données brutes, comme la colonne *Uncalibrated* de la Table 4 de Babu ;
la colonne « après » porte alors les deux corrections enchaînées.
"""

from __future__ import annotations

import argparse
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
from src.crosstalk.imbalance import (  # noqa: E402
    ChannelImbalance,
    crosspol_imbalance,
    estimate_copol,
)
from src.crosstalk.metrics import corner_reflector_crosstalk  # noqa: E402
from src.crosstalk.parameters import CrosstalkParameters  # noqa: E402
from src.crosstalk.physics import POL_ORDER, apply_matrix, inverse_distortion_matrix  # noqa: E402

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

#: Hauteur des tranches pour l'estimation de `g` sur l'image entière. Une tranche
#: couvre alors toute la largeur de la fauchée, 9 874 colonnes : 2 048 lignes
#: font environ 160 Mo par canal, quand 8 192 en feraient 650.
PAS_AZIMUT_IMAGE: int = 2048

#: Un trièdre a `|S_VV / S_HH| = 1` et une phase co-polaire nulle. Au-delà de
#: ces bornes, la cible n'est pas un trièdre — réflecteur masqué, déplacé, ou
#: cible parasite tombée dans la fenêtre de recherche.
RATIO_COPOL: tuple[float, float] = (0.7, 1.4)
PHASE_COPOL_DEG: float = 30.0

#: Identifiants des réflecteurs retenus par Babu et al. (2022), tels que les liste
#: sa *Data Availability Statement*. Ce sont tous des trièdres de 2,4384 m
#: (8 pieds), la seule taille qu'il utilise (§2). Les CR 16 et 19, de même taille
#: et qui répondent aussi, sont exclus sans justification dans l'article. Son
#: §4.6 parle par ailleurs de « six » réflecteurs : l'article se contredit, on
#: suit la liste explicite.
REFLECTEURS_BABU: frozenset[str] = frozenset(
    {"13", "14", "15", "17", "18", "20", "21", "22"}
)


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


def est_retenu_par_babu(reflecteur: Reflecteur) -> bool:
    """Le réflecteur fait-il partie des huit que mesure Babu ?"""
    return reflecteur.ident in REFLECTEURS_BABU


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
    correction: np.ndarray | None = None,
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
    `theta` des tranches sont moyennés par `CrosstalkParameters.mean`, selon la
    même règle que les fenêtres d'un patch. Les tranches étant de même hauteur,
    cela équivaut à la moyenne globale, à la dernière tranche près.

    Args:
        correction: matrice diagonale du déséquilibre de canaux
            (`ChannelImbalance.correction_matrix`), appliquée à chaque tranche
            avant l'estimation ; `None` pour estimer sur les données brutes.

    Returns:
        Le `theta` de la bande, ou `None` si aucune tranche n'est exploitable ou
        si la bande déborde de l'image.
    """
    c0, c1 = colonne - DEMI_BANDE, colonne + DEMI_BANDE + 1
    # Un indice négatif ne lève pas d'erreur en Python : il compte depuis la fin
    # et lirait silencieusement une autre zone. Une bande tronquée, elle, ne
    # serait plus la bande de Fore. Le réflecteur est donc écarté, sans
    # interrompre la campagne.
    if c0 < 0 or c1 > canaux["HH"].shape[1]:
        logger.warning("colonne %d : la bande de ±%d colonnes déborde de l'image", colonne, DEMI_BANDE)
        return None
    jeux: list[np.ndarray] = []
    for a0 in range(0, nombre_de_lignes, PAS_AZIMUT):
        a1 = min(a0 + PAS_AZIMUT, nombre_de_lignes)
        tranche = np.stack([np.asarray(canaux[pol][a0:a1, c0:c1]) for pol in POL_ORDER])
        if correction is not None:
            tranche = apply_matrix(correction, tranche, channel_axis=0)
        try:
            jeux.append(estimateur.estimate(tranche).to_vector())
        except CrosstalkError as erreur:
            logger.debug("tranche %d-%d, colonne %d : %s", a0, a1, colonne, erreur)
    if not jeux:
        return None
    return CrosstalkParameters.mean(np.array(jeux))


def mesure_un_reflecteur(
    canaux: dict[str, np.memmap],
    ligne: int,
    colonne: int,
    estimateur: CrosstalkEstimator,
    nombre_de_lignes: int,
    correction: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Fuite co-pol vers cross-pol au réflecteur, avant et après correction.

    La signature de trièdre a déjà été contrôlée par `localise_les_reflecteurs`.

    Args:
        correction: matrice du déséquilibre de canaux, appliquée avant `Sigma`
            au réflecteur comme à la bande ; `None` pour la chaîne sans §3.2.
            La mesure « avant » se fait toujours sur les valeurs brutes.

    Returns:
        Les deux mesures, leur différence et le `|alpha|` de la bande, ou `None`
        si l'estimation échoue.
    """
    valeurs = np.array([canaux[pol][ligne, colonne] for pol in POL_ORDER])
    avant = corner_reflector_crosstalk(*valeurs)

    theta = theta_de_la_bande(canaux, colonne, nombre_de_lignes, estimateur, correction)
    if theta is None:
        return None
    try:
        sigma = inverse_distortion_matrix(theta)
    except CrosstalkError as erreur:
        logger.debug("réflecteur (%d, %d) : %s", ligne, colonne, erreur)
        return None

    if correction is not None:
        valeurs = correction @ valeurs
    apres = corner_reflector_crosstalk(*(sigma @ valeurs))
    if not (np.isfinite(avant) and np.isfinite(apres)):
        return None
    return {"avant_db": avant, "apres_db": apres, "reduction_db": avant - apres,
            "alpha_abs": abs(theta.alpha)}


def localise_les_reflecteurs(
    catalogue: Path,
    canaux: dict[str, np.memmap],
    annotation: dict[str, float],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> list[tuple[Reflecteur, int, int]]:
    """Réflecteurs de Babu orientés vers le radar, avec la position de leur pic.

    Séparée de la mesure parce que `f` s'estime sur ces pics : il faut les
    connaître tous avant de corriger quoi que ce soit. Seuls les pics à
    signature de trièdre sont rendus, pour que ni `f` ni la mesure ne portent
    sur une cible parasite.

    Returns:
        Une liste de `(reflecteur, ligne, colonne)`, en pixels 1x1.
    """
    trouves: list[tuple[Reflecteur, int, int]] = []
    for reflecteur in lire_catalogue(catalogue):
        if not oriente_vers_le_radar(reflecteur, annotation["cap"]):
            continue
        if not est_retenu_par_babu(reflecteur):
            continue
        approche = position_radar(reflecteur, latitudes, longitudes)
        if approche is None:
            continue
        ligne, colonne = affine_sur_le_pic(canaux, *approche)
        if not est_un_triedre(canaux["HH"][ligne, colonne], canaux["VV"][ligne, colonne]):
            logger.info("réflecteur %s écarté (signature)", reflecteur.ident)
            continue
        trouves.append((reflecteur, ligne, colonne))
    return trouves


def desequilibre_de_l_acquisition(
    canaux: dict[str, np.memmap],
    pics: list[tuple[Reflecteur, int, int]],
    nombre_de_lignes: int,
) -> ChannelImbalance:
    """Babu §3.2 sur une acquisition : `f` aux pics, `g` sur l'image entière.

    Les moyennes `<|O_hv|^2>`, `<|O_vh|^2>` et `<O_hv O_vh*>` sont accumulées
    par tranches d'azimut, puis passées à `crosspol_imbalance` : l'image fait
    environ 4,8 Go par canal et ne tient pas en mémoire d'un bloc. C'est le choix
    de Fore et al. (2015, §IV-B), qui moyenne sur l'image entière ; Babu parle
    de cibles homogènes sans préciser lesquelles.
    """
    hh = np.array([canaux["HH"][ligne, colonne] for _, ligne, colonne in pics])
    vv = np.array([canaux["VV"][ligne, colonne] for _, ligne, colonne in pics])
    f, phi_somme = estimate_copol(hh, vv)

    somme_hv = somme_vh = 0.0
    correlation = 0j
    for a0 in range(0, nombre_de_lignes, PAS_AZIMUT_IMAGE):
        a1 = min(a0 + PAS_AZIMUT_IMAGE, nombre_de_lignes)
        hv = np.asarray(canaux["HV"][a0:a1])
        vh = np.asarray(canaux["VH"][a0:a1])
        somme_hv += float(np.sum(np.abs(hv) ** 2, dtype=np.float64))
        somme_vh += float(np.sum(np.abs(vh) ** 2, dtype=np.float64))
        correlation += complex(np.sum(hv * np.conj(vh), dtype=np.complex128))
    nombre = canaux["HV"].size
    g, phi_difference = crosspol_imbalance(
        somme_hv / nombre, somme_vh / nombre, correlation / nombre
    )
    return ChannelImbalance.from_estimates(f, phi_somme, g, phi_difference)


def mesure_une_acquisition(
    prefixe: str,
    catalogue: Path,
    racine_donnees: Path,
    estimateur: CrosstalkEstimator,
    desequilibre: bool = False,
) -> tuple[list[dict[str, float | str]], ChannelImbalance | None]:
    """Mesure les réflecteurs de Babu orientés vers le radar pour une acquisition.

    Returns:
        Une ligne par réflecteur mesuré, et le déséquilibre estimé si
        `desequilibre` est vrai, `None` sinon.
    """
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

    pics = localise_les_reflecteurs(catalogue, canaux, annotation, latitudes, longitudes)
    jeu = desequilibre_de_l_acquisition(canaux, pics, formes[0]) if desequilibre and pics else None
    correction = None if jeu is None else jeu.correction_matrix()

    lignes: list[dict[str, float | str]] = []
    for reflecteur, ligne, colonne in pics:
        mesure = mesure_un_reflecteur(
            canaux, ligne, colonne, estimateur, formes[0], correction
        )
        if mesure is None:
            logger.info("réflecteur %s écarté (estimation)", reflecteur.ident)
            continue
        lignes.append({"ident": reflecteur.ident, "cote": reflecteur.cote,
                       "ligne": ligne, "colonne": colonne, **mesure})
    return lignes, jeu


# ------------------------------------------------------------------- sorties


def agrege(lignes: list[dict[str, float | str]], fonction) -> tuple[float, float, float]:
    """`(avant, après, |alpha|)` agrégés sur les réflecteurs par `fonction`.

    Babu et al. (2022, §4.6) rapportent une *average crosstalk* par date, sans
    préciser si la moyenne porte sur les valeurs en dB ou sur les puissances.
    On moyenne les dB, la lecture la plus directe d'une moyenne annoncée en dB et
    cohérente avec ses minimum et maximum, eux aussi donnés en dB.
    """
    return tuple(
        float(fonction([l[cle] for l in lignes]))
        for cle in ("avant_db", "apres_db", "alpha_abs")
    )


def resume(lignes: list[dict[str, float | str]], titre: str) -> str:
    """Tableau par réflecteur, à la manière de la figure 9 de Babu et al."""
    entete = (f"{'CR':>4} {'côté m':>7} {'ligne':>7} {'colonne':>8} "
              f"{'avant dB':>10} {'après dB':>10} {'réduction dB':>13} {'|alpha|':>8}")
    sorties = [titre, entete, "-" * len(entete)]
    for ligne in sorted(lignes, key=lambda l: str(l["ident"])):
        sorties.append(
            f"{ligne['ident']:>4} {ligne['cote']:>7.2f} {ligne['ligne']:>7} "
            f"{ligne['colonne']:>8} {ligne['avant_db']:>10.2f} "
            f"{ligne['apres_db']:>10.2f} {ligne['reduction_db']:>13.2f} "
            f"{ligne['alpha_abs']:>8.3f}"
        )
    if lignes:
        sorties.append("-" * len(entete))
        # La moyenne est l'agrégat de Babu ; la médiane, robuste à un réflecteur
        # atypique comme le CR 15, sert de contrôle.
        for nom, fonction in (("moyenne", np.mean), ("médiane", np.median)):
            avant, apres, alpha = agrege(lignes, fonction)
            sorties.append(f"{nom:>20} {'':>8} "
                           f"{avant:>10.2f} {apres:>10.2f} {avant - apres:>13.2f} "
                           f"{alpha:>8.3f}")
        sorties.append(f"({len(lignes)} réflecteurs)")
    return "\n".join(sorties)


# ----------------------------------------------------------------------- CLI


def resume_desequilibre(jeu: ChannelImbalance) -> str:
    """Les quatre paramètres de Babu §3.2, avec l'écart à l'idéal de sa Table 3."""
    return (
        f"  déséquilibre (Babu §3.2) : f = {jeu.f:.4f} (écart {jeu.f - 1:+.4f})   "
        f"g = {jeu.g:.4f} (écart {jeu.g - 1:+.4f})\n"
        f"  phases : phi_t + phi_r = {np.degrees(jeu.phi_t + jeu.phi_r):+.2f} deg   "
        f"phi_t - phi_r = {np.degrees(jeu.phi_t - jeu.phi_r):+.2f} deg"
    )


def main(argv: list[str] | None = None) -> int:
    """Mesure les cinq acquisitions et affiche un tableau par date."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--desequilibre", action="store_true",
        help="corriger le déséquilibre de canaux avant le crosstalk (Babu §3.2)",
    )
    args = parser.parse_args(argv)
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
        lignes, jeu = mesure_une_acquisition(
            prefixe, catalogues / catalogue, donnees, estimateur, args.desequilibre
        )
        print("\n" + resume(lignes, f"=== {prefixe} ==="))
        if jeu is not None:
            print(resume_desequilibre(jeu))
        toutes.extend(lignes)

    if toutes:
        print(f"\n=== Les cinq acquisitions, {len(toutes)} mesures ===")
        print(f"  {'':>9} {'avant dB':>9} {'après dB':>9} {'réduction dB':>13}")
        for nom, fonction in (("moyenne", np.mean), ("médiane", np.median)):
            avant, apres, _ = agrege(toutes, fonction)
            print(f"  {nom:>9} {avant:>9.2f} {apres:>9.2f} {avant - apres:>13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
