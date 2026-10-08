#!/usr/bin/env python3
"""
Évalue les baselines statistiques de crosstalk sur la pile UAVSAR.
Chaque patch et chaque méthode donnent une ligne de résultat : crosstalk estimé, crosstalk résiduel après correction, et statut — un patch dégénéré est consigné,pas propagé en exception.
Les lignes sont écrites à chaque lancement dans `experiments/baselines.csv`, avec la position du patch dans la pile.
"""

from __future__ import annotations
import csv
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.crosstalk.estimators import create_from_config  # noqa: E402
from src.crosstalk.estimators.base import CrosstalkEstimator  # noqa: E402
from src.crosstalk.exceptions import CrosstalkError  # noqa: E402
from src.crosstalk.metrics import residual_crosstalk  # noqa: E402
from src.patchs import PositionPatch, build_dataloader, iter_patches  # noqa: E402

logger = logging.getLogger("run_baselines")

#: Configuration de la chaîne A : pour changer un réglage, modifier ce fichier.
CONFIG: Path = RACINE / "configs" / "data_uavsar.yaml"

#: Fichier de résultats, réécrit à chaque lancement. `experiments/` est ignoré par git.
SORTIE: Path = RACINE / "experiments" / "baselines.csv"

#: Colonnes du fichier, dans l'ordre. Une ligne qui porterait un autre champ fait échouer l'écriture : rien n'est calculé puis jeté.
COLONNES: tuple[str, ...] = (
    "acquisition", "ligne", "colonne", "name", "status",
    "initial_db", "residual_db", "reduction_db", "alpha_abs",
    "reciprocity_before", "reciprocity_after",
    "u_re", "u_im", "v_re", "v_im", "w_re", "w_im", "z_re", "z_im",
    "alpha_re", "alpha_im",
)


#Méthodes
def build_estimators(entrees: Sequence[dict[str, Any]],) -> dict[str, CrosstalkEstimator]:
    """
    Instancie les méthodes déclarées dans baselines.estimators, indexées par leur nom.
    """
    methodes: dict[str, CrosstalkEstimator] = {}
    for entree in entrees:
        estimateur = create_from_config(entree) # QueganEstimator(name='quegan', tol=1e-08)
        if estimateur.name in methodes:
            raise ValueError(f"méthode déclarée deux fois : {estimateur.name!r}")
        methodes[estimateur.name] = estimateur #{'quegan': QueganEstimator(name='quegan', tol=1e-08)}
    return methodes


#Évaluation
def evaluate_patches(patches: Iterable[tuple[PositionPatch, np.ndarray]],estimators: dict[str, CrosstalkEstimator],) -> list[dict[str, Any]]:
    """
    Applique chaque méthode à chaque patch et rend les lignes de résultat.
    Cœur pur de l'évaluation : il ne connaît ni fichiers ni dataloader, ce qui le rend testable sur des patchs synthétiques.

    Args:
        patches: itérable de couples `(position, patch)`, le patch (4, H, W) complexe dans l'ordre du pipeline.
        estimators: méthodes à appliquer, indexées par leur nom.

    Returns:
        Une ligne par couple patch/méthode, qui commence par la position du
        patch. Un patch sur lequel une méthode échoue donne une ligne de statut ill_conditioned, sans interrompre le parcours.
    """
    lignes: list[dict[str, Any]] = []
    for position, patch in patches:
        for nom, estimateur in estimators.items():
            ligne: dict[str, Any] = {**asdict(position), "name": nom, "status": "ok"}
            try:
                rapport = residual_crosstalk(patch, estimateur)
            except CrosstalkError as erreur:
                ligne["status"] = "ill_conditioned"
                logger.debug("patch %s, %s : %s", position, nom, erreur)
            else:
                ligne.update(rapport.as_row())
            lignes.append(ligne)
    #lignes[i] = {'acquisition': 'Rosamd_..._160211_L_090_04_BC', 'ligne': 30208, 'colonne': 4608, 'name': 'quegan', 'status': 'ok', 'initial_db': -20.0, ..., 'alpha_im': -0.03}
    #lignes[i+1] = {'acquisition': ..., 'ligne': ..., 'colonne': ..., 'name': 'quegan', 'status': 'ill_conditioned'}
    return lignes


def summarize(lignes: Sequence[dict[str, Any]]) -> str:
    """
    Tableau de synthèse, une ligne par méthode.
    """
    noms = sorted({ligne["name"] for ligne in lignes})
    entete = (
        f"{'méthode':>16} {'patchs':>7} {'échecs':>7} "
        f"{'initial dB':>11} {'résiduel dB':>12} {'réduction dB':>13} "
        f"{'|alpha|':>8} {'récip. avant':>13} {'récip. après':>13}"
    )
    sorties = [entete, "-" * len(entete)]
    for nom in noms:
        groupe = [ligne for ligne in lignes if ligne["name"] == nom]
        valides = [ligne for ligne in groupe if ligne["status"] == "ok"]
        if not valides:
            sorties.append(f"{nom:>16} {len(groupe):>7} {len(groupe):>7}"
                           f" {'-':>11} {'-':>12} {'-':>13} {'-':>8}"
                           f" {'-':>13} {'-':>13}")
            continue
        mediane = {
            cle: float(np.median([ligne[cle] for ligne in valides]))
            for cle in ("initial_db", "residual_db", "reduction_db", "alpha_abs",
                        "reciprocity_before", "reciprocity_after")
        }
        sorties.append(
            f"{nom:>16} {len(groupe):>7} {len(groupe) - len(valides):>7} "
            f"{mediane['initial_db']:>11.1f} {mediane['residual_db']:>12.1f} "
            f"{mediane['reduction_db']:>13.1f} {mediane['alpha_abs']:>8.3f} "
            f"{mediane['reciprocity_before']:>13.3f} {mediane['reciprocity_after']:>13.3f}"
        )
    return "\n".join(sorties)


def ecrire_resultats(lignes: Sequence[dict[str, Any]], chemin: Path) -> None:
    """
    Écrit les lignes en CSV, colonnes dans l'ordre de `COLONNES`, en écrasant le fichier.
    Une ligne `ill_conditioned` laisse ses cellules de mesure vides. 
    Un champ absent de `COLONNES` lève `ValueError` : le fichier porte tout ce que les lignes contiennent.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES, restval="")
        ecrivain.writeheader()
        ecrivain.writerows(lignes)


#Configuration
def lire_YAML(chemin: Path) -> dict[str, Any]:
    """
    Lit le YAML et ancre le chemin des données à la racine du dépôt.

    `trainpath` est écrit relatif (`data/Rosamd_35012_04`) pour que la
    configuration reste valable d'une machine à l'autre. Sans cet ancrage, `cvnn`
    le résoudrait depuis le dossier courant et ne trouverait pas les `.slc` dès que
    le script est lancé ailleurs qu'à la racine.
    """
    config = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    dataset = config["data"]["dataset"]
    if not Path(dataset["trainpath"]).is_absolute():
        dataset["trainpath"] = str(RACINE / dataset["trainpath"])
    return config


#---------------------------------Main------------------------------------------
def main() -> int:
    """Déroule la campagne : configuration, données, estimation, sorties."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        #2026-09-23 10:01:26,381 - run_baselines - INFO - méthodes évaluées : ['quegan']
    )
   
    config = lire_YAML(CONFIG)
    baselines = config.get("baselines") or {}
    estimateurs = build_estimators(baselines["estimators"])
    logger.info("méthodes évaluées : %s", list(estimateurs))

    loader = build_dataloader(config)
    lignes = evaluate_patches(iter_patches(loader), estimateurs)

    ecrire_resultats(lignes, SORTIE)
    print(summarize(lignes))
    logger.info("résultats par patch écrits dans %s", SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
