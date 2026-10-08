#!/usr/bin/env python3
"""
Évalue les baselines statistiques de crosstalk sur la pile UAVSAR.
Chaque patch et chaque méthode donnent une ligne de résultat : crosstalk estimé, crosstalk résiduel après correction, et statut — un patch dégénéré est consigné,pas propagé en exception.
Le script affiche la médiane de ces lignes, par méthode.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.crosstalk.estimators import create_from_config  # noqa: E402
from src.crosstalk.estimators.estimator import CrosstalkEstimator  # noqa: E402
from src.crosstalk.exceptions import CrosstalkError  # noqa: E402
from src.crosstalk.metrics import residual_crosstalk  # noqa: E402
from src.patchs import build_dataloader, iter_patches  # noqa: E402

logger = logging.getLogger("run_baselines")

#: Configuration de la chaîne A : pour changer un réglage, modifier ce fichier.
CONFIG: Path = RACINE / "configs" / "data_uavsar.yaml"


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


def evaluate_patches(patches: Iterable[np.ndarray],estimators: dict[str, CrosstalkEstimator],) -> list[dict[str, Any]]:
    """
    Applique chaque méthode à chaque patch et rend les lignes de résultat.
    Cœur pur de l'évaluation : il ne connaît ni fichiers ni dataloader, ce qui le rend testable sur des patchs synthétiques.

    Args:
        patches: itérable de patchs (4, H, W) complexes, dans l'ordre du pipeline.
        estimators: méthodes à appliquer, indexées par leur nom.

    Returns:
        Une ligne par couple patch/méthode. Un patch sur lequel une méthode échoue donne une ligne de statut ill_conditioned, sans interrompre le parcours.
    """
    lignes: list[dict[str, Any]] = []
    for rang, patch in enumerate(patches): #enumerate numérote les éléments d'un itérable au fil du parcours, à partir de 0. Il rend des couples (numéro, élément)
        for nom, estimateur in estimators.items(): #.items rend les couples (clé, valeur) d'un dictionnaire. Ici : ('quegan', QueganEstimator(name='quegan', tol=1e-08))
            ligne: dict[str, Any] = {"name": nom, "status": "ok"}
            try:
                bilan_correction = residual_crosstalk(patch, estimateur)  # Return une instance de ResidualReport : theta initial, theta résiduel, réciprocité avant/après
            except CrosstalkError as erreur:
                ligne["status"] = "ill_conditioned"
                logger.debug("patch n° %d, %s : %s", rang, nom, erreur)
                #lignes[i+1] = {'name': 'quegan', 'status': 'ill_conditioned'}
            else:
                ligne.update(bilan_correction.as_row())
            lignes.append(ligne)
            #lignes[i] = {'name': 'quegan', 'status': 'ok', 'initial_db': -20.0, ..., 'reciprocity_after': 0.79}
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
    """Déroule la campagne : configuration, données, estimation, synthèse."""
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

    print(summarize(lignes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
