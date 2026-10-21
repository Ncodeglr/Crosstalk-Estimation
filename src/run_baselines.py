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

#Colonnes du tableau de synthèse : clé du bilan, titre, largeur, format.
COLONNES: tuple[tuple[str, str, int, str], ...] = (
    ("patchs", "patchs", 7, "d"),
    ("echecs", "échecs", 7, "d"),
    ("initial_db", "initial dB", 11, ".1f"),
    ("residual_db", "résiduel dB", 12, ".1f"),
    ("reduction_db", "réduction dB", 13, ".1f"),
    ("alpha_abs", "|alpha|", 8, ".3f"),
    ("reciprocity_before", "reciprocity_before", 13, ".3f"),
    ("reciprocity_after", "reciprocity_after", 13, ".3f"),
)



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


def aggregate(lignes: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Bilan par méthode : nombre de patchs, nombre d'échecs, et médiane de chaque grandeur sur les patchs exploitables.
    Les grandeurs sont celles que portent les lignes (`ResidualReport.as_row`) : en ajouter une là-bas l'agrège ici sans retouche.
    La médiane est celle de numpy : un nan dans une colonne la rend nan, ce qui se voit dans le tableau au lieu d'être écarté en silence.

    Args:
        lignes: les lignes de evaluate_patches.

    Returns:
        {méthode: {grandeur: valeur}}, méthodes triées par nom. Une méthode en échec sur tous ses patchs n'a que ses deux compteurs, `patchs` et `echecs`.
    """
    groupes: dict[str, list[dict[str, Any]]] = {}
    for ligne in lignes:
        groupes.setdefault(ligne["name"], []).append(ligne) # un seul parcours : {'quegan': [ligne_0, ligne_1, ...]}
    bilan: dict[str, dict[str, float]] = {}
    for nom in sorted(groupes):
        groupe = groupes[nom]
        valides = [ligne for ligne in groupe if ligne["status"] == "ok"]
        resume: dict[str, float] = {"patchs": len(groupe), "echecs": len(groupe) - len(valides)}
        if valides:
            grandeurs = [cle for cle in valides[0] if cle not in ("name", "status")]
            resume.update({cle: float(np.median([ligne[cle] for ligne in valides])) for cle in grandeurs})
        bilan[nom] = resume
        #bilan['quegan'] = {'patchs': 4, 'echecs': 1, 'initial_db': -19.3, ..., 'reciprocity_after': 1.0}
    return bilan




def summarize(bilan: dict[str, dict[str, float]]) -> str:
    """
    Met le bilan de `aggregate` en tableau, une ligne par méthode.
    Mise en forme seule : aucun calcul ici. Une grandeur absente du bilan s'affiche `-`.
    """
    entete = " ".join([f"{'méthode':>16}"] + [f"{titre:>{largeur}}" for _, titre, largeur, _ in COLONNES])
    sorties = [entete, "-" * len(entete)]
    for nom, resume in bilan.items():
        cellules = [f"{nom:>16}"]
        for cle, _, largeur, fmt in COLONNES:
            valeur = resume.get(cle)
            cellules.append(f"{'-':>{largeur}}" if valeur is None else f"{valeur:>{largeur}{fmt}}")
        sorties.append(" ".join(cellules))
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

    print(summarize(aggregate(lignes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
