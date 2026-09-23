#!/usr/bin/env python3
"""Évalue les baselines statistiques de crosstalk sur la pile UAVSAR.

Orchestration seulement : les données viennent du pipeline existant
(`torchcvnn.datasets.SLCDataset` puis `cvnn.data.get_dataloaders`), les méthodes
du registre `crosstalk.estimators`, et les métriques de `crosstalk.metrics`. Ce
fichier ne contient ni physique ni estimation.

    python src/run_baselines.py

Aucune option de ligne de commande : tous les réglages vivent dans
`configs/data_uavsar.yaml`. `data.max_num_samples` limite le nombre de patchs
pour une mise au point rapide, `baselines.split` choisit la partition et
`baselines.output` la destination du CSV.

Chaque patch et chaque méthode donnent une ligne de CSV : crosstalk estimé,
crosstalk résiduel après correction, et statut — un patch dégénéré est consigné,
pas propagé en exception.
"""

from __future__ import annotations

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import yaml

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "cvnn" / "src"))

from src.crosstalk.estimators import create_from_config  # noqa: E402
from src.crosstalk.estimators.base import CrosstalkEstimator  # noqa: E402
from src.crosstalk.exceptions import CrosstalkError  # noqa: E402
from src.crosstalk.metrics import residual_crosstalk  # noqa: E402

logger = logging.getLogger("run_baselines")

#: Colonnes du CSV, dans l'ordre.
COLONNES = [
    "patch",
    "label",
    "method",
    "status",
    "initial_db",
    "residual_db",
    "reduction_db",
    "alpha_abs",
    "alpha_deg",
    "u_db",
    "v_db",
    "w_db",
    "z_db",
]


# méthodes

def build_estimators(entrees: Sequence[dict[str, Any]],) -> dict[str, CrosstalkEstimator]:
    """Instancie les méthodes déclarées dans `baselines.estimators`.

    L'étiquette d'une entrée vaut son `label`, ou le nom de la méthode s'il est
    absent.

    Raises:
        ValueError: si deux entrées portent la même étiquette.
        KeyError: si `name` manque ou n'est pas un nom enregistré.
    """
    methodes: dict[str, CrosstalkEstimator] = {}
    for entree in entrees:
        estimateur = create_from_config(entree) # QueganEstimator(window='full', alpha_phase='fore', …
        label = entree.get("label", estimateur.name)
        if label in methodes:
            raise ValueError(f"étiquette dupliquée dans la configuration : {label!r}")
        methodes[label] = estimateur #{'quegan_patch': QueganEstimator(window='full', alpha_phase='fore', …)}
    return methodes


# évaluation

def evaluate_patches(
    patches: Iterable[np.ndarray],
    estimators: dict[str, CrosstalkEstimator],
) -> list[dict[str, Any]]:
    """Applique chaque méthode à chaque patch et rend les lignes de résultat.

    Cœur pur de l'évaluation : il ne connaît ni fichiers ni dataloader, ce qui
    le rend testable sur des patchs synthétiques.

    Args:
        patches: itérable de patchs `(4, H, W)` complexes, dans l'ordre du
            pipeline.
        estimators: méthodes à appliquer, indexées par leur étiquette.

    Returns:
        Une ligne par couple patch/méthode. Un patch sur lequel une méthode
        échoue donne une ligne de statut `ill_conditioned`, sans interrompre le
        parcours.
    """
    lignes: list[dict[str, Any]] = []
    for index, patch in enumerate(patches):
        for label, estimateur in estimators.items():
            ligne: dict[str, Any] = {
                "patch": index,
                "label": label,
                "method": estimateur.name,
                "status": "ok",
            }
            try:
                rapport = residual_crosstalk(patch, estimateur)
            except CrosstalkError as erreur:
                ligne["status"] = "ill_conditioned"
                logger.debug("patch %d, %s : %s", index, label, erreur)
            else:
                ligne.update(rapport.as_row())
                ligne.update(
                    {
                        f"{nom}_db": valeur
                        for nom, valeur in rapport.initial.crosstalk_db().items()
                    }
                )
            lignes.append(ligne)
    return lignes


def summarize(lignes: Sequence[dict[str, Any]]) -> str:
    """Tableau de synthèse, une ligne par méthode."""
    labels = sorted({ligne["label"] for ligne in lignes})
    entete = (
        f"{'méthode':>16} {'patchs':>7} {'échecs':>7} "
        f"{'initial dB':>11} {'résiduel dB':>12} {'réduction dB':>13}"
    )
    sorties = [entete, "-" * len(entete)]
    for label in labels:
        groupe = [ligne for ligne in lignes if ligne["label"] == label]
        valides = [ligne for ligne in groupe if ligne["status"] == "ok"]
        if not valides:
            sorties.append(f"{label:>16} {len(groupe):>7} {len(groupe):>7}"
                           f" {'-':>11} {'-':>12} {'-':>13}")
            continue
        mediane = {
            cle: float(np.median([ligne[cle] for ligne in valides]))
            for cle in ("initial_db", "residual_db", "reduction_db")
        }
        sorties.append(
            f"{label:>16} {len(groupe):>7} {len(groupe) - len(valides):>7} "
            f"{mediane['initial_db']:>11.1f} {mediane['residual_db']:>12.1f} "
            f"{mediane['reduction_db']:>13.1f}"
        )
    return "\n".join(sorties)


def write_csv(lignes: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Écrit les lignes, en créant le répertoire au besoin."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as fichier:
        redacteur = csv.DictWriter(fichier, fieldnames=COLONNES, extrasaction="ignore")
        redacteur.writeheader()
        redacteur.writerows(lignes)


# -------------------------------------------------------------------- données


def _seed_global_rngs(seed: int) -> None:
    """Rend le découpage train/valid/test reproductible.

    `cvnn.data` découpe les indices avec `torch.randperm` et `random.shuffle`,
    qui tirent sur les générateurs **globaux** ; la clé `seed` de la
    configuration ne leur est jamais transmise. Sans cette fonction, deux
    exécutions de ce script évaluent deux partitions de test différentes —
    vérifié : trois appels successifs rendent trois premiers lots distincts.

    C'est rédhibitoire pour le protocole de comparaison : la partition de test
    des baselines doit être exactement celle sur laquelle le réseau est évalué,
    et surtout ne pas recouper celle sur laquelle il a été entraîné.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloader(config: dict[str, Any], split: str = "test"):
    """Construit le dataloader via `cvnn`, sans réécrire de pipeline.

    La configuration est passée telle quelle au framework : c'est le même objet
    qui décrit les données pour les baselines et pour le réseau, ce qui garantit
    une partition de test identique des deux côtés.

    Args:
        config: la configuration complète du projet.
        split: partition à rendre — `train`, `valid` ou `test`.

    Returns:
        Le `DataLoader` de la partition demandée.

    Raises:
        ValueError: si la partition demandée n'existe pas.
    """
    from cvnn.data import get_dataloaders, validate_and_correct_config

    _seed_global_rngs(int(config.get("seed", 0)))
    loaders = dict(
        zip(
            ("train", "valid", "test"),
            get_dataloaders(validate_and_correct_config(config), use_cuda=False),
        )
    )
    if split not in loaders:
        raise ValueError(f"split {split!r} inconnu : {sorted(loaders)}")
    return loaders[split]


def iter_patches(loader) -> Iterator[np.ndarray]:
    """Déplie les lots en patchs `(4, H, W)` numpy, dans l'ordre du pipeline.

    Générateur, et non liste : les patchs sont lus sur disque puis abandonnés au
    fil de l'eau. Une partition de test entière tiendrait sinon plusieurs
    dizaines de gigaoctets en mémoire.
    """
    for lot in loader:
        # Un dataset étiqueté rend (x, y) ; UavsarSLC rend le tenseur seul.
        tenseur = lot[0] if isinstance(lot, (list, tuple)) else lot
        for patch in tenseur.detach().cpu().numpy():
            yield patch


# ------------------------------------------------------------------------ CLI


def main() -> int:
    """Déroule la campagne : configuration, données, estimation, sorties."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    #2026-09-23 10:01:26,381 - src.crosstalk.estimators.quegan - WARNING - 2520 fenêtres sur 63504 écartées
    #└──── %(asctime)s ────┘   └────────── %(name)s ─────────┘   └ %(levelname)s ┘   └──── %(message)s ────┘

    config = yaml.safe_load(Path("configs/data_uavsar.yaml").read_text(encoding="utf-8"))
    baselines = config.get("baselines") or {}
    estimateurs = build_estimators(baselines["estimators"])
    logger.info("méthodes évaluées : %s", list(estimateurs))

    loader = build_dataloader(config, baselines.get("split", "test"))
    lignes = evaluate_patches(iter_patches(loader), estimateurs)

    destination = baselines.get("output", "experiments/baselines.csv")
    write_csv(lignes, destination)
    logger.info("%d lignes écrites dans %s", len(lignes), destination)
    print(summarize(lignes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
