#!/usr/bin/env python3
"""
Évalue les baselines statistiques de crosstalk sur la pile UAVSAR.
Chaque patch et chaque méthode donnent une ligne de résultat : crosstalk estimé, crosstalk résiduel après correction, et statut — un patch dégénéré est consigné,pas propagé en exception.
"""

from __future__ import annotations
import argparse
import logging
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

#: Configuration lue quand `--config` n'est pas donné.
CONFIG_PAR_DEFAUT: Path = RACINE / "configs" / "data_uavsar.yaml"


#Méthodes
def build_estimators(entrees: Sequence[dict[str, Any]],) -> dict[str, CrosstalkEstimator]:
    """
    Instancie les méthodes déclarées dans `baselines.estimators`, indexées par leur nom.
    """
    methodes: dict[str, CrosstalkEstimator] = {}
    for entree in entrees:
        estimateur = create_from_config(entree) # QueganEstimator(window='full', tol=1e-08)
        if estimateur.name in methodes:
            raise ValueError(f"méthode déclarée deux fois : {estimateur.name!r}")
        methodes[estimateur.name] = estimateur #{'quegan': QueganEstimator(window='full', tol=1e-08)}
    return methodes


#Évaluation
def evaluate_patches(patches: Iterable[np.ndarray],estimators: dict[str, CrosstalkEstimator],) -> list[dict[str, Any]]:
    """Applique chaque méthode à chaque patch et rend les lignes de résultat.

    Cœur pur de l'évaluation : il ne connaît ni fichiers ni dataloader, ce qui
    le rend testable sur des patchs synthétiques.

    Args:
        patches: itérable de patchs `(4, H, W)` complexes, dans l'ordre du pipeline.
        estimators: méthodes à appliquer, indexées par leur nom.

    Returns:
        Une ligne par couple patch/méthode. Un patch sur lequel une méthode
        échoue donne une ligne de statut `ill_conditioned`, sans interrompre le parcours.
    """
    lignes: list[dict[str, Any]] = []
    for index, patch in enumerate(patches):
        for nom, estimateur in estimators.items():
            ligne: dict[str, Any] = {"name": nom, "status": "ok"}
            try:
                rapport = residual_crosstalk(patch, estimateur)
            except CrosstalkError as erreur:
                ligne["status"] = "ill_conditioned"
                logger.debug("patch %d, %s : %s", index, nom, erreur)
            else:
                ligne.update(rapport.as_row())
            lignes.append(ligne)
    #lignes[i] = {'name': 'quegan', 'status': 'ok', 'initial_db': 20.0, 'residual_db': -30.0, 'reduction_db': 50.0, 'alpha_abs': 1.0, 'reciprocity_before': 0.83, 'reciprocity_after': 0.86}
    #lignes[i+1] = {'name': 'quegan', 'status': 'ill_conditioned'}
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


#Configuration
def chemin_de_configuration(argument: str | None) -> Path:
    """Résout `--config` : absolu, relatif au dossier courant, ou à défaut à la racine.

    Le repli sur la racine du dépôt permet de lancer le script depuis n'importe
    quel dossier — `src/` compris — avec le chemin tel qu'il figure dans le README.
    """
    if argument is None:
        return CONFIG_PAR_DEFAUT
    chemin = Path(argument)
    if chemin.is_absolute() or chemin.exists():
        return chemin
    return RACINE / chemin


def lire_configuration(chemin: Path) -> dict[str, Any]:
    """Lit le YAML et ancre le chemin des données à la racine du dépôt.

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


#Données
def build_dataloader(config: dict[str, Any]):
    """
    Construit le chargeur de la partition de test, via `cvnn`.
    L'objet rendu est un `torch.utils.data.DataLoader`, qui ne contient aucune donnée. 
    Itéré — ce que fait `iter_patches` — il produit des lots`torch.Tensor` de forme `(batch_size, 4, H, W)` en `complex64`.
    """
    from cvnn.data import get_dataloaders, validate_and_correct_config
    from cvnn.utils import set_seed

    set_seed(int(config.get("seed", 0)))
    _, _, test_loader = get_dataloaders(validate_and_correct_config(config), use_cuda=False)
    return test_loader #Retourne un DataLoader 


def iter_patches(loader) -> Iterator[np.ndarray]:
    """
    Déplie les lots en patchs `(4, H, W)` numpy, dans l'ordre du pipeline.
    Générateur, et non liste : les patchs sont lus sur disque puis abandonnés au fil de l'eau.
    """
    for lot in loader:
        # UavsarSLC n'a pas d'étiquettes : `lot` est un Tensor, pas un (x, y).
        # La branche `else` s'applique donc, et `tenseur` vaut le lot entier, de forme (batch_size, 4, patch_size, patch_size).
        tenseur = lot[0] if isinstance(lot, (list, tuple)) else lot
        for patch in tenseur.detach().cpu().numpy():
            yield patch  # produit un patch de forme (4, patch_size, patch_size)



#CLI
def main(argv: list[str] | None = None) -> int:
    """Déroule la campagne : configuration, données, estimation, sorties."""
    parser = argparse.ArgumentParser(description="Baselines statistiques de crosstalk sur la partition de test.")
    parser.add_argument(
        "--config", default=None,
        help=f"configuration YAML (défaut : {CONFIG_PAR_DEFAUT.relative_to(RACINE)})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    #2026-09-23 10:01:26,381 - src.crosstalk.estimators.quegan - WARNING - 2520 fenêtres sur 63504 écartées
    #└──── %(asctime)s ────┘   └────────── %(name)s ─────────┘   └ %(levelname)s ┘   └──── %(message)s ────┘

    config = lire_configuration(chemin_de_configuration(args.config))
    baselines = config.get("baselines") or {}
    estimateurs = build_estimators(baselines["estimators"])
    logger.info("méthodes évaluées : %s", list(estimateurs))

    loader = build_dataloader(config)
    lignes = evaluate_patches(iter_patches(loader), estimateurs)

    print(summarize(lignes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
