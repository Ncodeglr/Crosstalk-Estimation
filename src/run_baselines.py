#!/usr/bin/env python3
"""
Évalue les baselines statistiques de crosstalk sur la pile UAVSAR.
Chaque patch et chaque méthode donnent une ligne de résultat : crosstalk estimé, crosstalk résiduel après correction, et statut — un patch dégénéré est consigné,pas propagé en exception.
"""

from __future__ import annotations
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


#Méthodes
def build_estimators(entrees: Sequence[dict[str, Any]],) -> dict[str, CrosstalkEstimator]:
    """
    Instancie les méthodes déclarées dans `baselines.estimators`.
    L'étiquette d'une entrée vaut son `label`, ou le nom de la méthode s'il est absent.
    """
    methodes: dict[str, CrosstalkEstimator] = {}
    for entree in entrees:
        estimateur = create_from_config(entree) # QueganEstimator(window='full', tol=1e-08)
        label = entree.get("label", estimateur.name)
        if label in methodes:
            raise ValueError(f"étiquette dupliquée dans la configuration : {label!r}")
        methodes[label] = estimateur #{'quegan_patch': QueganEstimator(window='full', tol=1e-08)}
    return methodes


#Évaluation
def evaluate_patches(patches: Iterable[np.ndarray],estimators: dict[str, CrosstalkEstimator],) -> list[dict[str, Any]]:
    """Applique chaque méthode à chaque patch et rend les lignes de résultat.

    Cœur pur de l'évaluation : il ne connaît ni fichiers ni dataloader, ce qui
    le rend testable sur des patchs synthétiques.

    Args:
        patches: itérable de patchs `(4, H, W)` complexes, dans l'ordre du pipeline.
        estimators: méthodes à appliquer, indexées par leur étiquette.

    Returns:
        Une ligne par couple patch/méthode. Un patch sur lequel une méthode
        échoue donne une ligne de statut `ill_conditioned`, sans interrompre le parcours.
    """
    lignes: list[dict[str, Any]] = []
    for index, patch in enumerate(patches):
        for label, estimateur in estimators.items():
            ligne: dict[str, Any] = {"label": label, "status": "ok"}
            try:
                rapport = residual_crosstalk(patch, estimateur)
            except CrosstalkError as erreur:
                ligne["status"] = "ill_conditioned"
                logger.debug("patch %d, %s : %s", index, label, erreur)
            else:
                ligne.update(rapport.as_row())
            lignes.append(ligne)
    #lignes[i] = {'label': 'quegan_patch', 'status': 'ok', 'initial_db': 20.0, 'residual_db': -30.0, 'reduction_db': 50.0, 'alpha_abs': 1.0}
    #lignes[i+1] = {'label': 'quegan_patch', 'status': 'ill_conditioned'}
    return lignes


def summarize(lignes: Sequence[dict[str, Any]]) -> str:
    """
    Tableau de synthèse, une ligne par méthode.
    """
    labels = sorted({ligne["label"] for ligne in lignes})
    entete = (
        f"{'méthode':>16} {'patchs':>7} {'échecs':>7} "
        f"{'initial dB':>11} {'résiduel dB':>12} {'réduction dB':>13} "
        f"{'|alpha|':>8}"
    )
    sorties = [entete, "-" * len(entete)]
    for label in labels:
        groupe = [ligne for ligne in lignes if ligne["label"] == label]
        valides = [ligne for ligne in groupe if ligne["status"] == "ok"]
        if not valides:
            sorties.append(f"{label:>16} {len(groupe):>7} {len(groupe):>7}"
                           f" {'-':>11} {'-':>12} {'-':>13} {'-':>8}")
            continue
        mediane = {
            cle: float(np.median([ligne[cle] for ligne in valides]))
            for cle in ("initial_db", "residual_db", "reduction_db", "alpha_abs")
        }
        sorties.append(
            f"{label:>16} {len(groupe):>7} {len(groupe) - len(valides):>7} "
            f"{mediane['initial_db']:>11.1f} {mediane['residual_db']:>12.1f} "
            f"{mediane['reduction_db']:>13.1f} {mediane['alpha_abs']:>8.3f}"
        )
    return "\n".join(sorties)


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

    loader = build_dataloader(config)
    lignes = evaluate_patches(iter_patches(loader), estimateurs)

    print(summarize(lignes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
