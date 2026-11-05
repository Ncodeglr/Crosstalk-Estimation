"""Tests de `src/crosstalk/baselines/patchs.py` qui ne demandent pas la vraie pile (92 Go)."""

import numpy as np
import torch

from src.crosstalk.baselines.patchs import iter_patches


def test_les_lots_sont_deplies_en_patchs_dans_l_ordre():
    """Lots de 3, dernier lot incomplet : chaque patch sort seul, en `(4, H, W)`, dans l'ordre."""
    patchs = [np.full((4, 2, 2), rang, dtype=np.complex64) for rang in range(7)]
    chargeur = torch.utils.data.DataLoader(patchs, batch_size=3, shuffle=False)
    sortis = list(iter_patches(chargeur))
    assert [patch.shape for patch in sortis] == [(4, 2, 2)] * 7
    assert [int(patch[0, 0, 0].real) for patch in sortis] == list(range(7))


def test_le_decoupage_est_rendu_reproductible():
    """`cvnn.data` découpe avec les générateurs globaux, jamais graines.

    Le script appelle `cvnn.utils.set_seed`, la même fonction que
    `src/scripts/train_mlp.py` utilise pour le réseau : les deux voient donc la même
    partition. Sans ce réamorçage, deux exécutions évaluent deux partitions de
    test différentes.
    """
    import torch
    from cvnn.utils import set_seed

    set_seed(1670)
    premier = torch.randperm(50).tolist()
    set_seed(1670)
    assert torch.randperm(50).tolist() == premier
