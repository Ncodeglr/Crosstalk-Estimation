"""Tests du passage des index de `cvnn` aux positions dans la pile (`src/patchs.py`).

Ils tournent sur une pile factice qui imite les attributs de `SLCDataset` et de
`SLCFile` : la vraie pile pèse 92 Go.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.patchs import PositionPatch, iter_patches, position_du_patch, positions_de_test


# --------------------------------------------------- positions dans la pile


class PileFactice(torch.utils.data.Dataset):
    """Imite les attributs de `SLCDataset` et de `SLCFile` que lit le script.

    Deux acquisitions, rangées comme le ferait `glob` : la plus récente
    d'abord. Grille de 2 x 3 patchs ; chaque patch porte son index global.
    """

    def __init__(self):
        self.patch_counts = {"B_160610": 6, "A_160211": 6}
        fichier = SimpleNamespace(nsamples_per_cols=3, patch_stride=(256, 256))
        self.slc_polarizations = {cle: {"HH": fichier, "VV": fichier} for cle in self.patch_counts}

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return np.full((4, 2, 2), index, dtype=np.complex64)


@pytest.mark.parametrize(
    "index, attendu",
    [
        (0, ("B_160610", 0, 0)),
        (2, ("B_160610", 0, 512)),
        (3, ("B_160610", 256, 0)),
        (6, ("A_160211", 0, 0)),
        (11, ("A_160211", 256, 512)),
    ],
)
def test_position_du_patch(index, attendu):
    comptes = PileFactice().patch_counts
    assert position_du_patch(index, comptes, 3, (256, 256)) == PositionPatch(*attendu)


def test_un_index_hors_de_la_pile_est_refuse():
    with pytest.raises(IndexError):
        position_du_patch(12, PileFactice().patch_counts, 3, (256, 256))


def test_chaque_patch_recoit_la_position_de_son_index():
    """Le passage `Subset.indices` -> position, à travers un vrai `DataLoader` en lots."""
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=False)
    couples = list(iter_patches(chargeur))

    assert [int(patch[0, 0, 0].real) for _, patch in couples] == [7, 0, 11, 4]
    assert [position for position, _ in couples] == [
        PositionPatch("A_160211", 0, 256),
        PositionPatch("B_160610", 0, 0),
        PositionPatch("A_160211", 256, 512),
        PositionPatch("B_160610", 256, 256),
    ]


def test_un_chargeur_melange_est_refuse():
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=True)
    with pytest.raises(ValueError, match="mélangé"):
        next(iter_patches(chargeur))


def test_un_chargeur_qui_perd_des_patchs_est_refuse():
    """`drop_last` écarte le dernier lot incomplet : trois patchs pour quatre positions."""
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=False, drop_last=True)
    with pytest.raises(ValueError):
        list(iter_patches(chargeur))


def test_des_grilles_differentes_sont_refusees():
    pile = PileFactice()
    pile.slc_polarizations["A_160211"]["VV"] = SimpleNamespace(nsamples_per_cols=4, patch_stride=(256, 256))
    chargeur = torch.utils.data.DataLoader(torch.utils.data.Subset(pile, [0]), shuffle=False)
    with pytest.raises(ValueError, match="grilles"):
        positions_de_test(chargeur)


def test_le_decoupage_est_rendu_reproductible():
    """`cvnn.data` découpe avec les générateurs globaux, jamais graines.

    Le script appelle `cvnn.utils.set_seed`, la même fonction que
    `BaseExperiment` utilise pour le réseau : les deux voient donc la même
    partition. Sans ce réamorçage, deux exécutions évaluent deux partitions de
    test différentes.
    """
    import torch
    from cvnn.utils import set_seed

    set_seed(1670)
    premier = torch.randperm(50).tolist()
    set_seed(1670)
    assert torch.randperm(50).tolist() == premier
