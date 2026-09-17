"""Tests de l'ordre des polarisations."""

import numpy as np
import pytest

from src.crosstalk.conventions import (
    POL_ORDER_DATALOADER,
    POL_ORDER_ONERA,
    permutation,
    reorder,
    to_onera_order,
)


def test_permutation_dataloader_vers_onera_echange_les_canaux_croises():
    # [HH, HV, VH, VV] -> [HH, VH, HV, VV] : seuls 1 et 2 bougent.
    assert permutation(POL_ORDER_DATALOADER).tolist() == [0, 2, 1, 3]


def test_permutation_est_une_involution():
    perm = permutation(POL_ORDER_DATALOADER)
    assert perm[perm].tolist() == [0, 1, 2, 3]


def test_permutation_identite():
    assert permutation(POL_ORDER_ONERA).tolist() == [0, 1, 2, 3]


def test_permutation_rejette_des_ordres_incompatibles():
    with pytest.raises(ValueError, match="incompatibles"):
        permutation(("HH", "HV", "VH", "XX"))


def test_permutation_rejette_un_doublon():
    with pytest.raises(ValueError):
        permutation(("HH", "HH", "VH", "VV"))


def test_reorder_identite_ne_copie_pas():
    patch = np.zeros((4, 3, 3), dtype=np.complex64)
    assert reorder(patch, POL_ORDER_ONERA, POL_ORDER_ONERA) is patch


def test_to_onera_order_place_vh_en_position_1():
    # Chaque canal porte son propre indice d'origine, facile à suivre.
    patch = np.arange(4, dtype=np.complex64).reshape(4, 1, 1)
    converti = to_onera_order(patch)
    assert converti[:, 0, 0].real.tolist() == [0.0, 2.0, 1.0, 3.0]


def test_to_onera_order_gere_un_lot():
    patch = np.arange(2 * 4 * 5 * 6, dtype=np.complex64).reshape(2, 4, 5, 6)
    converti = to_onera_order(patch)
    assert converti.shape == patch.shape
    np.testing.assert_array_equal(converti[:, 1], patch[:, 2])
    np.testing.assert_array_equal(converti[:, 2], patch[:, 1])


def test_to_onera_order_est_involutif():
    patch = np.random.default_rng(0).normal(size=(4, 3, 3)).astype(np.complex64)
    aller = to_onera_order(patch)
    retour = reorder(aller, POL_ORDER_ONERA, POL_ORDER_DATALOADER)
    np.testing.assert_array_equal(retour, patch)


def test_reorder_refuse_un_axe_de_mauvaise_taille():
    with pytest.raises(ValueError, match="canaux"):
        to_onera_order(np.zeros((3, 8, 8), dtype=np.complex64))
