"""Tests de la mesure aux réflecteurs qui ne demandent pas les données réelles."""

import logging

import numpy as np
import pytest

from src.crosstalk.estimators.quegan import QueganEstimator
from src.crosstalk.matrix import POL_ORDER
from src.measure_reflectors import DEMI_BANDE, theta_de_la_bande

LARGEUR = 64


def canaux_factices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        pol: (rng.normal(size=(40, LARGEUR)) + 1j * rng.normal(size=(40, LARGEUR))).astype(np.complex64)
        for pol in POL_ORDER
    }


@pytest.mark.parametrize("colonne", [0, DEMI_BANDE - 1, LARGEUR - DEMI_BANDE])
def test_une_bande_qui_deborde_ecarte_le_reflecteur(colonne, caplog):
    """Sans ce garde, `colonne - DEMI_BANDE < 0` lirait l'autre bord de l'image."""
    with caplog.at_level(logging.WARNING, logger="measure_reflectors"):
        assert theta_de_la_bande(canaux_factices(), colonne, QueganEstimator()) is None
    assert "déborde" in caplog.text


@pytest.mark.parametrize("colonne", [DEMI_BANDE, LARGEUR - DEMI_BANDE - 1])
def test_une_bande_juste_dans_l_image_est_estimee(colonne):
    assert theta_de_la_bande(canaux_factices(), colonne, QueganEstimator()) is not None
