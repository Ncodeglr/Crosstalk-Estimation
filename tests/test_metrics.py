"""Tests des métriques d'évaluation."""

import numpy as np
import pytest

from src.crosstalk.conventions import POL_ORDER_ONERA
from src.crosstalk.estimators.quegan import QueganEstimator
from src.crosstalk.metrics import (
    ResidualReport,
    parameter_errors,
    residual_crosstalk,
)
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion
from src.crosstalk.synthetic import reflection_symmetric_scattering

THETA = CrosstalkParameters(
    u=0.08 + 0.05j,
    v=-0.06 + 0.09j,
    w=0.10 - 0.03j,
    z=-0.04 - 0.07j,
    alpha=1.03 * np.exp(1j * np.deg2rad(27.0)),
)


# ----------------------------------------------------------- erreurs sur theta


def test_erreurs_nulles_pour_un_jeu_identique():
    erreurs = parameter_errors(THETA, THETA)
    assert erreurs["max_rel_error"] == 0.0
    assert erreurs["alpha_error_db"] == pytest.approx(0.0)
    assert erreurs["alpha_error_deg"] == pytest.approx(0.0)


def test_erreur_relative_connue():
    decale = CrosstalkParameters(
        u=THETA.u * 1.1, v=THETA.v, w=THETA.w, z=THETA.z, alpha=THETA.alpha
    )
    erreurs = parameter_errors(decale, THETA)
    assert erreurs["rel_error_u"] == pytest.approx(0.1)
    assert erreurs["rel_error_v"] == pytest.approx(0.0)
    assert erreurs["max_rel_error"] == pytest.approx(0.1)


def test_reference_nulle_donne_une_erreur_relative_infinie():
    neutre = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)
    erreurs = parameter_errors(THETA, neutre)
    assert erreurs["rel_error_u"] == float("inf")
    assert erreurs["abs_error_u"] == pytest.approx(abs(THETA.u))


def test_erreur_de_phase_repliee():
    # 170° estimé contre -170° vrai : l'écart est de -20°, pas de 340°.
    estime = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, np.exp(1j * np.deg2rad(170.0)))
    vrai = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, np.exp(1j * np.deg2rad(-170.0)))
    assert parameter_errors(estime, vrai)["alpha_error_deg"] == pytest.approx(-20.0)


def test_ecart_de_module_de_alpha_en_db():
    estime = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 2.0)
    vrai = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 1.0)
    assert parameter_errors(estime, vrai)["alpha_error_db"] == pytest.approx(
        20 * np.log10(2.0)
    )


# --------------------------------------------------------------- résiduel


def test_rapport_de_residuel():
    initial = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 1.0)      # -20 dB
    residuel = CrosstalkParameters(0.01, 0.01, 0.01, 0.01, 1.0)  # -40 dB
    rapport = ResidualReport(initial=initial, residual=residuel)
    assert rapport.initial_db == pytest.approx(-20.0)
    assert rapport.residual_db == pytest.approx(-40.0)
    assert rapport.reduction_db == pytest.approx(20.0)


def test_rapport_vue_plate():
    rapport = ResidualReport(
        initial=CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 1.2),
        residual=CrosstalkParameters(0.01, 0.01, 0.01, 0.01, 1.0),
    )
    ligne = rapport.as_row()
    assert set(ligne) == {
        "initial_db",
        "residual_db",
        "reduction_db",
        "alpha_abs",
        "alpha_deg",
    }
    assert ligne["alpha_abs"] == pytest.approx(1.2)


def test_residuel_sur_donnees_synthetiques():
    signal = reflection_symmetric_scattering(
        (256, 256), beta=0.01, rng=np.random.default_rng(0)
    )
    patch = apply_distortion(signal, THETA)
    estimateur = QueganEstimator(source_order=POL_ORDER_ONERA)
    rapport = residual_crosstalk(patch, estimateur)
    assert rapport.initial_db == pytest.approx(THETA.max_crosstalk_db(), abs=1.0)
    assert rapport.reduction_db > 10.0


def test_residuel_propage_l_echec():
    from src.crosstalk.exceptions import IllConditionedCovarianceError

    estimateur = QueganEstimator(source_order=POL_ORDER_ONERA)
    with pytest.raises(IllConditionedCovarianceError):
        residual_crosstalk(np.ones((4, 16, 16), dtype=np.complex64), estimateur)
