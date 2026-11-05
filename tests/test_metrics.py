"""Tests des métriques d'évaluation."""

import numpy as np
import pytest

from src.crosstalk.baselines.quegan import QueganEstimator
from src.crosstalk.baselines.metrics import (
    ResidualReport,
    parameter_errors,
    reciprocity,
    residual_crosstalk,
)
from src.crosstalk.polarimetry.parameters import CrosstalkParameters
from src.crosstalk.polarimetry.matrix import apply_correction, apply_distortion
from synthetic import ScatteringParameters, sample_scattering

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
    rapport = ResidualReport(theta_initial=initial, theta_residual=residuel)
    assert rapport.initial_db == pytest.approx(-20.0)
    assert rapport.residual_db == pytest.approx(-40.0)
    assert rapport.reduction_db == pytest.approx(20.0)


def test_rapport_vue_plate():
    rapport = ResidualReport(
        theta_initial=CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 1.2 * np.exp(1j * np.deg2rad(30.0))),
        theta_residual=CrosstalkParameters(0.01, 0.01, 0.01, 0.01, 0.9 * np.exp(1j * np.deg2rad(-5.0))),
    )
    ligne = rapport.to_metrics_row()
    assert set(ligne) == {
        "initial_db",
        "residual_db",
        "reduction_db",
        "alpha_abs",
        "alpha_db_before",
        "alpha_db_after",
        "alpha_phase_deg_before",
        "alpha_phase_deg_after",
        "reciprocity_before",
        "reciprocity_after",
    }
    assert ligne["alpha_abs"] == pytest.approx(1.2)
    assert ligne["alpha_db_before"] == pytest.approx(20.0 * np.log10(1.2))
    assert ligne["alpha_db_after"] == pytest.approx(20.0 * np.log10(0.9))
    assert ligne["alpha_phase_deg_before"] == pytest.approx(30.0)
    assert ligne["alpha_phase_deg_after"] == pytest.approx(-5.0)


def test_residuel_sur_donnees_synthetiques():
    signal = sample_scattering(ScatteringParameters(beta=0.01), (256, 256), rng=np.random.default_rng(0)
    )
    patch = apply_distortion(signal, THETA)
    estimateur = QueganEstimator()
    rapport = residual_crosstalk(patch, estimateur)
    assert rapport.initial_db == pytest.approx(THETA.max_crosstalk_db(), abs=1.0)
    assert rapport.reduction_db > 10.0


def test_residuel_propage_l_echec():
    """Une correction impossible remonte telle quelle à l'appelant."""
    from src.crosstalk.baselines.estimator import CrosstalkEstimator
    from src.crosstalk.polarimetry.exceptions import SingularDmatrixError

    class Singulier(CrosstalkEstimator):
        name = "singulier"

        def estimate(self, patch):
            return CrosstalkParameters(1.0, 0.0, 1.0, 0.0, 1.0)  # uw - 1 = 0 : D singulière

    with pytest.raises(SingularDmatrixError):
        residual_crosstalk(np.ones((4, 16, 16), dtype=np.complex64), Singulier())


# ------------------------------------------------- réciprocité (Babu, §4.4)


def test_reciprocite_parfaite_vaut_un():
    """`S_HV = S_VH` exactement : c'est l'hypothèse de Quegan sur le signal vrai."""
    signal = sample_scattering(ScatteringParameters(beta=0.05), (64, 64), rng=np.random.default_rng(0)
    )
    assert reciprocity(signal) == pytest.approx(1.0)


def test_le_crosstalk_degrade_la_reciprocite():
    """C'est ce qui rend la métrique utile : elle réagit à la distorsion.

    Elle ne mesure pourtant rien que Quegan optimise : ses formes fermées en
    `u, v, w, z` ne lisent que des corrélations co-pol/cross-pol, et si son
    `alpha` lit `<HV VH*>`, la cohérence est insensible à `alpha`.
    """
    signal = sample_scattering(ScatteringParameters(beta=0.05), (256, 256), rng=np.random.default_rng(0)
    )
    assert reciprocity(apply_distortion(signal, THETA)) < reciprocity(signal)


def test_la_correction_restaure_la_reciprocite():
    """Validation indépendante : corriger avec le `theta` vrai doit la rendre à 1."""
    from src.crosstalk.polarimetry.matrix import apply_correction

    signal = sample_scattering(ScatteringParameters(beta=0.05), (256, 256), rng=np.random.default_rng(0)
    )
    observe = apply_distortion(signal, THETA)
    assert reciprocity(apply_correction(observe, THETA)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("alpha", [1.0, 1.25 * np.exp(-1j * np.deg2rad(23.0))])
def test_la_reciprocite_est_aveugle_a_u_egal_alpha_z_et_v_egal_alpha_w(alpha):
    """Angle mort exact, et non approché.

    Avec `u = alpha z` et `v = alpha w`, les lignes HV et VH de D donnent
    `O_VH = alpha O_HV` pixel par pixel : la cohérence vaut 1 alors que le
    crosstalk est bien présent. Elle ignore aussi `alpha` lui-même. Le cas
    `alpha` complexe distingue cette direction de `u = z, v = w`.
    """
    signal = sample_scattering(ScatteringParameters(beta=0.15), (64, 64), rng=np.random.default_rng(0)
    )
    z, w = 0.05 * np.exp(0.9j), 0.03 * np.exp(-1.7j)
    aveugle = CrosstalkParameters(u=alpha * z, v=alpha * w, w=w, z=z, alpha=alpha)
    assert aveugle.max_crosstalk_db() > -30.0
    assert reciprocity(apply_distortion(signal, aveugle)) == pytest.approx(1.0, abs=1e-12)

    seul = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, alpha)
    assert reciprocity(apply_distortion(signal, seul)) == pytest.approx(1.0, abs=1e-12)


def test_canal_croise_eteint_rend_nan():
    patch = np.ones((4, 16, 16), dtype=np.complex64)
    patch[1] = 0
    assert np.isnan(reciprocity(patch))


def test_la_phase_de_alpha_ne_change_aucune_metrique_comparable_a_babu():
    """Corriger avec `alpha` complexe (Fore, éq. 25) ou réel (Babu, éq. 15).

    Les deux corrections ne diffèrent que par `diag(1, e^{-i phi/2}, e^{+i phi/2}, 1)` :
    une rotation de phase de HV et VH, sans effet sur le retrait du crosstalk.
    Réciprocité et modules du résiduel doivent donc être identiques — ce qui
    garantit que passer à la phase de Fore n'a pas déplacé l'écart avec Babu.
    """
    sans_phase = CrosstalkParameters(THETA.u, THETA.v, THETA.w, THETA.z, abs(THETA.alpha))
    observe = apply_distortion(
        sample_scattering(ScatteringParameters(), (64, 64), rng=np.random.default_rng(5)), THETA
    )
    avec, sans = apply_correction(observe, THETA), apply_correction(observe, sans_phase)

    assert reciprocity(avec) == pytest.approx(reciprocity(sans), rel=1e-12)
    estimateur = QueganEstimator()
    assert estimateur.estimate(avec).crosstalk_db() == pytest.approx(
        estimateur.estimate(sans).crosstalk_db(), abs=1e-9
    )
