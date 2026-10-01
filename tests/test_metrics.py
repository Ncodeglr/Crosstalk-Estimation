"""Tests des métriques d'évaluation."""

import numpy as np
import pytest

from src.crosstalk.estimators.quegan import QueganEstimator
from src.crosstalk.metrics import (
    ResidualReport,
    corner_reflector_crosstalk,
    parameter_errors,
    reciprocity,
    residual_crosstalk,
)
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_correction, apply_distortion, distortion_matrix
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


def test_la_phase_de_alpha_n_est_pas_rapportee():
    """`alpha` estimé est réel par convention : son écart de phase ne mesure rien.

    Le jeu de référence porte pourtant une phase de 27°. Si la clé revenait un
    jour, elle vaudrait -27° quelle que soit la qualité de l'estimation.
    """
    assert "alpha_error_deg" not in parameter_errors(THETA, THETA)


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
        "reciprocity_before",
        "reciprocity_after",
    }
    assert ligne["alpha_abs"] == pytest.approx(1.2)


def test_residuel_sur_donnees_synthetiques():
    signal = reflection_symmetric_scattering(
        (256, 256), beta=0.01, rng=np.random.default_rng(0)
    )
    patch = apply_distortion(signal, THETA)
    estimateur = QueganEstimator()
    rapport = residual_crosstalk(patch, estimateur)
    assert rapport.initial_db == pytest.approx(THETA.max_crosstalk_db(), abs=1.0)
    assert rapport.reduction_db > 10.0


def test_residuel_propage_l_echec():
    from src.crosstalk.exceptions import IllConditionedCovarianceError

    estimateur = QueganEstimator()
    with pytest.raises(IllConditionedCovarianceError):
        residual_crosstalk(np.ones((4, 16, 16), dtype=np.complex64), estimateur)


# ------------------------------------------------- réciprocité (Babu, §4.4)


def test_reciprocite_parfaite_vaut_un():
    """`S_HV = S_VH` exactement : c'est l'hypothèse de Quegan sur le signal vrai."""
    signal = reflection_symmetric_scattering(
        (64, 64), beta=0.05, rng=np.random.default_rng(0)
    )
    assert reciprocity(signal) == pytest.approx(1.0)


def test_le_crosstalk_degrade_la_reciprocite():
    """C'est ce qui rend la métrique utile : elle réagit à la distorsion.

    Elle ne mesure pourtant rien que l'estimateur optimise — `estimate` ne lit
    que des corrélations co-pol/cross-pol, jamais l'accord entre HV et VH.
    """
    signal = reflection_symmetric_scattering(
        (256, 256), beta=0.05, rng=np.random.default_rng(0)
    )
    assert reciprocity(apply_distortion(signal, THETA)) < reciprocity(signal)


def test_la_correction_restaure_la_reciprocite():
    """Validation indépendante : corriger avec le `theta` vrai doit la rendre à 1."""
    from src.crosstalk.physics import apply_correction

    signal = reflection_symmetric_scattering(
        (256, 256), beta=0.05, rng=np.random.default_rng(0)
    )
    observe = apply_distortion(signal, THETA)
    assert reciprocity(apply_correction(observe, THETA)) == pytest.approx(1.0, abs=1e-9)


def test_canal_croise_eteint_rend_nan():
    patch = np.ones((4, 16, 16), dtype=np.complex64)
    patch[1] = 0
    assert np.isnan(reciprocity(patch))


# ------------------------------------- réflecteurs en coin (Babu, §4.6)


def test_triedre_parfait_rend_moins_l_infini():
    """`S = sqrt(sigma) I` : canaux croisés nuls, donc aucune fuite."""
    assert corner_reflector_crosstalk(1.0, 0.0, 0.0, 1.0) == float("-inf")


def test_une_fuite_connue_est_rendue_en_db():
    assert corner_reflector_crosstalk(1.0, 0.1, 0.0, 1.0) == pytest.approx(-20.0)
    assert corner_reflector_crosstalk(1.0, 0.0, 0.1, 1.0) == pytest.approx(-20.0)
    assert corner_reflector_crosstalk(1.0, 0.03, 0.1, 1.0) == pytest.approx(-20.0), (
        "c'est le pire des deux canaux croisés qui compte"
    )


def test_la_reference_copolaire_est_la_moyenne_geometrique():
    """`sqrt(|hh| |vv|)` et non `|hh|` : insensible au déséquilibre HH/VV."""
    assert corner_reflector_crosstalk(4.0, 0.2, 0.0, 1.0) == pytest.approx(
        corner_reflector_crosstalk(2.0, 0.2, 0.0, 2.0)
    )


def test_la_fuite_mesuree_vaut_max_z_plus_w_et_u_plus_v():
    """Ancre la métrique sur le modèle : `O = D I` donne `O_HV/O_HH = z + w`.

    C'est ce qui rend la mesure interprétable — elle porte sur des **sommes**
    de paramètres, pas sur les paramètres isolés. Les deux peuvent différer
    nettement par compensation entre termes.
    """
    observe = distortion_matrix(THETA) @ np.array([1, 0, 0, 1], dtype=np.complex128)
    attendu = 20 * np.log10(
        max(abs(THETA.z + THETA.w), abs(THETA.u + THETA.v))
    )
    assert corner_reflector_crosstalk(*observe) == pytest.approx(attendu, abs=0.05)


def test_canal_copolaire_eteint_rend_une_valeur_non_finie():
    """Un réflecteur absent ou masqué ne doit pas produire un chiffre exploitable.

    Deux cas distincts, tous deux non finis : `inf` si les canaux croisés
    subsistent, `nan` si tout est éteint. C'est `np.isfinite` qui filtre les
    deux, pas un test sur `nan`.
    """
    assert not np.isfinite(corner_reflector_crosstalk(0.0, 0.1, 0.1, 1.0))
    assert not np.isfinite(corner_reflector_crosstalk(0.0, 0.0, 0.0, 0.0))


def test_la_correction_fait_chuter_la_fuite_au_reflecteur():
    """Validation de bout en bout, indépendante de l'estimateur.

    On distord un trièdre par un `theta` connu, on corrige, et la fuite doit
    retomber au niveau numérique. Aucune ré-estimation n'intervient.
    """
    triedre = np.ones((4, 8, 8), dtype=np.complex128)
    triedre[1] = 0.0
    triedre[2] = 0.0
    observe = apply_distortion(triedre, THETA)
    avant = corner_reflector_crosstalk(*observe[:, 0, 0])
    apres = corner_reflector_crosstalk(
        *apply_correction(observe, THETA)[:, 0, 0]
    )
    assert avant > -25.0, "le cas de test doit présenter une fuite mesurable"
    assert apres < -250.0, f"fuite résiduelle {apres:.1f} dB, attendue négligeable"
