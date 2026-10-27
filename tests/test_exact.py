"""
Tests du solveur exact :

1. enregistrement dans le registre des estimateurs ;
2. exactitude : sur la covariance théorique D C_Q D^H, il retrouve theta à la précision numérique, là où les formes
   fermées de Quegan gardent leur biais au premier ordre ; la covariance corrigée respecte les 10 contraintes ;
3. invariances : aucun crosstalk donne theta neutre ; multiplier la covariance par un réel positif ne change rien ;
4. sur un patch fini, son erreur reste bien sous celle de Quegan ;
5. échecs rendus en nan, comme QueganEstimator.solve, et refus des formes invalides.
"""

import numpy as np
import pytest

from src.crosstalk.estimators import ExactEstimator, available_estimators, create_from_config
from src.crosstalk.estimators.quegan import QueganEstimator
from src.crosstalk.matrix import apply_distortion, distortion_matrix, sigma_calibration_matrix
from src.crosstalk.parameters import CrosstalkParameters
from synthetic import ScatteringParameters, sample_scattering

#: Direction de crosstalk de référence (module d'environ 1), mise à l'échelle par le niveau voulu.
DIRECTION = np.array([0.8 + 0.5j, -0.6 + 0.9j, 1.0 - 0.3j, -0.4 - 0.7j])
ALPHA = 1.03 * np.exp(1j * np.deg2rad(27.0))

#: Une scène non triviale : puissances différentes, corrélation HH-VV complexe.
SCENE = ScatteringParameters(sigma_hhhh=1.0, sigma_vvvv=0.8, sigma_hhvv=0.5 * np.sqrt(0.8) * np.exp(0.3j), beta=0.1)


def theta(niveau_db: float) -> np.ndarray:
    return np.r_[10 ** (niveau_db / 20) * DIRECTION, ALPHA]


def covariance_observee(t: np.ndarray) -> np.ndarray:
    """La covariance théorique observée D C_Q D^H : elle respecte exactement les hypothèses de Quegan une fois corrigée."""
    d = distortion_matrix(CrosstalkParameters.from_vector(t))
    return d @ SCENE.to_covariance() @ d.conj().T


def erreur(estime: np.ndarray, vrai: np.ndarray) -> float:
    """Plus grand écart absolu sur u, v, w, z et alpha."""
    return float(np.abs(estime - vrai).max())


# ----------------------------------------------------------------- enregistrement


def test_le_solveur_exact_est_enregistre():
    assert "exact" in available_estimators()
    assert isinstance(create_from_config({"name": "exact"}), ExactEstimator)


# ----------------------------------------------------------------- exactitude


@pytest.mark.parametrize("niveau_db", [-30.0, -20.0, -10.0])
def test_retrouve_theta_exactement_sur_la_covariance_theorique(niveau_db):
    """Là où Quegan garde son biais au premier ordre, le solveur exact retrouve theta à la précision numérique."""
    vrai = theta(niveau_db)
    cov = covariance_observee(vrai)
    exact = ExactEstimator().solve(cov)
    assert erreur(exact, vrai) < 1e-8
    assert erreur(exact, vrai) < 1e-4 * erreur(QueganEstimator().solve(cov), vrai)


def test_la_covariance_corrigee_respecte_les_dix_contraintes():
    cov = covariance_observee(theta(-15.0))
    sigma = sigma_calibration_matrix(CrosstalkParameters.from_vector(ExactEstimator().solve(cov)))
    corrigee = sigma @ cov @ sigma.conj().T
    p = corrigee.diagonal().real
    for i in (0, 3):                    # HH, VV
        for j in (1, 2):                # HV, VH
            assert abs(corrigee[i, j]) / np.sqrt(p[i] * p[j]) < 1e-9
    assert p[1] == pytest.approx(p[2], rel=1e-9)
    assert abs(np.angle(corrigee[1, 2])) < 1e-9


# ----------------------------------------------------------------- invariances


def test_sans_crosstalk_theta_est_neutre():
    neutre = np.array([0, 0, 0, 0, 1], dtype=complex)
    assert erreur(ExactEstimator().solve(SCENE.to_covariance()), neutre) < 1e-9


@pytest.mark.parametrize("facteur", [1e-6, 1e-3, 1e3, 1e6])
def test_l_echelle_de_la_covariance_ne_change_rien(facteur):
    cov = covariance_observee(theta(-20.0))
    assert erreur(ExactEstimator().solve(facteur * cov), ExactEstimator().solve(cov)) < 1e-9


# ----------------------------------------------------------------- patch fini


def test_sur_un_patch_fini_l_erreur_reste_bien_sous_celle_de_quegan():
    """
    512 x 512 pixels, crosstalk à -15 dB : le bruit d'estimation est de l'ordre de 1e-3, alors que le biais au premier
    ordre de Quegan dépasse 1e-2. Le solveur exact doit faire au moins cinq fois mieux.
    """
    vrai = theta(-15.0)
    signal = sample_scattering(SCENE, (512, 512), rng=np.random.default_rng(0))
    patch = apply_distortion(signal, CrosstalkParameters.from_vector(vrai)).astype(np.complex64)
    erreur_exact = erreur(ExactEstimator().estimate(patch).to_vector(), vrai)
    erreur_quegan = erreur(QueganEstimator().estimate(patch).to_vector(), vrai)
    assert erreur_exact < erreur_quegan / 5


def test_estimate_and_correct_rend_un_patch_de_meme_forme_et_de_meme_type():
    signal = sample_scattering(SCENE, (32, 32), rng=np.random.default_rng(1)).astype(np.complex64)
    corrige, parametres = ExactEstimator().estimate_and_correct(signal)
    assert corrige.shape == signal.shape and corrige.dtype == signal.dtype
    assert isinstance(parametres, CrosstalkParameters)


# ----------------------------------------------------------------- échecs et refus


def test_une_covariance_nulle_rend_des_nan():
    assert np.all(np.isnan(ExactEstimator().solve(np.zeros((4, 4), dtype=complex))))


def test_une_covariance_non_finie_rend_des_nan():
    cov = covariance_observee(theta(-20.0))
    cov[0, 0] = np.nan
    assert np.all(np.isnan(ExactEstimator().solve(cov)))


def test_un_depart_de_quegan_indefini_rend_des_nan():
    """HH et VV parfaitement corrélés : Delta = C11 C44 - |C14|^2 = 0, les formes fermées ne sont pas définies."""
    k = np.array([1.0, 0.1, 0.1, 1.0], dtype=complex)
    assert np.all(np.isnan(ExactEstimator().solve(np.outer(k, k.conj()))))


def test_un_patch_nul_rend_des_parametres_nan_sans_lever():
    parametres = ExactEstimator().estimate(np.zeros((4, 8, 8), dtype=np.complex64))
    assert np.all(np.isnan(parametres.to_vector()))


@pytest.mark.parametrize("forme", [(3, 3), (4,), (2, 4, 4)])
def test_une_covariance_de_mauvaise_forme_est_refusee(forme):
    with pytest.raises(ValueError):
        ExactEstimator().solve(np.zeros(forme, dtype=complex))


def test_un_lot_de_patchs_est_refuse():
    with pytest.raises(ValueError):
        ExactEstimator().estimate(np.zeros((2, 4, 8, 8), dtype=np.complex64))
