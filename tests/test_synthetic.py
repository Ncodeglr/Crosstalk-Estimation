"""Tests du générateur de signal vrai."""

import numpy as np
import pytest

from src.crosstalk.covariance import polarimetric_covariance
from src.crosstalk.physics import POL_ORDER
from src.crosstalk.synthetic import (
    IDX_HH,
    IDX_HV,
    IDX_VH,
    IDX_VV,
    quegan_covariance,
    reflection_symmetric_scattering,
)


def test_les_indices_locaux_suivent_l_ordre_du_modele():
    """Le garde-fou de la double déclaration.

    `synthetic` redéclare ses indices au lieu de les importer de `physics` ;
    ce test est ce qui empêche les deux déclarations de diverger en silence —
    une divergence qui ne lèverait aucune exception et se contenterait de
    renommer les paramètres estimés.
    """
    assert (IDX_HH, IDX_HV, IDX_VH, IDX_VV) == tuple(
        POL_ORDER.index(pol) for pol in ("HH", "HV", "VH", "VV")
    )


def test_structure_de_la_covariance_de_quegan():
    cov = quegan_covariance(sigma_hh=2.0, sigma_vv=3.0, rho_copol=0.5, beta=0.1)
    # Blocs co-pol / cross-pol nuls : c'est l'hypothèse de symétrie de réflexion.
    for i in (IDX_HH, IDX_VV):
        for j in (IDX_VH, IDX_HV):
            assert cov[i, j] == 0.0
            assert cov[j, i] == 0.0
    assert cov[IDX_HH, IDX_HH] == 2.0
    assert cov[IDX_VV, IDX_VV] == 3.0
    assert cov[IDX_HH, IDX_VV] == pytest.approx(0.5 * np.sqrt(6.0))
    # Bloc croisé constant : impose S_HV = S_VH.
    assert cov[IDX_VH, IDX_HV] == 0.1


def test_la_covariance_de_quegan_est_hermitienne_sdp_et_de_rang_3():
    cov = quegan_covariance()
    np.testing.assert_allclose(cov, cov.conj().T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)
    assert np.linalg.matrix_rank(cov) == 3  # les canaux croisés sont confondus


def test_correlation_copolaire_hors_domaine():
    with pytest.raises(ValueError, match="rho_copol"):
        quegan_covariance(rho_copol=1.0)


def test_puissance_negative_refusee():
    with pytest.raises(ValueError, match="positives"):
        quegan_covariance(beta=-0.1)


def test_reciprocite_stricte():
    signal = reflection_symmetric_scattering((32, 32), rng=np.random.default_rng(0))
    np.testing.assert_array_equal(signal[IDX_HV], signal[IDX_VH])


def test_le_tirage_converge_vers_la_covariance_visee():
    parametres = dict(sigma_hh=1.0, sigma_vv=0.7, rho_copol=0.6 + 0.2j, beta=0.05)
    signal = reflection_symmetric_scattering(
        (1024, 1024), **parametres, rng=np.random.default_rng(0)
    )
    empirique = polarimetric_covariance(signal)
    np.testing.assert_allclose(
        empirique, quegan_covariance(**parametres), atol=5e-3
    )


def test_forme_et_dtype():
    signal = reflection_symmetric_scattering(
        (8, 9), rng=np.random.default_rng(0), dtype=np.complex64
    )
    assert signal.shape == (4, 8, 9)
    assert signal.dtype == np.complex64


def test_reproductible_a_graine_egale():
    a = reflection_symmetric_scattering((8, 8), rng=np.random.default_rng(1))
    b = reflection_symmetric_scattering((8, 8), rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_la_phase_de_la_correlation_copolaire_est_respectee():
    """Non-régression : la convention C_ij = <O_i O_j*> impose la conjugaison.

    Une corrélation co-polaire réelle ne distinguerait pas les deux conventions ;
    il faut une phase non nulle pour que l'erreur soit visible.
    """
    rho = 0.6 * np.exp(1j * np.deg2rad(40.0))
    signal = reflection_symmetric_scattering(
        (1024, 1024), rho_copol=rho, rng=np.random.default_rng(0)
    )
    mesure = polarimetric_covariance(signal)[IDX_HH, IDX_VV]
    assert np.angle(mesure, deg=True) == pytest.approx(40.0, abs=1.0)
