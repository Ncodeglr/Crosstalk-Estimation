"""Tests de la covariance polarimétrique."""

import numpy as np
import pytest

from src.crosstalk.covariance import polarimetric_covariance
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.matrix import apply_distortion, distortion_matrix

def is_hermitian(cov: np.ndarray, tol: float = 1e-10) -> bool:
    """`C = C^H` sur les deux derniers axes."""
    return bool(np.allclose(cov, np.swapaxes(cov, -1, -2).conj(), atol=tol, rtol=0.0))


def is_positive_semidefinite(cov: np.ndarray, tol: float = 1e-10) -> bool:
    """Toutes les valeurs propres positives à `tol` près."""
    return bool(np.all(np.linalg.eigvalsh(cov) > -tol))


THETA = CrosstalkParameters(
    u=0.08 + 0.05j,
    v=-0.06 + 0.09j,
    w=0.10 - 0.03j,
    z=-0.04 - 0.07j,
    alpha=1.03 * np.exp(1j * np.deg2rad(27.0)),
)


def patch_aleatoire(forme=(4, 32, 40), dtype=np.complex64, graine=3):
    rng = np.random.default_rng(graine)
    bruit = rng.normal(size=forme) + 1j * rng.normal(size=forme)
    return bruit.astype(dtype)


# ------------------------------------------------------------ une C par patch


def test_rend_une_matrice_4x4():
    cov = polarimetric_covariance(patch_aleatoire())
    assert cov.shape == (4, 4)
    assert cov.dtype == np.complex128


def test_coincide_avec_le_calcul_direct():
    patch = patch_aleatoire((4, 5, 7), dtype=np.complex128)
    vecteurs = patch.reshape(4, -1)
    attendu = vecteurs @ vecteurs.conj().T / vecteurs.shape[1]
    np.testing.assert_allclose(polarimetric_covariance(patch), attendu, atol=1e-12)


def test_la_convention_de_conjugaison_est_bien_c_ij_egale_oi_conj_oj():
    # C_12 = < O_1 O_2* > et non < O_1* O_2 > : le choix fixe les conjugaisons
    # des formes fermées de alpha (Fore et al., 2015, équation 26).
    patch = np.zeros((4, 1, 1), dtype=np.complex128)
    patch[0, 0, 0] = 1.0 + 0.0j
    patch[1, 0, 0] = 0.0 + 1.0j
    cov = polarimetric_covariance(patch)
    assert cov[0, 1] == pytest.approx(-1j)  # 1 * conj(i) = -i
    assert cov[1, 0] == pytest.approx(+1j)


def test_hermitienne_et_semi_definie_positive():
    cov = polarimetric_covariance(patch_aleatoire())
    assert is_hermitian(cov)
    assert is_positive_semidefinite(cov)


def test_diagonale_reelle_et_positive():
    cov = polarimetric_covariance(patch_aleatoire())
    diagonale = np.diagonal(cov)
    np.testing.assert_allclose(diagonale.imag, 0.0, atol=1e-12)
    assert np.all(diagonale.real > 0.0)


def test_dtype_d_entree_sans_effet_sur_le_resultat():
    patch = patch_aleatoire(dtype=np.complex128)
    np.testing.assert_allclose(
        polarimetric_covariance(patch),
        polarimetric_covariance(patch.astype(np.complex64)),
        rtol=1e-6,
    )


# ------------------------------------------------- lien avec le modèle physique


def test_la_distorsion_transporte_la_covariance():
    """C_S' = D C_S D^H : le lien que les formes fermées inversent."""
    signal = patch_aleatoire((4, 64, 64), dtype=np.complex128)
    observe = apply_distortion(signal, THETA)
    matrice = distortion_matrix(THETA)
    np.testing.assert_allclose(
        polarimetric_covariance(observe),
        matrice @ polarimetric_covariance(signal) @ matrice.conj().T,
        atol=1e-12,
    )


def test_invariance_a_une_homothetie_reelle_globale():
    """Fonde la compatibilité avec `global_scalar_normalize` de la config.

    Une homothétie réelle globale `g` multiplie C par `g^2` : les formes fermées
    de Quegan, rapports homogènes de degré 4 sur degré 4, y sont insensibles.
    """
    patch = patch_aleatoire((4, 16, 16), dtype=np.complex128)
    facteur = 3.7
    np.testing.assert_allclose(
        polarimetric_covariance(facteur * patch),
        facteur**2 * polarimetric_covariance(patch),
        rtol=1e-12,
    )


# ------------------------------------------------------------------------- lots


def test_le_lot_est_traite_element_par_element():
    lot = patch_aleatoire((3, 4, 16, 16), dtype=np.complex128)
    cov = polarimetric_covariance(lot)
    assert cov.shape == (3, 4, 4)
    for index in range(lot.shape[0]):
        np.testing.assert_allclose(
            cov[index], polarimetric_covariance(lot[index]), atol=1e-12
        )


# -------------------------------------------------------- garde-fous et erreurs


def test_refuse_un_nombre_de_canaux_incorrect():
    with pytest.raises(ValueError, match="canaux"):
        polarimetric_covariance(patch_aleatoire((3, 8, 8)))


def test_refuse_un_patch_de_rang_trop_faible():
    with pytest.raises(ValueError, match=r"\(\.\.\., 4, H, W\)"):
        polarimetric_covariance(np.zeros((4, 8), dtype=np.complex64))
