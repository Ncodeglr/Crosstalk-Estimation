"""Tests de la covariance polarimétrique."""

import numpy as np
import pytest

from src.crosstalk.polarimetry.covariance import (
    HERMITIAN_ENTRIES,
    check_covariance,
    check_patch,
    covariance_entries,
    normalize_by_trace,
    polarimetric_covariance,
)
from src.crosstalk.polarimetry.parameters import CrosstalkParameters
from src.crosstalk.polarimetry.matrix import apply_distortion, distortion_matrix

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


# ----------------------------------------------------------------- covariance_entries


def test_les_dix_entrees_sont_le_triangle_superieur_en_pol_order():
    assert HERMITIAN_ENTRIES == ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))


def test_les_entrees_sont_celles_de_la_covariance_et_la_determinent():
    """Les 10 entrées redonnent toute la matrice : le triangle inférieur est le conjugué du triangle supérieur."""
    cov = polarimetric_covariance(patch_aleatoire())
    entrees = covariance_entries(cov)
    assert entrees.shape == (10,) and entrees.dtype == cov.dtype
    reconstruite = np.zeros((4, 4), dtype=complex)
    for valeur, (i, j) in zip(entrees, HERMITIAN_ENTRIES):
        reconstruite[i, j], reconstruite[j, i] = valeur, np.conj(valeur)
    np.testing.assert_array_equal(reconstruite, cov)


def test_les_entrees_conservent_les_dimensions_de_tete():
    lot = polarimetric_covariance(patch_aleatoire(forme=(3, 4, 8, 8)))
    assert covariance_entries(lot).shape == (3, 10)
    np.testing.assert_array_equal(covariance_entries(lot)[1], covariance_entries(lot[1]))


@pytest.mark.parametrize("forme", [(4,), (3, 3), (4, 3), (2, 4, 5)])
def test_les_entrees_refusent_une_matrice_qui_n_est_pas_4x4(forme):
    with pytest.raises(ValueError):
        covariance_entries(np.zeros(forme, dtype=complex))


# ----------------------------------------------------------------- normalize_by_trace


def test_la_covariance_normalisee_a_une_trace_de_un():
    normalisee = normalize_by_trace(polarimetric_covariance(patch_aleatoire()))
    assert np.trace(normalisee).real == pytest.approx(1.0, rel=1e-12)
    assert normalisee.dtype == np.complex128


@pytest.mark.parametrize("facteur", [1e-8, 0.5, 3.0, 1e8])
def test_la_normalisation_efface_l_echelle(facteur):
    cov = polarimetric_covariance(patch_aleatoire())
    np.testing.assert_allclose(normalize_by_trace(facteur * cov), normalize_by_trace(cov), rtol=1e-12)


def test_la_normalisation_conserve_les_dimensions_de_tete():
    lot = polarimetric_covariance(patch_aleatoire(forme=(3, 4, 8, 8)))
    assert normalize_by_trace(lot).shape == (3, 4, 4)
    np.testing.assert_allclose(normalize_by_trace(lot)[2], normalize_by_trace(lot[2]))


@pytest.mark.parametrize("trace", [0.0, -1.0, np.nan, np.inf])
def test_une_trace_nulle_negative_ou_non_finie_rend_une_matrice_nan(trace):
    """Patch vide ou invalide : l'échec reste visible, comme pour QueganEstimator.solve."""
    cov = np.diag([trace, 0.0, 0.0, 0.0]).astype(complex)
    assert np.all(np.isnan(normalize_by_trace(cov)))


def test_dans_un_lot_seules_les_matrices_invalides_deviennent_nan():
    lot = np.stack([polarimetric_covariance(patch_aleatoire()), np.zeros((4, 4), dtype=complex)])
    normalise = normalize_by_trace(lot)
    assert np.all(np.isfinite(normalise[0])) and np.all(np.isnan(normalise[1]))


@pytest.mark.parametrize("forme", [(4,), (3, 3), (2, 4, 5)])
def test_la_normalisation_refuse_une_matrice_qui_n_est_pas_4x4(forme):
    with pytest.raises(ValueError):
        normalize_by_trace(np.zeros(forme, dtype=complex))


# ----------------------------------------------------------------- check_patch et check_covariance


def test_check_patch_accepte_un_patch_seul():
    check_patch(np.zeros((4, 8, 8), dtype=np.complex64))


def test_check_patch_refuse_un_lot_sauf_si_on_l_autorise():
    lot = np.zeros((2, 4, 8, 8), dtype=np.complex64)
    with pytest.raises(ValueError, match="itérez"):
        check_patch(lot)
    check_patch(lot, lot=True)


@pytest.mark.parametrize("lot", [False, True])
def test_check_patch_refuse_un_mauvais_nombre_de_canaux(lot):
    with pytest.raises(ValueError, match="canaux"):
        check_patch(np.zeros((3, 8, 8), dtype=np.complex64), lot=lot)


def test_check_patch_refuse_un_patch_de_rang_trop_faible_meme_en_lot():
    with pytest.raises(ValueError, match=r"\(\.\.\., 4, H, W\)"):
        check_patch(np.zeros((4, 8), dtype=np.complex64), lot=True)


@pytest.mark.parametrize("objet", [[[1.0]], None, "patch"])
def test_check_patch_refuse_ce_qui_n_est_pas_un_tableau(objet):
    with pytest.raises(TypeError):
        check_patch(objet)


def test_check_covariance_accepte_une_matrice_et_un_lot_si_on_l_autorise():
    check_covariance(np.zeros((4, 4)))
    check_covariance(np.zeros((3, 4, 4)), lot=True)
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        check_covariance(np.zeros((3, 4, 4)))


@pytest.mark.parametrize("forme", [(4,), (3, 3), (4, 3), (2, 4, 5)])
@pytest.mark.parametrize("lot", [False, True])
def test_check_covariance_refuse_une_matrice_qui_n_est_pas_4x4(forme, lot):
    with pytest.raises(ValueError):
        check_covariance(np.zeros(forme), lot=lot)
