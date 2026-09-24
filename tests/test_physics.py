"""Tests du modèle physique : D, Sigma et leur application."""

import logging

import numpy as np
import pytest

from src.crosstalk.exceptions import DegenerateDistortionError
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import (
    CHANNEL_AXIS,
    POL_ORDER,
    apply_correction,
    apply_distortion,
    distortion_matrix,
    inverse_distortion_matrix,
)

THETA = CrosstalkParameters(
    u=0.08 + 0.05j,
    v=-0.06 + 0.09j,
    w=0.10 - 0.03j,
    z=-0.04 - 0.07j,
    alpha=1.03 * np.exp(1j * np.deg2rad(27.0)),
)


def jeux_aleatoires(nombre: int, echelle: float = 0.1, graine: int = 0) -> list:
    """Jeux de paramètres au niveau de crosstalk réaliste (~ -20 dB)."""
    rng = np.random.default_rng(graine)
    jeux = []
    for _ in range(nombre):
        crosstalk = echelle * (rng.normal(size=4) + 1j * rng.normal(size=4))
        alpha = (1.0 + 0.1 * rng.normal()) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        jeux.append(CrosstalkParameters.from_vector([*crosstalk, alpha]))
    return jeux


# --------------------------------------------------------------------- D et Sigma


@pytest.mark.parametrize("params", jeux_aleatoires(20))
def test_sigma_est_l_inverse_exact_de_d(params):
    # Fore et al. (2015), équation 15 : forme analytique, pas une approximation
    # au premier ordre. L'égalité doit tenir à la précision machine.
    produit = distortion_matrix(params) @ inverse_distortion_matrix(params)
    np.testing.assert_allclose(produit, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("params", jeux_aleatoires(20, graine=1))
def test_sigma_coincide_avec_l_inverse_numerique(params):
    np.testing.assert_allclose(
        inverse_distortion_matrix(params),
        np.linalg.inv(distortion_matrix(params)),
        atol=1e-12,
    )


def test_d_vaut_l_identite_sans_distorsion():
    neutre = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(distortion_matrix(neutre), np.eye(4), atol=1e-15)


def test_structure_de_d():
    """Coefficients imposés par le modèle, en ordre [HH, HV, VH, VV].

    Le canal HV porte `1/sqrt(alpha)` et le canal VH `sqrt(alpha)` — c'est
    l'inverse des positions qu'ils occupent chez Fore et al., dont l'ordre
    croise les deux canaux.
    """
    matrice = distortion_matrix(THETA)
    sa = np.sqrt(THETA.alpha)
    assert matrice[0, 0] == 1.0
    assert matrice[3, 3] == 1.0
    assert matrice[1, 1] == pytest.approx(1.0 / sa)   # HV
    assert matrice[2, 2] == pytest.approx(sa)         # VH
    assert matrice[0, 3] == pytest.approx(THETA.v * THETA.w)
    assert matrice[3, 0] == pytest.approx(THETA.u * THETA.z)


def test_melanger_les_deux_conventions_renomme_les_parametres():
    """Le piège que la convention retenue supprime, verrouillé par un test.

    Permuter simultanément lignes et colonnes 2 et 3 de D — c'est-à-dire lire
    notre matrice dans l'ordre de Fore et al. (2015) au lieu de celui de Babu
    et al. (2022) — redonne une matrice de la même famille, avec
    `(u, v, w, z, alpha) -> (z, w, v, u, 1/alpha)`. Aucune erreur n'est levée :
    seuls les noms changent. C'est pour cela que l'ordre des canaux doit rester
    celui que déclare `physics.POL_ORDER`.
    """
    perm = [0, 2, 1, 3]
    permutee = distortion_matrix(THETA)[np.ix_(perm, perm)]
    renommee = CrosstalkParameters(
        u=THETA.z, v=THETA.w, w=THETA.v, z=THETA.u, alpha=1.0 / THETA.alpha
    )
    np.testing.assert_allclose(permutee, distortion_matrix(renommee), atol=1e-12)


# ------------------------------------------------------------------- garde-fous


def test_determinant_nul_leve_une_exception():
    # u w = 1 exactement.
    degenere = CrosstalkParameters(u=2.0, v=0.1, w=0.5, z=0.1, alpha=1.0)
    with pytest.raises(DegenerateDistortionError) as erreur:
        inverse_distortion_matrix(degenere)
    assert "singulière" in str(erreur.value)


def test_le_seuil_de_singularite_est_reglable():
    presque = CrosstalkParameters(u=1.0, v=0.1, w=1.0 - 1e-6, z=0.1, alpha=1.0)
    inverse_distortion_matrix(presque, tol=1e-12)  # passe
    with pytest.raises(DegenerateDistortionError):
        inverse_distortion_matrix(presque, tol=1e-3)


def test_conditionnement_degrade_journalise_une_alerte(caplog):
    limite = CrosstalkParameters(u=0.8, v=0.1, w=0.8, z=0.1, alpha=1.0)
    with caplog.at_level(logging.WARNING, logger="src.crosstalk.physics"):
        inverse_distortion_matrix(limite)
    assert "conditionnement" in caplog.text


def test_pas_d_alerte_pour_un_crosstalk_realiste(caplog):
    with caplog.at_level(logging.WARNING, logger="src.crosstalk.physics"):
        inverse_distortion_matrix(THETA)
    assert caplog.text == ""


# ------------------------------------------------- application sur des patchs


def patch_aleatoire(forme=(4, 64, 64), dtype=np.complex64, graine=7):
    rng = np.random.default_rng(graine)
    reel = rng.normal(size=forme)
    imaginaire = rng.normal(size=forme)
    return (reel + 1j * imaginaire).astype(dtype)


def test_aller_retour_distorsion_correction():
    signal = patch_aleatoire()
    observe = apply_distortion(signal, THETA)
    corrige = apply_correction(observe, THETA)
    # complex64 : ~7 chiffres significatifs, et Sigma amplifie l'erreur d'arrondi.
    np.testing.assert_allclose(corrige, signal, rtol=1e-4, atol=1e-5)


def test_la_distorsion_modifie_reellement_le_patch():
    signal = patch_aleatoire()
    assert not np.allclose(apply_distortion(signal, THETA), signal)


def test_forme_et_dtype_preserves():
    for dtype in (np.complex64, np.complex128):
        signal = patch_aleatoire((4, 8, 8), dtype=dtype)
        sortie = apply_distortion(signal, THETA)
        assert sortie.shape == signal.shape
        assert sortie.dtype == dtype


def test_application_sur_un_lot():
    lot = patch_aleatoire((3, 4, 16, 16))
    sortie = apply_distortion(lot, THETA)
    assert sortie.shape == lot.shape
    # Chaque élément du lot doit valoir la distorsion appliquée seule.
    for index in range(lot.shape[0]):
        np.testing.assert_allclose(
            sortie[index], apply_distortion(lot[index], THETA), rtol=1e-5, atol=1e-6
        )


def test_axe_des_canaux_configurable():
    vecteurs = patch_aleatoire((4, 1000))  # (4, N), l'axe des canaux est -2
    sortie = apply_distortion(vecteurs, THETA, channel_axis=-2)
    attendu = distortion_matrix(THETA) @ vecteurs.astype(np.complex128)
    np.testing.assert_allclose(sortie, attendu, rtol=1e-4, atol=1e-6)


def test_axe_des_canaux_de_mauvaise_taille():
    with pytest.raises(ValueError, match="canaux"):
        apply_distortion(patch_aleatoire((3, 8, 8)), THETA)


def test_la_correction_est_lineaire():
    # Propriété qui fonde l'invariance des estimateurs à une homothétie globale
    # (cf. la normalisation de configs/data_uavsar.yaml).
    signal = patch_aleatoire((4, 16, 16), dtype=np.complex128)
    facteur = 3.7
    np.testing.assert_allclose(
        apply_correction(facteur * signal, THETA),
        facteur * apply_correction(signal, THETA),
        rtol=1e-10,
    )


# ------------------------------------------------ convention d'ordre des canaux


def test_l_ordre_du_modele_est_celui_de_babu():
    """Babu et al. (2022, éq. 8-9) ordonne `[HH, HV, VH, VV]`."""
    assert POL_ORDER == ("HH", "HV", "VH", "VV")


def test_l_ordre_du_modele_est_celui_du_pipeline():
    """`SLCDataset` trie les polarisations alphabétiquement avant de les empiler.

    C'est l'invariant qui rend toute permutation inutile — et la raison d'avoir
    retenu la convention de Babu plutôt que celle de Fore. S'il tombait, il
    faudrait réintroduire une conversion à la frontière du paquet.
    """
    assert POL_ORDER == tuple(sorted(POL_ORDER))


def test_axe_des_canaux_designe_les_polarisations():
    """`-3` désigne bien l'axe des polarisations d'un patch `(..., 4, H, W)`."""
    patch = np.zeros((2, 4, 8, 8), dtype=np.complex64)
    assert patch.shape[CHANNEL_AXIS] == len(POL_ORDER)
