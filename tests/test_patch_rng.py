"""
Tests de patch_rng : la même injection pour le même patch, une autre pour un autre patch ou une autre graine, et les
refus des arguments qui rendraient l'empreinte non déterministe.
"""

import numpy as np
import pytest

from src.crosstalk.crosstalk_sampler import CrosstalkSampler
from src.crosstalk.patch_rng import patch_rng


def patch_de(graine: int = 0) -> np.ndarray:
    rng = np.random.default_rng(graine)
    return (rng.normal(size=(4, 16, 16)) + 1j * rng.normal(size=(4, 16, 16))).astype(np.complex64)


def premier_theta(patch: np.ndarray, seed: int = 0):
    return CrosstalkSampler(no_injection_rate=0.0).generate(patch_rng(patch, seed))


def test_le_meme_patch_donne_la_meme_injection():
    assert premier_theta(patch_de()) == premier_theta(patch_de())


def test_une_copie_ou_une_autre_disposition_memoire_donne_la_meme_injection():
    """L'empreinte porte sur le contenu, pas sur l'objet : copie et tableau en ordre Fortran sont reconnus."""
    patch = patch_de()
    assert premier_theta(patch.copy()) == premier_theta(patch)
    assert premier_theta(np.asfortranarray(patch)) == premier_theta(patch)


def test_un_pixel_modifie_donne_une_autre_injection():
    patch = patch_de()
    modifie = patch.copy()
    modifie[0, 0, 0] += 1e-3
    assert premier_theta(modifie) != premier_theta(patch)


def test_une_autre_graine_donne_une_autre_injection():
    assert premier_theta(patch_de(), seed=1) != premier_theta(patch_de(), seed=0)


def test_le_type_du_tableau_entre_dans_l_empreinte():
    """Documenté dans patch_rng : le patch doit être passé tel qu'il est lu."""
    patch = patch_de()
    assert premier_theta(patch.astype(np.complex128)) != premier_theta(patch)


def test_un_tableau_d_objets_est_refuse():
    """
    Pour un tableau d'objets Python, tobytes() écrit des adresses mémoire : l'empreinte changerait d'un processus à
    l'autre, et le même patch recevrait des injections différentes sans aucune erreur.
    """
    patch = np.empty((4, 2, 2), dtype=object)
    patch[...] = 1.0
    with pytest.raises(TypeError, match="numérique"):
        patch_rng(patch, 0)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float32, np.int16])
def test_les_types_numeriques_sont_acceptes(dtype):
    patch_rng(np.zeros((4, 2, 2), dtype=dtype), 0)


@pytest.mark.parametrize("patch, seed, erreur", [
    ([[1.0]], 0, TypeError),             # pas un np.ndarray
    (np.zeros(3), 1.5, TypeError),       # graine non entière
    (np.zeros(3), True, TypeError),      # un booléen n'est pas une graine
    (np.zeros(3), -1, ValueError),       # graine négative
])
def test_les_arguments_invalides_sont_refuses(patch, seed, erreur):
    with pytest.raises(erreur):
        patch_rng(patch, seed)
