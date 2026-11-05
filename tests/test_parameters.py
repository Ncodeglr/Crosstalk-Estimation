"""Tests du porteur de paramètres."""

import numpy as np
import pytest

from src.crosstalk.polarimetry.parameters import PARAMETER_NAMES, CrosstalkParameters

THETA = CrosstalkParameters(
    u=0.08 + 0.05j,
    v=-0.06 + 0.09j,
    w=0.10 - 0.03j,
    z=-0.04 - 0.07j,
    alpha=1.03 * np.exp(1j * np.deg2rad(27.0)),
)


def test_immuable():
    with pytest.raises(Exception):
        THETA.u = 0.0


def test_aller_retour_vecteur():
    vecteur = THETA.to_vector()
    assert vecteur.shape == (5,)
    assert vecteur.dtype == np.complex128
    assert CrosstalkParameters.from_vector(vecteur) == THETA


def test_ordre_du_vecteur():
    vecteur = THETA.to_vector()
    for index, name in enumerate(PARAMETER_NAMES):
        assert vecteur[index] == getattr(THETA, name)


def test_from_vector_refuse_une_mauvaise_taille():
    with pytest.raises(ValueError, match=r"\(5,\)"):
        CrosstalkParameters.from_vector([1, 2, 3])


def test_alpha_nul_refuse():
    # D fait intervenir 1/sqrt(alpha).
    with pytest.raises(ValueError, match="alpha nul"):
        CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 0.0)


def test_entiers_et_flottants_convertis_en_complexes():
    params = CrosstalkParameters(0, 0.1, 0, 0, 1)
    assert all(isinstance(getattr(params, n), complex) for n in PARAMETER_NAMES)


def test_crosstalk_en_db():
    params = CrosstalkParameters(0.1, 0.1, 0.1, 0.1, 1.0)
    for valeur in params.crosstalk_db().values():
        assert valeur == pytest.approx(-20.0)
    assert params.max_crosstalk_db() == pytest.approx(-20.0)


def test_lecture_de_alpha():
    # L'ordre de grandeur mesuré par Fore et al. (2015), figures 7 et 9.
    assert THETA.alpha_db() == pytest.approx(20 * np.log10(1.03))
    assert THETA.alpha_phase_deg() == pytest.approx(27.0)


def test_representation_lisible():
    texte = str(THETA)
    assert "dB" in texte and "arg(alpha)" in texte


def test_un_terme_nul_rend_moins_l_infini_sans_avertissement():
    # Cas du jeu neutre (u = v = w = z = 0) : -inf est la valeur correcte en dB,
    # et NumPy ne doit pas émettre d'avertissement de division par zéro.
    neutre = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)
    with np.errstate(divide="raise"):
        valeurs = neutre.crosstalk_db()
    assert all(valeur == -np.inf for valeur in valeurs.values())
    assert neutre.max_crosstalk_db() == -np.inf
    assert neutre.alpha_db() == pytest.approx(0.0)
