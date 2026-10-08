"""Tests de l'interface commune aux estimateurs."""

import numpy as np
import pytest

from src.crosstalk.estimators.estimator import CrosstalkEstimator
from src.crosstalk.exceptions import SingularDmatrixError
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion

THETA = CrosstalkParameters(
    u=0.08 + 0.05j,
    v=-0.06 + 0.09j,
    w=0.10 - 0.03j,
    z=-0.04 - 0.07j,
    alpha=1.03 * np.exp(1j * np.deg2rad(27.0)),
)


class EstimateurConstant(CrosstalkEstimator):
    """Estimateur factice : rend toujours le même jeu, et compte ses appels."""

    name = "constant"

    def __init__(self, params: CrosstalkParameters = THETA, **kwargs) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.appels = 0

    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        self.check_patch(patch)
        self.appels += 1
        return self.params


def patch_aleatoire(forme=(4, 32, 32), dtype=np.complex64, graine=5):
    rng = np.random.default_rng(graine)
    return (rng.normal(size=forme) + 1j * rng.normal(size=forme)).astype(dtype)


# ------------------------------------------------------------- contrat abstrait


def test_la_base_n_est_pas_instanciable():
    with pytest.raises(TypeError):
        CrosstalkEstimator()


def test_une_sous_classe_sans_estimate_n_est_pas_instanciable():
    class Incomplet(CrosstalkEstimator):
        name = "incomplet"

    with pytest.raises(TypeError):
        Incomplet()


def test_une_sous_classe_sans_nom_est_refusee():
    class SansNom(CrosstalkEstimator):
        def estimate(self, patch):
            return THETA

    with pytest.raises(TypeError, match="`name`"):
        SansNom()


def test_representation_lisible():
    texte = repr(EstimateurConstant())
    assert "constant" in texte and "tol" in texte


# ------------------------------------------------------- contrat sur les formes


def test_refuse_un_lot():
    with pytest.raises(ValueError, match="itérez"):
        EstimateurConstant().estimate(patch_aleatoire((2, 4, 8, 8)))


def test_refuse_un_nombre_de_canaux_incorrect():
    with pytest.raises(ValueError, match="canaux"):
        EstimateurConstant().estimate(patch_aleatoire((3, 8, 8)))


# ------------------------------------------------------ contrat sur les canaux


def test_correct_inverse_exactement_la_distorsion():
    """Aucune conversion d'ordre : le patch entre et sort en `POL_ORDER`.

    C'est le bénéfice de la convention de Babu — l'ordre du modèle étant celui
    du pipeline, `correct` est une pure application de Sigma.
    """
    signal = patch_aleatoire((4, 64, 64), dtype=np.complex128)
    observe = apply_distortion(signal, THETA)
    np.testing.assert_allclose(
        EstimateurConstant().correct(observe), signal, rtol=1e-10, atol=1e-12
    )


# --------------------------------------------------------- correction héritée


def test_correct_avec_parametres_explicites_n_estime_pas():
    estimateur = EstimateurConstant()
    estimateur.correct(patch_aleatoire(), params=THETA)
    assert estimateur.appels == 0


def test_correct_sans_parametres_estime_une_fois():
    estimateur = EstimateurConstant()
    estimateur.correct(patch_aleatoire())
    assert estimateur.appels == 1


def test_estimate_and_correct_n_estime_qu_une_fois():
    estimateur = EstimateurConstant()
    corrige, params = estimateur.estimate_and_correct(patch_aleatoire())
    assert estimateur.appels == 1
    assert params == THETA
    np.testing.assert_allclose(corrige, estimateur.correct(patch_aleatoire(), THETA))


def test_forme_et_dtype_preserves():
    for dtype in (np.complex64, np.complex128):
        patch = patch_aleatoire((4, 8, 8), dtype=dtype)
        corrige = EstimateurConstant().correct(patch)
        assert corrige.shape == patch.shape
        assert corrige.dtype == dtype


def test_un_jeu_neutre_laisse_le_patch_inchange():
    neutre = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)
    patch = patch_aleatoire((4, 16, 16), dtype=np.complex128)
    np.testing.assert_allclose(
        EstimateurConstant(params=neutre).correct(patch), patch, atol=1e-15
    )


def test_un_jeu_degenere_propage_l_exception():
    degenere = CrosstalkParameters(u=2.0, v=0.1, w=0.5, z=0.1, alpha=1.0)
    with pytest.raises(SingularDmatrixError):
        EstimateurConstant(params=degenere).correct(patch_aleatoire())


def test_le_seuil_de_singularite_est_transmis():
    presque = CrosstalkParameters(u=1.0, v=0.1, w=1.0 - 1e-6, z=0.1, alpha=1.0)
    patch = patch_aleatoire((4, 8, 8))
    EstimateurConstant(params=presque, tol=1e-12).correct(patch)  # passe
    with pytest.raises(SingularDmatrixError):
        EstimateurConstant(params=presque, tol=1e-3).correct(patch)
