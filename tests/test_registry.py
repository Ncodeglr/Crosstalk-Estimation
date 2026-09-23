"""Tests du registre et de l'instanciation par configuration.

Ces tests valident aussi la promesse d'extensibilité : brancher une méthode
nouvelle ne demande de toucher à aucun module existant.
"""

import numpy as np
import pytest

from src.crosstalk.estimators import registry
from src.crosstalk.estimators.base import CrosstalkEstimator
from src.crosstalk.estimators.registry import (
    available_estimators,
    create_estimator,
    create_from_config,
    register_estimator,
)
from src.crosstalk.parameters import CrosstalkParameters


@pytest.fixture(autouse=True)
def registre_isole():
    """Le registre est un état global : chaque test repart d'une copie."""
    sauvegarde = dict(registry._ESTIMATORS)
    try:
        yield
    finally:
        registry._ESTIMATORS.clear()
        registry._ESTIMATORS.update(sauvegarde)


def fabrique_methode(nom: str = "factice", **attributs):
    """Définit et enregistre une méthode, comme le ferait un module nouveau."""

    @register_estimator
    class Factice(CrosstalkEstimator):
        name = nom

        def __init__(self, gain: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.gain = gain

        def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
            self.check_patch(patch)
            return CrosstalkParameters(0.0, 0.0, 0.0, 0.0, self.gain)

    for cle, valeur in attributs.items():
        setattr(Factice, cle, valeur)
    return Factice


# ------------------------------------------------------------- enregistrement


def test_enregistrer_puis_retrouver():
    classe = fabrique_methode("factice")
    assert "factice" in available_estimators()
    assert isinstance(create_estimator("factice"), classe)


def test_le_decorateur_rend_la_classe_inchangee():
    classe = fabrique_methode("factice")
    assert issubclass(classe, CrosstalkEstimator)
    assert classe.name == "factice"


def test_noms_tries():
    fabrique_methode("zeta")
    fabrique_methode("alpha")
    noms = available_estimators()
    assert noms == sorted(noms)


def test_reenregistrer_la_meme_classe_est_sans_effet():
    classe = fabrique_methode("factice")
    register_estimator(classe)  # idempotent
    assert available_estimators().count("factice") == 1
    assert isinstance(create_estimator("factice"), classe)


def test_nom_deja_pris_par_une_autre_classe():
    fabrique_methode("factice")
    with pytest.raises(ValueError, match="déjà enregistré"):
        fabrique_methode("factice")


def test_refuse_une_classe_hors_hierarchie():
    class Etranger:
        name = "etranger"

    with pytest.raises(TypeError, match="sous-classe"):
        register_estimator(Etranger)


def test_refuse_une_classe_sans_nom():
    class SansNom(CrosstalkEstimator):
        def estimate(self, patch):
            return CrosstalkParameters(0, 0, 0, 0, 1)

    with pytest.raises(TypeError, match="`name`"):
        register_estimator(SansNom)


def test_refuse_un_nom_vide():
    class NomVide(CrosstalkEstimator):
        name = ""

        def estimate(self, patch):
            return CrosstalkParameters(0, 0, 0, 0, 1)

    with pytest.raises(TypeError, match="`name`"):
        register_estimator(NomVide)


# ----------------------------------------------------------------- fabrication


def test_create_estimator_transmet_les_arguments():
    fabrique_methode("factice")
    estimateur = create_estimator("factice", gain=2.0, tol=1e-10)
    assert estimateur.gain == 2.0
    assert estimateur.tol == 1e-10


def test_nom_inconnu_liste_les_noms_valides():
    fabrique_methode("factice")
    with pytest.raises(KeyError) as erreur:
        create_estimator("inexistant")
    assert "factice" in str(erreur.value)


def test_registre_vide_message_explicite():
    registry._ESTIMATORS.clear()
    with pytest.raises(KeyError, match="aucun"):
        create_estimator("quoi_que_ce_soit")


# ---------------------------------------------------------- pilotage par config


def test_create_from_config():
    fabrique_methode("factice")
    estimateur = create_from_config({"name": "factice", "params": {"gain": 3.0}})
    assert estimateur.gain == 3.0


def test_create_from_config_sans_params():
    fabrique_methode("factice")
    assert create_from_config({"name": "factice"}).gain == 1.0


def test_create_from_config_params_nul():
    fabrique_methode("factice")
    assert create_from_config({"name": "factice", "params": None}).gain == 1.0


def test_create_from_config_sans_nom():
    with pytest.raises(KeyError, match="'name'"):
        create_from_config({"params": {}})


def test_create_from_config_params_mal_typees():
    fabrique_methode("factice")
    with pytest.raises(TypeError, match="mapping"):
        create_from_config({"name": "factice", "params": ["gain", 3.0]})


def test_une_liste_de_configurations_comme_en_yaml():
    # La forme visée dans configs/ : plusieurs méthodes comparées en un passage.
    fabrique_methode("factice")
    fabrique_methode("autre")
    configuration = [
        {"name": "factice", "params": {"gain": 1.0}},
        {"name": "autre", "params": {"gain": 2.0}},
    ]
    estimateurs = [create_from_config(bloc) for bloc in configuration]
    assert [e.name for e in estimateurs] == ["factice", "autre"]
    assert [e.gain for e in estimateurs] == [1.0, 2.0]


# ------------------------------------------------- la méthode branchée fonctionne


def test_une_methode_fraichement_branchee_corrige_un_patch():
    """Bout en bout : définir, enregistrer, instancier, corriger."""
    fabrique_methode("factice")
    estimateur = create_from_config({"name": "factice"})
    rng = np.random.default_rng(0)
    patch = (rng.normal(size=(4, 16, 16)) + 1j * rng.normal(size=(4, 16, 16))).astype(
        np.complex64
    )
    corrige, params = estimateur.estimate_and_correct(patch)
    # gain = 1.0 -> alpha = 1, crosstalk nul -> Sigma = identite.
    assert params == CrosstalkParameters(0, 0, 0, 0, 1)
    np.testing.assert_allclose(corrige, patch, rtol=1e-6, atol=1e-7)
