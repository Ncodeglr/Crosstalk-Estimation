"""
Tests de src/crosstalk/deep_learning/training/experiment.py, ce que partagent les scripts d'entraînement du MLP et du ResNet :

1. la transformation d'injection du YAML : la même loi et la même graine pour le MLP et le ResNet ;
2. les chemins du YAML, lus depuis la racine du dépôt ;
3. evaluate : erreurs médianes par tranche sur un cas construit à la main.
Les tests qui lisent la pile sont sautés si elle est absente.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.deep_learning.crosstalk_injection import injection_transform_yaml
from src.crosstalk.deep_learning.training.experiment import INITIALISATIONS, RACINE as RACINE_DU_MODULE, evaluate, from_repo_root, init_weights, resolve_trainpath

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc(nom: str = "cvnn_mlp.yaml") -> dict:
    """Un YAML du banc d'essai : celui du MLP par défaut, ou celui du ResNet (cvnn_resnet.yaml)."""
    return yaml.safe_load((RACINE / "configs" / nom).read_text())


# ----------------------------------------------------------------- la transformation d'injection du YAML


def test_le_yaml_du_resnet_a_la_loi_et_la_graine_d_injection_du_mlp():
    resnet = injection_transform_yaml(config_du_banc("cvnn_resnet.yaml"))
    assert resnet["name"] == "crosstalk_injection_resnet"
    assert resnet["params"] == injection_transform_yaml(config_du_banc())["params"]


def test_une_config_sans_la_transformation_est_refusee():
    config = config_du_banc()
    config["data"]["transforms"] = []
    with pytest.raises(ValueError, match="0 transformations d'injection"):
        injection_transform_yaml(config)


def test_une_config_avec_deux_transformations_d_injection_est_refusee():
    config = config_du_banc()
    mlp = config["data"]["transforms"][0]
    config["data"]["transforms"] = [mlp, {**mlp, "name": "crosstalk_injection_resnet"}]
    with pytest.raises(ValueError, match="2 transformations d'injection"):
        injection_transform_yaml(config)


# ----------------------------------------------------------------- les chemins du YAML


@sur_la_pile
def test_un_chemin_relatif_de_la_pile_se_lit_depuis_la_racine_du_depot(monkeypatch):
    monkeypatch.chdir(RACINE / "src" / "scripts")
    assert resolve_trainpath(config_du_banc()) == PILE


@sur_la_pile
def test_un_chemin_absolu_de_la_pile_est_garde_tel_quel():
    config = config_du_banc()
    config["data"]["dataset"]["trainpath"] = str(PILE)
    assert resolve_trainpath(config) == PILE


def test_un_chemin_relatif_du_yaml_se_lit_depuis_la_racine_et_un_absolu_reste_tel_quel(tmp_path, monkeypatch):
    monkeypatch.chdir(RACINE / "src" / "scripts")
    assert RACINE_DU_MODULE == RACINE
    assert from_repo_root("./logs") == RACINE / "logs"
    assert from_repo_root(tmp_path) == tmp_path


def test_un_dossier_sans_slc_est_refuse_avec_un_message_clair(tmp_path):
    config = config_du_banc()
    config["data"]["dataset"]["trainpath"] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="aucun fichier .slc"):
        resolve_trainpath(config)


# ----------------------------------------------------------------- evaluate


class Recopie(torch.nn.Module):
    """Réseau factice : sa sortie est la première colonne de son entrée, ce qui fixe les prédictions du test."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :5, 0, 0]


def lot(predictions: list[list[complex]], cibles: list[list[complex]]):
    entrees = torch.zeros(len(predictions), 10, 1, 1, dtype=torch.complex64)
    entrees[:, :5, 0, 0] = torch.tensor(predictions, dtype=torch.complex64)
    return entrees, torch.tensor(cibles, dtype=torch.complex64)


def test_evaluate_rend_la_mediane_par_tranche_et_ignore_les_patchs_ecartes():
    a_22 = 10 ** (-22 / 20)   # crosstalk injecté à -22 dB : tranche -25/-20
    e_30 = 10 ** (-30 / 20)   # erreur de -30 dB sur chaque terme
    nan = complex(np.nan, np.nan)
    chargeur = [
        lot([[0.01] * 4 + [1.2]], [[0.0] * 4 + [1.0]]),                                      # sans injection, erreur -40 dB ; alpha à 0,2 près
        lot([[a_22 + e_30] * 4 + [1.0 + 0.1j], [5.0] * 5], [[a_22] * 4 + [1.0], [nan] * 5]),  # -30 dB, alpha à 0,1 près ; puis un patch écarté
    ]
    resultats = evaluate(Recopie(), chargeur, torch.device("cpu"))
    assert resultats["patchs"] == 2
    assert resultats["aucune"] == pytest.approx(-40.0, abs=1e-4)
    assert resultats["-25/-20"] == pytest.approx(-30.0, abs=1e-4)
    assert all(np.isnan(resultats[t]) for t in ("-35/-30", "-30/-25", "-20/-15", "-15/-10"))
    assert resultats["alpha"] == pytest.approx(0.15, abs=1e-6)
    assert resultats["alpha_aucune"] == pytest.approx(20 * np.log10(0.2), abs=1e-4)
    assert resultats["alpha_-25/-20"] == pytest.approx(-20.0, abs=1e-4)
    assert all(np.isnan(resultats[f"alpha_{t}"]) for t in ("-35/-30", "-30/-25", "-20/-15", "-15/-10"))
    assert resultats["n_aucune"] == resultats["n_-25/-20"] == 1
    assert all(resultats[f"n_{t}"] == 0 for t in ("-35/-30", "-30/-25", "-20/-15", "-15/-10"))


def test_evaluate_rend_une_erreur_finie_si_alpha_estime_est_nul():
    """Le prédicteur nul : son erreur sur alpha vaut |alpha|."""
    resultats = evaluate(Recopie(), [lot([[0.01] * 4 + [0.0]], [[0.0] * 4 + [1.0]])], torch.device("cpu"))
    assert resultats["alpha"] == pytest.approx(1.0)


# ----------------------------------------------------------------- l'initialisation


def petit_reseau(nom_yaml: str, **modele):
    """(configuration, réseau) : le réseau du YAML, réduit, construit comme par create_model de cvnn, sans initialisation."""
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = config_du_banc(nom_yaml)
    canaux, taille = (10, 1) if "mlp" in nom_yaml else (4, 16)
    config["data"].update(inferred_input_channels=canaux, inferred_input_size=taille)
    config["model"].update(num_layers=2, channels_width=4, **modele)
    return config, create_model(config["model"]["class"], **get_model_params(config))


LES_DEUX_YAML = ["cvnn_mlp.yaml", "cvnn_resnet.yaml"]
AUTRES_INITIALISATIONS = [nom for nom, fonction in INITIALISATIONS.items() if fonction is not None]


def couches_a_poids(reseau: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in reseau.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d))]


@pytest.mark.parametrize("nom_yaml", LES_DEUX_YAML)
def test_les_yaml_declarent_une_initialisation_acceptee(nom_yaml):
    assert config_du_banc(nom_yaml)["model"]["init_weights"] in INITIALISATIONS


@pytest.mark.parametrize("nom_yaml", LES_DEUX_YAML)
def test_kaiming_normal_est_exactement_l_initialisation_de_cvnn(nom_yaml):
    """Le résultat des entraînements déjà faits ne dépend donc pas du passage par model.init_weights."""
    from cvnn.models.utils import init_weights_mode_aware
    from cvnn.utils import set_seed

    set_seed(1)
    config, par_le_yaml = petit_reseau(nom_yaml, init_weights="kaiming_normal")
    init_weights(par_le_yaml, config)
    set_seed(1)
    _, par_cvnn = petit_reseau(nom_yaml)
    par_cvnn.apply(lambda m: init_weights_mode_aware(m, "complex"))
    a, b = par_le_yaml.state_dict(), par_cvnn.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[cle], b[cle]) for cle in a)


@pytest.mark.parametrize("nom_yaml", LES_DEUX_YAML)
@pytest.mark.parametrize("nom", AUTRES_INITIALISATIONS)
def test_une_autre_initialisation_retire_les_poids_et_garde_les_biais(nom_yaml, nom):
    from cvnn.utils import set_seed

    set_seed(1)
    config, reference = petit_reseau(nom_yaml, init_weights="kaiming_normal")
    init_weights(reference, config)
    set_seed(1)
    config, reseau = petit_reseau(nom_yaml, init_weights=nom)
    init_weights(reseau, config)
    assert len(couches_a_poids(reseau)) >= 3
    for couche, couche_reference in zip(couches_a_poids(reseau), couches_a_poids(reference)):
        assert torch.isfinite(torch.view_as_real(couche.weight)).all() and not torch.equal(couche.weight, couche_reference.weight)
        if couche.bias is not None:
            assert torch.equal(couche.bias, couche_reference.bias)


@pytest.mark.parametrize("nom", AUTRES_INITIALISATIONS)
def test_une_autre_initialisation_garde_le_seuil_de_modrelu(nom):
    config, reseau = petit_reseau("cvnn_mlp.yaml", init_weights=nom, activation="modReLU")
    init_weights(reseau, config)
    seuils = [m.b for m in reseau.modules() if type(m).__name__.lower() == "modrelu"]
    assert seuils and all(torch.all(b == 0.25) for b in seuils)


@pytest.mark.parametrize("nom", list(INITIALISATIONS))
def test_meme_graine_meme_initialisation(nom):
    from cvnn.utils import set_seed

    etats = []
    for _ in range(2):
        set_seed(3)
        config, reseau = petit_reseau("cvnn_mlp.yaml", init_weights=nom)
        init_weights(reseau, config)
        etats.append(reseau.state_dict())
    assert all(torch.equal(etats[0][cle], etats[1][cle]) for cle in etats[0])


@pytest.mark.parametrize("valeur", ["xavier", "Kaiming_Normal", None])
def test_une_initialisation_inconnue_est_refusee(valeur):
    config, reseau = petit_reseau("cvnn_mlp.yaml", init_weights=valeur)
    with pytest.raises(ValueError, match="init_weights"):
        init_weights(reseau, config)


def test_une_initialisation_absente_est_refusee():
    config, reseau = petit_reseau("cvnn_mlp.yaml")
    del config["model"]["init_weights"]
    with pytest.raises(ValueError, match="init_weights"):
        init_weights(reseau, config)
