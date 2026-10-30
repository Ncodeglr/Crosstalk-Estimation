"""
Tests de src/crosstalk/deep_learning/experiment.py, ce que partagent les scripts d'entraînement du MLP et du ResNet :

1. la loi d'injection lue dans le YAML, celui du MLP comme celui du ResNet, sans la graine de la transformation ;
2. les chemins du YAML, lus depuis la racine du dépôt ;
3. evaluate : erreurs médianes par tranche sur un cas construit à la main.
Les tests qui lisent la pile sont sautés si elle est absente.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.crosstalk_sampler import CrosstalkSampler
from src.crosstalk.deep_learning.experiment import RACINE as RACINE_DU_MODULE, evaluate, from_repo_root, resolve_trainpath, sampler_params_yaml

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc(nom: str = "cvnn_mlp.yaml") -> dict:
    """Un YAML du banc d'essai : celui du MLP par défaut, ou celui du ResNet (cvnn_resnet.yaml)."""
    return yaml.safe_load((RACINE / "configs" / nom).read_text())


# ----------------------------------------------------------------- la loi d'injection


def test_la_loi_lue_est_celle_de_la_transformation_sans_sa_graine():
    config = config_du_banc()
    params = config["data"]["transforms"][0]["params"]
    loi = sampler_params_yaml(config)
    assert "seed" not in loi and loi == {cle: valeur for cle, valeur in params.items() if cle != "seed"}
    assert CrosstalkSampler(**loi) == CrosstalkSampler(**{cle: valeur for cle, valeur in params.items() if cle != "seed"})


def test_la_loi_lue_dans_le_yaml_du_resnet_est_celle_du_mlp():
    resnet = config_du_banc("cvnn_resnet.yaml")
    assert resnet["data"]["transforms"][0]["name"] == "crosstalk_injection_resnet"
    assert sampler_params_yaml(resnet) == sampler_params_yaml(config_du_banc())


def test_une_config_sans_la_transformation_est_refusee():
    config = config_du_banc()
    config["data"]["transforms"] = []
    with pytest.raises(ValueError, match="0 transformations d'injection"):
        sampler_params_yaml(config)


def test_une_config_avec_deux_transformations_d_injection_est_refusee():
    config = config_du_banc()
    mlp = config["data"]["transforms"][0]
    config["data"]["transforms"] = [mlp, {**mlp, "name": "crosstalk_injection_resnet"}]
    with pytest.raises(ValueError, match="2 transformations d'injection"):
        sampler_params_yaml(config)


# ----------------------------------------------------------------- les chemins du YAML


@sur_la_pile
def test_un_chemin_relatif_de_la_pile_se_lit_depuis_la_racine_du_depot(monkeypatch):
    monkeypatch.chdir(RACINE / "src")
    assert resolve_trainpath(config_du_banc()) == PILE


@sur_la_pile
def test_un_chemin_absolu_de_la_pile_est_garde_tel_quel():
    config = config_du_banc()
    config["data"]["dataset"]["trainpath"] = str(PILE)
    assert resolve_trainpath(config) == PILE


def test_un_chemin_relatif_du_yaml_se_lit_depuis_la_racine_et_un_absolu_reste_tel_quel(tmp_path, monkeypatch):
    monkeypatch.chdir(RACINE / "src")
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
    a_25 = 10 ** (-25 / 20)   # crosstalk injecté à -25 dB : tranche -30/-20
    e_30 = 10 ** (-30 / 20)   # erreur de -30 dB sur chaque terme
    nan = complex(np.nan, np.nan)
    chargeur = [
        lot([[0.01] * 4 + [0.2]], [[0.0] * 4 + [0.0]]),                                  # sans injection, erreur -40 dB
        lot([[a_25 + e_30] * 4 + [0.1j], [5.0] * 5], [[a_25] * 4 + [0.0], [nan] * 5]),    # -30 dB ; puis un patch écarté
    ]
    resultats = evaluate(Recopie(), chargeur, torch.device("cpu"))
    assert resultats["patchs"] == 2
    assert resultats["aucune"] == pytest.approx(-40.0, abs=1e-4)
    assert resultats["-30/-20"] == pytest.approx(-30.0, abs=1e-4)
    assert all(np.isnan(resultats[t]) for t in ("-40/-30", "-20/-15", "-15/-10", "-10/-5"))
    assert resultats["log_alpha"] == pytest.approx(0.15, abs=1e-6)
    assert resultats["n_aucune"] == resultats["n_-30/-20"] == 1
    assert all(resultats[f"n_{t}"] == 0 for t in ("-40/-30", "-20/-15", "-15/-10", "-10/-5"))
