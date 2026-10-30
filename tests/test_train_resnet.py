"""
Tests du script src/train_resnet.py :

1. le YAML du banc d'essai : le ResNet de référence, échéance cohérente avec nepochs, et le même découpage et la même
   injection que le YAML du MLP, condition pour comparer les deux réseaux sur les mêmes patchs de test ;
2. sur la vraie pile, un entraînement complet de deux époques sur un petit échantillon, lancé depuis src/, sans wandb,
   avec un cache temporaire ; deux entraînements de même graine sont identiques, convolutions sur GPU comprises.
   Sautés si la pile est absente.
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.crosstalk.deep_learning import ResNet
from src.train_resnet import CONFIG, main

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc() -> dict:
    return yaml.safe_load(CONFIG.read_text())


# ----------------------------------------------------------------- le YAML du banc d'essai


def test_le_yaml_construit_le_resnet_de_reference():
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = config_du_banc()
    config["data"].update(inferred_input_channels=4, inferred_input_size=64)  # ce que cvnn déduit du premier lot
    reseau = create_model(config["model"]["class"], **get_model_params(config))
    assert isinstance(reseau, ResNet)
    # CReLU, largeur 16, 4 blocs Down : 16, 32, 64, 128, 256 canaux, puis la tête 256 -> 256 -> 5.
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in reseau.parameters()) == 2_575_338


def test_l_echeance_cosinus_finit_avec_l_entrainement():
    config = config_du_banc()
    assert config["scheduler"]["params"]["T_max"] == config["nepochs"]


def decoupage_et_injection(config: dict) -> tuple:
    """
    Ce qui détermine les patchs de test et leurs injections : la graine du découpage et la section data, sans ce qui
    peut différer d'un réseau à l'autre (taille des lots, workers, sous-ensemble d'entraînement, nom de la
    transformation).
    """
    donnees = copy.deepcopy(config["data"])
    for cle in ("batch_size", "num_workers", "max_train_samples"):
        donnees.pop(cle, None)
    for transformation in donnees["transforms"]:
        del transformation["name"]
    return config["seed"], donnees


def test_le_resnet_et_le_mlp_ont_le_meme_decoupage_et_la_meme_injection():
    """Condition pour que leurs evaluation.yaml portent sur les mêmes patchs de test, avec les mêmes theta_k."""
    resnet = config_du_banc()
    mlp = yaml.safe_load((RACINE / "configs" / "cvnn_mlp.yaml").read_text())
    assert [t["name"] for t in resnet["data"]["transforms"]] == ["crosstalk_injection_resnet"]
    assert [t["name"] for t in mlp["data"]["transforms"]] == ["crosstalk_injection_mlp"]
    assert decoupage_et_injection(resnet) == decoupage_et_injection(mlp)


# ----------------------------------------------------------------- sur la vraie pile


def petite_config(dossier: Path, train_seed: int = 1) -> Path:
    """Le YAML du banc d'essai réduit à 400 patchs, dont 200 d'entraînement, deux époques et la graine d'entraînement
    train_seed, sans wandb, écrit dans dossier."""
    config = config_du_banc()
    config["train_seed"] = train_seed
    config["data"].update(max_num_samples=400, max_train_samples=200, num_workers=0, batch_size=64)
    config["nepochs"] = config["scheduler"]["params"]["T_max"] = 2
    config["logging"] = {"logdir": str(dossier / "logs")}
    chemin = dossier / "config.yaml"
    chemin.write_text(yaml.safe_dump(config, allow_unicode=True))
    return chemin


@sur_la_pile
def test_un_entrainement_complet_ecrit_ses_sorties_et_evalue_le_test(tmp_path, monkeypatch):
    monkeypatch.chdir(RACINE / "src")   # comme « cd src ; python3 train_resnet.py » : le chemin relatif de la pile doit tenir
    monkeypatch.setenv("WANDB_DIR", "provisoire")
    monkeypatch.delenv("WANDB_DIR")      # absent pendant le test, rétabli ensuite par monkeypatch
    resultats = main(petite_config(tmp_path), dossier_cache=tmp_path / "cache")

    (cache,) = (tmp_path / "cache").iterdir()
    assert cache.name.startswith("crosstalk_injection_resnet_")
    assert len(np.load(cache / "train_indices.npy")) == 200   # max_train_samples
    assert os.environ["WANDB_DIR"] == str(RACINE)   # wandb écrit à la racine, et non dans src/

    (sortie,) = (tmp_path / "logs").iterdir()
    assert {p.name for p in sortie.iterdir()} == {"config.yml", "best_model.pt", "last_model.pt", "historique.yaml",
                                                 "evaluation.yaml"}
    donnees = yaml.safe_load((sortie / "config.yml").read_text())["data"]
    assert (donnees["inferred_input_channels"], donnees["inferred_input_size"]) == (4, 64)   # le patch (4, 64, 64)
    historique = yaml.safe_load((sortie / "historique.yaml").read_text())
    assert {"train_loss", "valid_loss", "train_u", "valid_log_alpha"} <= set(historique)
    assert all(len(valeurs) == 2 for valeurs in historique.values())
    assert yaml.safe_load((sortie / "evaluation.yaml").read_text()) == pytest.approx(resultats, nan_ok=True)
    assert resultats["patchs"] > 0 and np.isfinite(resultats["log_alpha"])


@sur_la_pile
def test_deux_entrainements_de_meme_graine_sont_identiques(tmp_path, monkeypatch):
    """À train_seed égale, les convolutions sur GPU ne doivent introduire aucun hasard : mêmes historiques et mêmes
    résultats de test, au bit près (les fichiers YAML sont identiques)."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))   # main ne le pose alors pas pour les tests suivants
    sorties = []
    for nom in ("a", "b"):
        (tmp_path / nom).mkdir()
        main(petite_config(tmp_path / nom), dossier_cache=tmp_path / "cache")
        (sortie,) = (tmp_path / nom / "logs").iterdir()
        sorties.append({fichier: (sortie / fichier).read_text() for fichier in ("historique.yaml", "evaluation.yaml")})
    assert sorties[0] == sorties[1]
