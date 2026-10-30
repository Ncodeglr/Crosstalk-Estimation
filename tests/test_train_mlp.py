"""
Tests du script src/train_mlp.py :

1. la loi d'injection lue dans le YAML, celle de la transformation, et les chemins du YAML lus depuis la racine ;
2. evaluate : erreurs médianes par tranche sur un cas construit à la main ;
3. le YAML du banc d'essai : réseau des simulations, échéance cohérente avec nepochs ;
4. sur la vraie pile, un entraînement complet de deux époques sur un petit échantillon, lancé depuis src/, sans wandb,
   avec un cache temporaire ; la graine d'entraînement change l'entraînement sans toucher au découpage. Sautés si la
   pile est absente.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.deep_learning import MLP
from src.crosstalk.crosstalk_sampler import CrosstalkSampler
from src.train_mlp import CONFIG, RACINE as RACINE_DU_SCRIPT, evaluate, from_repo_root, main, resolve_trainpath, sampler_params_yaml

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc() -> dict:
    return yaml.safe_load(CONFIG.read_text())


# ----------------------------------------------------------------- la loi d'injection


def test_la_loi_lue_est_celle_de_la_transformation_sans_sa_graine():
    config = config_du_banc()
    params = config["data"]["transforms"][0]["params"]
    loi = sampler_params_yaml(config)
    assert "seed" not in loi and loi == {cle: valeur for cle, valeur in params.items() if cle != "seed"}
    assert CrosstalkSampler(**loi) == CrosstalkSampler(**{cle: valeur for cle, valeur in params.items() if cle != "seed"})


def test_une_config_sans_la_transformation_est_refusee():
    config = config_du_banc()
    config["data"]["transforms"] = []
    with pytest.raises(ValueError):
        sampler_params_yaml(config)


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
    assert RACINE_DU_SCRIPT == RACINE
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


# ----------------------------------------------------------------- le YAML du banc d'essai


def test_le_yaml_construit_le_mlp_des_simulations():
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = config_du_banc()
    config["data"].update(inferred_input_channels=10, inferred_input_size=1)  # ce que cvnn déduit du premier lot
    reseau = create_model(config["model"]["class"], **get_model_params(config))
    assert isinstance(reseau, MLP)
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in reseau.parameters()) == 86_669


def test_l_echeance_cosinus_finit_avec_l_entrainement():
    config = config_du_banc()
    assert config["scheduler"]["params"]["T_max"] == config["nepochs"]


# ----------------------------------------------------------------- sur la vraie pile


def petite_config(dossier: Path, train_seed: int = 1, **reglages_data) -> Path:
    """Le YAML du banc d'essai réduit à 400 patchs, deux époques et la graine d'entraînement train_seed, sans wandb,
    écrit dans dossier."""
    config = config_du_banc()
    config["train_seed"] = train_seed
    config["data"].update(max_num_samples=400, num_workers=0, batch_size=64, **reglages_data)
    config["nepochs"] = config["scheduler"]["params"]["T_max"] = 2
    config["logging"] = {"logdir": str(dossier / "logs")}
    chemin = dossier / "config.yaml"
    chemin.write_text(yaml.safe_dump(config, allow_unicode=True))
    return chemin


@sur_la_pile
def test_un_entrainement_complet_ecrit_ses_sorties_et_evalue_le_test(tmp_path, monkeypatch):
    monkeypatch.chdir(RACINE / "src")   # comme « cd src ; python3 train_mlp.py » : le chemin relatif de la pile doit tenir
    monkeypatch.setenv("WANDB_DIR", "provisoire")
    monkeypatch.delenv("WANDB_DIR")      # absent pendant le test, rétabli ensuite par monkeypatch
    resultats = main(petite_config(tmp_path), dossier_cache=tmp_path / "cache")

    assert len(list((tmp_path / "cache").glob("crosstalk_injection_mlp_*"))) == 1
    assert os.environ["WANDB_DIR"] == str(RACINE)   # wandb écrit à la racine, et non dans src/

    (sortie,) = (tmp_path / "logs").iterdir()
    assert {p.name for p in sortie.iterdir()} == {"config.yml", "best_model.pt", "last_model.pt", "historique.yaml",
                                                 "evaluation.yaml"}
    assert yaml.safe_load((sortie / "config.yml").read_text())["data"]["inferred_input_channels"] == 10
    historique = yaml.safe_load((sortie / "historique.yaml").read_text())
    assert {"train_loss", "valid_loss", "train_u", "valid_log_alpha"} <= set(historique)
    assert all(len(valeurs) == 2 for valeurs in historique.values())
    assert yaml.safe_load((sortie / "evaluation.yaml").read_text()) == pytest.approx(resultats, nan_ok=True)
    assert resultats["patchs"] > 0 and np.isfinite(resultats["log_alpha"])


@sur_la_pile
def test_la_graine_d_entrainement_change_l_entrainement_mais_pas_le_decoupage(tmp_path, monkeypatch):
    """Même train_seed : le même entraînement ; une autre : un autre entraînement, sur les mêmes couples."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))   # main ne le pose alors pas pour les tests suivants
    historiques = []
    for nom, graine in (("a", 1), ("b", 1), ("c", 2)):
        (tmp_path / nom).mkdir()
        main(petite_config(tmp_path / nom, train_seed=graine), dossier_cache=tmp_path / "cache")
        (sortie,) = (tmp_path / nom / "logs").iterdir()
        historiques.append(yaml.safe_load((sortie / "historique.yaml").read_text()))

    meme, meme_bis, autre = historiques
    assert all(meme[cle] == pytest.approx(meme_bis[cle], rel=1e-6) for cle in meme)
    assert meme["train_loss"] != pytest.approx(autre["train_loss"], rel=1e-6)
    # Un seul cache, relu par les trois runs : il vérifie le découpage à chaque relecture, qui n'a donc pas changé.
    assert len(list((tmp_path / "cache").glob("crosstalk_injection_mlp_*"))) == 1


@sur_la_pile
def test_une_config_sans_partition_de_test_est_refusee(tmp_path):
    with pytest.raises(ValueError, match="test_ratio"):
        main(petite_config(tmp_path, test_ratio=0.0), dossier_cache=tmp_path / "cache")
