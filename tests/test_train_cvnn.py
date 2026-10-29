"""
Tests du script src/train_cvnn.py :

1. la loi d'injection lue dans le YAML, celle de la transformation ;
2. evaluate : erreurs médianes par tranche sur un cas construit à la main ;
3. le YAML du banc d'essai : réseau des simulations, échéance cohérente avec nepochs ;
4. sur la vraie pile, un entraînement complet de deux époques sur un petit échantillon, sans wandb. Sautés si la pile
   est absente.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.deep_learning import CrosstalkLoss, MLP
from src.crosstalk.injection import InjectionLaw
from src.train_cvnn import CONFIG, evaluate, injection_law, main

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc() -> dict:
    return yaml.safe_load(CONFIG.read_text())


# ----------------------------------------------------------------- la loi d'injection


def test_la_loi_lue_est_celle_de_la_transformation_sans_sa_graine():
    config = config_du_banc()
    params = config["data"]["transforms"][0]["params"]
    loi = injection_law(config)
    assert "seed" not in loi and loi == {cle: valeur for cle, valeur in params.items() if cle != "seed"}
    assert InjectionLaw(**loi) == InjectionLaw(**{cle: valeur for cle, valeur in params.items() if cle != "seed"})


def test_une_config_sans_la_transformation_est_refusee():
    config = config_du_banc()
    config["data"]["transforms"] = []
    with pytest.raises(ValueError):
        injection_law(config)


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


# ----------------------------------------------------------------- le YAML du banc d'essai


def test_le_yaml_construit_le_mlp_des_simulations():
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = config_du_banc()
    config["data"].update(inferred_input_channels=10, inferred_input_size=1)  # ce que cvnn déduit du premier lot
    reseau = create_model(config["model"]["class"], **get_model_params(config))
    assert isinstance(reseau, MLP)
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in reseau.parameters()) == 70_157


def test_l_echeance_cosinus_finit_avec_l_entrainement():
    config = config_du_banc()
    assert config["scheduler"]["params"]["T_max"] == config["nepochs"]


# ----------------------------------------------------------------- sur la vraie pile


def petite_config(dossier: Path, **reglages_data) -> Path:
    """Le YAML du banc d'essai réduit à 400 patchs et deux époques, sans wandb, écrit dans dossier."""
    config = config_du_banc()
    config["data"]["dataset"]["trainpath"] = str(PILE)
    config["data"].update(max_num_samples=400, num_workers=0, batch_size=64, **reglages_data)
    config["nepochs"] = config["scheduler"]["params"]["T_max"] = 2
    config["logging"] = {"logdir": str(dossier / "logs")}
    chemin = dossier / "config.yaml"
    chemin.write_text(yaml.safe_dump(config, allow_unicode=True))
    return chemin


@sur_la_pile
def test_un_entrainement_complet_ecrit_ses_sorties_et_evalue_le_test(tmp_path):
    resultats = main(petite_config(tmp_path))

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
def test_une_config_sans_partition_de_test_est_refusee(tmp_path):
    with pytest.raises(ValueError, match="test_ratio"):
        main(petite_config(tmp_path, test_ratio=0.0))
