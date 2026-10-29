#!/usr/bin/env python3
"""
Le script enchaîne les briques de cvnn, sans BaseExperiment :
    1. configuration : le YAML (load_config de cvnn), le device ;
    2. données : les chargeurs de cvnn, puis check_pipeline sur chacun ;
    3. wandb : setup_wandb de cvnn, qui ne fait rien si la section logging.wandb est absente ;
    4. modèle : la classe nommée par model.class, construite par create_model de cvnn, initialisée comme dans cvnn ;
    5. perte et optimiseur : CrosstalkLoss, qui lit la loi d'injection de la transformation ; l'optimiseur de torch.optim
       nommé par optim.algo ;
    6. entraînement : train_model de cvnn (échéance, écrêtage, sauvegarde du meilleur modèle, envoi à wandb) ;
    7. évaluation : le meilleur modèle en validation, sur la partition de test ;
    8. fin du run wandb.

Sorties, dans logging.logdir/<task>_<nom du run wandb, ou date et heure> :
    config.yml       la configuration complète, avec ce que cvnn a déduit des données ;
    best_model.pt    le meilleur modèle en validation, et last_model.pt le dernier, écrits par train_model ;
    historique.yaml  la perte et le terme de chaque composante, à chaque époque ;
    evaluation.yaml  les erreurs médianes sur le test, par tranche de crosstalk injecté (voir evaluate).
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from tqdm import tqdm

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

# Importer ce paquet ajoute cvnn au chemin et enregistre la transformation et le réseau dans ses registres.
from src.crosstalk.deep_learning import CrosstalkLoss, check_pipeline  # noqa: E402
from cvnn.config import load_config  # noqa: E402
from cvnn.config_utils import get_model_params  # noqa: E402
from cvnn.data import get_dataloaders, validate_and_correct_config  # noqa: E402
from cvnn.models import create_model  # noqa: E402
from cvnn.models.utils import init_weights_mode_aware  # noqa: E402
from cvnn.train import train_model  # noqa: E402
from cvnn.utils import set_seed  # noqa: E402
from cvnn.wandb_utils import finish_wandb_run, log_metrics, setup_wandb  # noqa: E402

logger = logging.getLogger("train_mlp")

#: Configuration par défaut : celle du banc d'essai par réinjection.
CONFIG: Path = RACINE / "configs" / "cvnn_mlp.yaml"

#: Tranches de crosstalk injecté, en dB, celles des simulations. Le niveau d'un patch est 20 log10 de la moyenne
#: quadratique de |u|, |v|, |w|, |z| de sa cible ; un patch laissé sans injection forme la tranche « aucune ».
TRANCHES_DB: tuple[tuple[int, int], ...] = ((-40, -30), (-30, -20), (-20, -15), (-15, -10), (-10, -5))


def sampler_params_yaml(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Les paramètres de CrosstalkSampler (la loi d'injection), lus dans la transformation « crosstalk_injection » du YAML,
    sans sa graine.

    Raises: ValueError: si data.transforms ne contient pas la transformation « crosstalk_injection ».
    """
    for transformation in cfg["data"].get("transforms") or []:
        if transformation.get("name") == "crosstalk_injection":
            return {cle: valeur for cle, valeur in (transformation.get("params") or {}).items() if cle != "seed"}
    raise ValueError("data.transforms ne contient pas la transformation « crosstalk_injection »")



def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    """
    Erreurs médianes du réseau sur test_loader, comme dans les simulations. Les patchs écartés (cible nan) sont ignorés.

    Returns:
        pour la tranche « aucune » et chaque tranche de TRANCHES_DB (clé « -40/-30 », etc.) : la médiane, en dB, de
        10 log10 de la moyenne de |ŷ - y|² sur u, v, w, z ; nan si la tranche est vide.
        « log_alpha » : la médiane de |ŷ - y| sur log alpha, sur tous les patchs valides.
        « patchs » : le nombre de patchs valides.
    """
    #Étape 1 : prédictions et cibles de tout le chargeur, sur CPU.
    model.eval()
    sorties, cibles = [], []
    with torch.no_grad():
        for entrees, cible in tqdm(test_loader, desc="Test"):
            sorties.append(model(entrees.to(device)).cpu())
            cibles.append(cible)
    sorties, cibles = torch.cat(sorties).numpy(), torch.cat(cibles).numpy()

    #Étape 2 : seuls les patchs valides.
    valides = np.isfinite(cibles).all(axis=1)
    sorties, cibles = sorties[valides], cibles[valides]

    #Étape 3 : erreur et niveau de crosstalk injecté de chaque patch, en dB ; un patch sans injection a un niveau -inf.
    erreur_db = 10 * np.log10(np.mean(np.abs(sorties[:, :4] - cibles[:, :4]) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        niveau_db = 10 * np.log10(np.mean(np.abs(cibles[:, :4]) ** 2, axis=1))

    #Étape 4 : médiane par tranche.
    tranches = {"aucune": niveau_db == -np.inf}
    tranches.update({f"{a}/{b}": (niveau_db >= a) & (niveau_db < b) for a, b in TRANCHES_DB})
    resultats = {nom: float(np.median(erreur_db[dans])) if dans.any() else float("nan") for nom, dans in tranches.items()}
    resultats["log_alpha"] = float(np.median(np.abs(sorties[:, 4] - cibles[:, 4])))
    resultats["patchs"] = int(valides.sum())
    return resultats


def main(chemin_config: Path = CONFIG) -> dict[str, float]:
    """Déroule l'entraînement puis l'évaluation (étapes de la docstring du module) ; rend les résultats du test."""
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    cfg = load_config(chemin_config)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ==========================================
    # 2. DONNÉES (train_loader, valid_loader, test_loader)
    # ==========================================
    # set_seed juste avant get_dataloaders, comme src/patchs.py : même découpage que la chaîne A.
    set_seed(int(cfg["seed"]))
    cfg = validate_and_correct_config(cfg)
    chargeurs = get_dataloaders(cfg, use_cuda)
    if len(chargeurs) != 3:
        raise ValueError("data.test_ratio doit être > 0 : l'évaluation se fait sur la partition de test")
    train_loader, valid_loader, test_loader = chargeurs
    for chargeur in chargeurs:
        check_pipeline(chargeur)

    # ==========================================
    # 3. WANDB ET DOSSIER DE SORTIE
    # ==========================================
    # Après les données : la configuration envoyée à wandb contient ce que cvnn en a déduit.
    wandb_log, run_name = setup_wandb(cfg)
    logdir = Path(cfg["logging"]["logdir"]) / f"{cfg['task']}_{run_name or time.strftime('%Y%m%d_%H%M%S')}"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.yml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    logger.info("dossier de sortie : %s", logdir)

    # ==========================================
    # 4. INITIALISATION DU MODÈLE
    # ==========================================
    model = create_model(cfg["model"]["class"], **get_model_params(cfg))
    model.apply(lambda m: init_weights_mode_aware(m, cfg["model"]["layer_mode"]))
    model.to(device)

    # ==========================================
    # 5. PERTE ET OPTIMISEUR
    # ==========================================
    loss_fn = CrosstalkLoss(**sampler_params_yaml(cfg)).to(device)
    optimizer = getattr(torch.optim, cfg["optim"]["algo"])(model.parameters(), **cfg["optim"]["params"])

    # ==========================================
    # 6. ENTRAÎNEMENT ET VALIDATION (train_model de cvnn)
    # ==========================================
    # À chaque époque : phase d'entraînement sur train_loader, puis phase de validation sur valid_loader.
    # Le meilleur modèle en validation est sauvegardé dans best_model.pt.
    historique = train_model(model, train_loader, valid_loader, cfg, logdir, device, loss_fn, optimizer)
    (logdir / "historique.yaml").write_text(yaml.safe_dump(dict(historique), sort_keys=False))

    # ==========================================
    # 7. ÉVALUATION FINALE (test_loader, une seule fois, sur best_model.pt)
    # ==========================================
    model.load_state_dict(torch.load(logdir / "best_model.pt", map_location=device)["model_state_dict"])
    resultats = evaluate(model, test_loader, device)
    (logdir / "evaluation.yaml").write_text(yaml.safe_dump(resultats, sort_keys=False))
    log_metrics(wandb_log, resultats, prefix="test")
    logger.info("test (%d patchs valides) : %s", resultats["patchs"]," | ".join(f"{nom} {valeur:.2f}" for nom, valeur in resultats.items() if nom != "patchs"))

    # ==========================================
    # 8. FIN DU RUN WANDB
    # ==========================================
    finish_wandb_run()
    return resultats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parseur = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parseur.add_argument("config", nargs="?", type=Path, default=CONFIG, help="YAML de l'expérience (défaut : %(default)s)")
    main(parseur.parse_args().config)
