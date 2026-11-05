#!/usr/bin/env python3
"""
Entraîne le ResNet sur le patch brut, puis l'évalue sur la partition de test, comme src/train_mlp.py le fait pour le MLP.
Le ResNet (src/crosstalk/deep_learning/resnet_model.py) reçoit le patch injecté (4, H, W). Le script enchaîne les mêmes briques de cvnn, sans BaseExperiment :
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import torch
import yaml

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

# Importer ce paquet ajoute cvnn au chemin et enregistre la transformation et le réseau dans ses registres.
from src.crosstalk.deep_learning import CrosstalkLoss, cached_loaders, check_pipeline  # noqa: E402
from src.crosstalk.deep_learning.cache import DOSSIER_CACHE  # noqa: E402
from src.crosstalk.deep_learning.experiment import evaluate, from_repo_root, init_weights, resolve_trainpath  # noqa: E402
from src.crosstalk.matrix import POL_ORDER  # noqa: E402
from cvnn.config import load_config  # noqa: E402
from cvnn.config_utils import get_model_params  # noqa: E402
from cvnn.data import get_dataloaders, validate_and_correct_config  # noqa: E402
from cvnn.models import create_model  # noqa: E402
from cvnn.train import train_model  # noqa: E402
from cvnn.utils import set_seed  # noqa: E402
from cvnn.wandb_utils import finish_wandb_run, log_metrics, setup_wandb  # noqa: E402

logger = logging.getLogger("train_resnet")

#: Configuration par défaut : celle du banc d'essai par réinjection, pour le ResNet.
CONFIG: Path = RACINE / "configs" / "cvnn_resnet.yaml"


def main(chemin_config: Path = CONFIG, dossier_cache: Path = DOSSIER_CACHE) -> dict[str, float]:
    """Déroule l'entraînement puis l'évaluation (étapes de la docstring du module) ; rend les résultats du test."""
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    cfg = load_config(chemin_config)
    # Les chemins relatifs du YAML se lisent depuis la racine du dépôt, d'où qu'on lance le script.
    cfg["data"]["dataset"]["trainpath"] = str(resolve_trainpath(cfg))
    cfg["logging"]["logdir"] = str(from_repo_root(cfg["logging"]["logdir"]))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ==========================================
    # 2. DONNÉES (train_loader, valid_loader, test_loader)
    # ==========================================
    # La graine seed fixe le découpage : posée juste avant get_dataloaders, comme dans src/patchs.py, elle donne le même découpage que la chaîne A.
    set_seed(int(cfg["seed"]))
    cfg = validate_and_correct_config(cfg)
    chargeurs = get_dataloaders(cfg, use_cuda)
    if len(chargeurs) != 3:
        raise ValueError("data.test_ratio doit être > 0 : l'évaluation se fait sur la partition de test")
    
    # L'entrée du ResNet est le patch injecté : ses 4 canaux de polarisation, à la taille des patchs du YAML.
    taille = cfg["data"]["dataset"]["patch_size"]
    for chargeur in chargeurs:
        check_pipeline(chargeur, forme_entree=(len(POL_ORDER), taille, taille))
    # Les mêmes couples (entrée, cible), servis depuis le cache plutôt que relus dans la pile à chaque époque.
    train_loader, valid_loader, test_loader = cached_loaders(chargeurs, cfg, dossier_cache)

    # ==========================================
    # 3. WANDB ET DOSSIER DE SORTIE
    # ==========================================
    # Après les données : la configuration envoyée à wandb contient ce que cvnn en a déduit.
    # wandb range ses fichiers dans <WANDB_DIR>/wandb, par défaut dans le dossier de lancement : on le fixe à la racine,
    # sauf si l'utilisateur a déjà défini WANDB_DIR.
    os.environ.setdefault("WANDB_DIR", str(RACINE))
    wandb_log, run_name = setup_wandb(cfg)
    logdir = Path(cfg["logging"]["logdir"]) / f"{cfg['task']}_{run_name or time.strftime('%Y%m%d_%H%M%S')}"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.yml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    logger.info("dossier de sortie : %s", logdir)

    # ==========================================
    # 4. INITIALISATION DU MODÈLE
    # ==========================================
    # La graine d'entraînement fixe l'initialisation des poids, puis l'ordre des lots à chaque époque. Posée ici, après
    # les données et wandb, elle ne dépend pas de ce que ces étapes ont tiré dans les générateurs aléatoires.
    set_seed(int(cfg["seed_train"])) #set_seed ne passe dans aucune fonction parce qu'elle n'en a pas besoin : elle agit sur un objet global, le générateur aléatoire de torch, que toutes les fonctions qui tirent au hasard consultent ensuite d'elles-mêmes.
    model = create_model(cfg["model"]["class"], **get_model_params(cfg))
    init_weights(model, cfg)
    model.to(device)

    # ==========================================
    # 5. PERTE ET OPTIMISEUR
    # ==========================================
    loss_fn = CrosstalkLoss()
    optimizer = getattr(torch.optim, cfg["optim"]["algo"])(model.parameters(), **cfg["optim"]["params"])

    # ==========================================
    # 6. ENTRAÎNEMENT ET VALIDATION (train_model de cvnn)
    # ==========================================
    # À chaque époque : phase d'entraînement sur train_loader, puis phase de validation sur valid_loader.
    # Le meilleur modèle en validation est sauvegardé dans best_model.pt.
    historique = train_model(model, train_loader, valid_loader, cfg, logdir, device, loss_fn, optimizer)
    (logdir / "historique.yaml").write_text(yaml.safe_dump(dict(historique), sort_keys=False)) #historique est un dictionnaire de listes, avec les clés « train_loss », « valid_loss » et « valid_<composante> » pour chaque composante de la perte. Chaque liste contient la valeur de la métrique à chaque époque.

    # ==========================================
    # 7. ÉVALUATION FINALE (test_loader, une seule fois, sur best_model.pt)
    # ==========================================
    model.load_state_dict(torch.load(logdir / "best_model.pt", map_location=device)["model_state_dict"])
    resultats = evaluate(model, test_loader, device)
    (logdir / "evaluation.yaml").write_text(yaml.safe_dump(resultats, sort_keys=False))
    log_metrics(wandb_log, resultats, prefix="test")
    tranches = [nom for nom in resultats if f"n_{nom}" in resultats]
    logger.info("test (%d patchs valides) : %s | alpha %.4f", resultats["patchs"],
                " | ".join(f"{nom} {resultats[nom]:.2f} dB ({resultats[f'n_{nom}']} patchs)" for nom in tranches), resultats["alpha"])

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
