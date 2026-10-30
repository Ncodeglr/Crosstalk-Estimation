#!/usr/bin/env python3
"""
Le script enchaîne les briques de cvnn, sans BaseExperiment :
    1. configuration : le YAML (load_config de cvnn), le device ;
    2. données : les chargeurs de cvnn, découpés selon la graine seed, puis check_pipeline sur chacun, puis le cache des
       couples (entrée, cible) (src/crosstalk/deep_learning/cache.py) : la première exécution parcourt la pile une fois,
       les suivantes relisent le cache ;
    3. wandb : setup_wandb de cvnn, qui ne fait rien si la section logging.wandb est absente ;
    4. modèle : la classe nommée par model.class, construite par create_model de cvnn, initialisée comme dans cvnn à
       partir de la graine d'entraînement train_seed, qui fixe aussi l'ordre des lots ;
    5. perte et optimiseur : CrosstalkLoss, qui lit la loi d'injection de la transformation ; l'optimiseur de torch.optim
       nommé par optim.algo ;
    6. entraînement : train_model de cvnn (échéance, écrêtage, sauvegarde du meilleur modèle, envoi à wandb) ;
    7. évaluation : le meilleur modèle en validation, sur la partition de test ;
    8. fin du run wandb.

Sorties, dans logging.logdir/<task>_<nom du run wandb, ou date et heure> (un logdir relatif se lit depuis la racine du
dépôt, comme data.dataset.trainpath ; les fichiers de wandb vont dans <racine>/wandb) :
    config.yml       la configuration complète, avec ce que cvnn a déduit des données ;
    best_model.pt    le meilleur modèle en validation, et last_model.pt le dernier, écrits par train_model ;
    historique.yaml  la perte et le terme de chaque composante, à chaque époque ;
    evaluation.yaml  les erreurs médianes sur le test, par tranche de crosstalk injecté (voir evaluate).
Le cache, lui, est écrit dans cache/<transformation>_<empreinte>/ (src/crosstalk/deep_learning/cache.py), et réutilisé
tant que les réglages des données ne changent pas.

Le script du ResNet, src/train_resnet.py, suit les mêmes étapes. Les fonctions communes aux deux scripts (evaluate,
sampler_params_yaml, from_repo_root, resolve_trainpath) sont dans src/crosstalk/deep_learning/experiment.py.
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
from src.crosstalk.deep_learning.experiment import evaluate, from_repo_root, resolve_trainpath, sampler_params_yaml  # noqa: E402
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
    # La graine seed fixe le découpage : posée juste avant get_dataloaders, comme dans src/patchs.py, elle donne le même
    # découpage que la chaîne A.
    set_seed(int(cfg["seed"]))
    cfg = validate_and_correct_config(cfg)
    chargeurs = get_dataloaders(cfg, use_cuda)
    if len(chargeurs) != 3:
        raise ValueError("data.test_ratio doit être > 0 : l'évaluation se fait sur la partition de test")
    for chargeur in chargeurs:
        check_pipeline(chargeur)
    # Les mêmes couples (entrées (B, 10, 1, 1), cibles (B, 5)), servis depuis le cache plutôt que relus dans la pile à chaque époque.
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
    set_seed(int(cfg["train_seed"])) #set_seed ne passe dans aucune fonction parce qu'elle n'en a pas besoin : elle agit sur un objet global, le générateur aléatoire de torch, que toutes les fonctions qui tirent au hasard consultent ensuite d'elles-mêmes.
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
    tranches = [nom for nom in resultats if f"n_{nom}" in resultats]
    logger.info("test (%d patchs valides) : %s | log_alpha %.4f", resultats["patchs"],
                " | ".join(f"{nom} {resultats[nom]:.2f} dB ({resultats[f'n_{nom}']} patchs)" for nom in tranches), resultats["log_alpha"])

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
