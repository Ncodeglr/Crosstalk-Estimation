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
Le cache, lui, est écrit dans cache/mlp_<empreinte>.pt, et réutilisé tant que les réglages des données ne changent pas.
"""

from __future__ import annotations
import argparse
import logging
import os
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
from src.crosstalk.deep_learning import CrosstalkLoss, cached_loaders, check_pipeline  # noqa: E402
from src.crosstalk.deep_learning.cache import DOSSIER_CACHE  # noqa: E402
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


def from_repo_root(chemin: str | Path) -> Path:
    """
    Un chemin du YAML : relatif, il se lit depuis la racine du dépôt, et non depuis le dossier de lancement, pour que
    le script marche aussi bien lancé depuis la racine que depuis src/ ; absolu, il est gardé tel quel.
    """
    chemin = Path(chemin)
    return chemin if chemin.is_absolute() else RACINE / chemin


def resolve_trainpath(cfg: dict[str, Any]) -> Path:
    """
    Le dossier de la pile (data.dataset.trainpath du YAML), lu par from_repo_root.

    Raises: FileNotFoundError: si le dossier ne contient aucun fichier .slc. Sans ce contrôle, torchcvnn construirait un
        jeu de données vide, et cvnn échouerait plus loin sur une erreur obscure (« num_samples=0 »).
    """
    pile = from_repo_root(cfg["data"]["dataset"]["trainpath"])
    if not any(pile.glob("*.slc")):
        raise FileNotFoundError(f"aucun fichier .slc dans {pile} (data.dataset.trainpath du YAML)")
    return pile


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    """
    Erreurs médianes du réseau sur test_loader, comme dans les simulations. Les patchs écartés (cible nan) sont ignorés.

    Returns:
        pour la tranche « aucune » et chaque tranche de TRANCHES_DB (clé « -40/-30 », etc.) : la médiane, en dB, de
        10 log10 de la moyenne de |ŷ - y|² sur u, v, w, z ; nan si la tranche est vide.
        « n_<tranche> » (« n_aucune », « n_-40/-30 », etc.) : le nombre de patchs de chaque tranche.
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

    #Étape 2 : Filtre les patchs avec des valeurs NaN dans la cible
    valides = np.isfinite(cibles).all(axis=1) #Renvoie un tableau booléen (B,). Les patchs invalides sont ceux qui contiennent un zéro ou une valeur non finie dans la covariance, et qui ont été écartés par CrosstalkInjectionTransform. 
    sorties, cibles = sorties[valides], cibles[valides] #Applique le filtre aux prédictions et aux cibles. Exemple : si 10 % des patchs sont invalides, sorties et cibles passent de (1000, 5) à (900, 5).

    #Étape 3 : erreur et niveau de crosstalk injecté de chaque patch, en dB ; un patch sans injection a un niveau -inf.
    erreur_db = 10 * np.log10(np.mean(np.abs(sorties[:, :4] - cibles[:, :4]) ** 2, axis=1))
    with np.errstate(divide="ignore"): #Si un patch n'a pas de crosstalk injecté, sa cible vaut zéro et son niveau est -inf. La médiane de la tranche « aucune » est donc -inf, ce qui est correct.
        niveau_db = 10 * np.log10(np.mean(np.abs(cibles[:, :4]) ** 2, axis=1)) #Calcul du niveau de crosstalk injecté, en dB, pour chaque patch. Un patch sans injection a un niveau -inf.

    #Étape 4 : médiane par tranche.
    tranches = {"aucune": niveau_db == -np.inf} #Création d'un dictionnaire à une seule entrée. La clé est "aucune". La valeur est un masque, cad un tableau de booléens avec une case par patch
    tranches.update({f"{a}/{b}": (niveau_db >= a) & (niveau_db < b) for a, b in TRANCHES_DB})
    resultats = {nom: float(np.median(erreur_db[dans])) if dans.any() else float("nan") for nom, dans in tranches.items()}
    resultats.update({f"n_{nom}": int(dans.sum()) for nom, dans in tranches.items()}) #Effectif de chaque tranche : le nombre de True de son masque. Une médiane sur 3 000 patchs pèse moins qu'une médiane sur 20 000.
    resultats["log_alpha"] = float(np.median(np.abs(sorties[:, 4] - cibles[:, 4])))
    resultats["patchs"] = int(valides.sum())
    return resultats


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
