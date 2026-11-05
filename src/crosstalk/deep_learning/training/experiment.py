"""
Ce que partagent les scripts d'entraînement du MLP (src/scripts/train_mlp.py) et du ResNet (src/scripts/train_resnet.py) : les chemins
du YAML lus depuis la racine du dépôt, l'initialisation du réseau, et l'évaluation sur la partition de test.

Les deux réseaux sont ainsi évalués de la même façon. Quand leurs YAML ont le même découpage et la même loi
d'injection, ils le sont aussi sur les mêmes patchs de test, avec les mêmes theta_k : leurs evaluation.yaml se
comparent tranche par tranche.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torchcvnn.nn.modules as c_nn
from cvnn.models.utils import init_weights_mode_aware
from tqdm import tqdm
from ...paths import RACINE

#: Tranches de crosstalk injecté, en dB : elles découpent par pas de 5 dB l'enveloppe crosstalk_db = [-35, -10] des YAML.
#: Le niveau d'un patch est 20 log10 de la moyenne quadratique de |u|, |v|, |w|, |z| de sa cible ; un patch laissé sans
#: injection forme la tranche « aucune ».
TRANCHES_DB: tuple[tuple[int, int], ...] = ((-35, -30), (-30, -25), (-25, -20), (-20, -15), (-15, -10))

#: Les valeurs acceptées pour model.init_weights, et la fonction de torchcvnn qui remplace les poids tirés par cvnn
#: (None : ceux de cvnn sont gardés).
INITIALISATIONS: dict[str, Any] = {
    "kaiming_normal": None,
    "kaiming_uniform": c_nn.init.complex_kaiming_uniform_,
    "xavier_normal": c_nn.init.complex_xavier_normal_,
    "xavier_uniform": c_nn.init.complex_xavier_uniform_,
}


def from_repo_root(chemin: str | Path) -> Path:
    """
    Un chemin du YAML : relatif, il se lit depuis la racine du dépôt, et non depuis le dossier de lancement, pour que
    les scripts marchent aussi bien lancés depuis la racine que depuis src/scripts/ ; absolu, il est gardé tel quel.
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


def init_weights(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    """
    Initialise le réseau selon model.init_weights du YAML.

    cvnn n'offre aucun choix : init_weights_mode_aware tire les poids en Kaiming normal, met les biais à 0,01 et le seuil
    de modReLU à 0,25. Elle est donc toujours appliquée, ce qui garde ces deux derniers réglages ; pour une autre
    initialisation, seuls les poids sont ensuite retirés, par la fonction de torchcvnn et pour la même activation (ReLU).

    Raises: ValueError: si model.init_weights est absent ou inconnu.
    """
    nom = cfg["model"].get("init_weights")
    if nom not in INITIALISATIONS:
        raise ValueError(f"model.init_weights = {nom!r} : valeurs acceptées {list(INITIALISATIONS)}")
    model.apply(lambda module: init_weights_mode_aware(module, cfg["model"]["layer_mode"]))
    if INITIALISATIONS[nom] is not None:
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                INITIALISATIONS[nom](module.weight, nonlinearity="relu")


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    """
    Erreurs médianes du réseau sur test_loader, comme dans les simulations. Les patchs écartés (cible nan) sont ignorés.

    Returns:
        pour la tranche « aucune » et chaque tranche de TRANCHES_DB (clé « -35/-30 », etc.) : la médiane, en dB, de
        10 log10 de la moyenne de |ŷ - y|² sur u, v, w, z ; nan si la tranche est vide.
        « n_<tranche> » (« n_aucune », « n_-35/-30 », etc.) : le nombre de patchs de chaque tranche.
        « alpha_<tranche> » (« alpha_aucune », « alpha_-35/-30 », etc.) : la médiane, en dB, de 10 log10 de
        |alpha estimé - alpha|² ; nan si la tranche est vide. Alpha reste hors de la moyenne sur u, v, w, z : proche de 1,
        il y masquerait l'erreur sur le crosstalk.
        « alpha » : la médiane de |alpha estimé - alpha|, sur tous les patchs valides.
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
    valides = np.isfinite(cibles).all(axis=1) #Renvoie un tableau booléen (B,). Les patchs invalides sont ceux qui contiennent un zéro ou une valeur non finie (bord de fauchée), et qui ont été écartés par la transformation d'injection.
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
    erreur_alpha_db = 10 * np.log10(np.abs(sorties[:, 4] - cibles[:, 4]) ** 2) #Même calcul que erreur_db, pour alpha seul.
    resultats.update({f"alpha_{nom}": float(np.median(erreur_alpha_db[dans])) if dans.any() else float("nan") for nom, dans in tranches.items()})
    resultats["alpha"] = float(np.median(np.abs(sorties[:, 4] - cibles[:, 4]))) #Erreur de chaque patch, |alpha estimé - alpha|, puis médiane.
    resultats["patchs"] = int(valides.sum())
    return resultats
