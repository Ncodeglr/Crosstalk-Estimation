"""
La perte du banc d'essai par réinjection : CrosstalkLoss.

Formule
-------
Soit un lot de B patchs, dont V valides (cible finie), et les 5 composantes j de la cible y = (u, v, w, z, alpha).
La perte vaut :

    L = (1 / V) * Σ_{i valide} Σ_j |ŷ_ij - y_ij|²

|ŷ - y|² est la MSE complexe de torchcvnn (ComplexMSELoss, sans réduction). CrosstalkLoss n'y ajoute que le retrait des
patchs écartés. Elle somme les composantes au lieu de les moyenner : le terme de chaque composante est ainsi une
métrique, et leur somme vaut L.

Échelle des composantes
-----------------------
Aucune normalisation : chaque composante garde son échelle. Sous la loi par défaut, E[|u|²] vaut 0,0114 par terme de
crosstalk et Var(alpha) vaut 0,133 : alpha pèse donc environ 11,6 fois plus que chaque terme de crosstalk.
Mesure faite le 2026-11-05 (MLP 3 x 128, patchs 128 x 128, seed 16, 3 graines, 50 époques) : par rapport à l'ancienne
perte, normalisée par E[|y|²] avec log alpha en cible, u, v, w, z perdent de 0,34 à 0,49 dB selon la tranche, sur les
3 graines et les 5 tranches, et l'erreur sur alpha gagne environ 1 %. Cette mesure n'a pas été faite sur le ResNet.
Les valeurs de la perte ne se comparent donc pas à celles des runs entraînés avec l'ancienne perte.

Patchs écartés
--------------
La transformation donne une cible nan aux patchs écartés (zéro ou valeur non finie). Ils sont retirés AVANT la
soustraction : (ŷ - nan) * 0 vaut nan, et un seul nan dans le gradient corrompt tous les poids. Un lot sans aucun
patch valide donne L = 0, rattaché au graphe, et non nan.

Avec cvnn
---------
train_model de cvnn appelle loss_fn(sorties, cibles) et accepte un couple (perte, métriques). Il moyenne les métriques
sur l'époque (pondérées par la taille des lots), les range dans l'historique (train_u, valid_u, etc.) et les envoie à
wandb (training/u, validation/u, etc.) si la section logging.wandb existe. CrosstalkLoss renvoie donc, en plus de L,
le terme de chaque composante ; leur somme vaut L.
CrosstalkLoss n'hérite pas d'une MSE : la boucle d'entraînement de cvnn comparerait sinon les sorties aux entrées
(cvnn/src/cvnn/train.py, train_one_epoch).
"""

from __future__ import annotations
import torch
from torch import nn
from torchcvnn.nn.modules import ComplexMSELoss
from ...polarimetry.parameters import PARAMETER_NAMES
from ..crosstalk_injection import CARD_THETA

#: Noms des composantes de la cible, dans l'ordre de CrosstalkInjector.inject. Ce sont les clés des métriques.
COMPOSANTES: tuple[str, ...] = PARAMETER_NAMES


class CrosstalkLoss(nn.Module):
    """(sorties (B, 5), cibles (B, 5)) -> (L, {composante: terme}) : la formule de la docstring du module."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = ComplexMSELoss(reduction="none")

    def forward(self, theta_estimate: torch.Tensor, theta_true: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if theta_estimate.ndim != 2 or theta_estimate.shape[1] != CARD_THETA or theta_estimate.shape != theta_true.shape:
            raise ValueError(f"sorties {tuple(theta_estimate.shape)} et cibles {tuple(theta_true.shape)} : (B, {CARD_THETA}) attendu pour les deux")

        valides = torch.isfinite(theta_true).all(dim=1) #Étape 1 : (B,), les patchs dont la cible est finie.
        erreur = self.mse(theta_estimate[valides], theta_true[valides])   #Étape 2 : (V, 5), |ŷ - y|² sur les seuls patchs valides.
        termes = erreur.sum(dim=0) / valides.sum().clamp(min=1) #Étape 3 : (5,), moyenne sur les patchs valides ; 0 si V = 0.
        return termes.sum(), {nom: terme.detach() for nom, terme in zip(COMPOSANTES, termes)}
