"""
La perte du banc d'essai par réinjection : CrosstalkLoss.

Formule
-------
Soit un lot de B patchs, dont V valides (cible finie), et les 5 composantes j de la cible y = (u, v, w, z, log alpha).
La perte vaut :

    L = (1 / V) * Σ_{i valide} Σ_j |ŷ_ij - y_ij|² / σ_j²,        avec σ_j² = E[|y_j|²] sous la loi d'injection

|ŷ - y|² est la MSE complexe de torchcvnn (ComplexMSELoss, sans réduction). CrosstalkLoss n'y ajoute que deux
choses : le retrait des patchs écartés et la division par σ_j².

Pourquoi diviser par σ_j²
-------------------------
Les composantes n'ont pas la même échelle. Sous la loi par défaut, σ² vaut 0,0186 pour chaque terme de crosstalk et
0,105 pour log alpha : avec la MSE complexe seule, log alpha pèserait 5,6 fois plus que chaque terme de crosstalk.
Mesure faite le 2026-10-29 (MLP, 580 000 patchs, deux graines) : la MSE complexe seule perd 0,6, 1,0 et 1,3 dB sur
les tranches -20/-15, -15/-10 et -10/-5 dB, pour un écart entre graines d'au plus 0,16 dB.
Une fois divisée par σ_j², chaque composante coûte exactement 1 au prédicteur nul, qui prédit « sans crosstalk,
alpha = 1 » : ce prédicteur a donc L = 5.

σ_j² est estimé une seule fois, à la construction, sur N_TIRAGES tirages de draw_injection à la graine GRAINE_SIGMA2.
Il suit ainsi la loi d'injection telle qu'elle est codée, sans formule à tenir à jour. tests/test_loss.py vérifie
l'estimation contre la formule fermée de la loi actuelle.

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
import numpy as np
import torch
from torch import nn
from torchcvnn.nn.modules import ComplexMSELoss
from ..injection import InjectionLaw, draw_injection
from .injection_transform import CARD_THETA

#: Noms des composantes de la cible, dans l'ordre de CrosstalkInjectionTransform. Ce sont les clés des métriques.
COMPOSANTES: tuple[str, ...] = ("u", "v", "w", "z", "log_alpha")

#: Nombre de tirages et graine de l'estimation de σ². Avec 200 000 tirages, l'estimation prend 1,4 s, et deux graines
#: différentes donnent des σ² qui diffèrent d'environ 1 %.
N_TIRAGES: int = 200_000
GRAINE_SIGMA2: int = 0


def target_power(law: InjectionLaw, n_tirages: int = N_TIRAGES, graine: int = GRAINE_SIGMA2) -> np.ndarray:
    """
    σ² = E[|y_j|²] pour chaque composante de la cible (u, v, w, z, log alpha), estimé sur n_tirages tirages de law.

    La cible est construite comme dans CrosstalkInjectionTransform : log alpha est le logarithme complexe de alpha.

    Returns:
        tableau (5,) de réels positifs.
    """
    rng = np.random.default_rng(graine)
    cibles = np.array([draw_injection(law, rng).to_vector() for _ in range(n_tirages)])  #Étape 1 : (n_tirages, 5), (u, v, w, z, alpha).
    cibles[:, 4] = np.log(cibles[:, 4])                                                    #Étape 2 : alpha -> log alpha, comme la cible.
    return (np.abs(cibles) ** 2).mean(axis=0)                                              #Étape 3 : moyenne de |y_j|² par composante.


class CrosstalkLoss(nn.Module):
    """
    (sorties (B, 5), cibles (B, 5)) -> (L, {composante: terme}) : la formule de la docstring du module.

    Args:
        **loi: les champs de injection.InjectionLaw, c'est-à-dire les paramètres de la transformation
            « crosstalk_injection » du YAML, sans sa graine. Ceux qui manquent prennent leur valeur par défaut.
    """

    def __init__(self, **loi) -> None:
        super().__init__()
        self.mse = ComplexMSELoss(reduction="none")
        #Buffer et non paramètre : σ² suit le réseau sur le device (.to(device)) sans être appris.
        self.register_buffer("sigma2", torch.from_numpy(target_power(InjectionLaw(**loi))).float())

    def forward(self, sorties: torch.Tensor, cibles: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sorties.ndim != 2 or sorties.shape[1] != CARD_THETA or sorties.shape != cibles.shape:
            raise ValueError(f"sorties {tuple(sorties.shape)} et cibles {tuple(cibles.shape)} : (B, {CARD_THETA}) attendu pour les deux")

        valides = torch.isfinite(cibles).all(dim=1)                                #Étape 1 : (B,), les patchs dont la cible est finie.
        erreur = self.mse(sorties[valides], cibles[valides]) / self.sigma2         #Étape 2 : (V, 5), |ŷ - y|² / σ² sur les seuls patchs valides.
        termes = erreur.sum(dim=0) / valides.sum().clamp(min=1)                    #Étape 3 : (5,), moyenne sur les patchs valides ; 0 si V = 0.
        return termes.sum(), {nom: terme.detach() for nom, terme in zip(COMPOSANTES, termes)}
