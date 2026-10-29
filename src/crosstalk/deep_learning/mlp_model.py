"""
class MLP

Il reçoit l'entrée produite par CrosstalkInjectionTransform : les 10 entrées distinctes de la covariance normalisée par
sa trace, rangées en (B, 10, 1, 1). Il prédit (u, v, w, z, log alpha), soit (B, 5) en complex64. C'est une estimation
par patch, à information égale avec Quegan, qui ne voit lui aussi que la covariance.


Choix par nom
-------------

L'initialisation n'est pas faite ici : src/train_mlp.py applique init_weights_mode_aware de cvnn (Kaiming complexe)
après la construction.

"""

from __future__ import annotations
import torch
from torch import nn
from cvnn.models import register_model
from cvnn.models.linear import SingleLinear
from .injection_transform import CARD_THETA

#: Facteur appliqué à la sortie. Mesure faite sur un patch réel : à l'initialisation par init_weights_mode_aware, la
#: couche de sortie rend des valeurs dont l'écart quadratique va de 0,2 à 1,8 (médiane 0,7) selon la composante et la
#: graine, pour 64, 128, 256 comme pour 128, 128, 128. Sous la loi par
#: défaut, les cibles valent 0,14 (u, v, w, z) et 0,33 (log alpha). Avec ce facteur, les premières prédictions sont
#: proches de zéro, c'est-à-dire de la prédiction « sans crosstalk », plutôt qu'un décalage aléatoire plus grand que
#: les cibles.
ECHELLE_SORTIE: float = 0.1

#: Les valeurs acceptées pour normalization_layer
NORMALISATIONS: tuple[str | None, ...] = (None, "none", "layer", "batch")


@register_model("MLP")
class MLP(nn.Module):
    """
    Entrées de la covariance (B, 10, 1, 1) -> (u, v, w, z, log alpha) (B, 5), en complex64.

    Args: **kwargs: les autres champs de get_model_params, ignorés.
    """

    def __init__(self,num_channels: int,input_size: int,num_layers: int,channels_width: int,activation: str,layer_mode: str = "complex",normalization_layer: str | None = None,**kwargs,) -> None:
        super().__init__()
        if layer_mode != "complex":
            raise ValueError(f"layer_mode = {layer_mode!r} : seul 'complex' est accepté")
        if normalization_layer not in NORMALISATIONS:
            raise ValueError(f"normalization_layer = {normalization_layer!r} : valeurs acceptées {NORMALISATIONS}")
        for nom, valeur in (("num_layers", num_layers), ("channels_width", channels_width)):
            if isinstance(valeur, bool) or not isinstance(valeur, int) or valeur < 1:
                raise ValueError(f"{nom} = {valeur!r} : entier strictement positif attendu (section model du YAML)")

        #La première couche reçoit les num_channels * input_size**2 valeurs de l'entrée aplatie ; le bloc i a channels_width * 2**i neurones.
        largeurs = [num_channels * input_size**2] + [channels_width * 2**i for i in range(num_layers)] #Donne [10, 64, 128, 256] pour num_layers = 3, channels_width = 64.
        self.layers = nn.Sequential(*[
            SingleLinear(n_entree, n_sortie, linear_mode=layer_mode, activation=activation, normalization=normalization_layer)
            for n_entree, n_sortie in zip(largeurs[:-1], largeurs[1:])
        ])
        #Tête : depuis la largeur du dernier bloc, sans normalisation ni activation, pour que la sortie puisse prendre toute valeur complexe.
        self.final_layer = SingleLinear(largeurs[-1], CARD_THETA, linear_mode=layer_mode, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)            #Étape 1 : (B, 10, 1, 1) -> (B, 10), les 10 entrées de la covariance de chaque patch.
        x = self.layers(x)          #Étape 2 : (B, 10) -> (B, 64) -> (B, 128) -> (B, 256), blocs linéaire -> normalisation -> activation.
        x = self.final_layer(x)     #Étape 3 : (B, 256) -> (B, 5), une valeur complexe par composante de theta_k.
        return ECHELLE_SORTIE * x   #Étape 4 : les premières prédictions partent de près de zéro (voir ECHELLE_SORTIE).
