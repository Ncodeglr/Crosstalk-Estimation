"""
MLP

Rôle
----
Il reçoit l'entrée produite par CrosstalkInjectionTransform : les 10 entrées distinctes de la covariance normalisée par
sa trace, rangées en (B, 10, 1, 1). Il prédit (u, v, w, z, log alpha), soit (B, 5) en complex64. C'est une estimation
par patch, à information égale avec Quegan, qui ne voit lui aussi que la covariance.

Architecture
------------
Chaque couche cachée est un bloc SingleLinear de cvnn (cvnn/src/cvnn/models/linear.py) :
linéaire complexe -> normalisation -> activation, chacune choisie par son nom dans la section model du YAML.

    Étape 1 : aplatissement                          (B, 10, 1, 1)       -> (B, 10)
    Étape 2 : num_layers blocs cachés                (B, 10)             -> (B, channels_width)
    Étape 3 : tête, sans normalisation ni activation (B, channels_width) -> (B, 5)
    Étape 4 : mise à l'échelle                       x ECHELLE_SORTIE

Exemple numérique, avec num_layers = 3, channels_width = 128 et modReLU, pour un batch = 1024 patchs :

    [1024, 10, 1, 1] -> [1024, 10] -> [1024, 128] -> [1024, 128] -> [1024, 128] -> [1024, 5]

soit 70 157 paramètres réels (un paramètre complexe en compte deux) : le MLP des simulations.

Choix par nom
-------------

L'initialisation n'est pas faite ici : src/train_cvnn.py applique init_weights_mode_aware de cvnn (Kaiming complexe)
après la construction.

"""

from __future__ import annotations
import torch
from torch import nn
from cvnn.models import register_model
from cvnn.models.linear import SingleLinear
from .injection_transform import CARD_THETA

#: Facteur appliqué à la sortie. Mesure faite sur un patch réel : à l'initialisation par init_weights_mode_aware, la
#: couche de sortie rend des valeurs dont l'écart quadratique va de 0,2 à 1,5 selon la composante. Sous la loi par
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

    Args:
        num_channels: canaux de l'entrée, 10 (inferred_input_channels).
        input_size: côté de l'image d'entrée, 1 (inferred_input_size).
        num_layers: nombre de blocs cachés.
        channels_width: largeur des blocs cachés.
        activation: nom de l'activation des blocs cachés, cherché par get_activation.
        layer_mode: seul « complex » est accepté. L'entrée est complexe, et ni « real » ni « split » ne
            l'accepteraient.
        normalization_layer: normalisation des blocs cachés, parmi NORMALISATIONS.
        **kwargs: les autres champs de get_model_params, ignorés.

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

        #Blocs cachés : la première couche reçoit les num_channels * input_size**2 valeurs de l'entrée aplatie, les suivantes channels_width.
        largeurs = [num_channels * input_size**2] + [channels_width] * num_layers
        self.couches = nn.Sequential(*[
            SingleLinear(n_entree, n_sortie, linear_mode=layer_mode, activation=activation, normalization=normalization_layer)
            for n_entree, n_sortie in zip(largeurs[:-1], largeurs[1:])
        ])
        #Tête : sans normalisation ni activation, pour que la sortie puisse prendre toute valeur complexe.
        self.tete = SingleLinear(channels_width, CARD_THETA, linear_mode=layer_mode, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)            #Étape 1 : (B, 10, 1, 1) -> (B, 10), les 10 entrées de la covariance de chaque patch.
        x = self.couches(x)         #Étape 2 : (B, 10) -> (B, channels_width), num_layers blocs linéaire -> normalisation -> activation.
        x = self.tete(x)            #Étape 3 : (B, channels_width) -> (B, 5), une valeur complexe par composante de theta_k.
        return ECHELLE_SORTIE * x   #Étape 4 : les premières prédictions partent de près de zéro (voir ECHELLE_SORTIE).
