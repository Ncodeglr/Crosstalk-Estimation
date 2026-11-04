"""
class ResNet

Le ResNet de cvnn (cvnn.models.models.ResNet), avec une tête de régression à la place de sa tête de classification.

Il reçoit l'entrée produite par ResNetInjectionTransform : le patch injecté, normalisé pour que sa covariance soit de
trace 1, soit (B, 4, H, W) en complex64. Il prédit (u, v, w, z, log alpha), soit (B, 5) en complex64, comme le MLP.
À la différence du MLP, il voit les pixels du patch, et pas seulement leur covariance.


Ce qui vient de cvnn
--------------------
Tout l'encodeur, construit par cvnn depuis la section model du YAML (get_model_params) : un SingleConv qui fait passer
les num_channels canaux à channels_width, puis num_layers blocs Down (résiduels si residual: true), qui doublent les
canaux et divisent la taille par deux. Le forward aussi : l'encodeur, puis la tête (convnet.bottleneck).
Sa construction exige la correction de ConvNet décrite dans sa docstring (cvnn/src/cvnn/models/models.py).


Ce qui change : la tête
-----------------------
La tête de classification de cvnn (FullyConnected : moyenne spatiale, puis DoubleLinear) applique l'activation après
sa dernière couche : avec CReLU, ses sorties ne peuvent avoir ni partie réelle ni partie imaginaire négative. C'est
sans conséquence pour une classification, que cvnn projette ensuite sur |.|, mais rédhibitoire pour theta_k, dont les
phases sont uniformes : sur 256 patchs injectés, aucune de ses sorties n'était négative, et 9 % étaient nulles
(mesure du 2026-10-30).
Elle est remplacée par la même tête, sans l'activation finale : la moyenne spatiale, un SingleLinear de la largeur de
l'encodeur vers elle-même (avec activation et normalisation), puis un SingleLinear vers CARD_THETA, sans activation ni
normalisation, comme la dernière couche du MLP. Le nombre de paramètres ne change pas.
La sortie est enfin multipliée par ECHELLE_SORTIE, comme celle du MLP.

L'initialisation n'est pas faite ici : comme pour le MLP, le script d'entraînement applique init_weights (experiment.py) après
la construction, selon model.init_weights du YAML, à tout le réseau, nouvelle tête comprise.
"""

from __future__ import annotations
import torch
from torch import nn
import torchcvnn.nn.modules as c_nn
from cvnn.models import register_model
from cvnn.models.linear import SingleLinear
from cvnn.models.models import ResNet as CvnnResNet
from .crosstalk_injection import CARD_THETA, ECHELLE_SORTIE


@register_model("ResNet") #Ajouté au registre de cvnn, pour que create_model("ResNet", ...) le construise : cvnn ne construit son ResNet que dans sa tâche de classification (model_utils.build_resnet).
class ResNet(CvnnResNet):
    """
    Patch injecté (B, 4, H, W) -> (u, v, w, z, log alpha) (B, 5), en complex64.

    Args:
        channels_width, num_layers, activation, normalization_layer: ceux de l'encodeur de cvnn, que la tête reprend.
        layer_mode: seul « complex » est accepté : l'entrée et la cible sont complexes.
        upsampling_layer: rendu par get_model_params, sans objet pour un encodeur ; ignoré, comme dans build_resnet de cvnn.
        **params: les autres paramètres du ResNet de cvnn, tirés de la section model du YAML par get_model_params
            (num_channels, input_size, num_blocks, downsampling_layer, residual, dropout, projection_layer,
            projection_config, gumbel_softmax). num_classes est fixé à CARD_THETA.

    Raises: ValueError: si layer_mode n'est pas « complex ».
    """

    def __init__(self, channels_width: int, num_layers: int, activation: str, layer_mode: str = "complex",
                 normalization_layer: str | None = None, upsampling_layer: str | None = None, **params) -> None:
        if layer_mode != "complex":
            raise ValueError(f"layer_mode = {layer_mode!r} : seul 'complex' est accepté")
        #Étape 1 : le ResNet de cvnn, encodeur et tête de classification, avec CARD_THETA sorties.
        super().__init__(channels_width=channels_width, num_layers=num_layers, activation=activation,
                         layer_mode=layer_mode, normalization_layer=normalization_layer, num_classes=CARD_THETA,
                         **params)
        #Étape 2 : sa tête est remplacée par la même, sans l'activation finale.
        largeur = channels_width * 2**num_layers #Canaux en sortie de l'encodeur : 256 pour channels_width = 16 et num_layers = 4.
        self.convnet.bottleneck = nn.Sequential(
            c_nn.AdaptiveAvgPool2d(1), #La moyenne spatiale de la tête de cvnn : (B, 256, 4, 4) -> (B, 256, 1, 1).
            nn.Flatten(1),
            SingleLinear(largeur, largeur, linear_mode=layer_mode, activation=activation, normalization=normalization_layer),
            SingleLinear(largeur, CARD_THETA, linear_mode=layer_mode, activation=None), #Sans activation : la sortie peut prendre toute valeur complexe.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ECHELLE_SORTIE * super().forward(x) #Le forward de cvnn (encodeur, puis tête), puis l'échelle commune (voir ECHELLE_SORTIE).
