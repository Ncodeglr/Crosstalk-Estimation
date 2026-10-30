"""
Tests du réseau « ResNet » (le ResNet de cvnn, avec une tête de régression) :

1. sur les lots de la transformation : forme et type de la sortie, valeurs de tout signe, gradients, échelle ;
2. l'architecture : l'encodeur de cvnn, la tête sans activation finale, le nombre de paramètres ;
3. la construction comme cvnn la fait (create_model sur get_model_params), l'enregistrement et les refus.
"""

import numpy as np
import pytest
import torch
from torchcvnn.nn.modules import CReLU

from src.crosstalk.deep_learning import ResNet, ResNetInjectionTransform
from src.crosstalk.deep_learning.crosstalk_injection import CARD_THETA, ECHELLE_SORTIE
from cvnn.models.models import ResNet as CvnnResNet  # après src : l'import de src ajoute cvnn au chemin

#: Les réglages de la section model du YAML, tels que get_model_params les transmet, avec ce que cvnn déduit du
#: premier lot de la transformation (4 canaux, patchs 64 x 64).
REGLAGES = dict(num_channels=4, input_size=64, num_layers=4, channels_width=16, num_blocks=1, activation="CReLU",
                layer_mode="complex", normalization_layer=None, downsampling_layer=None, residual=True, dropout=0.0)


def lot_de_la_transformation(taille: int = 8) -> torch.Tensor:
    """Un lot (B, 4, 64, 64) tel que le DataLoader de cvnn l'assemble à partir de ResNetInjectionTransform."""
    rng = np.random.default_rng(0)
    transformation = ResNetInjectionTransform()
    patchs = rng.normal(size=(taille, 4, 64, 64)) + 1j * rng.normal(size=(taille, 4, 64, 64))
    return torch.stack([transformation(patch)[0] for patch in patchs])


def reseau_initialise() -> ResNet:
    """Le réseau tel que le script d'entraînement le prépare : construit, puis initialisé par init_weights_mode_aware."""
    from cvnn.models.utils import init_weights_mode_aware

    torch.manual_seed(0)
    reseau = ResNet(**REGLAGES)
    reseau.apply(lambda m: init_weights_mode_aware(m, "complex"))
    return reseau


# ----------------------------------------------------------------- sur les lots de la transformation


def test_un_lot_de_la_transformation_donne_cinq_sorties_complexes_par_patch():
    sortie = reseau_initialise()(lot_de_la_transformation(8))
    assert sortie.shape == (8, CARD_THETA) and sortie.dtype == torch.complex64
    assert torch.isfinite(sortie.real).all() and torch.isfinite(sortie.imag).all()


def test_les_sorties_prennent_des_valeurs_de_tout_signe():
    """Ce que la tête de classification de cvnn ne permettait pas : avec CReLU, elle ne sortait rien de négatif."""
    with torch.no_grad():
        sortie = reseau_initialise()(lot_de_la_transformation(32))
    for partie in (sortie.real, sortie.imag):
        assert partie.min() < 0 < partie.max()


def test_tous_les_parametres_recoivent_un_gradient():
    reseau = reseau_initialise()
    reseau(lot_de_la_transformation()).abs().pow(2).sum().backward()
    for nom, parametre in reseau.named_parameters():
        assert parametre.grad is not None and torch.isfinite(parametre.grad).all(), nom


def test_la_sortie_est_celle_de_la_tete_multipliee_par_l_echelle():
    reseau = reseau_initialise()
    entree = lot_de_la_transformation()
    with torch.no_grad():
        attendu = ECHELLE_SORTIE * reseau.convnet.bottleneck(reseau.encode(entree))
        assert torch.equal(reseau(entree), attendu)


# ----------------------------------------------------------------- l'architecture


def test_l_encodeur_est_celui_du_resnet_de_cvnn():
    reseau = ResNet(**REGLAGES)
    assert isinstance(reseau, CvnnResNet)
    assert [type(bloc).__name__ for bloc in reseau.convnet.encoder] == ["SingleConv"] + ["Down"] * 4


def test_la_tete_finit_par_une_couche_lineaire_sans_activation():
    tete = ResNet(**REGLAGES).convnet.bottleneck
    lineaires = [m for m in tete.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lineaires] == [(256, 256), (256, CARD_THETA)]
    assert any(isinstance(m, CReLU) for m in tete[-2].modules())
    assert not any(isinstance(m, CReLU) for m in tete[-1].modules())


def test_le_nombre_de_parametres_est_celui_du_resnet_de_cvnn():
    # 2 575 338 paramètres réels : la tête de régression a les couches de celle de cvnn, sans son activation finale.
    reseau = ResNet(**REGLAGES)
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in reseau.parameters()) == 2_575_338


def test_le_resnet_de_cvnn_se_construit_grace_a_la_correction_de_convnet():
    """Sans la correction de ConvNet (cvnn/src/cvnn/models/models.py), sa tête lève TypeError à la construction."""
    CvnnResNet(**REGLAGES, num_classes=CARD_THETA)


# ----------------------------------------------------------------- construction par cvnn


def test_le_reseau_est_enregistre_dans_cvnn():
    from cvnn.models import get_model

    assert get_model("ResNet") is ResNet


def test_cvnn_le_construit_depuis_la_section_model_du_yaml():
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = {"model": {"class": "ResNet", "layer_mode": "complex", "num_layers": 4, "channels_width": 16,
                        "num_blocks": 1, "activation": "CReLU", "normalization_layer": "none",
                        "downsampling_layer": None, "residual": True, "dropout": 0.0},
              "data": {"inferred_input_channels": 4, "inferred_input_size": 64}}
    reseau = create_model(config["model"]["class"], **get_model_params(config))
    assert isinstance(reseau, ResNet)
    assert sum(p.numel() for p in reseau.parameters()) == sum(p.numel() for p in ResNet(**REGLAGES).parameters())


@pytest.mark.parametrize("layer_mode", ["real", "split"])
def test_un_layer_mode_autre_que_complexe_est_refuse(layer_mode):
    with pytest.raises(ValueError):
        ResNet(**{**REGLAGES, "layer_mode": layer_mode})
