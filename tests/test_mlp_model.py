"""
Tests du réseau « MLP » :

1. sur les lots de la transformation : forme et type de la sortie, gradients ;
2. l'architecture : couches, activations, normalisations et nombre de paramètres des simulations ;
3. la construction comme cvnn la fait (create_model sur get_model_params), l'enregistrement et les refus.
"""

import numpy as np
import pytest
import torch
import torchcvnn.nn.modules as c_nn
from torchcvnn.nn.modules import modReLU

from src.crosstalk.deep_learning import MLPInjectionTransform, MLP
from src.crosstalk.deep_learning.mlp_model import ECHELLE_SORTIE

#: Les réglages de la section model du YAML, tels que get_model_params les transmet, avec ce que cvnn déduit du
#: premier lot de la transformation (10 canaux, image 1 x 1).
REGLAGES = dict(num_channels=10, input_size=1, num_layers=3, channels_width=64, activation="modReLU",
                layer_mode="complex")


def lot_de_la_transformation(taille: int = 8) -> torch.Tensor:
    """Un lot (B, 10, 1, 1) tel que le DataLoader de cvnn l'assemble à partir de MLPInjectionTransform."""
    rng = np.random.default_rng(0)
    transformation = MLPInjectionTransform()
    patchs = rng.normal(size=(taille, 4, 16, 16)) + 1j * rng.normal(size=(taille, 4, 16, 16))
    return torch.stack([transformation(patch)[0] for patch in patchs])


def reseau_initialise() -> MLP:
    """Le réseau tel que src/train_mlp.py le prépare : construit, puis initialisé par init_weights_mode_aware."""
    from cvnn.models.utils import init_weights_mode_aware

    torch.manual_seed(0)
    reseau = MLP(**REGLAGES)
    reseau.apply(lambda m: init_weights_mode_aware(m, "complex"))
    return reseau


# ----------------------------------------------------------------- sur les lots de la transformation


def test_un_lot_de_la_transformation_donne_cinq_sorties_complexes_par_patch():
    sortie = reseau_initialise()(lot_de_la_transformation(8))
    assert sortie.shape == (8, 5) and sortie.dtype == torch.complex64
    assert torch.isfinite(sortie.real).all() and torch.isfinite(sortie.imag).all()


def test_tous_les_parametres_recoivent_un_gradient():
    reseau = reseau_initialise()
    reseau(lot_de_la_transformation()).abs().pow(2).sum().backward()
    for nom, parametre in reseau.named_parameters():
        assert parametre.grad is not None and torch.isfinite(parametre.grad).all(), nom


def test_la_sortie_est_celle_de_la_tete_multipliee_par_l_echelle():
    reseau = reseau_initialise()
    entree = lot_de_la_transformation()
    with torch.no_grad():
        attendu = ECHELLE_SORTIE * reseau.final_layer(reseau.layers(entree.flatten(1)))
        assert torch.equal(reseau(entree), attendu)


# ----------------------------------------------------------------- l'architecture


def test_trois_couches_cachees_de_64_128_256_puis_une_tete_vers_5_sans_activation():
    reseau = MLP(**REGLAGES)
    lineaires = [m for m in reseau.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lineaires] == [(10, 64), (64, 128), (128, 256), (256, 5)]
    assert all(m.weight.dtype == torch.complex64 for m in lineaires)
    assert sum(isinstance(m, modReLU) for m in reseau.layers.modules()) == 3
    assert not any(isinstance(m, modReLU) for m in reseau.final_layer.modules())


def test_sans_reglage_les_blocs_caches_n_ont_pas_de_normalisation():
    reseau = MLP(**REGLAGES)
    assert not any(isinstance(m, (c_nn.LayerNorm, c_nn.BatchNorm1d)) for m in reseau.modules())


@pytest.mark.parametrize("normalisation, classe", [("layer", c_nn.LayerNorm), ("batch", c_nn.BatchNorm1d)])
def test_une_normalisation_acceptee_est_placee_dans_chaque_bloc_cache_et_pas_dans_la_tete(normalisation, classe):
    reseau = MLP(**REGLAGES, normalization_layer=normalisation)
    assert sum(isinstance(m, classe) for m in reseau.layers.modules()) == 3
    assert not any(isinstance(m, classe) for m in reseau.final_layer.modules())
    sortie = reseau(lot_de_la_transformation(8))
    assert sortie.shape == (8, 5) and torch.isfinite(sortie).all()


def test_le_nombre_de_parametres_est_celui_des_simulations():
    # 86 669 paramètres réels, comme le MLP_C_64_128_256 des simulations : un paramètre complexe en compte deux.
    reseau = MLP(**REGLAGES)
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in reseau.parameters()) == 86_669


# ----------------------------------------------------------------- construction par cvnn


def test_le_reseau_est_enregistre_dans_cvnn():
    from cvnn.models import get_model

    assert get_model("MLP") is MLP


def test_cvnn_le_construit_depuis_la_section_model_du_yaml():
    from cvnn.config_utils import get_model_params
    from cvnn.models import create_model

    config = {"model": {"class": "MLP", "layer_mode": "complex", "num_layers": 3, "channels_width": 64,
                        "activation": "modReLU"},
              "data": {"inferred_input_channels": 10, "inferred_input_size": 1}}
    reseau = create_model(config["model"]["class"], **get_model_params(config))
    assert isinstance(reseau, MLP)
    assert sum(p.numel() for p in reseau.parameters()) == sum(p.numel() for p in MLP(**REGLAGES).parameters())


@pytest.mark.parametrize("reglage", [dict(layer_mode="real"), dict(layer_mode="split"), dict(num_layers=None),
                                     dict(num_layers=0), dict(channels_width=None), dict(channels_width=True),
                                     dict(normalization_layer="group"), dict(normalization_layer="instance"),
                                     dict(normalization_layer="BatchNorm")])
def test_une_configuration_invalide_est_refusee_a_la_construction(reglage):
    with pytest.raises(ValueError):
        MLP(**{**REGLAGES, **reglage})
