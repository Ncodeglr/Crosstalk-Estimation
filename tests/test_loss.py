"""
Tests de la perte « CrosstalkLoss » :

1. la formule sur un lot : la MSE complexe seule, sans normalisation, et les métriques par composante ;
2. les patchs écartés : retirés avant la soustraction, gradient sans nan, lot sans patch valide ;
3. avec cvnn : les boucles d'entraînement et de validation de cvnn reçoivent la perte et les métriques ; les refus.
"""

import numpy as np
import pytest
import torch

from src.crosstalk.deep_learning import MLPInjectionTransform, MLP
from src.crosstalk.deep_learning.training.loss import COMPOSANTES, CrosstalkLoss
from src.crosstalk.injection.crosstalk_sampler import CrosstalkSampler


@pytest.fixture(scope="module")
def perte() -> CrosstalkLoss:
    return CrosstalkLoss()


def cibles_de_la_loi(n: int, graine: int, sampler: CrosstalkSampler = CrosstalkSampler()) -> torch.Tensor:
    """n cibles (u, v, w, z, alpha) tirées de la loi, comme CrosstalkInjector.inject les construit."""
    rng = np.random.default_rng(graine)
    return torch.from_numpy(np.array([sampler.generate(rng).to_vector() for _ in range(n)]).astype(np.complex64))


def lot_aleatoire(b: int = 6, graine: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Des sorties (avec gradient) et des cibles complexes (b, 5) quelconques."""
    g = torch.Generator().manual_seed(graine)
    sorties = (0.1 * torch.randn(b, 5, dtype=torch.complex64, generator=g)).requires_grad_()
    return sorties, 0.1 * torch.randn(b, 5, dtype=torch.complex64, generator=g)


# ----------------------------------------------------------------- la formule


def test_la_perte_suit_la_formule(perte):
    sorties, cibles = lot_aleatoire()
    valeur, _ = perte(sorties, cibles)
    attendu = (np.abs(sorties.detach().numpy() - cibles.numpy()) ** 2).sum(axis=1).mean()
    assert valeur.item() == pytest.approx(attendu, rel=1e-5)


def test_les_composantes_ne_sont_pas_normalisees(perte):
    """Une erreur de 0,1 sur u coûte 0,01 et une erreur de 1 sur alpha coûte 1 : chacune garde son échelle."""
    sorties = torch.zeros(1, 5, dtype=torch.complex64)
    cibles = torch.tensor([[0.1, 0, 0, 0, 1]], dtype=torch.complex64)
    _, termes = perte(sorties, cibles)
    assert float(termes["u"]) == pytest.approx(0.01, rel=1e-5)
    assert float(termes["alpha"]) == pytest.approx(1.0, rel=1e-5)


def test_les_composantes_sont_celles_de_la_cible():
    assert COMPOSANTES == ("u", "v", "w", "z", "alpha")


def test_la_perte_n_a_ni_parametre_ni_buffer(perte):
    assert list(perte.parameters()) == [] and list(perte.buffers()) == []


def test_les_metriques_sont_les_termes_de_chaque_composante_et_somment_a_la_perte(perte):
    sorties, cibles = lot_aleatoire()
    valeur, termes = perte(sorties, cibles)
    assert list(termes) == list(COMPOSANTES)
    assert all(not terme.requires_grad for terme in termes.values())
    assert float(sum(termes.values())) == pytest.approx(valeur.item(), rel=1e-6)


def test_la_perte_est_reelle_et_rattachee_au_graphe(perte):
    sorties, cibles = lot_aleatoire()
    valeur, _ = perte(sorties, cibles)
    assert valeur.ndim == 0 and not valeur.is_complex() and valeur.requires_grad


# ----------------------------------------------------------------- patchs écartés


def test_un_patch_ecarte_est_ignore(perte):
    sorties, cibles = lot_aleatoire()
    avec_ecarte = cibles.clone()
    avec_ecarte[2] = complex(np.nan, np.nan)
    garde = [0, 1, 3, 4, 5]
    assert perte(sorties, avec_ecarte)[0].item() == pytest.approx(perte(sorties[garde], cibles[garde])[0].item(), rel=1e-6)


def test_un_patch_ecarte_ne_met_pas_de_nan_dans_le_gradient(perte):
    sorties, cibles = lot_aleatoire()
    cibles[2] = complex(np.nan, np.nan)
    perte(sorties, cibles)[0].backward()
    assert torch.isfinite(sorties.grad).all()
    assert torch.equal(sorties.grad[2], torch.zeros(5, dtype=torch.complex64))


def test_un_lot_sans_patch_valide_donne_zero_et_un_gradient_nul(perte):
    sorties, _ = lot_aleatoire()
    valeur, termes = perte(sorties, torch.full((6, 5), complex(np.nan, np.nan), dtype=torch.complex64))
    assert valeur.item() == 0.0 and all(float(t) == 0.0 for t in termes.values())
    valeur.backward()
    assert torch.equal(sorties.grad, torch.zeros_like(sorties))


def test_les_patchs_ecartes_par_la_transformation_sont_ignores(perte):
    transformation = MLPInjectionTransform()
    patch_ecarte = np.zeros((4, 8, 8), dtype=np.complex64)
    _, cible = transformation(patch_ecarte)
    valeur, _ = perte(torch.zeros(1, 5, dtype=torch.complex64, requires_grad=True), cible[None])
    assert valeur.item() == 0.0


# ----------------------------------------------------------------- avec cvnn, refus


def test_les_boucles_de_cvnn_recoivent_la_perte_et_les_metriques(perte):
    from cvnn.train import train_one_epoch, validate_one_epoch

    torch.manual_seed(0)
    reseau = MLP(num_channels=10, input_size=1, num_layers=1, channels_width=16, activation="modReLU")
    cibles = cibles_de_la_loi(64, graine=2)
    cibles[::8] = complex(np.nan, np.nan)
    lots = [(torch.randn(16, 10, 1, 1, dtype=torch.complex64), cibles[i:i + 16]) for i in range(0, 64, 16)]
    optimiseur = torch.optim.Adam(reseau.parameters(), lr=1e-3)

    for resultats in (train_one_epoch(reseau, lots, perte, optimiseur, torch.device("cpu")),
                      validate_one_epoch(reseau, lots, perte, torch.device("cpu"))):
        assert set(resultats) == {"loss", *COMPOSANTES}
        assert np.isfinite(list(resultats.values())).all()
        assert sum(resultats[nom] for nom in COMPOSANTES) == pytest.approx(resultats["loss"], rel=1e-5)


def test_la_perte_n_herite_pas_de_mse():
    # Sinon train_one_epoch de cvnn comparerait les sorties aux entrées.
    from torchcvnn.nn.modules import ComplexMSELoss  # celle que teste cvnn/src/cvnn/train.py

    assert not issubclass(CrosstalkLoss, (torch.nn.MSELoss, ComplexMSELoss))


@pytest.mark.parametrize("sorties, cibles", [((6, 5), (5, 5)), ((6, 4), (6, 4)), ((6, 5, 1), (6, 5, 1))])
def test_des_formes_incompatibles_sont_refusees(perte, sorties, cibles):
    with pytest.raises(ValueError):
        perte(torch.zeros(sorties, dtype=torch.complex64), torch.zeros(cibles, dtype=torch.complex64))


def test_la_perte_ne_prend_aucun_reglage():
    with pytest.raises(TypeError):
        CrosstalkLoss(crosstalk_db=(-35.0, -10.0))
