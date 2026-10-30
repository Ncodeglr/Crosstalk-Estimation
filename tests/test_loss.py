"""
Tests de la perte « CrosstalkLoss » :

1. m = E|y|² : l'estimation par tirages contre la formule fermée de la loi, et le prédicteur nul qui coûte 1 par composante ;
2. la formule sur un lot, et les métriques par composante ;
3. les patchs écartés : retirés avant la soustraction, gradient sans nan, lot sans patch valide ;
4. avec cvnn : les boucles d'entraînement et de validation de cvnn reçoivent la perte et les métriques ; les refus.
"""

import numpy as np
import pytest
import torch

from src.crosstalk.deep_learning import MLPInjectionTransform, MLP
from src.crosstalk.deep_learning.loss import COMPOSANTES, CrosstalkLoss, mean_squared_parameters
from src.crosstalk.crosstalk_sampler import CrosstalkSampler

LN10 = np.log(10.0)


@pytest.fixture(scope="module")
def perte() -> CrosstalkLoss:
    """La perte telle que l'entraînement la construit : loi par défaut, N_TIRAGES tirages (1,4 s, construite une fois)."""
    return CrosstalkLoss()


def mean_squared_exact(sampler: CrosstalkSampler) -> np.ndarray:
    """
    m_j = E|y_j|² par la formule fermée de CrosstalkSampler (docstring de crosstalk_sampler.CrosstalkSampler).

    Crosstalk : |u|² = 10^((L + e) / 10), L et e indépendants et uniformes en dB, donc E|u|² = E[10^(L/10)] E[10^(e/10)],
    avec E[10^(x/10)] = (10^(b/10) - 10^(a/10)) / ((b - a) ln(10) / 10) pour x uniforme sur [a, b].
    log alpha : |log alpha|² = (ln|alpha|)² + arg(alpha)², avec ln|alpha| = m ln(10) / 20 et E[x²] = (a² + ab + b²) / 3.
    Un patch neutre (probabilité no_injection_rate) a une cible nulle.
    """
    def moyenne_puissance(a, b):
        return (10 ** (b / 10) - 10 ** (a / 10)) / ((b - a) * LN10 / 10)

    def moment2(a, b):
        return (a * a + a * b + b * b) / 3

    s = sampler.term_spread_db
    crosstalk = moyenne_puissance(sampler.crosstalk_db[0] + s, sampler.crosstalk_db[1] - s) * moyenne_puissance(-s, s)
    log_alpha = (LN10 / 20) ** 2 * moment2(*sampler.alpha_module_db) + moment2(*np.deg2rad(sampler.alpha_phase_deg))
    return (1 - sampler.no_injection_rate) * np.array([crosstalk] * 4 + [log_alpha])


def cibles_de_la_loi(n: int, graine: int, sampler: CrosstalkSampler = CrosstalkSampler()) -> torch.Tensor:
    """n cibles (u, v, w, z, log alpha) tirées de la loi, comme CrosstalkInjector.inject les construit."""
    rng = np.random.default_rng(graine)
    cibles = np.array([sampler.generate(rng).to_vector() for _ in range(n)])
    cibles[:, 4] = np.log(cibles[:, 4])
    return torch.from_numpy(cibles.astype(np.complex64))


def lot_aleatoire(b: int = 6, graine: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Des sorties (avec gradient) et des cibles complexes (b, 5) quelconques."""
    g = torch.Generator().manual_seed(graine)
    sorties = (0.1 * torch.randn(b, 5, dtype=torch.complex64, generator=g)).requires_grad_()
    return sorties, 0.1 * torch.randn(b, 5, dtype=torch.complex64, generator=g)


# ----------------------------------------------------------------- m = E|y|²


def test_m_par_tirages_suit_la_formule_fermee_de_la_loi_par_defaut(perte):
    np.testing.assert_allclose(perte.mean_squared.numpy(), mean_squared_exact(CrosstalkSampler()), rtol=0.03)


def test_m_par_tirages_suit_la_formule_fermee_d_une_autre_loi():
    loi = dict(crosstalk_db=(-30.0, -10.0), term_spread_db=2.0, alpha_module_db=(-1.0, 2.0),
               alpha_phase_deg=(-10.0, 40.0), no_injection_rate=0.0)
    np.testing.assert_allclose(mean_squared_parameters(CrosstalkSampler(**loi), 50_000), mean_squared_exact(CrosstalkSampler(**loi)), rtol=0.03)


def test_le_predicteur_nul_coute_un_par_composante(perte):
    cibles = cibles_de_la_loi(20_000, graine=1)
    valeur, termes = perte(torch.zeros_like(cibles), cibles)
    np.testing.assert_allclose([float(termes[nom]) for nom in COMPOSANTES], 1.0, rtol=0.05)
    assert valeur.item() == pytest.approx(5.0, rel=0.03)


def test_m_est_un_buffer_et_non_un_parametre_appris(perte):
    assert "mean_squared" in dict(perte.named_buffers())
    assert list(perte.parameters()) == []


# ----------------------------------------------------------------- la formule


def test_la_perte_suit_la_formule(perte):
    sorties, cibles = lot_aleatoire()
    valeur, _ = perte(sorties, cibles)
    attendu = (np.abs(sorties.detach().numpy() - cibles.numpy()) ** 2 / perte.mean_squared.numpy()).sum(axis=1).mean()
    assert valeur.item() == pytest.approx(attendu, rel=1e-5)


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


@pytest.mark.parametrize("reglages, erreur", [(dict(no_injection_rate=1.0), ValueError),
                                              (dict(champ_inconnu=1), TypeError)])
def test_une_configuration_invalide_est_refusee_a_la_construction(reglages, erreur):
    with pytest.raises(erreur):
        CrosstalkLoss(**reglages)
