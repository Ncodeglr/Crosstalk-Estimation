"""
Tests des transformations « crosstalk_injection_mlp » et « crosstalk_injection_resnet », et de check_pipeline :

1. le couple du MLP : forme, type, déterminisme, cible = theta_k de patch_rng, entrée = covariance injectée normalisée ;
2. l'injection commune (inject) et l'entrée du ResNet, le patch injecté normalisé : pour un même patch, les deux
   réseaux ont la même cible, et la covariance de l'entrée du ResNet est l'entrée du MLP ;
3. les patchs écartés (zéro ou valeur non finie) et les refus ;
4. check_pipeline : il laisse passer les lots de la transformation attendue et arrête les autres ;
5. l'enregistrement dans cvnn ;
6. sur la vraie pile, à travers les chargeurs de cvnn : le format des lots, le découpage identique à celui de la
   chaîne A, la détection d'une transformation que cvnn a ignorée en silence, et les mêmes données pour les deux
   réseaux. Sautés si la pile est absente.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.covariance import HERMITIAN_ENTRIES, covariance_entries, polarimetric_covariance
from src.crosstalk.deep_learning import CrosstalkInjector, MLPInjectionTransform, ResNetInjectionTransform, check_pipeline
from src.crosstalk.crosstalk_sampler import NEUTRE, CrosstalkSampler
from src.crosstalk.patch_rng import patch_rng
from src.crosstalk.matrix import apply_distortion, distortion_matrix

RACINE = Path(__file__).resolve().parent.parent
PUISSANCES = [k for k, (i, j) in enumerate(HERMITIAN_ENTRIES) if i == j]


def patch_aleatoire(graine: int = 0, forme=(4, 16, 16)) -> torch.Tensor:
    """Un patch tel que cvnn le donne à la transformation : tenseur complex64 (4, H, W), sans zéro."""
    rng = np.random.default_rng(graine)
    return torch.from_numpy((rng.normal(size=forme) + 1j * rng.normal(size=forme)).astype(np.complex64))


def patch_injecte(transformation: CrosstalkInjector) -> torch.Tensor:
    """Le premier patch aléatoire qui reçoit une injection (non neutre)."""
    for graine in range(100):
        patch = patch_aleatoire(graine)
        if transformation.sampler.generate(patch_rng(patch.numpy(), transformation.seed_injection)) != NEUTRE:
            return patch
    raise AssertionError("aucun patch injecté sur 100 essais")


def patch_brut(transformation: CrosstalkInjector) -> torch.Tensor:
    """Le premier patch aléatoire laissé brut (theta_k neutre)."""
    for graine in range(100):
        patch = patch_aleatoire(graine)
        if transformation.sampler.generate(patch_rng(patch.numpy(), transformation.seed_injection)) == NEUTRE:
            return patch
    raise AssertionError("aucun patch brut sur 100 essais")


# ----------------------------------------------------------------- le couple produit


def test_forme_et_type_du_couple():
    entree, cible = MLPInjectionTransform()(patch_aleatoire())
    assert entree.shape == (10, 1, 1) and entree.dtype == torch.complex64
    assert cible.shape == (5,) and cible.dtype == torch.complex64


def test_le_meme_patch_donne_toujours_le_meme_couple():
    transformation = MLPInjectionTransform(seed_injection=3)
    premier, second = transformation(patch_aleatoire()), transformation(patch_aleatoire())
    assert torch.equal(premier[0], second[0]) and torch.equal(premier[1], second[1])


def test_un_tableau_numpy_donne_le_meme_couple_qu_un_tenseur():
    transformation = MLPInjectionTransform()
    patch = patch_aleatoire()
    assert all(torch.equal(a, b) for a, b in zip(transformation(patch), transformation(patch.numpy())))


def test_une_autre_graine_donne_une_autre_injection():
    patch = patch_injecte(MLPInjectionTransform(seed_injection=0))
    assert not torch.equal(MLPInjectionTransform(seed_injection=0)(patch)[1], MLPInjectionTransform(seed_injection=1)(patch)[1])


def test_la_cible_est_le_theta_k_de_patch_rng():
    transformation = MLPInjectionTransform(seed_injection=5)
    patch = patch_injecte(transformation)
    theta = transformation.sampler.generate(patch_rng(patch.numpy(), 5)).to_vector()
    np.testing.assert_allclose(transformation(patch)[1].numpy(), theta, rtol=1e-6)


def test_l_entree_est_la_covariance_injectee_normalisee_par_sa_trace():
    transformation = MLPInjectionTransform()
    patch = patch_injecte(transformation)
    theta = transformation.sampler.generate(patch_rng(patch.numpy(), 0))
    d = distortion_matrix(theta)
    covariance = d @ polarimetric_covariance(patch.numpy()) @ d.conj().T
    attendu = covariance_entries(covariance / np.trace(covariance).real)
    np.testing.assert_allclose(transformation(patch)[0].numpy().ravel(), attendu, rtol=1e-5, atol=1e-7)


def test_injecter_la_covariance_revient_a_injecter_les_pixels():
    """D agit linéairement sur chaque pixel : < (D k)(D k)^H > = D < k k^H > D^H."""
    transformation = MLPInjectionTransform()
    patch = patch_injecte(transformation)
    theta = transformation.sampler.generate(patch_rng(patch.numpy(), 0))
    covariance = polarimetric_covariance(apply_distortion(patch.numpy().astype(np.complex128), theta))
    attendu = covariance_entries(covariance / np.trace(covariance).real)
    np.testing.assert_allclose(transformation(patch)[0].numpy().ravel(), attendu, rtol=1e-5, atol=1e-7)


def test_les_quatre_puissances_somment_a_un():
    entree = MLPInjectionTransform()(patch_aleatoire())[0].numpy().ravel()
    assert entree[PUISSANCES].real.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(entree[PUISSANCES].imag == 0)


def test_un_patch_brut_a_une_cible_neutre_et_sa_covariance_seule():
    transformation = MLPInjectionTransform(no_injection_rate=0.2)
    patch = patch_brut(transformation)
    entree, cible = transformation(patch)
    assert torch.all(cible[:4] == 0) and cible[4] == 1
    covariance = polarimetric_covariance(patch.numpy())
    np.testing.assert_allclose(entree.numpy().ravel(), covariance_entries(covariance / np.trace(covariance).real),
                               rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------- l'injection commune et l'entrée du ResNet


def test_inject_rend_le_patch_injecte_normalise_et_la_cible():
    """Le patch injecté est D(theta_k) appliqué à chaque pixel, divisé par la racine de la trace de sa covariance."""
    injecteur = CrosstalkInjector(seed_injection=5)
    patch = patch_injecte(injecteur)
    injecte, cible = injecteur.inject(patch)
    theta = injecteur.sampler.generate(patch_rng(patch.numpy(), 5))
    distordu = apply_distortion(patch.numpy().astype(np.complex128), theta)
    np.testing.assert_allclose(injecte, distordu / np.sqrt(np.trace(polarimetric_covariance(distordu)).real), rtol=1e-12)
    assert np.trace(polarimetric_covariance(injecte)).real == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(cible, theta.to_vector(), rtol=1e-12)


@pytest.mark.parametrize("valeur", [0.0, np.nan, np.inf])
def test_inject_ecarte_un_patch_contenant_un_zero_ou_une_valeur_non_finie(valeur):
    patch = patch_aleatoire()
    patch[1, 3, 4] = valeur
    assert CrosstalkInjector().inject(patch) is None


def test_les_deux_transformations_sont_soeurs_et_partagent_l_injection():
    """Chacune n'ajoute que sa représentation du patch injecté : aucune n'hérite de l'autre."""
    assert MLPInjectionTransform.inject is ResNetInjectionTransform.inject is CrosstalkInjector.inject
    assert not issubclass(ResNetInjectionTransform, MLPInjectionTransform)
    assert not issubclass(MLPInjectionTransform, ResNetInjectionTransform)
    assert not callable(CrosstalkInjector())     # l'injecteur seul n'est pas une transformation de cvnn


def test_l_entree_du_resnet_est_le_patch_injecte():
    transformation = ResNetInjectionTransform(seed_injection=5)
    patch = patch_injecte(transformation)
    entree, cible = transformation(patch)
    injecte, cible_injectee = transformation.inject(patch)
    assert entree.shape == (4, 16, 16) and entree.dtype == torch.complex64
    assert cible.shape == (5,) and cible.dtype == torch.complex64
    np.testing.assert_allclose(entree.numpy(), injecte, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(cible.numpy(), cible_injectee, rtol=1e-6)


def test_pour_un_meme_patch_le_mlp_et_le_resnet_voient_les_memes_donnees():
    """La garantie de l'appariement : même cible, et la covariance de l'entrée du ResNet est l'entrée du MLP."""
    mlp, resnet = MLPInjectionTransform(seed_injection=2, no_injection_rate=0.2), ResNetInjectionTransform(seed_injection=2, no_injection_rate=0.2)
    for patch in (patch_injecte(mlp), patch_brut(mlp)):
        (entree_mlp, cible_mlp), (entree_resnet, cible_resnet) = mlp(patch), resnet(patch)
        assert torch.equal(cible_resnet, cible_mlp)
        covariance = polarimetric_covariance(entree_resnet.numpy())
        np.testing.assert_allclose(covariance_entries(covariance), entree_mlp.numpy().ravel(), rtol=1e-5, atol=1e-7)


def test_un_patch_ecarte_donne_au_resnet_une_entree_nulle_et_une_cible_nan():
    patch = patch_aleatoire()
    patch[0, 0, 0] = 0
    entree, cible = ResNetInjectionTransform()(patch)
    assert entree.shape == (4, 16, 16) and torch.all(entree == 0)
    assert torch.all(torch.isnan(cible.real)) and torch.all(torch.isnan(cible.imag))


# ----------------------------------------------------------------- patchs écartés et refus


@pytest.mark.parametrize("valeur", [0.0, np.nan, np.inf])
def test_un_patch_contenant_un_zero_ou_une_valeur_non_finie_est_ecarte(valeur):
    """Bord de fauchée : entrée nulle, cible nan ; la perte et l'évaluation ignorent les cibles nan."""
    patch = patch_aleatoire()
    patch[2, 5, 7] = valeur
    entree, cible = MLPInjectionTransform()(patch)
    assert torch.all(entree == 0)
    assert torch.all(torch.isnan(cible.real)) and torch.all(torch.isnan(cible.imag))


@pytest.mark.parametrize("forme", [(3, 16, 16), (16, 16), (2, 4, 16, 16)])
def test_un_patch_mal_forme_est_refuse(forme):
    with pytest.raises(ValueError):
        MLPInjectionTransform()(torch.ones(forme, dtype=torch.complex64))


@pytest.mark.parametrize("reglages, erreur", [
    (dict(seed_injection=-1), ValueError),
    (dict(seed_injection=1.5), TypeError),
    (dict(seed_injection=True), TypeError),
    (dict(crosstalk_db=(-5.0, -40.0)), ValueError),     # loi invalide
    (dict(crosstalk=(-40.0, -5.0)), TypeError),         # champ inconnu de CrosstalkSampler
])
def test_une_configuration_invalide_est_refusee_a_la_construction(reglages, erreur):
    with pytest.raises(erreur):
        CrosstalkInjector(**reglages)


def test_les_parametres_du_yaml_construisent_la_loi_du_yaml():
    """cvnn construit la transformation avec MLPInjectionTransform(**params)."""
    params = yaml.safe_load((RACINE / "configs" / "cvnn_mlp.yaml").read_text())["data"]["transforms"][0]["params"]
    transformation = MLPInjectionTransform(**params)
    assert transformation.seed_injection == 0
    assert transformation.sampler == CrosstalkSampler()


# ----------------------------------------------------------------- check_pipeline


def test_check_pipeline_laisse_passer_les_lots_de_la_transformation():
    transformation = MLPInjectionTransform()
    couples = [transformation(patch_aleatoire(graine)) for graine in range(6)]
    check_pipeline(torch.utils.data.DataLoader(couples, batch_size=4))


def test_check_pipeline_arrete_des_patchs_bruts():
    """Ce que verrait le réseau si cvnn avait ignoré la transformation."""
    patchs = [patch_aleatoire(graine) for graine in range(6)]
    with pytest.raises(RuntimeError, match="crosstalk_injection"):
        check_pipeline(torch.utils.data.DataLoader(patchs, batch_size=4))


def test_check_pipeline_arrete_des_entrees_vectorielles():
    """(B, 10) au lieu de (B, 10, 1, 1) : cvnn refuserait ce lot à la construction des chargeurs."""
    couples = [(torch.zeros(10, dtype=torch.complex64), torch.zeros(5, dtype=torch.complex64))] * 4
    with pytest.raises(RuntimeError, match="entrées"):
        check_pipeline(torch.utils.data.DataLoader(couples, batch_size=4))


def test_check_pipeline_arrete_un_couple_de_mauvaise_forme():
    couples = [(torch.zeros((9, 1, 1), dtype=torch.complex64), torch.zeros(5, dtype=torch.complex64))] * 4
    with pytest.raises(RuntimeError, match="entrées"):
        check_pipeline(torch.utils.data.DataLoader(couples, batch_size=4))


def test_check_pipeline_laisse_passer_les_lots_de_la_transformation_du_resnet():
    couples = [ResNetInjectionTransform()(patch_aleatoire(graine)) for graine in range(6)]
    check_pipeline(torch.utils.data.DataLoader(couples, batch_size=4), forme_entree=(4, 16, 16))


def test_check_pipeline_arrete_les_lots_du_mlp_quand_il_attend_ceux_du_resnet():
    couples = [MLPInjectionTransform()(patch_aleatoire(graine)) for graine in range(6)]
    with pytest.raises(RuntimeError, match="entrées"):
        check_pipeline(torch.utils.data.DataLoader(couples, batch_size=4), forme_entree=(4, 16, 16))


# ----------------------------------------------------------------- enregistrement dans cvnn


@pytest.mark.parametrize("nom, classe", [("crosstalk_injection_mlp", MLPInjectionTransform),
                                         ("crosstalk_injection_resnet", ResNetInjectionTransform)])
def test_les_transformations_sont_enregistrees_dans_cvnn(nom, classe):
    from cvnn.transform_registry import get_transform_class

    assert get_transform_class("complex", nom, "polsar") is classe


# ----------------------------------------------------------------- sur la vraie pile

PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def chargeurs(nom_yaml: str, **reglages_data):
    """Les trois chargeurs de cvnn pour un YAML du projet, comme les construisent patchs.py et train_mlp.py."""
    from cvnn.data import get_dataloaders, validate_and_correct_config
    from cvnn.utils import set_seed

    config = yaml.safe_load((RACINE / "configs" / nom_yaml).read_text())
    # Lots de 16 : ces tests n'examinent qu'un lot, inutile d'en transformer batch_size = 1024 comme l'entraînement.
    config["data"].update(num_workers=0, max_num_samples=2000, batch_size=16, **reglages_data)
    set_seed(int(config["seed"]))
    return get_dataloaders(validate_and_correct_config(config), use_cuda=False)


@sur_la_pile
def test_les_chargeurs_de_cvnn_rendent_le_format_attendu():
    for chargeur in chargeurs("cvnn_mlp.yaml"):
        check_pipeline(chargeur)


@sur_la_pile
def test_le_decoupage_est_identique_a_celui_de_la_chaine_a():
    banc = chargeurs("cvnn_mlp.yaml")
    chaine_a = chargeurs("data_uavsar.yaml")
    for partition_banc, partition_a in zip(banc, chaine_a):
        assert list(partition_banc.dataset.indices) == list(partition_a.dataset.indices)


@sur_la_pile
def test_une_loi_invalide_dans_le_yaml_est_detectee_par_check_pipeline():
    """cvnn ignore la transformation qu'il ne peut pas construire : check_pipeline doit s'en apercevoir."""
    transformations = [{"name": "crosstalk_injection_mlp", "params": {"crosstalk_db": [-5.0, -40.0]}}]
    _, _, test = chargeurs("cvnn_mlp.yaml", transforms=transformations)
    with pytest.raises(RuntimeError, match="crosstalk_injection"):
        check_pipeline(test)


@sur_la_pile
def test_sur_la_pile_le_mlp_et_le_resnet_voient_les_memes_donnees():
    """À travers les chargeurs de cvnn : la transformation du ResNet se construit depuis le YAML, et son premier lot de
    test a les cibles de celui du MLP, avec pour covariances les entrées du MLP."""
    params = yaml.safe_load((RACINE / "configs" / "cvnn_mlp.yaml").read_text())["data"]["transforms"][0]["params"]
    *_, test_resnet = chargeurs("cvnn_mlp.yaml", transforms=[{"name": "crosstalk_injection_resnet", "params": params}])
    *_, test_mlp = chargeurs("cvnn_mlp.yaml")
    check_pipeline(test_resnet, forme_entree=(4, 64, 64))

    (entrees_resnet, cibles_resnet), (entrees_mlp, cibles_mlp) = next(iter(test_resnet)), next(iter(test_mlp))
    torch.testing.assert_close(cibles_resnet, cibles_mlp, rtol=0, atol=0, equal_nan=True)
    covariances = covariance_entries(polarimetric_covariance(entrees_resnet.numpy()))
    np.testing.assert_allclose(covariances, entrees_mlp.numpy().reshape(len(entrees_mlp), -1), rtol=1e-5, atol=1e-7)
