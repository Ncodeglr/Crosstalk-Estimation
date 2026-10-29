"""
Tests de la transformation « crosstalk_injection » et de check_pipeline :

1. le couple produit : forme, type, déterminisme, cible = theta_k de patch_rng, entrée = covariance injectée normalisée ;
2. les patchs écartés (zéro ou valeur non finie) et les refus ;
3. check_pipeline : il laisse passer les lots de la transformation et arrête les patchs bruts ;
4. l'enregistrement dans cvnn ;
5. sur la vraie pile, à travers les chargeurs de cvnn : le format des lots, le découpage identique à celui de la
   chaîne A, et la détection d'une transformation que cvnn a ignorée en silence. Sautés si la pile est absente.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.covariance import HERMITIAN_ENTRIES, covariance_entries, polarimetric_covariance
from src.crosstalk.deep_learning import CrosstalkInjectionTransform, check_pipeline
from src.crosstalk.injection import NEUTRE, InjectionLaw, draw_injection, patch_rng
from src.crosstalk.matrix import apply_distortion, distortion_matrix

RACINE = Path(__file__).resolve().parent.parent
PUISSANCES = [k for k, (i, j) in enumerate(HERMITIAN_ENTRIES) if i == j]


def patch_aleatoire(graine: int = 0, forme=(4, 16, 16)) -> torch.Tensor:
    """Un patch tel que cvnn le donne à la transformation : tenseur complex64 (4, H, W), sans zéro."""
    rng = np.random.default_rng(graine)
    return torch.from_numpy((rng.normal(size=forme) + 1j * rng.normal(size=forme)).astype(np.complex64))


def patch_injecte(transformation: CrosstalkInjectionTransform) -> torch.Tensor:
    """Le premier patch aléatoire qui reçoit une injection (non neutre)."""
    for graine in range(100):
        patch = patch_aleatoire(graine)
        if draw_injection(transformation.law, patch_rng(patch.numpy(), transformation.seed)) != NEUTRE:
            return patch
    raise AssertionError("aucun patch injecté sur 100 essais")


def patch_brut(transformation: CrosstalkInjectionTransform) -> torch.Tensor:
    """Le premier patch aléatoire laissé brut (theta_k neutre)."""
    for graine in range(100):
        patch = patch_aleatoire(graine)
        if draw_injection(transformation.law, patch_rng(patch.numpy(), transformation.seed)) == NEUTRE:
            return patch
    raise AssertionError("aucun patch brut sur 100 essais")


# ----------------------------------------------------------------- le couple produit


def test_forme_et_type_du_couple():
    entree, cible = CrosstalkInjectionTransform()(patch_aleatoire())
    assert entree.shape == (10, 1, 1) and entree.dtype == torch.complex64
    assert cible.shape == (5,) and cible.dtype == torch.complex64


def test_le_meme_patch_donne_toujours_le_meme_couple():
    transformation = CrosstalkInjectionTransform(seed=3)
    premier, second = transformation(patch_aleatoire()), transformation(patch_aleatoire())
    assert torch.equal(premier[0], second[0]) and torch.equal(premier[1], second[1])


def test_un_tableau_numpy_donne_le_meme_couple_qu_un_tenseur():
    transformation = CrosstalkInjectionTransform()
    patch = patch_aleatoire()
    assert all(torch.equal(a, b) for a, b in zip(transformation(patch), transformation(patch.numpy())))


def test_une_autre_graine_donne_une_autre_injection():
    patch = patch_injecte(CrosstalkInjectionTransform(seed=0))
    assert not torch.equal(CrosstalkInjectionTransform(seed=0)(patch)[1], CrosstalkInjectionTransform(seed=1)(patch)[1])


def test_la_cible_est_le_theta_k_de_patch_rng_avec_log_alpha():
    transformation = CrosstalkInjectionTransform(seed=5)
    patch = patch_injecte(transformation)
    theta = draw_injection(transformation.law, patch_rng(patch.numpy(), 5)).to_vector()
    attendu = np.r_[theta[:4], np.log(theta[4])]
    np.testing.assert_allclose(transformation(patch)[1].numpy(), attendu, rtol=1e-6)


def test_l_entree_est_la_covariance_injectee_normalisee_par_sa_trace():
    transformation = CrosstalkInjectionTransform()
    patch = patch_injecte(transformation)
    theta = draw_injection(transformation.law, patch_rng(patch.numpy(), 0))
    d = distortion_matrix(theta)
    covariance = d @ polarimetric_covariance(patch.numpy()) @ d.conj().T
    attendu = covariance_entries(covariance / np.trace(covariance).real)
    np.testing.assert_allclose(transformation(patch)[0].numpy().ravel(), attendu, rtol=1e-5, atol=1e-7)


def test_injecter_la_covariance_revient_a_injecter_les_pixels():
    """D agit linéairement sur chaque pixel : < (D k)(D k)^H > = D < k k^H > D^H."""
    transformation = CrosstalkInjectionTransform()
    patch = patch_injecte(transformation)
    theta = draw_injection(transformation.law, patch_rng(patch.numpy(), 0))
    covariance = polarimetric_covariance(apply_distortion(patch.numpy().astype(np.complex128), theta))
    attendu = covariance_entries(covariance / np.trace(covariance).real)
    np.testing.assert_allclose(transformation(patch)[0].numpy().ravel(), attendu, rtol=1e-5, atol=1e-7)


def test_les_quatre_puissances_somment_a_un():
    entree = CrosstalkInjectionTransform()(patch_aleatoire())[0].numpy().ravel()
    assert entree[PUISSANCES].real.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(entree[PUISSANCES].imag == 0)


def test_un_patch_brut_a_une_cible_neutre_et_sa_covariance_seule():
    transformation = CrosstalkInjectionTransform()
    patch = patch_brut(transformation)
    entree, cible = transformation(patch)
    assert torch.all(cible == 0)
    covariance = polarimetric_covariance(patch.numpy())
    np.testing.assert_allclose(entree.numpy().ravel(), covariance_entries(covariance / np.trace(covariance).real),
                               rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------- patchs écartés et refus


@pytest.mark.parametrize("valeur", [0.0, np.nan, np.inf])
def test_un_patch_contenant_un_zero_ou_une_valeur_non_finie_est_ecarte(valeur):
    """Bord de fauchée : entrée nulle, cible nan ; la perte et l'évaluation ignorent les cibles nan."""
    patch = patch_aleatoire()
    patch[2, 5, 7] = valeur
    entree, cible = CrosstalkInjectionTransform()(patch)
    assert torch.all(entree == 0)
    assert torch.all(torch.isnan(cible.real)) and torch.all(torch.isnan(cible.imag))


@pytest.mark.parametrize("forme", [(3, 16, 16), (16, 16), (2, 4, 16, 16)])
def test_un_patch_mal_forme_est_refuse(forme):
    with pytest.raises(ValueError):
        CrosstalkInjectionTransform()(torch.ones(forme, dtype=torch.complex64))


@pytest.mark.parametrize("reglages, erreur", [
    (dict(seed=-1), ValueError),
    (dict(seed=1.5), TypeError),
    (dict(seed=True), TypeError),
    (dict(crosstalk_db=(-5.0, -40.0)), ValueError),     # loi invalide
    (dict(crosstalk=(-40.0, -5.0)), TypeError),         # champ inconnu d'InjectionLaw
])
def test_une_configuration_invalide_est_refusee_a_la_construction(reglages, erreur):
    with pytest.raises(erreur):
        CrosstalkInjectionTransform(**reglages)


def test_les_parametres_du_yaml_construisent_la_loi_du_yaml():
    """cvnn construit la transformation avec CrosstalkInjectionTransform(**params)."""
    params = yaml.safe_load((RACINE / "configs" / "cvnn_injection.yaml").read_text())["data"]["transforms"][0]["params"]
    transformation = CrosstalkInjectionTransform(**params)
    assert transformation.seed == 0
    assert transformation.law == InjectionLaw()


# ----------------------------------------------------------------- check_pipeline


def test_check_pipeline_laisse_passer_les_lots_de_la_transformation():
    transformation = CrosstalkInjectionTransform()
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


# ----------------------------------------------------------------- enregistrement dans cvnn


def test_la_transformation_est_enregistree_dans_cvnn():
    from cvnn.transform_registry import get_transform_class

    assert get_transform_class("complex", "crosstalk_injection", "polsar") is CrosstalkInjectionTransform


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
    for chargeur in chargeurs("cvnn_injection.yaml"):
        check_pipeline(chargeur)


@sur_la_pile
def test_le_decoupage_est_identique_a_celui_de_la_chaine_a():
    banc = chargeurs("cvnn_injection.yaml")
    chaine_a = chargeurs("data_uavsar.yaml")
    for partition_banc, partition_a in zip(banc, chaine_a):
        assert list(partition_banc.dataset.indices) == list(partition_a.dataset.indices)


@sur_la_pile
def test_une_loi_invalide_dans_le_yaml_est_detectee_par_check_pipeline():
    """cvnn ignore la transformation qu'il ne peut pas construire : check_pipeline doit s'en apercevoir."""
    transformations = [{"name": "crosstalk_injection", "params": {"crosstalk_db": [-5.0, -40.0]}}]
    _, _, test = chargeurs("cvnn_injection.yaml", transforms=transformations)
    with pytest.raises(RuntimeError, match="crosstalk_injection"):
        check_pipeline(test)
