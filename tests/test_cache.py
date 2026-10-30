"""
Tests du cache des couples (entrée, cible), commun au MLP et au ResNet :

1. l'empreinte : ce qui la change (données, découpage, sous-ensemble d'entraînement, loi d'injection) et ce qui ne la
   change pas (modèle, optimiseur, époques, taille de lot, graine d'entraînement) ;
2. sur la vraie pile : le cache contient exactement la sortie de la transformation (celle du MLP comme celle du
   ResNet), dans le découpage de cvnn, l'entraînement réduit à ses max_train_samples premiers patchs ; il est
   réutilisé, et refusé s'il a été altéré ou s'il vient d'un autre découpage ; ses chargeurs ont les réglages de ceux
   de cvnn ; il ne touche pas aux générateurs aléatoires globaux. Sautés si la pile est absente.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import RandomSampler

from src.crosstalk.deep_learning import cached_loaders, check_pipeline
from src.crosstalk.deep_learning import cache as module_cache
from src.crosstalk.deep_learning.cache import PARTITIONS, fingerprint

RACINE = Path(__file__).resolve().parent.parent
PILE = RACINE / "data" / "Rosamd_35012_04"
sur_la_pile = pytest.mark.skipif(not PILE.is_dir(), reason="pile UAVSAR absente")


def config_du_banc() -> dict:
    return yaml.safe_load((RACINE / "configs" / "cvnn_mlp.yaml").read_text())


# ----------------------------------------------------------------- l'empreinte


def test_la_meme_configuration_donne_la_meme_empreinte():
    assert fingerprint(config_du_banc()) == fingerprint(config_du_banc())


@pytest.mark.parametrize("modification", [
    lambda c: c["data"]["dataset"].update(patch_size=32, patch_stride=32),
    lambda c: c["data"]["dataset"].update(trainpath="data/autre_pile"),
    lambda c: c.update(seed=301),
    lambda c: c["data"].update(test_ratio=0.2),
    lambda c: c["data"].update(max_num_samples=1000),
    lambda c: c["data"].update(max_train_samples=100_000),
    lambda c: c["data"]["transforms"][0]["params"].update(crosstalk_db=[-30.0, -5.0]),
    lambda c: c["data"]["transforms"][0]["params"].update(seed=1),
    lambda c: c["data"]["transforms"][0].update(name="crosstalk_injection_resnet"),
])
def test_ce_qui_determine_les_couples_change_l_empreinte(modification):
    config = config_du_banc()
    modification(config)
    assert fingerprint(config) != fingerprint(config_du_banc())


@pytest.mark.parametrize("modification", [
    lambda c: c["model"].update(channels_width=16),
    lambda c: c["optim"]["params"].update(lr=0.001),
    lambda c: c.update(nepochs=30),
    lambda c: c["data"].update(batch_size=256, num_workers=16),
    lambda c: c.update(train_seed=2),
])
def test_le_modele_l_optimiseur_et_le_chargement_ne_changent_pas_l_empreinte(modification):
    config = config_du_banc()
    modification(config)
    assert fingerprint(config) == fingerprint(config_du_banc())


def test_le_nom_du_cache_est_celui_de_la_transformation_suivi_de_l_empreinte():
    config = config_du_banc()
    assert module_cache.cache_name(config) == f"crosstalk_injection_mlp_{fingerprint(config)}"


def test_une_configuration_sans_transformation_d_injection_n_a_pas_de_cache():
    config = config_du_banc()
    config["data"]["transforms"] = []
    with pytest.raises(ValueError, match="transformations d'injection"):
        module_cache.cache_name(config)


# ----------------------------------------------------------------- sur la vraie pile


def chargeurs_de_cvnn(**reglages_data):
    """Les chargeurs de cvnn pour le YAML du banc réduit à 200 patchs, construits comme dans train_mlp.py."""
    from cvnn.data import get_dataloaders, validate_and_correct_config
    from cvnn.utils import set_seed

    config = config_du_banc()
    config["data"].update({"num_workers": 0, "max_num_samples": 200, "batch_size": 16, **reglages_data})
    set_seed(int(config["seed"]))
    config = validate_and_correct_config(config)
    return get_dataloaders(config, use_cuda=False), config


@pytest.fixture(scope="module")
def banc(tmp_path_factory):
    """Les chargeurs de cvnn, la configuration, le dossier des caches, et les chargeurs servis par le cache."""
    chargeurs, config = chargeurs_de_cvnn()
    dossier = tmp_path_factory.mktemp("cache")
    return chargeurs, config, dossier, cached_loaders(chargeurs, config, dossier)


def dossier_du_cache(dossier: Path) -> Path:
    """Le seul cache construit dans dossier."""
    (chemin,) = dossier.iterdir()
    return chemin


def lire(cache: Path, partition: str, champ: str) -> np.ndarray:
    return np.load(cache / f"{partition}_{champ}.npy")


@sur_la_pile
def test_le_cache_contient_exactement_la_sortie_de_la_transformation(banc):
    chargeurs, _, _, servis = banc
    for chargeur, servi in zip(chargeurs, servis):
        entrees, cibles = servi.dataset.tensors
        assert len(entrees) == len(chargeur.dataset) > 0
        for j in range(len(chargeur.dataset)):
            entree, cible = chargeur.dataset[j]
            torch.testing.assert_close(entrees[j], entree, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(cibles[j], cible, rtol=0, atol=0, equal_nan=True)


@sur_la_pile
def test_les_patchs_du_cache_sont_ceux_du_decoupage_et_le_dossier_est_complet(banc):
    chargeurs, config, dossier, _ = banc
    cache = dossier_du_cache(dossier)
    assert cache.name == module_cache.cache_name(config) and cache.name.startswith("crosstalk_injection_mlp_")
    for nom, chargeur in zip(PARTITIONS, chargeurs):
        assert lire(cache, nom, "indices").tolist() == list(chargeur.dataset.indices)
    assert sorted(p.name for p in cache.iterdir()) == sorted(f"{nom}_{champ}.npy" for nom in PARTITIONS
                                                             for champ in ("entrees", "cibles", "indices"))


@sur_la_pile
def test_le_cache_du_resnet_contient_exactement_la_sortie_de_sa_transformation(tmp_path):
    params = config_du_banc()["data"]["transforms"][0]["params"]
    chargeurs, config = chargeurs_de_cvnn(max_num_samples=60,
                                          transforms=[{"name": "crosstalk_injection_resnet", "params": params}])
    servis = cached_loaders(chargeurs, config, tmp_path)
    assert dossier_du_cache(tmp_path).name.startswith("crosstalk_injection_resnet_")
    for chargeur, servi in zip(chargeurs, servis):
        check_pipeline(servi, forme_entree=(4, 64, 64))
        entrees, cibles = servi.dataset.tensors
        for j in (0, len(chargeur.dataset) - 1):
            entree, cible = chargeur.dataset[j]
            torch.testing.assert_close(entrees[j], entree, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(cibles[j], cible, rtol=0, atol=0, equal_nan=True)


@sur_la_pile
def test_max_train_samples_garde_les_premiers_patchs_d_entrainement_et_tout_le_reste(tmp_path):
    chargeurs, config = chargeurs_de_cvnn(max_train_samples=50)
    servis = cached_loaders(chargeurs, config, tmp_path)
    cache = dossier_du_cache(tmp_path)
    assert len(servis[0].dataset) == 50
    assert lire(cache, "train", "indices").tolist() == list(chargeurs[0].dataset.indices)[:50]
    for nom, chargeur, servi in zip(PARTITIONS[1:], chargeurs[1:], servis[1:]):
        assert len(servi.dataset) == len(chargeur.dataset)
        assert lire(cache, nom, "indices").tolist() == list(chargeur.dataset.indices)


@sur_la_pile
@pytest.mark.parametrize("valeur", [0, 10_000, 12.5, True])
def test_un_max_train_samples_invalide_est_refuse(tmp_path, valeur):
    chargeurs, config = chargeurs_de_cvnn(max_train_samples=valeur)
    with pytest.raises(ValueError, match="max_train_samples"):
        cached_loaders(chargeurs, config, tmp_path)


@sur_la_pile
def test_les_chargeurs_du_cache_ont_les_reglages_de_ceux_de_cvnn(banc):
    chargeurs, _, _, servis = banc
    for chargeur, servi in zip(chargeurs, servis):
        check_pipeline(servi)
        assert servi.batch_size == chargeur.batch_size
        assert isinstance(servi.sampler, RandomSampler) == isinstance(chargeur.sampler, RandomSampler)
    assert isinstance(servis[0].sampler, RandomSampler)          # seul l'entraînement mélange ses lots


@sur_la_pile
def test_un_second_appel_relit_le_cache_sans_le_reconstruire(banc, monkeypatch):
    chargeurs, config, dossier, _ = banc

    def interdit(*args, **kwargs):
        raise AssertionError("le cache a été reconstruit")

    monkeypatch.setattr(module_cache, "build_cache", interdit)
    cached_loaders(chargeurs, config, dossier)


def copie_alteree(banc, dossier: Path, alteration) -> None:
    """Copie le cache du banc dans dossier, sous le même nom, puis applique alteration au dossier copié."""
    _, _, origine, _ = banc
    copie = dossier / dossier_du_cache(origine).name
    shutil.copytree(dossier_du_cache(origine), copie)
    alteration(copie)


@sur_la_pile
def test_un_cache_altere_est_refuse(banc, tmp_path):
    copie_alteree(banc, tmp_path, lambda cache: np.save(cache / "train_entrees.npy",
                                                        lire(cache, "train", "entrees") + 1))
    chargeurs, config, _, _ = banc
    with pytest.raises(RuntimeError, match="transformation"):
        cached_loaders(chargeurs, config, tmp_path)


@sur_la_pile
def test_un_cache_d_un_autre_decoupage_est_refuse(banc, tmp_path):
    copie_alteree(banc, tmp_path, lambda cache: np.save(cache / "valid_indices.npy",
                                                        lire(cache, "valid", "indices")[::-1]))
    chargeurs, config, _, _ = banc
    with pytest.raises(RuntimeError, match="découpage"):
        cached_loaders(chargeurs, config, tmp_path)


@sur_la_pile
def test_construire_ou_relire_le_cache_ne_touche_pas_aux_generateurs_globaux(banc, tmp_path):
    """Sinon, l'initialisation du réseau dépendrait de ce que le cache vienne d'être construit ou non."""
    chargeurs, config, _, _ = banc
    for _ in range(2):                                           # construction, puis relecture
        torch_avant, numpy_avant = torch.get_rng_state(), np.random.get_state()[1].copy()
        cached_loaders(chargeurs, config, tmp_path)
        assert torch.equal(torch.get_rng_state(), torch_avant)
        assert np.array_equal(np.random.get_state()[1], numpy_avant)
