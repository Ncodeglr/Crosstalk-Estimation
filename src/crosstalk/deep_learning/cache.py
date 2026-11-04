"""
Le cache du banc d'essai par réinjection, commun au MLP et au ResNet : les couples (entrée, cible) de la
transformation d'injection, calculés une seule fois et gardés sur disque.

Pourquoi
--------
À chaque époque, les chargeurs de cvnn relisent la pile pour recalculer les mêmes couples : theta_k est figé par patch
(patch_rng), et l'entrée ne dépend que du patch. La lecture prend 99 % du temps : avec des patchs 64 x 64 et
8 workers, une époque dure environ 53 min (mesure du 2026-10-29). On calcule donc les couples une fois, avec les
chargeurs de cvnn et la transformation elle-même, puis on entraîne à partir du cache.

Format
------
Un dossier par cache, cache/<transformation>_<empreinte>/ (crosstalk_injection_mlp_... ou
crosstalk_injection_resnet_...), avec trois fichiers .npy par partition (train, valid, test) :
    <partition>_entrees.npy   les entrées : (N, 10, 1, 1) pour le MLP, (N, 4, H, W) pour le ResNet, en complex64 ;
    <partition>_cibles.npy    les cibles (N, 5), en complex64 ;
    <partition>_indices.npy   la position de chaque patch dans la pile.
Les fichiers sont lus par memmap : seules les pages utilisées sont lues sur le disque. Le même format sert au cache du
MLP (environ 89 Mo pour toute la pile) et à celui du ResNet (environ 32 Go pour 100 000 patchs d'entraînement et des
partitions de validation et de test complètes).

Sous-ensemble d'entraînement
----------------------------
data.max_train_samples (null par défaut : tout) ne garde que les N premiers patchs de la partition d'entraînement,
dans l'ordre du découpage de cvnn. Cet ordre est aléatoire par emplacement au sol (les dates d'un même emplacement se
suivent) : on garde donc un sous-ensemble aléatoire d'emplacements, chacun avec toutes ses dates. La validation et le
test restent complets : les erreurs du MLP et du ResNet se comparent sur les mêmes patchs. Seuls les N patchs gardés
sont lus à la construction.

Garanties
---------
- Mêmes données : le cache contient la sortie de la transformation pour chaque patch des partitions de cvnn, avec les
  indices de ces patchs dans la pile.
- Jamais périmé : le nom du dossier contient l'empreinte de tout ce qui détermine son contenu (fingerprint). À chaque
  chargement, N_CONTROLE patchs de chaque partition, tirés au hasard, sont recalculés par la transformation et doivent
  être identiques au cache : un changement du code d'injection est ainsi détecté même si le YAML n'a pas changé.
- Jamais à moitié écrit : le cache est écrit dans un dossier temporaire (<nom>.tmp), puis renommé.
- Mémoire constante : les fichiers sont créés à leur taille finale, puis remplis lot par lot.
- Sans effet sur la suite : la construction n'utilise pas les générateurs aléatoires globaux. L'initialisation du
  réseau est donc la même, que le cache vienne d'être construit ou qu'il soit seulement relu.

Limites
-------
Le cache suppose que l'injection reste figée par patch.
L'ordre de mélange des lots n'est pas celui des chargeurs de cvnn ; on n'en attend aucun effet sur le résultat.
"""

from __future__ import annotations
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, TensorDataset
from tqdm import tqdm
from .crosstalk_injection import injection_transform_yaml

logger = logging.getLogger(__name__)

#: Dossier des caches, à la racine du dépôt (ignoré par git).
DOSSIER_CACHE: Path = Path(__file__).resolve().parents[3] / "cache"

#: Nombre de patchs de chaque partition recalculés à chaque chargement pour contrôler le cache.
N_CONTROLE: int = 8

PARTITIONS: tuple[str, ...] = ("train", "valid", "test")


def fingerprint(cfg: dict[str, Any]) -> str:
    """
    Empreinte de tout ce qui détermine le contenu du cache : la pile, la taille des patchs, le découpage (graine,
    proportions, max_num_samples), le sous-ensemble d'entraînement (max_train_samples) et les transformations (dont la
    loi d'injection et sa graine). Le modèle, l'optimiseur, nepochs et seed_train n'y entrent pas : les changer
    réutilise le même cache.
    """
    donnees = cfg["data"]
    contenu = {
        "trainpath": str(Path(donnees["dataset"]["trainpath"]).resolve()),
        "dataset": {cle: donnees["dataset"].get(cle) for cle in ("name", "patch_size", "patch_stride")},
        "seed": cfg["seed"],
        "decoupage": {cle: donnees.get(cle) for cle in ("valid_ratio", "test_ratio", "max_num_samples",
                                                        "max_train_samples")},
        "transforms": donnees.get("transforms"),
    }
    return hashlib.sha256(json.dumps(contenu, sort_keys=True).encode()).hexdigest()[:16]


def cache_name(cfg: dict[str, Any]) -> str:
    """
    Le nom du dossier du cache : le nom de la transformation d'injection du YAML (injection_transform_yaml), puis
    l'empreinte, par exemple crosstalk_injection_mlp_8d53430c7aecbff4.

    Raises: ValueError: si data.transforms ne contient pas exactement une transformation d'injection.
    """
    return f"{injection_transform_yaml(cfg)['name']}_{fingerprint(cfg)}"


def _partitions(chargeurs, cfg: dict[str, Any]) -> list[tuple[str, DataLoader, Dataset, list[int]]]:
    """
    Les patchs à mettre en cache, partition par partition : (nom, chargeur de cvnn, jeu de données, indices des patchs
    dans la pile). L'entraînement est réduit à ses data.max_train_samples premiers patchs ; la validation et le test
    sont complets.

    Raises: ValueError: si max_train_samples n'est ni null ni un entier de 1 à la taille de la partition d'entraînement.
    """
    parties = [(nom, chargeur, chargeur.dataset, list(chargeur.dataset.indices))
               for nom, chargeur in zip(PARTITIONS, chargeurs)]
    n = cfg["data"].get("max_train_samples")
    if n is None:
        return parties
    nom, chargeur, entrainement, indices = parties[0]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(entrainement):
        raise ValueError(f"data.max_train_samples = {n!r} : null, ou un entier de 1 à {len(entrainement)} (la taille de "
                         f"la partition d'entraînement), attendu")
    parties[0] = (nom, chargeur, Subset(entrainement, range(n)), indices[:n])
    return parties


def build_cache(parties: list[tuple[str, DataLoader, Dataset, list[int]]], chemin: Path) -> None:
    """
    Parcourt une fois chaque partition (_partitions), dans l'ordre et sans mélange, et écrit ses entrées, ses cibles et
    les indices de ses patchs dans la pile, dans un dossier temporaire renommé à la fin.
    """
    temporaire = chemin.with_name(chemin.name + ".tmp")
    shutil.rmtree(temporaire, ignore_errors=True)   # reste éventuel d'une construction interrompue
    temporaire.mkdir(parents=True)
    for nom, chargeur, donnees, indices in parties:
        np.save(temporaire / f"{nom}_indices.npy", np.asarray(indices, dtype=np.int64))
        #Les fichiers sont créés à leur taille finale, avec la forme d'un premier couple, puis remplis lot par lot.
        entree, cible = donnees[0]
        entrees = np.lib.format.open_memmap(temporaire / f"{nom}_entrees.npy", mode="w+", dtype=np.complex64,
                                            shape=(len(donnees), *entree.shape))
        cibles = np.lib.format.open_memmap(temporaire / f"{nom}_cibles.npy", mode="w+", dtype=np.complex64,
                                           shape=(len(donnees), *cible.shape))
        #Générateur propre : sans lui, le DataLoader tirerait la graine de ses workers dans le générateur global de torch.
        lecteur = DataLoader(donnees, batch_size=chargeur.batch_size, shuffle=False, num_workers=chargeur.num_workers,
                             generator=torch.Generator().manual_seed(0))
        debut = 0
        for x, y in tqdm(lecteur, desc=f"Cache {nom}"):
            entrees[debut:debut + len(x)] = x.numpy()
            cibles[debut:debut + len(y)] = y.numpy()
            debut += len(x)
        entrees.flush()
        cibles.flush()
        del entrees, cibles
    temporaire.replace(chemin)


def cached_loaders(chargeurs, cfg: dict[str, Any], dossier: Path = DOSSIER_CACHE) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Les trois chargeurs (entraînement, validation, test), servis depuis le cache. Le cache est construit s'il
    n'existe pas, puis contrôlé à chaque chargement.

    Args:
        chargeurs: les trois chargeurs de cvnn (get_dataloaders), déjà vérifiés par check_pipeline.
        cfg: la configuration, pour l'empreinte et max_train_samples.
        dossier: le dossier des caches.

    Returns:
        trois chargeurs dont les données restent sur disque (memmap), de lots (B, *forme d'entrée) et (B, 5), avec la
        taille de lot des chargeurs de cvnn ; seul celui qui mélangeait ses lots chez cvnn (l'entraînement) les mélange.

    Raises:
        RuntimeError: si le cache ne correspond pas au découpage ou à la transformation actuels.
        ValueError: si data.max_train_samples est invalide (_partitions).
    """
    #Étape 1 : construire le cache s'il n'existe pas (une passe sur les patchs à mettre en cache).
    parties = _partitions(chargeurs, cfg)
    chemin = dossier / cache_name(cfg)
    if not chemin.exists():
        logger.info("cache absent, construction (une passe sur la pile) : %s", chemin)
        build_cache(parties, chemin)

    #Étape 2 : le relire par memmap, et contrôler qu'il correspond au découpage et à la transformation actuels.
    tirage = np.random.default_rng()   # générateur propre : ne touche pas au générateur global de numpy
    chargeurs_du_cache = []
    for nom, chargeur, donnees, indices in parties:
        #mmap_mode="c" : lu à la demande sur le disque, et modifiable en mémoire sans toucher au fichier, ce qu'attend torch.from_numpy.
        entrees, cibles = (torch.from_numpy(np.load(chemin / f"{nom}_{champ}.npy", mmap_mode="c"))
                           for champ in ("entrees", "cibles"))
        if np.load(chemin / f"{nom}_indices.npy").tolist() != indices:
            raise RuntimeError(f"{chemin.name}, partition {nom} : les patchs du cache ne sont pas ceux du découpage "
                               f"actuel. Supprimer ce dossier pour le reconstruire.")
        for j in tirage.choice(len(indices), size=min(N_CONTROLE, len(indices)), replace=False):
            entree, cible = donnees[int(j)]
            if not (_identiques(entree, entrees[j]) and _identiques(cible, cibles[j])):
                raise RuntimeError(f"{chemin.name}, partition {nom}, patch {indices[j]} : le cache ne correspond plus à "
                                   f"la transformation (code d'injection modifié ?). Supprimer ce dossier pour le "
                                   f"reconstruire.")

        #Étape 3 : le chargeur de la partition, aux réglages du chargeur de cvnn.
        chargeurs_du_cache.append(DataLoader(TensorDataset(entrees, cibles), batch_size=chargeur.batch_size,
                                             shuffle=isinstance(chargeur.sampler, RandomSampler),
                                             pin_memory=chargeur.pin_memory))
    logger.info("cache chargé et contrôlé : %s", chemin)
    return tuple(chargeurs_du_cache)


def _identiques(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Égalité à l'arrondi du complex64 près, nan compris (cibles des patchs écartés)."""
    return a.shape == b.shape and bool(torch.isclose(torch.view_as_real(a), torch.view_as_real(b), rtol=1e-5,
                                                     atol=1e-7, equal_nan=True).all())
