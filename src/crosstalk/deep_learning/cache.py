"""
Le cache du banc d'essai par réinjection : les couples (entrée, cible) de CrosstalkInjectionTransform, calculés une
seule fois.

Pourquoi
--------
À chaque époque, les chargeurs de cvnn relisent toute la pile pour recalculer les mêmes couples : theta_k est figé
par patch (patch_rng) et la covariance ne dépend que du patch. La lecture prend 99 % du temps : avec des patchs
64 x 64 et 8 workers, une époque dure environ 53 min (mesure du 2026-10-29). Les couples, eux, tiennent en 15
complexes par patch, soit environ 89 Mo pour toute la pile. On les calcule donc une fois, avec les chargeurs de cvnn
et la transformation elle-même, puis on entraîne à partir de ce fichier.

Garanties
---------
- Mêmes données : le cache contient la sortie de la transformation pour chaque patch des partitions de cvnn, avec les
  indices de ces patchs dans la pile.
- Jamais périmé : le nom du fichier contient l'empreinte de tout ce qui détermine son contenu (fingerprint). À chaque
  chargement, N_CONTROLE patchs de chaque partition, tirés au hasard, sont recalculés par la transformation et doivent
  être identiques au cache : un changement du code d'injection est ainsi détecté même si le YAML n'a pas changé.
- Jamais à moitié écrit : le fichier est écrit sous un nom temporaire, puis renommé.
- Sans effet sur la suite : la construction n'utilise pas les générateurs aléatoires globaux. L'initialisation du
  réseau est donc la même, que le cache vienne d'être construit ou qu'il soit seulement relu.

Limites
-------
Le cache ne sert qu'aux réseaux qui prennent la covariance en entrée (le MLP) : les patchs bruts d'un ResNet ne
tiendraient pas en mémoire. Il suppose aussi que l'injection reste figée par patch.
L'ordre de mélange des lots n'est pas celui des chargeurs de cvnn ; on n'en attend aucun effet sur le résultat.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Dossier des caches, à la racine du dépôt (ignoré par git).
DOSSIER_CACHE: Path = Path(__file__).resolve().parents[3] / "cache"

#: Nombre de patchs de chaque partition recalculés à chaque chargement pour contrôler le cache.
N_CONTROLE: int = 8

PARTITIONS: tuple[str, ...] = ("train", "valid", "test")


def fingerprint(cfg: dict[str, Any]) -> str:
    """
    Empreinte de tout ce qui détermine le contenu du cache : la pile, la taille des patchs, le découpage (graine,
    proportions, max_num_samples) et les transformations (dont la loi d'injection et sa graine). Le modèle,
    l'optimiseur et nepochs n'y entrent pas : les changer réutilise le même cache.
    """
    donnees = cfg["data"]
    contenu = {
        "trainpath": str(Path(donnees["dataset"]["trainpath"]).resolve()),
        "dataset": {cle: donnees["dataset"].get(cle) for cle in ("name", "patch_size", "patch_stride")},
        "seed": cfg["seed"],
        "decoupage": {cle: donnees.get(cle) for cle in ("valid_ratio", "test_ratio", "max_num_samples")},
        "transforms": donnees.get("transforms"),
    }
    return hashlib.sha256(json.dumps(contenu, sort_keys=True).encode()).hexdigest()[:16]


def build_cache(chargeurs, chemin: Path) -> None:
    """
    Parcourt une fois chaque partition des chargeurs de cvnn, dans l'ordre et sans mélange, et enregistre ses entrées,
    ses cibles et les indices de ses patchs dans la pile.
    """
    contenu: dict[str, Any] = {}
    for nom, chargeur in zip(PARTITIONS, chargeurs):
        #Générateur propre : sans lui, le DataLoader tirerait la graine de ses workers dans le générateur global de torch.
        lecteur = DataLoader(chargeur.dataset, batch_size=chargeur.batch_size, shuffle=False,
                             num_workers=chargeur.num_workers, generator=torch.Generator().manual_seed(0))
        entrees, cibles = [], []
        for x, y in tqdm(lecteur, desc=f"Cache {nom}"):
            entrees.append(x)
            cibles.append(y)
        contenu[nom] = {"entrees": torch.cat(entrees), "cibles": torch.cat(cibles),
                        "indices": torch.as_tensor(list(chargeur.dataset.indices))}
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(".tmp")
    torch.save(contenu, temporaire)
    temporaire.replace(chemin)


def cached_loaders(chargeurs, cfg: dict[str, Any], dossier: Path = DOSSIER_CACHE) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Les trois chargeurs (entraînement, validation, test), servis depuis le cache. Le cache est construit s'il
    n'existe pas, puis contrôlé à chaque chargement.

    Args:
        chargeurs: les trois chargeurs de cvnn (get_dataloaders), déjà vérifiés par check_pipeline.
        cfg: la configuration, pour l'empreinte.
        dossier: le dossier des caches.

    Returns:
        trois chargeurs en mémoire, de lots (B, 10, 1, 1) et (B, 5), avec la taille de lot des chargeurs de cvnn ;
        seul celui qui mélangeait ses lots chez cvnn (l'entraînement) les mélange.

    Raises:
        RuntimeError: si le cache ne correspond pas au découpage ou à la transformation actuels.
    """
    #Étape 1 : construire le cache s'il n'existe pas (une passe sur la pile).
    chemin = dossier / f"mlp_{fingerprint(cfg)}.pt"
    if not chemin.exists():
        logger.info("cache absent, construction (une passe sur la pile) : %s", chemin)
        build_cache(chargeurs, chemin)
    cache = torch.load(chemin)

    #Étape 2 : contrôler qu'il correspond au découpage et à la transformation actuels.
    tirage = np.random.default_rng()   # générateur propre : ne touche pas au générateur global de numpy
    for nom, chargeur in zip(PARTITIONS, chargeurs):
        partition = cache[nom]
        if partition["indices"].tolist() != list(chargeur.dataset.indices):
            raise RuntimeError(f"{chemin.name}, partition {nom} : les patchs du cache ne sont pas ceux du découpage "
                               f"actuel. Supprimer ce fichier pour le reconstruire.")
        for j in tirage.choice(len(partition["indices"]), size=min(N_CONTROLE, len(partition["indices"])), replace=False):
            entree, cible = chargeur.dataset[int(j)]
            if not (_identiques(entree, partition["entrees"][j]) and _identiques(cible, partition["cibles"][j])):
                raise RuntimeError(f"{chemin.name}, partition {nom}, patch {int(partition['indices'][j])} : le cache ne "
                                   f"correspond plus à la transformation (code d'injection modifié ?). Supprimer ce "
                                   f"fichier pour le reconstruire.")
    logger.info("cache chargé et contrôlé : %s", chemin)

    #Étape 3 : les chargeurs en mémoire, aux réglages des chargeurs de cvnn.
    return tuple(
        DataLoader(TensorDataset(cache[nom]["entrees"], cache[nom]["cibles"]), batch_size=chargeur.batch_size,
                   shuffle=isinstance(chargeur.sampler, RandomSampler), pin_memory=chargeur.pin_memory)
        for nom, chargeur in zip(PARTITIONS, chargeurs)
    )


def _identiques(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Égalité à l'arrondi du complex64 près, nan compris (cibles des patchs écartés)."""
    return a.shape == b.shape and bool(torch.isclose(torch.view_as_real(a), torch.view_as_real(b), rtol=1e-5,
                                                     atol=1e-7, equal_nan=True).all())
