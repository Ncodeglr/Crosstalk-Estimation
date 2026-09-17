"""Ordre des polarisations : la convention et sa traduction.

Le modèle physique de ce paquet suit **Fore et al. (2015)**, dont l'équation
(11) fixe le vecteur observé ::

    o = [O_HH, O_VH, O_HV, O_VV]^T          (POL_ORDER_ONERA, indices 1..4)

Le pipeline de données, lui, sert un autre ordre. `torchcvnn.datasets.SLCDataset`
trie les polarisations alphabétiquement (`sorted(slcs.keys())`) avant de les
empiler, ce qui donne ::

    [HH, HV, VH, VV]                        (POL_ORDER_DATALOADER)

Les canaux 2 et 3 sont donc permutés entre les deux conventions, et l'oubli de
cette permutation ne produit **aucune erreur visible** : permuter simultanément
les lignes et les colonnes 2 et 3 de la matrice de distorsion redonne une
matrice de la même famille, avec les paramètres renommés ::

    (u, v, w, z, alpha)  ->  (z, w, v, u, 1/alpha)

C'est exactement la différence entre les formes fermées de Quegan publiées par
Fore et al. (2015, équation 24) et par Babu et al. (2022, équation 13) : les
deux articles sont corrects, chacun dans son propre étiquetage. Ce module rend
la traduction explicite pour que ce choix cesse d'être implicite.

Corollaire à garder en tête lors des comparaisons bibliographiques : les
paramètres `u, v, w, z` publiés par Babu et al. correspondent à nos `z, w, v, u`.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

#: Ordre du modèle physique, celui de Fore et al. (2015), équation (11).
POL_ORDER_ONERA: tuple[str, ...] = ("HH", "VH", "HV", "VV")

#: Ordre servi par le pipeline (tri alphabétique de `SLCDataset`).
POL_ORDER_DATALOADER: tuple[str, ...] = ("HH", "HV", "VH", "VV")

#: Indices 0-based dans `POL_ORDER_ONERA` (les articles indexent à partir de 1).
IDX_HH, IDX_VH, IDX_HV, IDX_VV = 0, 1, 2, 3

#: Axe portant les polarisations dans un patch `(..., 4, H, W)`.
CHANNEL_AXIS: int = -3


def permutation(
    source: Sequence[str],
    target: Sequence[str] = POL_ORDER_ONERA,
) -> np.ndarray:
    """Indices réordonnant `source` en `target`.

    `source[permutation(source, target)[i]] == target[i]` pour tout `i`.

    Args:
        source: ordre des canaux en entrée, p. ex. `POL_ORDER_DATALOADER`.
        target: ordre voulu, `POL_ORDER_ONERA` par défaut.

    Returns:
        Tableau d'indices entiers de longueur `len(target)`.

    Raises:
        ValueError: si les deux séquences ne décrivent pas les mêmes canaux.
    """
    source = tuple(source)
    target = tuple(target)
    if sorted(source) != sorted(target):
        raise ValueError(
            f"ordres de polarisation incompatibles : {source} vs {target}"
        )
    if len(set(source)) != len(source):
        raise ValueError(f"polarisation dupliquée dans {source}")
    return np.array([source.index(pol) for pol in target], dtype=np.intp)


def reorder(
    patch: np.ndarray,
    source_order: Sequence[str] = POL_ORDER_DATALOADER,
    target_order: Sequence[str] = POL_ORDER_ONERA,
    channel_axis: int = CHANNEL_AXIS,
) -> np.ndarray:
    """Réordonne l'axe des polarisations d'un patch.

    Args:
        patch: tableau dont l'axe `channel_axis` porte les 4 polarisations,
            typiquement `(4, H, W)` ou `(B, 4, H, W)`.
        source_order: ordre des canaux de `patch`.
        target_order: ordre voulu.
        channel_axis: axe des polarisations, `-3` par défaut.

    Returns:
        Le patch réordonné. Si la permutation est l'identité, `patch` est rendu
        tel quel, sans copie ; sinon une copie est faite (indexation avancée).

    Raises:
        ValueError: si l'axe visé n'a pas la longueur attendue.
    """
    perm = permutation(source_order, target_order)
    if patch.shape[channel_axis] != len(perm):
        raise ValueError(
            f"l'axe {channel_axis} porte {patch.shape[channel_axis]} canaux, "
            f"{len(perm)} attendus pour l'ordre {tuple(target_order)}"
        )
    if np.array_equal(perm, np.arange(len(perm))):
        return patch
    return np.take(patch, perm, axis=channel_axis)


def to_onera_order(
    patch: np.ndarray,
    source_order: Sequence[str] = POL_ORDER_DATALOADER,
    channel_axis: int = CHANNEL_AXIS,
) -> np.ndarray:
    """Convertit un patch vers `POL_ORDER_ONERA`, l'ordre du modèle physique.

    C'est la frontière du paquet : tout ce qui entre passe par ici, et tout le
    code en aval peut supposer l'ordre `[HH, VH, HV, VV]`.
    """
    return reorder(patch, source_order, POL_ORDER_ONERA, channel_axis)
