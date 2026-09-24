"""Covariance polarimétrique du signal **observé**.

Toutes les méthodes statistiques d'estimation du crosstalk partent de la même
quantité (Babu et al., 2022 ; Fore et al., 2015, équation 16) ::

    C_ij = < O_i O_j* >,        i, j dans 1..4, ordre POL_ORDER = [HH, HV, VH, VV]

où `<.>` est une moyenne spatiale locale. C est hermitienne et semi-définie
positive. Elle est calculée sur `O`, le signal **observé**, et non sur le signal
vrai `S` : c'est bien le crosstalk qui s'y trouve, et c'est de là que les formes
fermées l'extraient. Le lien avec le modèle direct est ::

    C_O = D C_S D^H          (sans bruit)

Géométrie des patchs. Les `.slc` UAVSAR stockent 61 349 lignes d'azimut de
9 874 cases distance : dans un patch `(4, H, W)`, **l'axe -2 est l'azimut et
l'axe -1 la distance**. Une « ligne de distance » au sens de Babu et al. (2022)
et de Fore et al. (2015) est donc une *colonne* — distance fixée, azimut
variable.

Deux échelles de moyennage, à ne pas confondre :

`window="full"`
    Le patch entier est traité comme une unique fenêtre d'estimation. Un seul C.
    C'est le mode de la comparaison réseau contre baselines, où chaque méthode
    doit produire un jeu `(u, v, w, z, alpha)` par patch 64x64.

`window=(5, 5)` — `BABU_WINDOW`
    Fenêtre glissante, un C par position valide. Le champ rendu a la forme
    `(H', W', 4, 4)` : l'axe 0 est l'azimut, l'axe 1 la distance. Reproduire
    Babu et al. (2022, §3.3.1) consiste alors à estimer un `theta` par fenêtre,
    puis à **moyenner ces theta le long de l'axe 0** (azimut) pour obtenir un
    jeu par ligne de distance. La réduction porte sur les `theta` et non sur C —
    moyenner des rapports n'est pas le rapport des moyennes — c'est pourquoi
    elle appartient à l'estimateur et non à ce module.

Ce module attend ses entrées en `physics.POL_ORDER`, qui est déjà l'ordre
produit par le pipeline : aucune conversion n'a lieu nulle part dans le paquet.
"""

from __future__ import annotations

import logging
from typing import Literal, Union

import numpy as np

logger = logging.getLogger(__name__)

#: Fenêtre glissante de Babu et al. (2022, §3.3.1).
BABU_WINDOW: tuple[int, int] = (5, 5)

#: Spécification de fenêtre acceptée par `polarimetric_covariance`.
WindowSpec = Union[Literal["full"], tuple[int, int]]

#: En deçà, la covariance 4x4 est estimée sur trop peu d'échantillons.
MIN_SAMPLES: int = 16


def _check_patch(patch: np.ndarray) -> None:
    if patch.ndim < 3:
        raise ValueError(
            f"patch de forme {patch.shape} : (..., 4, H, W) attendu"
        )
    if patch.shape[-3] != 4:
        raise ValueError(
            f"l'axe -3 porte {patch.shape[-3]} canaux, 4 attendus "
            "(le patch doit être en ordre POL_ORDER)"
        )


def _warn_if_too_few(n_samples: int, contexte: str) -> None:
    if n_samples < MIN_SAMPLES:
        logger.warning(
            "%s : %d échantillons pour estimer une covariance 4x4 "
            "(%d recommandés) ; l'estimation sera bruitée",
            contexte,
            n_samples,
            MIN_SAMPLES,
        )


def polarimetric_covariance(
    patch: np.ndarray, window: WindowSpec = "full"
) -> np.ndarray:
    """Covariance polarimétrique du signal observé.

    Args:
        patch: `(..., 4, H, W)` complexe, en ordre `POL_ORDER`. Les
            dimensions de tête éventuelles (lot) sont conservées.
        window: `"full"` pour traiter le patch comme une unique fenêtre, ou un
            couple `(kh, kw)` pour une fenêtre glissante — `(5, 5)` reproduit
            Babu et al. (2022).

    Returns:
        En mode `"full"`, un tableau `(..., 4, 4)` complex128.
        En mode glissant, `(..., H - kh + 1, W - kw + 1, 4, 4)` complex128 ;
        l'axe `-4` est l'azimut, l'axe `-3` la distance.

    Raises:
        ValueError: forme de patch invalide, ou fenêtre plus grande que le patch.

    Note:
        Le calcul est mené en complex128 quel que soit le dtype d'entrée : C est
        un accumulateur, et la précision y importe davantage que la mémoire.
        Le mode glissant passe par une table de sommes cumulées, dont le coût
        est en `O(H W)` **indépendamment de la taille de fenêtre** : le 5x5 de
        Babu et al. (2022) et les fenêtres 201x201 de Fore et al. (2015) coûtent
        le même temps.
    """
    _check_patch(patch)
    data = patch.astype(np.complex128, copy=False)

    if window == "full":
        height, width = data.shape[-2:]
        _warn_if_too_few(height * width, "mode 'full'")
        # Repliés en (..., 4, N), les canaux se contractent par un produit
        # matriciel, donc par BLAS — trois fois plus rapide qu'un einsum.
        vecteurs = data.reshape(*data.shape[:-3], 4, height * width)
        return vecteurs @ vecteurs.conj().swapaxes(-1, -2) / (height * width)

    kh, kw = window
    if kh < 1 or kw < 1:
        raise ValueError(f"fenêtre {window} : dimensions strictement positives attendues")
    height, width = data.shape[-2:]
    if kh > height or kw > width:
        raise ValueError(
            f"fenêtre {window} plus grande que le patch ({height}, {width})"
        )
    _warn_if_too_few(kh * kw, f"fenêtre glissante {window}")

    # (..., 4, 4, H, W) : les 16 produits O_i O_j* avant moyennage.
    produits = np.einsum("...ihw,...jhw->...ijhw", data, data.conj())
    # Table de sommes cumulées : la somme sur une fenêtre se lit en quatre accès,
    # quelle que soit sa taille. Le zéro-padding initial évite un cas particulier
    # sur la première ligne et la première colonne.
    cumul = produits.cumsum(axis=-2).cumsum(axis=-1)
    cumul = np.pad(cumul, [(0, 0)] * (cumul.ndim - 2) + [(1, 0), (1, 0)])
    field = (
        cumul[..., kh:, kw:]
        - cumul[..., :-kh, kw:]
        - cumul[..., kh:, :-kw]
        + cumul[..., :-kh, :-kw]
    ) / (kh * kw)
    # (..., 4, 4, H', W') -> (..., H', W', 4, 4)
    return np.moveaxis(field, (-4, -3), (-2, -1))
