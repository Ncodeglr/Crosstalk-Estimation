"""
Covariance polarimétrique du signal **observé**.

Toutes les méthodes statistiques d'estimation du crosstalk partent de la même quantité (Babu et al., 2022 ; Fore et al., 2015, équation 16):

    C_ij = < S'_i S'_j* >,      i, j dans 1..4, = [HH, HV, VH, VV]

Elle est calculée sur s', le signal **observé** — le S' de Babu (éq. 8), étalonné en gain et en phase mais pas en crosstalk.
"""

from __future__ import annotations
import numpy as np
from .matrix import POL_ORDER

def _check_patch(patch: np.ndarray) -> None:
    if patch.ndim < 3:
        raise ValueError(f"patch de forme {patch.shape} : (..., 4, H, W) attendu")
    if patch.shape[-3] != len(POL_ORDER):
        raise ValueError(f"l'axe -3 porte {patch.shape[-3]} canaux, {len(POL_ORDER)} attendus "
            f"(le patch doit être en ordre {POL_ORDER})")


def polarimetric_covariance(patch: np.ndarray) -> np.ndarray:
    """
    Covariance polarimétrique du signal observé, moyennée sur tout le patch.

    Args: patch(..., 4, H, W) complexe. Les dimensions de tête éventuelles (lot) sont conservées.
    Returns: (..., 4, 4) complex128 : (1/N) somme_n k_n k_n^H, où k_n est le vecteur des 4 canaux du pixel n et N = H W.
    Raises: ValueError: forme de patch invalide.
    """
    _check_patch(patch)
    height, width = patch.shape[-2:]
    data = patch.astype(np.complex128, copy=False) # complex128 pour la précision, sans copie si c'est déjà le cas
    vecteurs = data.reshape(*data.shape[:-3], 4, height * width) # [:-3] = tout sauf les trois derniers axes
    return vecteurs @ vecteurs.conj().swapaxes(-1, -2) / (height * width)
