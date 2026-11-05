"""
Covariance polarimétrique du signal **observé**.

Toutes les méthodes statistiques d'estimation du crosstalk partent de la même quantité (Babu et al., 2022 ; Fore et al., 2015, équation 16):

    C_ij = < S'_i S'_j* >,      i, j dans 1..4, = [HH, HV, VH, VV]

Elle est calculée sur s', le signal **observé** — le S' de Babu (éq. 8), étalonné en gain et en phase mais pas en crosstalk.
"""

from __future__ import annotations
import numpy as np
from .matrix import POL_ORDER

#: Les 10 entrées distinctes d'une covariance 4 x 4 hermitienne, en indices POL_ORDER : (0, 0), (0, 1), ..., (3, 3).
HERMITIAN_ENTRIES: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(len(POL_ORDER)) for j in range(i, len(POL_ORDER))
)

def check_patch(patch: np.ndarray, *, lot: bool = False) -> None:
    """
    Valide un patch en ordre POL_ORDER : un seul patch (4, H, W) ; ou, si lot, un patch ou un lot (..., 4, H, W).

    Une seule définition pour tout le projet : la covariance accepte un lot, les estimateurs et la transformation du
    banc d'essai n'acceptent qu'un patch.

    Raises:
        TypeError: si patch n'est pas un np.ndarray.
        ValueError: si la forme ne convient pas, ou si l'axe des polarisations ne porte pas 4 canaux.
    """
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"patch de type {type(patch).__name__} : np.ndarray attendu")
    if lot and patch.ndim < 3:
        raise ValueError(f"patch de forme {patch.shape} : (..., 4, H, W) attendu")
    if not lot and patch.ndim != 3:
        raise ValueError(f"patch de forme {patch.shape} : (4, H, W) attendu. Un seul patch à la fois : "
                         "pour un lot, itérez.")
    if patch.shape[-3] != len(POL_ORDER):
        raise ValueError(f"l'axe -3 porte {patch.shape[-3]} canaux, {len(POL_ORDER)} attendus "
            f"(le patch doit être en ordre {POL_ORDER})")


def check_covariance(cov: np.ndarray, *, lot: bool = False) -> None:
    """
    Valide une covariance en ordre POL_ORDER : une seule matrice (4, 4) ; ou, si lot, une matrice ou un lot (..., 4, 4).
    Le pendant de check_patch pour les covariances, et la seule définition du projet.

    Raises: ValueError: si la forme ne convient pas.
    """
    n = len(POL_ORDER)
    if lot and (cov.ndim < 2 or cov.shape[-2:] != (n, n)):
        raise ValueError(f"covariance de forme {cov.shape} : (..., {n}, {n}) attendu")
    if not lot and cov.shape != (n, n):
        raise ValueError(f"covariance de forme {cov.shape} : ({n}, {n}) attendu")


def polarimetric_covariance(patch: np.ndarray) -> np.ndarray:
    """
    Covariance polarimétrique du signal observé, moyennée sur tout le patch.

    Args: patch(..., 4, H, W) complexe. Les dimensions de tête éventuelles (lot) sont conservées.
    Returns: (..., 4, 4) complex128 : (1/N) somme_n k_n k_n^H, où k_n est le vecteur des 4 canaux du pixel n et N = H W.
    Raises: TypeError, ValueError : patch invalide (voir check_patch).
    """
    check_patch(patch, lot=True)
    height, width = patch.shape[-2:]
    data = patch.astype(np.complex128, copy=False) # complex128 pour la précision, sans copie si c'est déjà le cas
    vecteurs = data.reshape(*data.shape[:-3], 4, height * width) # [:-3] = tout sauf les trois derniers axes
    return vecteurs @ vecteurs.conj().swapaxes(-1, -2) / (height * width)


def normalize_by_trace(cov: np.ndarray) -> np.ndarray:
    """
    La covariance divisée par sa trace, la puissance totale des 4 canaux : la trace de la covariance rendue vaut 1.

    Le crosstalk ne dépend pas de l'échelle du patch : multiplier C par un réel positif ne change ni les contraintes
    de Quegan, ni theta. La normalisation met un patch sombre et un patch brillant à la même échelle.

    Args: cov: (..., 4, 4). Les dimensions de tête éventuelles (lot) sont conservées.
    Returns: (..., 4, 4) complex128. Une matrice dont la trace est nulle, négative ou non finie (patch vide ou
        invalide) est rendue entièrement en nan : l'échec reste visible, comme pour QueganEstimator.solve.
    Raises: ValueError: si cov n'est pas de forme (..., 4, 4) (voir check_covariance).
    """
    cov = np.asarray(cov, dtype=np.complex128)
    check_covariance(cov, lot=True)
    trace = np.trace(cov, axis1=-2, axis2=-1).real
    trace = np.where(np.isfinite(trace) & (trace > 0.0), trace, np.nan)
    # Diviser par nan est voulu pour une trace invalide : NumPy l'annonce par un avertissement « invalid value », inutile ici.
    with np.errstate(invalid="ignore"):
        return cov / trace[..., np.newaxis, np.newaxis]


def covariance_entries(cov: np.ndarray) -> np.ndarray:
    """
    Les 10 entrées distinctes d'une covariance, dans l'ordre de HERMITIAN_ENTRIES.

    Une covariance est hermitienne (C_ji = C_ij*) : ses 4 puissances, sur la diagonale et réelles, et ses 6 entrées
    au-dessus de la diagonale la déterminent entièrement. C'est l'entrée du réseau du banc d'essai.

    Args: cov: (..., 4, 4). Les dimensions de tête éventuelles (lot) sont conservées.
    Returns: (..., 10), du même type que cov.
    Raises: ValueError: si cov n'est pas de forme (..., 4, 4) (voir check_covariance).
    """
    cov = np.asarray(cov)
    check_covariance(cov, lot=True)
    return np.stack([cov[..., i, j] for i, j in HERMITIAN_ENTRIES], axis=-1)
