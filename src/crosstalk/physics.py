from __future__ import annotations
import logging
import numpy as np
from .exceptions import SingularDmatrixError
from .parameters import CrosstalkParameters

logger = logging.getLogger(__name__)

POL_ORDER: tuple[str, ...] = ("HH", "HV", "VH", "VV") # Ordre du modèle physique et du pipeline — Babu équations 8-9
CHANNEL_AXIS: int = -3 # Axe portant les polarisations dans un patch `(..., 4, H, W)`
DEFAULT_TOL: float = 1e-8 # Seuil sous lequel `|(uw-1)(vz-1)| ` fait lever `SingularDmatrixError`
WARN_CONDITIONING: float = 0.5 # Seuil d'alerte : en deçà, D est inversible mais le crosstalk n'est plus petit.


def distortion_matrix(params: CrosstalkParameters) -> np.ndarray:
    """
    Matrice de distorsion D, en ordre `POL_ORDER`.
    Ses coefficients sont ceux de l'équation 13 de Fore et al. (2015), **déjà réindexés** :
    lignes et colonnes 2 et 3 ont été échangées une fois pour toutes à l'écriture, puisque Fore ordonne [HH, VH, HV, VV].
    Aucune permutation n'a donc lieu à l'exécution — ni ici, ni ailleurs dans le paquet.
   
    Args: params: le jeu `(u, v, w, z, alpha)`.
    
    Returns: Tableau `(4, 4)` complex128, dans l'ordre `POL_ORDER`.
    """
    u, v, w, z, alpha = params.to_vector() # Utilise la méthode to_vector de parameters.py pour obtenir les valeurs des paramètres du vecteur
    sa = np.sqrt(alpha)
    return np.array(
        [
            [1.0, v / sa, w * sa, v * w],
            [z, 1.0 / sa, w * z * sa, w],
            [u, u * v / sa, sa, v],
            [u * z, u / sa, z * sa, 1.0],
        ],
        dtype=np.complex128,
    )


def inverse_distortion_matrix(params: CrosstalkParameters, tol: float = DEFAULT_TOL) -> np.ndarray:
    """
    Matrice de calibration Sigma, en ordre `POL_ORDER`.
    Ses coefficients sont ceux de l'équation 15 de Fore, déjà réindexés de la même façon que ceux de distortion_matrix.

    Forme analytique exacte de `D^-1`, et non une approximation au premier ordre :
        `D @ Sigma == I` à la précision machine. L'équation 12 de Babu et al.
        porte la même faute de transcription que son équation 10 et n'inverse
        ni l'une ni l'autre de ses matrices de distorsion ; celle de Fore est exacte.

    Args:
        params: le jeu `(u, v, w, z, alpha)`.
        tol: seuil sur `|(uw-1)(vz-1)|` sous lequel D est déclarée singulière.

    Returns: Sigma, Tableau `(4, 4)` complex128, dans l'ordre `POL_ORDER`

    Raises: SingularDmatrixError: si le déterminant réduit est trop petit.
    """
    u, v, w, z, alpha = params.to_vector()
    sa = np.sqrt(alpha)
    determinant = (u * w - 1.0) * (v * z - 1.0)

    if abs(determinant) < tol:
        raise SingularDmatrixError(determinant, tol)
    if abs(determinant) < WARN_CONDITIONING:
        logger.warning(
            "conditionnement dégradé : |(uw-1)(vz-1)| = %.3e ; le crosstalk "
            "n'est plus petit devant 1, l'hypothèse du modèle est douteuse",
            abs(determinant),
        )

    return (
        np.array(
            [
                [1.0, -v, -w, v * w],
                [-z * sa, sa, w * z * sa, -w * sa],
                [-u / sa, u * v / sa, 1.0 / sa, -v / sa],
                [u * z, -u, -z, 1.0],
            ],
            dtype=np.complex128,
        )
        / determinant
    )


def apply_matrix(matrix: np.ndarray, data: np.ndarray, channel_axis: int) -> np.ndarray:
    """
    Contracte matrix avec l'axe des polarisations de data, sans boucle.
    Moteur commun des fonctions apply_distortion et apply_correction

    Args:
        matrix: (4, 4) complexe, en ordre POL_ORDER :  
            -Soit D, rendue par distortion_matrix pour distordre 
            -Soit Sigma, rendue par inverse_distortion_matrix pour corriger.
        
        data: les données polarimétriques, de forme quelconque pourvu que
            `channel_axis` porte les 4 polarisations en ordre `POL_ORDER`.
            Dans ce projet c'est soit le signal vrai `s`, synthétique, que
            l'on distord, soit le patch observé `o` du dataloader, `complex64`
            et de forme `(4, H, W)` ou `(lot, 4, H, W)`, que l'on corrige.
        
        channel_axis: position de l'axe des polarisations, `-3` pour un patch.

    Returns:
        Le résultat `matrix @ data`, appliqué au vecteur des 4 polarisations de
        chaque pixel. Même forme et même dtype que `data` — le calcul se fait
        en complex128 pour la précision, mais un patch `complex64` le reste.

    Raises: ValueError: si `channel_axis` ne porte pas 4 canaux.
    """
    if data.shape[channel_axis] != 4:
        raise ValueError(f"l'axe {channel_axis} porte {data.shape[channel_axis]} canaux, 4 attendus")
    moved = np.moveaxis(data, channel_axis, -1)
    result = np.einsum("ij,...j->...i", matrix, moved.astype(np.complex128))
    result = np.moveaxis(result, -1, channel_axis)
    if np.iscomplexobj(data):
        return result.astype(data.dtype, copy=False)
    return result


def apply_distortion(sVector_scattering: np.ndarray,params: CrosstalkParameters,channel_axis: int = CHANNEL_AXIS,) -> np.ndarray:
    """
    Applique le modèle direct sans bruit : `o = D s`.
    Sert à simuler des observations à theta connu — c'est le seul moyen de disposer d'une vérité terrain, les données réelles n'en ayant pas.

    Args:
        sVector_scattering: patch `(..., 4, H, W)` du signal vrai, ordre POL_ORDER.
        params: le jeu `(u, v, w, z, alpha)` à appliquer.
        channel_axis: axe des polarisations.

    Returns: Le patch distordu, de même forme et même dtype.
    """
    return apply_matrix(distortion_matrix(params), sVector_scattering, channel_axis)


def apply_correction(observed: np.ndarray,params: CrosstalkParameters,channel_axis: int = CHANNEL_AXIS,tol: float = DEFAULT_TOL,) -> np.ndarray:
    """
    Corrige un patch observé : s_corrigé = Sigma @ o (Fore équation 14).

    Le bruit n'étant pas connu, il n'est pas retranché et se trouve amplifié par Sigma. 
    La correction est donc exacte sur données sans bruit uniquement.

    Args:
        observed: patch (..., 4, H, W) du signal observé, ordre POL_ORDER.
        params: le jeu estimé.
        channel_axis: axe des polarisations.
        tol: seuil de singularité passé à `inverse_distortion_matrix`.

    Returns: Le patch corrigé, de même forme et même dtype.

    Raises: SingularDmatrixError: si D est singulière.
    """
    sigma = inverse_distortion_matrix(params, tol=tol)
    return apply_matrix(sigma, observed, channel_axis)
