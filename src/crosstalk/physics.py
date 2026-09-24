"""Modèle physique du crosstalk : D, Sigma, et leur application aux données.

Ce module est **méthode-agnostique** : il ne contient rien de Quegan,
d'Ainsworth ou d'un réseau. Toutes les méthodes d'estimation partagent le même
modèle direct et la même correction ; seule l'estimation de `theta` les
distingue.

Ordre des polarisations
-----------------------

`POL_ORDER`, défini ici, est l'ordre du vecteur observé chez **Babu et al.
(2022, équations 8-9)** ::

    o = [O_HH, O_HV, O_VH, O_VV]^T          (indices 1 à 4 dans les articles)

C'est exactement l'ordre que sert le pipeline : `torchcvnn.datasets.SLCDataset`
trie les polarisations alphabétiquement (`sorted(slcs.keys())`) avant de les
empiler, ce qui donne `[HH, HV, VH, VV]`. **Aucune permutation n'est donc
nécessaire**, et c'est la raison d'avoir retenu cette convention plutôt que
celle de Fore et al. (2015, équation 11), qui ordonne `[HH, VH, HV, VV]`.

Les deux conventions sont équivalentes, et non contradictoires : appliquées
chacune avec sa propre matrice de distorsion et ses propres formes fermées,
elles rendent les mêmes `(u, v, w, z, alpha)`, aux mêmes noms. Vérifié
numériquement. Ce qui produit un renommage silencieux — `(u,v,w,z,alpha)`
devenant `(z,w,v,u,1/alpha)` — c'est de les **mélanger**, par exemple en
appliquant les formes fermées de l'un à des données rangées selon l'autre.

Attention en lisant Babu et al. : son équation 10 reproduit la matrice D de Fore
telle quelle, sur un vecteur pourtant ordonné autrement, et ses équations 16 à
18 font de même pour `alpha`. Prises littéralement avec son équation 13, elles
sont incohérentes. Les matrices ci-dessous sont donc dérivées de celles de Fore
par permutation, puis vérifiées numériquement, plutôt que recopiées de son
article.

`synthetic.py` redéclare les indices `IDX_*` de ce même ordre ; leur cohérence
avec `POL_ORDER` est vérifiée par `tests/test_synthetic.py`.

Modèle direct
-------------

Données déjà étalonnées en gain absolu et en déséquilibre co-polaire
(`A = 1`, `k = 1/sqrt(alpha)`) ::

    o = D s + n,   avec o, s dans l'ordre POL_ORDER = [HH, HV, VH, VV]

        | 1      v/sa     w.sa     vw   |
    D = | z      1/sa     wz.sa    w    |,      sa = sqrt(alpha)
        | u      uv/sa    sa       v    |
        | uz     u/sa     z.sa     1    |

Correction (équation 15), inverse analytique exact de D ::

                  1        | 1        -v       -w       vw     |
    Sigma = --------------- | -z.sa    sa       wz.sa    -w.sa  |
            (uw-1)(vz-1)   | -u/sa    uv/sa    1/sa     -v/sa  |
                           | uz       -u       -z       1      |

La correction réalisable sur données est `s_chapeau = Sigma o` : le bruit `n`
étant inconnu, Sigma l'amplifie au lieu de le retrancher (équation 14). C'est
une limite du modèle, pas de cette implémentation, et elle doit être prise en
compte dans l'évaluation.

Branche de la racine carrée : `sqrt(alpha)` est pris sur la branche principale
de NumPy. Le choix de branche n'est pas neutre — remplacer `sa` par `-sa` donne
`D diag(1, -1, -1, 1)`, soit un changement de signe des canaux croisés — mais il
est sans conséquence tant que la même convention sert à distordre et à corriger,
ce que garantit le passage systématique par ce module.
"""

from __future__ import annotations
import logging
import numpy as np

from .exceptions import DegenerateDistortionError
from .parameters import CrosstalkParameters

logger = logging.getLogger(__name__)

#: Ordre du modèle physique et du pipeline — Babu et al. (2022), équations 8-9.
POL_ORDER: tuple[str, ...] = ("HH", "HV", "VH", "VV")

#: Axe portant les polarisations dans un patch `(..., 4, H, W)`.
CHANNEL_AXIS: int = -3

#: Seuil sous lequel `|(uw-1)(vz-1)| ` fait lever `DegenerateDistortionError`.
DEFAULT_TOL: float = 1e-8

#: Seuil d'alerte : en deçà, D est inversible mais le crosstalk n'est plus petit.
WARN_CONDITIONING: float = 0.5


def distortion_matrix(params: CrosstalkParameters) -> np.ndarray:
    """Matrice de distorsion D (Babu et al., 2022, équation 10).

    Args:
        params: le jeu `(u, v, w, z, alpha)`.

    Returns:
        Tableau `(4, 4)` complex128, dans l'ordre `POL_ORDER`.
    """
    u, v, w, z, alpha = params.to_vector()
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


def inverse_distortion_matrix(
    params: CrosstalkParameters, tol: float = DEFAULT_TOL
) -> np.ndarray:
    """Matrice de calibration Sigma (Babu et al., 2022, équation 12).

    Forme analytique exacte de `D^-1`, et non une approximation au premier
    ordre : `D @ Sigma == I` à la précision machine.

    Args:
        params: le jeu `(u, v, w, z, alpha)`.
        tol: seuil sur `|(uw-1)(vz-1)|` sous lequel D est déclarée singulière.

    Returns:
        Tableau `(4, 4)` complex128, dans l'ordre `POL_ORDER`.

    Raises:
        DegenerateDistortionError: si le déterminant réduit est trop petit.
    """
    u, v, w, z, alpha = params.to_vector()
    sa = np.sqrt(alpha)
    determinant = (u * w - 1.0) * (v * z - 1.0)

    if abs(determinant) < tol:
        raise DegenerateDistortionError(determinant, tol)
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


def _apply_matrix(
    matrix: np.ndarray, data: np.ndarray, channel_axis: int
) -> np.ndarray:
    """Contracte `matrix` avec l'axe des polarisations de `data`, sans boucle.

    Le calcul se fait en complex128 ; le dtype d'entrée est restitué en sortie
    pour que des patchs complex64 le restent.
    """
    if data.shape[channel_axis] != 4:
        raise ValueError(
            f"l'axe {channel_axis} porte {data.shape[channel_axis]} canaux, 4 attendus"
        )
    moved = np.moveaxis(data, channel_axis, -1)
    result = np.einsum("ij,...j->...i", matrix, moved.astype(np.complex128))
    result = np.moveaxis(result, -1, channel_axis)
    if np.iscomplexobj(data):
        return result.astype(data.dtype, copy=False)
    return result


def apply_distortion(
    scattering: np.ndarray,
    params: CrosstalkParameters,
    channel_axis: int = CHANNEL_AXIS,
) -> np.ndarray:
    """Applique le modèle direct sans bruit : `o = D s`.

    Sert à simuler des observations à `theta` connu — c'est le seul moyen de
    disposer d'une vérité terrain, les données réelles n'en ayant pas.

    Args:
        scattering: patch `(..., 4, H, W)` du signal vrai, ordre
            `POL_ORDER`.
        params: le jeu `(u, v, w, z, alpha)` à appliquer.
        channel_axis: axe des polarisations.

    Returns:
        Le patch distordu, de même forme et même dtype.
    """
    return _apply_matrix(distortion_matrix(params), scattering, channel_axis)


def apply_correction(
    observed: np.ndarray,
    params: CrosstalkParameters,
    channel_axis: int = CHANNEL_AXIS,
    tol: float = DEFAULT_TOL,
) -> np.ndarray:
    """Corrige un patch observé : `s_chapeau = Sigma o` (équation 14).

    Le bruit n'étant pas connu, il n'est pas retranché et se trouve amplifié par
    Sigma. La correction est donc exacte sur données sans bruit uniquement.

    Args:
        observed: patch `(..., 4, H, W)` du signal observé, ordre
            `POL_ORDER`.
        params: le jeu estimé.
        channel_axis: axe des polarisations.
        tol: seuil de singularité passé à `inverse_distortion_matrix`.

    Returns:
        Le patch corrigé, de même forme et même dtype.

    Raises:
        DegenerateDistortionError: si D est singulière.
    """
    sigma = inverse_distortion_matrix(params, tol=tol)
    return _apply_matrix(sigma, observed, channel_axis)
