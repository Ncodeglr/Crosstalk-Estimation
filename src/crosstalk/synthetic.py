"""Génération de signal vrai conforme aux hypothèses de Quegan.

Les données réelles n'ont pas de vérité terrain : le seul moyen de vérifier
qu'un estimateur retrouve `theta` est de partir d'un `S` synthétique, de lui
appliquer une distorsion connue, et de comparer.

Encore faut-il que ce `S` respecte les hypothèses de la méthode, sans quoi
l'échec du test ne prouverait rien. Quegan suppose la covariance vraie
(Fore et al., 2015, équation 17), dans l'ordre `POL_ORDER_ONERA` ::

              | sigma_hh    0       0      sigma_hhvv |
        C_Q = |    0      beta    beta         0      |
              |    0      beta    beta         0      |
              | sigma_hhvv*  0       0      sigma_vv  |

Deux propriétés s'y lisent :

- **symétrie de réflexion** : les blocs co-pol/cross-pol sont nuls, c'est-à-dire
  `<S_HH S_VH*> = <S_VV S_HV*> = 0`. Sur les données observées `O`, ces termes
  ne sont pas nuls — ils valent le crosstalk, et c'est précisément ce que les
  formes fermées y lisent ;
- **réciprocité stricte** : le bloc croisé est constant égal à `beta`, ce qui
  impose `S_HV = S_VH` exactement, et non seulement en moyenne.

`C_Q` est donc de rang 3, ce qui interdit une factorisation de Cholesky : le
tirage est construit explicitement, bloc par bloc.
"""

from __future__ import annotations

import numpy as np

from .conventions import IDX_HH, IDX_HV, IDX_VH, IDX_VV


def quegan_covariance(
    sigma_hh: float = 1.0,
    sigma_vv: float = 1.0,
    rho_copol: complex = 0.6,
    beta: float = 0.05,
) -> np.ndarray:
    """Covariance vraie supposée par Quegan (Fore et al., 2015, équation 17).

    Args:
        sigma_hh: puissance du canal HH, `<|S_HH|^2>`.
        sigma_vv: puissance du canal VV.
        rho_copol: corrélation co-polaire complexe entre HH et VV, de module
            inférieur à 1. Quegan ne fait aucune hypothèse dessus.
        beta: puissance des canaux croisés, `<|S_HV|^2> = <|S_VH|^2>`.

    Returns:
        `(4, 4)` complex128, dans l'ordre `POL_ORDER_ONERA`.

    Raises:
        ValueError: si `|rho_copol| >= 1` ou si une puissance est négative.
    """
    if abs(rho_copol) >= 1.0:
        raise ValueError(f"|rho_copol| = {abs(rho_copol):.3f} doit être < 1")
    if min(sigma_hh, sigma_vv, beta) < 0.0:
        raise ValueError("les puissances doivent être positives")

    cov = np.zeros((4, 4), dtype=np.complex128)
    sigma_hhvv = rho_copol * np.sqrt(sigma_hh * sigma_vv)
    cov[IDX_HH, IDX_HH] = sigma_hh
    cov[IDX_VV, IDX_VV] = sigma_vv
    cov[IDX_HH, IDX_VV] = sigma_hhvv
    cov[IDX_VV, IDX_HH] = np.conj(sigma_hhvv)
    for i in (IDX_VH, IDX_HV):
        for j in (IDX_VH, IDX_HV):
            cov[i, j] = beta
    return cov


def reflection_symmetric_scattering(
    shape: tuple[int, ...],
    sigma_hh: float = 1.0,
    sigma_vv: float = 1.0,
    rho_copol: complex = 0.6,
    beta: float = 0.05,
    rng: np.random.Generator | None = None,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Tire un signal vrai gaussien circulaire de covariance `quegan_covariance`.

    Args:
        shape: forme spatiale, p. ex. `(64, 64)` ou `(1_000_000,)`.
        sigma_hh, sigma_vv, rho_copol, beta: voir `quegan_covariance`.
        rng: générateur ; un générateur non graine est créé si omis.
        dtype: type complexe de sortie.

    Returns:
        `(4, *shape)` dans l'ordre `POL_ORDER_ONERA`, avec `S_HV = S_VH`
        exactement et une corrélation co-pol/cross-pol nulle en espérance.
    """
    if rng is None:
        rng = np.random.default_rng()

    def circulaire() -> np.ndarray:
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)

    # Bloc co-polaire : (HH, VV) corrélés par rho_copol. La conjugaison est
    # imposée par la convention C_ij = <O_i O_j*> : pour obtenir
    # <S_HH S_VV*> = rho_copol, il faut construire VV avec conj(rho_copol).
    hh = circulaire()
    vv = np.conj(rho_copol) * hh + np.sqrt(1.0 - abs(rho_copol) ** 2) * circulaire()

    # Bloc croisé : un unique tirage, partagé par VH et HV (réciprocité stricte).
    croise = circulaire()

    signal = np.empty((4, *shape), dtype=np.complex128)
    signal[IDX_HH] = np.sqrt(sigma_hh) * hh
    signal[IDX_VV] = np.sqrt(sigma_vv) * vv
    signal[IDX_VH] = np.sqrt(beta) * croise
    signal[IDX_HV] = signal[IDX_VH]
    return signal.astype(dtype, copy=False)
