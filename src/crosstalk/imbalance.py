"""
Déséquilibre de canaux et biais de phase, à corriger **avant** le crosstalk.

C'est l'étape §3.2 de Babu et al. (2022), reprise de Fore et al. (2015, §IV-B).
Le modèle relie le signal observé `O` au signal `S'` débarrassé du déséquilibre
mais pas encore du crosstalk (Fore, équation 4 ; Babu, équation 7) :

    O_hh = A                     S'_hh
    O_hv = A  f g  e^{i phi_t}   S'_hv
    O_vh = A  f/g  e^{i phi_r}   S'_vh
    O_vv = A  f^2  e^{i(phi_t + phi_r)} S'_vv

avec `f` le déséquilibre co-polaire, `g` le déséquilibre cross-polaire et
`phi_t`, `phi_r` les biais de phase à l'émission et à la réception.

Deux familles de cibles se partagent l'estimation :

- les **trièdres**, de matrice `S = sqrt(sigma) I`, donnent `f` et
  `phi_t + phi_r` (Babu, équations 3 et 5) ;
- les **cibles distribuées**, réciproques en moyenne (`S_hv = S_vh`), donnent
  `g` et `phi_t - phi_r` (Babu, équations 4 et 6).

Le gain absolu `A` n'est pas estimé : le crosstalk mesuré aux réflecteurs est
un rapport de canaux, et les seuils de Quegan sont relatifs, si bien qu'un gain
global s'annule partout. L'estimer demanderait la SER théorique de chaque
réflecteur (Babu, équation 2) pour un effet nul sur la mesure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import POL_ORDER


@dataclass(frozen=True, slots=True)
class ChannelImbalance:
    """Jeu immuable `(f, g, phi_t, phi_r)` pour une acquisition.

    Attributes:
        f: déséquilibre co-polaire, réel positif proche de 1.
        g: déséquilibre cross-polaire, réel positif proche de 1.
        phi_t: biais de phase à l'émission, en radians.
        phi_r: biais de phase à la réception, en radians.
    """

    f: float
    g: float
    phi_t: float
    phi_r: float

    def __post_init__(self) -> None:
        for nom in ("f", "g"):
            valeur = getattr(self, nom)
            if not (np.isfinite(valeur) and valeur > 0):
                raise ValueError(f"{nom} = {valeur} : un déséquilibre est un réel strictement positif")
        for nom in ("phi_t", "phi_r"):
            if not np.isfinite(getattr(self, nom)):
                raise ValueError(f"{nom} = {getattr(self, nom)} : phase non finie")

    @classmethod
    def from_estimates(
        cls, f: float, phi_somme: float, g: float, phi_difference: float
    ) -> "ChannelImbalance":
        """Assemble le jeu à partir des deux estimations partielles.

        Les trièdres donnent `phi_t + phi_r`, les cibles distribuées
        `phi_t - phi_r` ; on en tire chaque phase par demi-somme et
        demi-différence.

        Ambiguïté : la somme et la différence ne sont connues qu'à `2 pi` près,
        donc `phi_t` et `phi_r` ne le sont qu'à `pi` près, et ensemble. Les deux
        solutions diffèrent par un signe commun sur `HV` et `VH`. Ni Fore ni
        Babu ne lèvent cette ambiguïté ; elle est sans effet sur le crosstalk
        aux réflecteurs, qui ne lit que des modules, ni sur la réciprocité
        `<S'_hv S'_vh*>`, où le signe se compense.
        """
        return cls(
            f=float(f),
            g=float(g),
            phi_t=float((phi_somme + phi_difference) / 2.0),
            phi_r=float((phi_somme - phi_difference) / 2.0),
        )

    def correction_matrix(self) -> np.ndarray:
        """Matrice diagonale `(4, 4)` qui ramène `O` à `S'`, dans `POL_ORDER`.

        C'est l'inverse de Babu, équation 7, au gain `A` près. Elle s'applique
        avec `physics.apply_matrix`, comme `Sigma`.
        """
        gains = {
            "HH": 1.0,
            "HV": self.f * self.g * np.exp(1j * self.phi_t),
            "VH": self.f / self.g * np.exp(1j * self.phi_r),
            "VV": self.f**2 * np.exp(1j * (self.phi_t + self.phi_r)),
        }
        return np.diag([1.0 / gains[pol] for pol in POL_ORDER]).astype(np.complex128)


def estimate_copol(hh_pics: np.ndarray, vv_pics: np.ndarray) -> tuple[float, float]:
    """`f` et `phi_t + phi_r` sur les pics des trièdres (Babu, équations 3 et 5).

    Chaque trièdre donne `f_i = (|O_vv|^2 / |O_hh|^2)^(1/4)` et
    `phi_i = arg(O_vv O_hh*)`. Babu ne dit pas comment il agrège les
    réflecteurs, et Fore ajuste ces quantités en fonction de l'incidence
    (équations 7 et 9). Nos réflecteurs couvrant une plage d'incidence trop
    étroite pour un ajustement, on retient une valeur par acquisition : la
    médiane des `f_i`, robuste à un réflecteur dégradé, et la moyenne
    circulaire des `phi_i`, qui ne souffre pas du repliement à `+-pi`.

    Args:
        hh_pics, vv_pics: valeurs complexes au pic de chaque trièdre, même forme.

    Returns:
        `(f, phi_t + phi_r)`, la phase en radians dans `]-pi, pi]`.

    Raises:
        ValueError: si aucun réflecteur n'est fourni ou si un pic HH est nul.
    """
    hh = np.asarray(hh_pics, dtype=np.complex128).ravel()
    vv = np.asarray(vv_pics, dtype=np.complex128).ravel()
    if hh.size == 0 or hh.shape != vv.shape:
        raise ValueError(f"pics de formes {hh.shape} et {vv.shape} : au moins un réflecteur attendu de chaque côté")
    if np.any(hh == 0):
        raise ValueError("pic HH nul : f n'est pas défini")
    f = float(np.median((np.abs(vv) ** 2 / np.abs(hh) ** 2) ** 0.25))
    phases = np.angle(vv * np.conj(hh))
    return f, float(np.angle(np.mean(np.exp(1j * phases))))


def crosspol_imbalance(
    puissance_hv: float, puissance_vh: float, correlation: complex
) -> tuple[float, float]:
    """`g` et `phi_t - phi_r` à partir des trois moyennes (Babu, équations 4 et 6).

    Séparée de `estimate_crosspol` pour qu'une moyenne accumulée par tranches,
    sur une image trop grande pour la mémoire, passe par la même formule.

    Args:
        puissance_hv: `<|O_hv|^2>`.
        puissance_vh: `<|O_vh|^2>`.
        correlation: `<O_hv O_vh*>`.

    Returns:
        `(g, phi_t - phi_r)`, la phase en radians dans `]-pi, pi]`.
    """
    if not (puissance_hv > 0 and puissance_vh > 0):
        raise ValueError(f"puissances cross-pol {puissance_hv}, {puissance_vh} : strictement positives attendues")
    g = float((puissance_hv / puissance_vh) ** 0.25)
    return g, float(np.angle(correlation))


def estimate_crosspol(hv: np.ndarray, vh: np.ndarray) -> tuple[float, float]:
    """`g` et `phi_t - phi_r` sur des cibles distribuées (Babu, équations 4 et 6).

    Repose sur la réciprocité `S_hv = S_vh` : toute différence de puissance ou
    de phase entre `O_hv` et `O_vh` est alors imputée au déséquilibre. La fuite
    des co-polaires dans les cross-polaires (crosstalk, ~ -24 dB) n'est pas
    retirée à ce stade ; c'est l'ordre de Babu, qui corrige le déséquilibre
    avant le crosstalk.

    Args:
        hv, vh: valeurs complexes des mêmes pixels, de forme quelconque.

    Returns:
        `(g, phi_t - phi_r)`.
    """
    hv = np.asarray(hv)
    vh = np.asarray(vh)
    if hv.size == 0 or hv.shape != vh.shape:
        raise ValueError(f"canaux de formes {hv.shape} et {vh.shape} : mêmes pixels attendus")
    return crosspol_imbalance(
        float(np.mean(np.abs(hv) ** 2)),
        float(np.mean(np.abs(vh) ** 2)),
        complex(np.mean(hv * np.conj(vh))),
    )
