"""Les cinq paramètres de distorsion polarimétrique.

`theta = (u, v, w, z, alpha)` : quatre termes de crosstalk complexes et le
déséquilibre de canal de polarisation croisée. Ce sont les seules inconnues du
modèle direct `o = D s + n` une fois les données étalonnées en gain absolu et en
déséquilibre co-polaire (`A = 1`, `k = 1/sqrt(alpha)` ; Fore et al., 2015,
équations 11 à 13).

Le champ `alpha` est **complexe**, parce que le modèle direct l'autorise : Fore
et al. (2015, figures 7 et 9) mesurent `arg(alpha) ~ 25-30°` sur des données
UAVSAR étalonnées en radiométrie et en phase mais non corrigées du crosstalk.
Les formes fermées de ce projet suivent en revanche l'équation 15 de Babu et al.
(2022), qui s'arrête au module : `QueganEstimator` rend donc un `alpha` réel
positif. Le type reste complexe pour que la distorsion et la correction de
`physics` restent générales, et pour que les jeux de test puissent porter une
phase.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Iterable
import numpy as np

#: Ordre canonique des paramètres dans les vecteurs `(5,)`.
PARAMETER_NAMES: tuple[str, ...] = ("u", "v", "w", "z", "alpha")

#: Les quatre termes de crosstalk, sans `alpha`.
CROSSTALK_NAMES: tuple[str, ...] = ("u", "v", "w", "z")


@dataclass(frozen=True, slots=True)
class CrosstalkParameters:
    """Jeu immuable `(u, v, w, z, alpha)` pour un patch ou une ligne de distance.

    Attributes:
        u, v, w, z: termes de crosstalk complexes, petits devant 1 (~ -20 dB sur UAVSAR bande L).
        alpha: déséquilibre de canal de polarisation croisée, complexe de module proche de 1.
    """

    u: complex
    v: complex
    w: complex
    z: complex
    alpha: complex

    def __post_init__(self) -> None:
        for field in fields(self):
            value = complex(getattr(self, field.name))
            object.__setattr__(self, field.name, value)
        if self.alpha == 0:
            raise ValueError("alpha nul : D fait intervenir 1/sqrt(alpha) et n'est pas définie")

    # ------------------------------------------------------------------ vecteurs

    @classmethod
    def from_vector(cls, theta: Iterable[complex]) -> "CrosstalkParameters":
        """Construit depuis un vecteur ordonné selon `PARAMETER_NAMES`."""
        values = np.asarray(list(theta))
        if values.shape != (5,):
            raise ValueError(f"vecteur de forme {values.shape}, (5,) attendu")
        return cls(*(complex(value) for value in values))

    def to_vector(self) -> np.ndarray:
        """Vecteur `(5,)` complex128 ordonné selon `PARAMETER_NAMES`."""
        return np.array(
            [getattr(self, name) for name in PARAMETER_NAMES], dtype=np.complex128
        )

    # ------------------------------------------------------- lectures physiques

    def crosstalk_db(self) -> dict[str, float]:
        """Module des quatre termes de crosstalk en dB : `20 log10 |.|`.

        C'est l'échelle des figures 7 à 9 de Fore et al. (2015) et des tableaux
        de Babu et al. (2022) : -20 dB environ avant calibration.

        Un terme exactement nul rend `-inf`, qui est sa valeur en dB : c'est le
        cas d'un jeu neutre, pas une erreur, et l'avertissement NumPy associé
        est donc réduit au silence.
        """
        with np.errstate(divide="ignore"):
            return {
                name: float(20.0 * np.log10(abs(getattr(self, name))))
                for name in CROSSTALK_NAMES
            }

    def max_crosstalk_db(self) -> float:
        """Le pire des quatre termes, en dB. Résumé scalaire d'un jeu."""
        return max(self.crosstalk_db().values())

    def alpha_db(self) -> float:
        """Module du déséquilibre croisé en dB : `20 log10 |alpha|`."""
        return 20.0 * np.log10(abs(self.alpha))

    def alpha_phase_deg(self) -> float:
        """Phase du déséquilibre croisé en degrés, dans `]-180, 180]`."""
        return float(np.angle(self.alpha, deg=True))

    def __str__(self) -> str:
        terms = ", ".join(
            f"{name}={value:+.1f} dB" for name, value in self.crosstalk_db().items()
        )
        return (
            f"CrosstalkParameters({terms}, "
            f"|alpha|={self.alpha_db():+.2f} dB, "
            f"arg(alpha)={self.alpha_phase_deg():+.1f}°)"
        )
