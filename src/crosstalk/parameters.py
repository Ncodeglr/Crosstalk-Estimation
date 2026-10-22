"""
Les cinq paramètres de distorsion polarimétrique.theta = (u, v, w, z, alpha)
Ce sont les seules inconnues du modèle direct `s' = D s + n` (le S' de Babu, éq. 8) 
Les data sont Polarimetrically Calibrated = Yes donc : gain absolu et en déséquilibre co-polaire (`A = 1`, `k = 1/sqrt(alpha) Fore et al., 2015, équations 11 à 13.
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
    """
    @dataclass permet de rendre le jeu de paramètres immuable et plus léger (slots), elles sont dédiées au stockage de données, avec très peu de code.
    frozen = True : Permet l'immuabilité et la sécurité
    slots = True : Permet de réduire la mémoire et d'accélérer l'accès aux attributs, mais empêche l'ajout d'attributs dynamiques.
    
    Jeu immuable (u, v, w, z, alpha) pour un patch, ou pour la bande de distance d'un réflecteur.

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
        # post_init est appelé après l'initialisation de l'objet, il permet de vérifier les valeurs des attributs et de les transformer si nécessaire.
        # fields(self) renvoie une liste des champs de la dataclass, on peut donc itérer sur eux pour vérifier les valeurs
        for field in fields(self):
            value = complex(getattr(self, field.name)) # Convertir en complexe 
            object.__setattr__(self, field.name, value) # Mettre à jour l'attribut avec la valeur convertie en complexe
        if self.alpha == 0:
            raise ValueError("alpha nul : D fait intervenir 1/sqrt(alpha) et n'est pas définie")

    #--------------------------------Méthodes : Conversion en vecteurs----------------------------------

    @classmethod #Signifie que la méthode est liée à la classe et non à une instance de la classe. Elle peut être appelée sur la classe elle-même, sans avoir besoin de créer une instance de la classe.
    def from_vector(cls, theta: Iterable[complex]) -> "CrosstalkParameters":
        """
        Permet de créer une instance de la CrosstalkParameters à partir d'un vecteur ordonné selon `PARAMETER_NAMES`.
        """
        values = np.asarray(list(theta))
        if values.shape != (5,):
            raise ValueError(f"vecteur de forme {values.shape}, (5,) attendu")
        return cls(*(complex(value) for value in values))

    def to_vector(self) -> np.ndarray:
        """Vecteur (5,) complex128 ordonné selon `PARAMETER_NAMES`."""
        return np.array([getattr(self, name) for name in PARAMETER_NAMES], dtype=np.complex128)

    #------------------------------Méthodes : Lectures physiques----------------------------------------- 

    def crosstalk_db(self) -> dict[str, float]:
        """
        Module des quatre termes de crosstalk en dB : `20 log10 |.|`.

        C'est l'échelle des figures 7 à 9 de Fore et des tableaux de Babu : -20 dB environ avant calibration.
        Un terme exactement nul rend `-inf`, qui est sa valeur en dB : c'est le cas d'un jeu neutre, pas une erreur.
        """
        with np.errstate(divide="ignore"):
            #np.errstate(divide="ignore") permet de supprimer les avertissements de division par zéro, ce qui est utile ici car on peut avoir des termes de crosstalk nuls
            return {name: float(20.0 * np.log10(abs(getattr(self, name)))) for name in CROSSTALK_NAMES}

    def max_crosstalk_db(self) -> float:
        """Le pire des quatre termes, en dB."""
        return max(self.crosstalk_db().values())

    def alpha_db(self) -> float:
        """Module du déséquilibre croisé en dB : `20 log10 |alpha|`."""
        return 20.0 * np.log10(abs(self.alpha))

    def alpha_phase_deg(self) -> float:
        """Phase du déséquilibre croisé en degrés, dans `]-180, 180]`."""
        return float(np.angle(self.alpha, deg=True))

    def __str__(self) -> str:
        terms = ", ".join(f"{name}={value:+.1f} dB" for name, value in self.crosstalk_db().items())
        return (
            f"CrosstalkParameters({terms}, "
            f"|alpha|={self.alpha_db():+.2f} dB, "
            f"arg(alpha)={self.alpha_phase_deg():+.1f}°)"
        )

