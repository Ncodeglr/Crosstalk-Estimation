"""Interface commune aux méthodes d'estimation du crosstalk.

Le principe directeur du paquet : le modèle physique (D, Sigma, correction) et
la covariance sont **communs** à toutes les méthodes ; seule l'estimation de
`theta = (u, v, w, z, alpha)` les distingue. Cette classe de base matérialise
cette frontière — une méthode nouvelle n'écrit que `estimate`, et hérite du
reste.

Ordre des canaux. Le modèle suit Babu et al. (2022), dont l'ordre`[HH, HV, VH, VV]` est exactement celui que sert le pipeline.
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar
from ..parameters import CrosstalkParameters
from ..physics import DEFAULT_TOL, apply_correction


class CrosstalkEstimator(ABC):
    """
    Base abstraite : une méthode d'estimation du crosstalk.

    Attributes:
        name: identifiant de la méthode, utilisé par le registre et les configurations. Toute sous-classe concrète doit le définir.
    """

    name: ClassVar[str]

    def __init__(self, tol: float = DEFAULT_TOL) -> None:
        """
        Args:
            tol: seuil de singularité de D, passé à `apply_correction`.
        """
        if not hasattr(type(self), "name"):
            raise TypeError(f"{type(self).__name__} doit définir un attribut de classe `name`")
        self.tol = float(tol)

    # à implémenter

    @abstractmethod
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """Estime un jeu `(u, v, w, z, alpha)` pour ce patch.

        C'est le seul point que les méthodes ont à écrire.

        Args:
            patch: `(4, H, W)` complexe, en ordre `POL_ORDER`.

        Returns:
            Les paramètres `(u, v, w, z, alpha)`, dans la convention `POL_ORDER`.
        """

    # hérité, commun

    def check_patch(self, patch: np.ndarray) -> None:
        """Valide la forme d'un patch : un seul patch `(4, H, W)`.

        Raises:
            ValueError: si le patch n'est pas de rang 3 ou n'a pas 4 canaux.
        """
        if patch.ndim != 3:
            raise ValueError(f"patch de forme {patch.shape} : (4, H, W) attendu. Un estimateur "
                "rend un jeu de paramètres par patch ; pour un lot, itérez."
            )
        if patch.shape[0] != 4:
            raise ValueError(f"patch à {patch.shape[0]} canaux, 4 attendus")

    def correct(self, patch: np.ndarray, params: CrosstalkParameters | None = None) -> np.ndarray:
        """Corrige un patch du crosstalk : `s_chapeau = Sigma o`.

        Args:
            patch: `(4, H, W)` complexe, en ordre `POL_ORDER`.
            params: paramètres à appliquer. S'ils sont omis, ils sont estimés
                sur ce patch même.

        Returns:
            Le patch corrigé, **dans l'ordre d'entrée** et de même dtype.

        Raises:
            DegenerateDistortionError: si D est singulière.
        """
        if params is None:
            params = self.estimate(patch)
        self.check_patch(patch)
        return apply_correction(patch, params, tol=self.tol)

    def estimate_and_correct(self, patch: np.ndarray) -> tuple[np.ndarray, CrosstalkParameters]:
        """Estime puis corrige, sans estimer deux fois.

        Returns:
            Le couple `(patch corrigé, paramètres)`.
        """
        params = self.estimate(patch)
        return self.correct(patch, params), params

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, tol={self.tol:g})")
