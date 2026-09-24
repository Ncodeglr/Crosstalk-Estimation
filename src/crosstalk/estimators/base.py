"""
Interface commune aux méthodes d'estimation du crosstalk.
Le principe directeur du paquet : le modèle physique (D, Sigma, correction) et
la covariance sont **communs** à toutes les méthodes ; seule l'estimation de `theta = (u, v, w, z, alpha)` les distingue.

Ordre des canaux `[HH, HV, VH, VV]`
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar
from ..parameters import CrosstalkParameters
from ..physics import DEFAULT_TOL, apply_correction

# Class CrosstalkEstimator(ABC) est une classe abstraite qui définit l'interface commune pour toutes les méthodes d'estimation du crosstalk. 
# Attribut de classe `name` qui doit être défini par chaque sous-classe concrète

class CrosstalkEstimator(ABC):
    name: ClassVar[str] #Partage le nom de la méthode, utilisé pour l'enregistrement et l'identification.

    def __init__(self, tol: float = DEFAULT_TOL) -> None:
        self.tol = float(tol)
        if not hasattr(type(self), "name"):
            raise TypeError(f"{type(self).__name__} doit définir un attribut de classe `name`")

    @abstractmethod
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """
        Estime un jeu `(u, v, w, z, alpha)` pour ce patch.
        Args: patch: `(4, H, W)` complexe, en ordre `POL_ORDER`.
        Returns: Les paramètres `(u, v, w, z, alpha)`.
        """

    #------------------------------Méthodes Héritées------------------------------
    
    def check_patch(self, patch: np.ndarray) -> None:
        """
        Valide la forme d'un patch : un seul patch `(4, H, W)`.
        """
        if patch.ndim != 3:
            raise ValueError(f"patch de forme {patch.shape} : (4, H, W) attendu. Un estimateur "
            "rend un jeu de paramètres par patch ; pour un lot, itérez.")
        if patch.shape[0] != 4:
            raise ValueError(f"patch à {patch.shape[0]} canaux, 4 attendus")

    def correct(self, patch: np.ndarray, params: CrosstalkParameters | None = None) -> np.ndarray:
        """
        Corrige un patch du crosstalk : `s_chapeau = Sigma o`.
        Returns: Le patch corrigé, **dans l'ordre d'entrée** et de même dtype.
        """
        if params is None:
            params = self.estimate(patch)
        self.check_patch(patch)
        return apply_correction(patch, params, tol=self.tol)

    def estimate_and_correct(self, patch: np.ndarray) -> tuple[np.ndarray, CrosstalkParameters]:
        """
        Estime puis corrige, sans estimer deux fois.
        Returns: Le couple `(patch corrigé, paramètres)`.
        """
        params = self.estimate(patch)
        return self.correct(patch, params), params

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, tol={self.tol:g})")
