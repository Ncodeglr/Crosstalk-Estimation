"""
Interface commune aux méthodes d'estimation du crosstalk.
Le principe directeur du paquet : le modèle physique (D, Sigma, correction) est
**commun** à toutes les méthodes ; seule l'estimation de theta = (u, v, w, z, alpha) les distingue.

Ordre des canaux [HH, HV, VH, VV]
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar
from ..polarimetry.covariance import check_patch
from ..polarimetry.parameters import CrosstalkParameters
from ..polarimetry.matrix import DEFAULT_TOL, apply_correction


class CrosstalkEstimator(ABC):
    """
    Interface commune aux méthodes d'estimation du crosstalk.
    Le modèle physique (D, Sigma, correction) est commun à toutes les méthodes ;
    seule l'estimation de theta = (u, v, w, z, alpha) les distingue.

    Attributes:
        name: nom de la méthode, attribut de classe que chaque sous-classe concrète doit définir ; sert à l'enregistrement.

    Args:
        tol: seuil de singularité de D, transmis à la correction.
    """
    
    name: ClassVar[str] #Partage le nom de la méthode, utilisé pour l'enregistrement et l'identification.

    def __init__(self, tol: float = DEFAULT_TOL) -> None:
        self.tol = float(tol)
        if not hasattr(type(self), "name"):
            raise TypeError(f"{type(self).__name__} doit définir un attribut de classe `name`")

    @abstractmethod
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """
        Estime un jeu (u, v, w, z, alpha) pour un patch.
        Args: patch: (4, H, W) complexe, en ordre POL_ORDER
        Returns: Les paramètres (u, v, w, z, alpha)
        """

    #------------------------------Méthodes------------------------------
    
    def check_patch(self, patch: np.ndarray) -> None:
        """
        Valide un patch : un seul patch `(4, H, W)`, puisqu'un estimateur rend un jeu de paramètres par patch.
        La vérification elle-même est celle de tout le projet : covariance.check_patch.
        """
        check_patch(patch)

    def correct(self, patch: np.ndarray, theta: CrosstalkParameters | None = None) -> np.ndarray:
        """
        Corrige un patch du crosstalk : `s_chapeau = Sigma s'`.
        Args: theta: le jeu (u, v, w, z, alpha) qui définit `Sigma` ; estimé sur `patch` s'il n'est pas fourni.
        Returns: Le patch corrigé, **dans l'ordre d'entrée** et de même dtype.
        """
        if theta is None:
            theta = self.estimate(patch)
        self.check_patch(patch)
        return apply_correction(patch, theta, tol=self.tol)

    def estimate_and_correct(self, patch: np.ndarray) -> tuple[np.ndarray, CrosstalkParameters]:
        """
        Estime puis corrige, sans estimer deux fois.
        Returns: Le couple `(patch_corrige, theta)` : le patch corrigé, et le jeu (u, v, w, z, alpha) estimé sur le patch d'entrée.
        """
        theta = self.estimate(patch)
        patch_corrige = self.correct(patch, theta)
        return patch_corrige, theta

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, tol={self.tol:g})")
