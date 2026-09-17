"""Interface commune aux méthodes d'estimation du crosstalk.

Le principe directeur du paquet : le modèle physique (D, Sigma, correction) et
la covariance sont **communs** à toutes les méthodes ; seule l'estimation de
`theta = (u, v, w, z, alpha)` les distingue. Cette classe de base matérialise
cette frontière — une méthode nouvelle n'écrit que `estimate`, et hérite du
reste.

Ordre des canaux. Un estimateur reçoit les patchs dans l'ordre du pipeline
(`POL_ORDER_DATALOADER` par défaut) et les convertit vers l'ordre du modèle
physique. `correct` **rend le patch corrigé dans l'ordre d'entrée** : c'est une
transformation donnée-vers-donnée, elle n'a pas à changer la disposition des
canaux au passage. Seuls les `CrosstalkParameters` rendus par `estimate` sont,
eux, exprimés dans la convention du modèle (celle de Fore et al., 2015).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

import numpy as np

from ..conventions import (
    POL_ORDER_DATALOADER,
    POL_ORDER_ONERA,
    reorder,
)
from ..parameters import CrosstalkParameters
from ..physics import DEFAULT_TOL, apply_correction


class CrosstalkEstimator(ABC):
    """Base abstraite : une méthode d'estimation du crosstalk.

    Attributes:
        name: identifiant de la méthode, utilisé par le registre et les
            configurations. Toute sous-classe concrète doit le définir.
    """

    name: ClassVar[str]

    def __init__(
        self,
        source_order: Sequence[str] = POL_ORDER_DATALOADER,
        tol: float = DEFAULT_TOL,
    ) -> None:
        """
        Args:
            source_order: ordre des canaux des patchs reçus. Le défaut est
                celui que sert `torchcvnn.datasets.SLCDataset`.
            tol: seuil de singularité de D, passé à `apply_correction`.
        """
        if not hasattr(type(self), "name"):
            raise TypeError(
                f"{type(self).__name__} doit définir un attribut de classe `name`"
            )
        self.source_order = tuple(source_order)
        self.tol = float(tol)

    # ------------------------------------------------------------- à implémenter

    @abstractmethod
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """Estime un jeu `(u, v, w, z, alpha)` pour ce patch.

        C'est le seul point que les méthodes ont à écrire.

        Args:
            patch: `(4, H, W)` complexe, dans `self.source_order`.

        Returns:
            Les paramètres, exprimés dans la convention `POL_ORDER_ONERA`.
        """

    # --------------------------------------------------------- hérité, commun

    def to_model_order(self, patch: np.ndarray) -> np.ndarray:
        """Convertit un patch de `self.source_order` vers `POL_ORDER_ONERA`."""
        self.check_patch(patch)
        return reorder(patch, self.source_order, POL_ORDER_ONERA)

    def check_patch(self, patch: np.ndarray) -> None:
        """Valide la forme d'un patch : un seul patch `(4, H, W)`.

        Raises:
            ValueError: si le patch n'est pas de rang 3 ou n'a pas 4 canaux.
        """
        if patch.ndim != 3:
            raise ValueError(
                f"patch de forme {patch.shape} : (4, H, W) attendu. Un estimateur "
                "rend un jeu de paramètres par patch ; pour un lot, itérez."
            )
        if patch.shape[0] != 4:
            raise ValueError(
                f"patch à {patch.shape[0]} canaux, 4 attendus "
                f"(ordre annoncé : {self.source_order})"
            )

    def correct(
        self, patch: np.ndarray, params: CrosstalkParameters | None = None
    ) -> np.ndarray:
        """Corrige un patch du crosstalk : `s_chapeau = Sigma o`.

        Args:
            patch: `(4, H, W)` complexe, dans `self.source_order`.
            params: paramètres à appliquer. S'ils sont omis, ils sont estimés
                sur ce patch même.

        Returns:
            Le patch corrigé, **dans l'ordre d'entrée** et de même dtype.

        Raises:
            DegenerateDistortionError: si D est singulière.
        """
        if params is None:
            params = self.estimate(patch)
        corrige = apply_correction(
            self.to_model_order(patch), params, tol=self.tol
        )
        return reorder(corrige, POL_ORDER_ONERA, self.source_order)

    def estimate_and_correct(
        self, patch: np.ndarray
    ) -> tuple[np.ndarray, CrosstalkParameters]:
        """Estime puis corrige, sans estimer deux fois.

        Returns:
            Le couple `(patch corrigé, paramètres)`.
        """
        params = self.estimate(patch)
        return self.correct(patch, params), params

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"source_order={self.source_order}, tol={self.tol:g})"
        )
