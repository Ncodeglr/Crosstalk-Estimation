"""Le côté statistique : les méthodes d'estimation du crosstalk, leurs métriques (`metrics`) et les patchs de test (`patchs`).

Importer ce paquet peuple le registre : chaque module de méthode se déclare via
`@register_estimator` et est importé ici.

Exemple d'extension — brancher une méthode sur l'interface ne demande de
modifier aucun fichier existant ::

    from typing import override
    import numpy as np
    from src.crosstalk.baselines import CrosstalkEstimator, register_estimator
    from src.crosstalk.polarimetry.parameters import CrosstalkParameters

    @register_estimator
    class SansCrosstalkEstimator(CrosstalkEstimator):
        '''Méthode de référence triviale : déclare l'absence de distorsion.'''

        name = "sans_crosstalk"

        @override
        def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
            self.check_patch(patch)
            return CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)

    # Sigma vaut alors l'identité : `correct` rend le patch inchangé.
    estimateur = create_from_config({"name": "sans_crosstalk"})
    corrige, theta = estimateur.estimate_and_correct(patch)

Le registre est ensuite adressable depuis une configuration ::

    create_from_config({"name": "sans_crosstalk", "params": {"tol": 1e-10}})
"""

from .estimator import CrosstalkEstimator
from .quegan import QueganEstimator
from .registry import (
    available_estimators,
    create_from_config,
    register_estimator,
)

__all__ = [
    "CrosstalkEstimator",
    "QueganEstimator",
    "available_estimators",
    "create_from_config",
    "register_estimator",
]
