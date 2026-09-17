"""Estimation et correction du crosstalk polarimétrique.

Le paquet sépare quatre responsabilités :

- `conventions` : ordre des polarisations et traduction depuis le pipeline ;
- `parameters`  : le jeu `(u, v, w, z, alpha)` ;
- `physics`     : D, Sigma et leur application — commun à toutes les méthodes ;
- `covariance`  : C = < o o^H >, l'entrée des méthodes statistiques ;
- `metrics`     : erreurs sur theta et crosstalk résiduel ;
- `estimators`  : ce qui distingue les méthodes, c'est-à-dire l'estimation de
  `theta` et elle seule, plus le registre qui les rend adressables par nom.

Référence d'implémentation : Fore, A. G., Chapman, B. D., Hawkins, B. P.,
Hensley, S., Jones, C. E., Michel, T. R., & Muellerschoen, R. J. (2015).
UAVSAR Polarimetric Calibration. *IEEE TGRS*, 53(6), 3481-3491.
"""

from .covariance import (
    BABU_WINDOW,
    WindowSpec,
    polarimetric_covariance,
)
from .conventions import (
    POL_ORDER_DATALOADER,
    POL_ORDER_ONERA,
    to_onera_order,
)
from .exceptions import (
    CrosstalkError,
    DegenerateDistortionError,
    IllConditionedCovarianceError,
)
from .estimators import (
    CrosstalkEstimator,
    available_estimators,
    create_estimator,
    create_from_config,
    register_estimator,
)
from .metrics import ResidualReport, parameter_errors, residual_crosstalk
from .parameters import CrosstalkParameters
from .physics import (
    apply_correction,
    apply_distortion,
    distortion_matrix,
    inverse_distortion_matrix,
)

__all__ = [
    "BABU_WINDOW",
    "WindowSpec",
    "polarimetric_covariance",
    "POL_ORDER_DATALOADER",
    "POL_ORDER_ONERA",
    "to_onera_order",
    "CrosstalkError",
    "DegenerateDistortionError",
    "IllConditionedCovarianceError",
    "CrosstalkEstimator",
    "CrosstalkParameters",
    "ResidualReport",
    "parameter_errors",
    "residual_crosstalk",
    "available_estimators",
    "create_estimator",
    "create_from_config",
    "register_estimator",
    "apply_correction",
    "apply_distortion",
    "distortion_matrix",
    "inverse_distortion_matrix",
]
