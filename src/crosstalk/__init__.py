"""Estimation et correction du crosstalk polarimétrique.

Le paquet sépare cinq responsabilités :

- `parameters`  : le jeu `(u, v, w, z, alpha)` ;
- `matrix`      : l'ordre des polarisations, D, Sigma et leur application —
  commun à toutes les méthodes ;
- `covariance`  : C = < s' s'^H >, moyennée sur le patch entier, l'entrée des
  méthodes statistiques ;
- `metrics`     : erreurs sur theta et crosstalk résiduel ;
- `estimators`  : ce qui distingue les méthodes, c'est-à-dire l'estimation de
  `theta` et elle seule, plus le registre qui les rend adressables par nom.

Référence d'implémentation : Babu, A., Kumar, S., & Agrawal, S. (2022).
Polarimetric calibration and spatio-temporal polarimetric distortion analysis
of UAVSAR PolSAR data. *Earth and Space Science*, 9(4), e2020EA001629.
L'ordre des polarisations suit ses équations 8 et 9, les formes fermées de
Quegan son équation 13, et le module de `alpha` son équation 15. La matrice de
distorsion, son inverse, la réindexation de `alpha` et sa phase viennent en
revanche de Fore, réindexés dans cet ordre : les équations 10, 12 et 16 à 18 de
Babu sont erronées, et son équation 15 omet la phase (voir `matrix.py`). Règle suivie : Babu, sauf là où ses équations sont incohérentes ou
incomplètes, auquel cas Fore.

Référence secondaire, citée là où elle est la source d'un fait que Babu ne
formule pas : Fore, A. G., Chapman, B. D., Hawkins, B. P., Hensley, S.,
Jones, C. E., Michel, T. R., & Muellerschoen, R. J. (2015). UAVSAR Polarimetric
Calibration. *IEEE TGRS*, 53(6), 3481-3491.
"""

from .covariance import polarimetric_covariance
from .exceptions import (
    CrosstalkError,
    SingularDmatrixError,
)
from .estimators import (
    CrosstalkEstimator,
    available_estimators,
    create_from_config,
    register_estimator,
)
from .metrics import (
    ResidualReport,
    corner_reflector_crosstalk,
    parameter_errors,
    reciprocity,
    residual_crosstalk,
)
from .parameters import CrosstalkParameters
from .matrix import (
    POL_ORDER,
    apply_correction,
    apply_distortion,
    distortion_matrix,
    sigma_calibration_matrix,
)

__all__ = [
    "polarimetric_covariance",
    "POL_ORDER",
    "CrosstalkError",
    "SingularDmatrixError",
    "CrosstalkEstimator",
    "CrosstalkParameters",
    "ResidualReport",
    "corner_reflector_crosstalk",
    "parameter_errors",
    "reciprocity",
    "residual_crosstalk",
    "available_estimators",
    "create_from_config",
    "register_estimator",
    "apply_correction",
    "apply_distortion",
    "distortion_matrix",
    "sigma_calibration_matrix",
]
