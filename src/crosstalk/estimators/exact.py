"""
Solveur exact : les hypothèses de Quegan, résolues sans approximation.

Rôle
----
Les formes fermées de Quegan (1994) supposent la cible réciproque (hypothèse 2) et à symétrie de réflexion
(hypothèse 3), puis négligent les termes du second ordre en crosstalk : leur estimation est biaisée dès que le
crosstalk n'est plus très petit. Le solveur exact garde exactement les mêmes hypothèses, mais les impose sans
approximation : il cherche le theta dont la correction rend la covariance du patch conforme à ces deux hypothèses.

C'est le témoin « sans apprentissage » du banc d'essai : il reçoit la même covariance C que Quegan et que le CVNN.
L'écart entre Quegan et lui mesure l'effet de l'approximation au premier ordre ; l'écart entre lui et le CVNN mesure
ce qu'apporte l'apprentissage.

Les 10 contraintes
------------------
Sur la covariance corrigée Ĉ = Sigma(theta) C Sigma(theta)^H, avec Sigma = D^-1 :
- symétrie de réflexion : les 4 corrélations normalisées entre canaux co-polaires et croisés,
  Ĉ_ij / sqrt(Ĉ_ii Ĉ_jj) pour i dans {HH, VV} et j dans {HV, VH}, sont nulles (8 équations réelles) ;
- réciprocité : HV et VH ont la même puissance, log(Ĉ_HV,HV / Ĉ_VH,VH) = 0, et la même phase, arg Ĉ_HV,VH = 0
  (2 équations réelles).
Les 10 inconnues réelles sont u, v, w, z (complexes) et alpha, paramétré par log|alpha| et arg(alpha), ce qui le garde
non nul. Le système est résolu par moindres carrés (scipy.optimize.least_squares), en partant de la solution de Quegan.

Échec
-----
Comme QueganEstimator.solve, solve rend des nan quand il ne peut pas conclure : covariance de trace nulle ou non
finie, départ de Quegan non défini, D singulière au cours de la résolution, ou échec de l'optimiseur. Un nan se voit
dans les bilans au lieu d'être écarté en silence.
"""

from __future__ import annotations
from typing import override
import numpy as np
from scipy.optimize import least_squares
from ..covariance import check_covariance, normalize_by_trace, polarimetric_covariance
from ..matrix import POL_ORDER, distortion_matrix
from ..parameters import CrosstalkParameters
from .estimator import CrosstalkEstimator
from .quegan import QueganEstimator
from .registry import register_estimator

IDX_HH, IDX_HV, IDX_VH, IDX_VV = (POL_ORDER.index(pol) for pol in ("HH", "HV", "VH", "VV"))

#: Tolérances d'arrêt de least_squares, sur le pas et sur la baisse relative du coût.
XTOL: float = 1e-10
FTOL: float = 1e-10


def _vers_theta(x: np.ndarray) -> np.ndarray:
    """(10,) réels [Re u, Im u, ..., Re z, Im z, log|alpha|, arg alpha] -> (5,) complexes (u, v, w, z, alpha)."""
    return np.r_[x[0:8:2] + 1j * x[1:8:2], np.exp(x[8] + 1j * x[9])]


def _depuis_theta(theta: np.ndarray) -> np.ndarray:
    """(5,) complexes (u, v, w, z, alpha) -> (10,) réels, l'inverse de _vers_theta."""
    return np.r_[np.column_stack([theta[:4].real, theta[:4].imag]).ravel(), np.log(abs(theta[4])), np.angle(theta[4])]


def _residus(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Les 10 contraintes, nulles quand la covariance corrigée respecte la symétrie de réflexion et la réciprocité.

    Sigma est obtenue par inversion numérique de D, et non par matrix.sigma_calibration_matrix : celle-ci émet une
    alerte de conditionnement, qui se répéterait ici à chaque évaluation. D singulière lève np.linalg.LinAlgError.
    """
    sigma = np.linalg.inv(distortion_matrix(CrosstalkParameters.from_vector(_vers_theta(x))))
    corrigee = sigma @ cov @ sigma.conj().T
    puissances = corrigee.diagonal().real
    residus = []
    for i in (IDX_HH, IDX_VV):
        for j in (IDX_HV, IDX_VH):
            correlation = corrigee[i, j] / np.sqrt(puissances[i] * puissances[j])
            residus += [correlation.real, correlation.imag]
    residus += [np.log(puissances[IDX_HV] / puissances[IDX_VH]), np.angle(corrigee[IDX_HV, IDX_VH])]
    return np.array(residus)


@register_estimator
class ExactEstimator(CrosstalkEstimator):
    """
    Estimation du crosstalk par résolution exacte des contraintes de Quegan (voir la docstring du module).
    Un jeu (u, v, w, z, alpha) par patch, tiré de l'unique covariance du patch (covariance.polarimetric_covariance).
    """

    name = "exact"

    def solve(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Résout les 10 contraintes sur une covariance.

        La covariance est d'abord divisée par sa trace (covariance.normalize_by_trace) : les contraintes ne dépendent
        pas de l'échelle, et la résolution est ainsi la même pour un patch sombre et pour un patch brillant.

        Args:
            cov_matrix: (4, 4) complexe, en POL_ORDER. Une seule covariance : contrairement aux formes fermées de
                QueganEstimator.solve, la résolution itérative ne se vectorise pas sur un lot.

        Returns:
            theta, (5,) complex128 dans l'ordre (u, v, w, z, alpha), ou cinq nan en cas d'échec (voir la docstring
            du module).

        Raises:
            ValueError: si cov_matrix n'est pas de forme (4, 4).
        """
        cov = np.asarray(cov_matrix, dtype=np.complex128)
        check_covariance(cov)
        echec = np.full(5, np.nan + 0j)
        cov = normalize_by_trace(cov)
        if not np.all(np.isfinite(cov)):       # trace nulle ou non finie, ou entrée non finie
            return echec
        # Quand un dénominateur des formes fermées s'annule, Quegan rend des nan en émettant des RuntimeWarning de
        # NumPy. Ce cas est traité juste après (départ non fini -> échec) : les avertissements n'apportent rien ici.
        with np.errstate(divide="ignore", invalid="ignore"):
            depart = QueganEstimator().solve(cov)
        if not np.all(np.isfinite(depart)) or depart[4] == 0:
            return echec
        try:
            resultat = least_squares(_residus, _depuis_theta(depart), args=(cov,), xtol=XTOL, ftol=FTOL)
        except np.linalg.LinAlgError:          # D singulière au cours de la résolution
            return echec
        except ValueError:                     # résidus non finis dès le départ (puissance corrigée nulle)
            return echec
        if resultat.status <= 0 or not np.all(np.isfinite(resultat.x)):
            return echec
        return _vers_theta(resultat.x)

    @override
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """Estime un jeu (u, v, w, z, alpha) pour ce patch."""
        self.check_patch(patch)
        return CrosstalkParameters.from_vector(self.solve(polarimetric_covariance(patch)))
