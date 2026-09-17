"""Métriques d'évaluation des estimateurs de crosstalk.

Deux situations, deux familles de métriques.

**Vérité terrain disponible** — uniquement sur données synthétiques, où `theta`
est connu parce qu'il a été appliqué : `parameter_errors` compare directement.

**Pas de vérité terrain** — le cas des données réelles : `residual_crosstalk`
applique le protocole de Fore et al. (2015, §V), qui corrige les données puis
ré-estime le crosstalk sur les données corrigées. C'est ainsi que Fore compare
Quegan et Ainsworth (résiduels de -15 à -20 dB contre -30 dB) et que Babu et al.
(2022, tableau 4) classent les trois méthodes.

.. warning::
   Le résiduel est **optimiste par construction** : c'est le même estimateur qui
   juge son propre ajustement, sur le même patch. Mesuré sur la pile Rosamond,
   il donne des réductions de 8 à 15 dB là où Babu et al. en rapportent 2,8 dB
   aux réflecteurs en coin — la différence tient à la nature de la mesure, pas à
   la qualité de la correction. Cette métrique sert à **comparer des méthodes
   entre elles**, pas à annoncer une performance absolue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .parameters import CROSSTALK_NAMES, CrosstalkParameters

if TYPE_CHECKING:  # pragma: no cover
    from .estimators.base import CrosstalkEstimator


def parameter_errors(
    estimate: CrosstalkParameters, reference: CrosstalkParameters
) -> dict[str, float]:
    """Écarts entre un jeu estimé et la vérité terrain.

    N'a de sens que sur données synthétiques, `reference` étant le `theta`
    effectivement appliqué par `physics.apply_distortion`.

    Args:
        estimate: jeu estimé.
        reference: jeu vrai.

    Returns:
        Dictionnaire plat, prêt pour un CSV :

        - `abs_error_u` .. `abs_error_z` : `|theta_estime - theta_vrai|` ;
        - `rel_error_u` .. `rel_error_z` : le même, rapporté à `|theta_vrai|`
          (`inf` si le terme vrai est nul) ;
        - `max_rel_error` : le pire des quatre ;
        - `alpha_error_db` : écart des modules de alpha, en dB ;
        - `alpha_error_deg` : écart des phases de alpha, replié dans
          `]-180, 180]`.
    """
    resultats: dict[str, float] = {}
    relatives: list[float] = []
    for name in CROSSTALK_NAMES:
        estime = getattr(estimate, name)
        vrai = getattr(reference, name)
        absolue = abs(estime - vrai)
        resultats[f"abs_error_{name}"] = float(absolue)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = float(absolue / abs(vrai)) if vrai != 0 else float("inf")
        resultats[f"rel_error_{name}"] = relative
        relatives.append(relative)
    resultats["max_rel_error"] = max(relatives)
    resultats["alpha_error_db"] = estimate.alpha_db() - reference.alpha_db()
    ecart_phase = estimate.alpha_phase_deg() - reference.alpha_phase_deg()
    resultats["alpha_error_deg"] = float((ecart_phase + 180.0) % 360.0 - 180.0)
    return resultats


@dataclass(frozen=True, slots=True)
class ResidualReport:
    """Résultat du protocole de résiduel de Fore et al. (2015, §V).

    Attributes:
        initial: jeu estimé sur les données d'origine.
        residual: jeu ré-estimé sur les données corrigées.
    """

    initial: CrosstalkParameters
    residual: CrosstalkParameters

    @property
    def initial_db(self) -> float:
        """Crosstalk initial, `max 20 log10 |u, v, w, z|`."""
        return self.initial.max_crosstalk_db()

    @property
    def residual_db(self) -> float:
        """Crosstalk résiduel après correction."""
        return self.residual.max_crosstalk_db()

    @property
    def reduction_db(self) -> float:
        """Réduction obtenue, positive quand la correction améliore."""
        return self.initial_db - self.residual_db

    def as_row(self) -> dict[str, float]:
        """Vue plate, prête pour un CSV."""
        return {
            "initial_db": self.initial_db,
            "residual_db": self.residual_db,
            "reduction_db": self.reduction_db,
            "alpha_abs": float(abs(self.initial.alpha)),
            "alpha_deg": self.initial.alpha_phase_deg(),
        }


def residual_crosstalk(
    patch: np.ndarray, estimator: "CrosstalkEstimator"
) -> ResidualReport:
    """Estime, corrige, puis ré-estime sur les données corrigées.

    Args:
        patch: `(4, H, W)` complexe, dans `estimator.source_order`.
        estimator: la méthode évaluée. C'est elle qui juge sa propre correction,
            d'où la réserve énoncée en tête de module.

    Returns:
        Le couple `(jeu initial, jeu résiduel)`.

    Raises:
        IllConditionedCovarianceError: si l'une des deux estimations échoue.
        DegenerateDistortionError: si la correction est impossible.
    """
    corrige, initial = estimator.estimate_and_correct(patch)
    return ResidualReport(initial=initial, residual=estimator.estimate(corrige))
