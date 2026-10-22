"""
Métriques d'évaluation des estimateurs de crosstalk.

Deux situations, deux familles de métriques.

**Vérité terrain disponible** — uniquement sur données synthétiques, où `theta`
est connu parce qu'il a été appliqué : `parameter_errors` compare directement.

**Pas de vérité terrain** — le cas des données réelles. Trois contrôles :
- `corner_reflector_crosstalk` mesure la fuite au pic d'un trièdre, dont les
canaux croisés sont nuls par construction (`measure_reflectors.py`).

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from .parameters import CROSSTALK_NAMES, CrosstalkParameters
from .matrix import POL_ORDER

if TYPE_CHECKING:  # pragma: no cover
    from .estimators.estimator import CrosstalkEstimator


def parameter_errors(estimate: CrosstalkParameters, ground_truth: CrosstalkParameters) -> dict[str, float]:
    """
    Écarts entre un jeu estimé et la vérité terrain.
    N'a de sens que sur données synthétiques, `ground_truth` étant le `theta` effectivement appliqué par `matrix.apply_distortion`.
    
    Args:
        estimate: jeu estimé.
        ground_truth: jeu vrai (potentiellement synthétique)
    
    Returns:
        Dictionnaire plat, prêt pour un CSV :
        - abs_error_u .. abs_error_z : |theta_estime - theta_vrai|
        - rel_error_u .. rel_error_z : le même, rapporté à |theta_vrai| (inf si le terme vrai est nul)
        - max_rel_error : le pire des quatre ;
        - alpha_error_db : écart des modules de alpha, en dB.

    Note:
        L'écart des phases de alpha n'est pas rendu. Il ne mesurait rien tant
        que alpha était estimé réel (Babu, éq. 15) ; depuis que sa phase l'est
        aussi (Fore, éq. 25), il aurait un sens, mais n'a pas été ajouté.
    """
    resultats: dict[str, float] = {}
    relatives: list[float] = []
    for parametre in CROSSTALK_NAMES:
        parametre_estime = getattr(estimate, parametre)
        parametre_vrai = getattr(ground_truth, parametre)
        resultats[f"abs_error_{parametre}"] = float(abs(parametre_estime - parametre_vrai))
        
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = float( abs(parametre_estime - parametre_vrai) / abs(parametre_vrai)) if parametre_vrai != 0 else float("inf")
        resultats[f"rel_error_{parametre}"] = relative
        relatives.append(relative)
    resultats["max_rel_error"] = max(relatives) #max_rel_error prend le pire des quatre, pas la moyenne — un seul paramètre mal estimé suffit à fausser la correction, donc on ne veut pas qu'il soit noyé.
    
    #Pour alpha, on ne rend que l'écart des modules en dB
    resultats["alpha_error_db"] = estimate.alpha_db() - ground_truth.alpha_db()
    return resultats


def reciprocity(patch: np.ndarray) -> float:
    """
    Mesure la cohérence entre les canaux croisés soit HV et VH : |<HV VH*>| / sqrt(<|HV|^2><|VH|^2>)
    Validation indépendante de Quegan, contrairement au résiduel.
    C'est le contrôle du §4.4 de Babu, dont les figures 7a et 8a tracent HV contre VH avant et après calibration : un nuage dispersé devient une droite.
    Args : patch: `(4, H, W)` complexe, en `matrix.POL_ORDER`.

    Returns: Un réel dans [0, 1], 1 pour une réciprocité parfaite. Rend nan si l'un des deux canaux croisés est éteint.
    """
    hv = patch[POL_ORDER.index("HV")].ravel() #.ravel() rend un tableau 1D à partir d'un tableau multidimensionnel, ici on a un patch (4, H, W) et on veut juste les valeurs du canal HV, donc on prend le canal HV et on le met en 1D pour faire les calculs.
    vh = patch[POL_ORDER.index("VH")].ravel()
    denominateur = np.sqrt(np.mean(np.abs(hv) ** 2) * np.mean(np.abs(vh) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.abs(np.mean(hv * np.conj(vh))) / denominateur)


def corner_reflector_crosstalk(hh: complex, hv: complex, vh: complex, vv: complex) -> float:
    """
    Fuite co-pol vers cross-pol mesurée à un trièdre, en dB. C'est la mesure de Babu et al. (2022, §4.6 et figure 9).

    La seule vérité terrain disponible sur ces données. Un trièdre a pour
    matrice de diffusion `S = sqrt(sigma) * I` (Fore, équation 5) :
    ses canaux croisés sont **nuls par construction**. Tout ce qu'on y mesure
    est donc du crosstalk, sans aucune hypothèse de modèle — contrairement au
    résiduel de `residual_crosstalk`, que le même estimateur juge.

   
    Args: hh, hv, vh, vv: les quatre valeurs complexes observées au réflecteur, en ordre `matrix.POL_ORDER`.

    Returns:
        20 log10( max(|hv|, |vh|) / sqrt(|hh| |vv|) ). 
        La référence co-polaire est la moyenne géométrique des deux canaux, et non `|hh|` seul, pour ne pas dépendre du déséquilibre résiduel entre HH et VV.

        Rend `-inf` sur un trièdre parfait, et une valeur non finie quand la
        référence co-polaire s'annule — `inf` si les canaux croisés ne sont pas
        nuls, nan si les canaux croisés le sont aussi. L'appelant doit donc filtrer par
        `np.isfinite` : un réflecteur absent ou masqué ne doit pas produire un chiffre.
    """
    copol = np.sqrt(abs(hh) * abs(vv))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(20.0 * np.log10(max(abs(hv), abs(vh)) / copol))


@dataclass(frozen=True, slots=True)
class ResidualReport:
    """
    @dataclass permet de rendre le jeu de paramètres immuable et plus léger (slots), elles sont dédiées au stockage de données, avec très peu de code.
        -frozen=True permet l'immuabilité et la sécurité
        -slots=True permet de réduire la mémoire et d'accélérer l'accès aux attributs, mais empêche l'ajout d'attributs dynamiques.
        
    Résultat du protocole de résiduel de Fore (§V).

    Attributes:
        theta_initial: jeu estimé sur les données d'origine.
        theta_residual: jeu ré-estimé sur les données corrigées.
        reciprocity_before: cohérence HV/VH avant correction.
        reciprocity_after: la même après correction. C'est la seule des quatre grandeurs que l'estimateur ne juge pas lui-même.
    """

    theta_initial: CrosstalkParameters
    theta_residual: CrosstalkParameters
    reciprocity_before: float = float("nan")
    reciprocity_after: float = float("nan")

    @property # @property transforme une méthode en attribut calculé : on la lit sans parenthèses, comme une donnée, mais elle est recalculée à chaque lecture.
    def initial_db(self) -> float:
        """Crosstalk initial, `max 20 log10 |u, v, w, z|`."""
        return self.theta_initial.max_crosstalk_db()

    @property
    def residual_db(self) -> float:
        """Crosstalk résiduel après correction."""
        return self.theta_residual.max_crosstalk_db()

    @property
    def reduction_db(self) -> float:
        """Réduction obtenue, positive quand la correction améliore."""
        return self.initial_db - self.residual_db

    def to_metrics_row(self) -> dict[str, float]:
        """
        Les six grandeurs résumées du rapport, en un dictionnaire plat : une ligne
        de résultats, dont `run_baselines.aggregate` prend la médiane.
        """
        return {
            "initial_db": self.initial_db,
            "residual_db": self.residual_db,
            "reduction_db": self.reduction_db,
            "alpha_abs": float(abs(self.theta_initial.alpha)),
            "reciprocity_before": self.reciprocity_before,
            "reciprocity_after": self.reciprocity_after,
        }


def residual_crosstalk(patch: np.ndarray, estimator: "CrosstalkEstimator") -> ResidualReport:
    """
    Estime, corrige, puis ré-estime sur les données corrigées.
    """
    patch_corrige, theta_initial = estimator.estimate_and_correct(patch) #Applique la méthode de type CrosstalkEstimator
    return ResidualReport(
        theta_initial=theta_initial,
        theta_residual=estimator.estimate(patch_corrige), #Ré-estime le crosstalk sur les données corrigées, avec le même estimateur. Cela crée un biais de validation, mais c'est le protocole de Fore et al. (2015, §V) et Babu et al. (2022, §4.4).
        reciprocity_before=reciprocity(patch),
        reciprocity_after=reciprocity(patch_corrige),
    )

