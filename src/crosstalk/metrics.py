"""Métriques d'évaluation des estimateurs de crosstalk.

Deux situations, deux familles de métriques.

**Vérité terrain disponible** — uniquement sur données synthétiques, où `theta`
est connu parce qu'il a été appliqué : `parameter_errors` compare directement.

**Pas de vérité terrain** — le cas des données réelles. Trois contrôles :

- `residual_crosstalk` applique le protocole de Fore et al. (2015, §V) : il
  corrige les données, puis ré-estime le crosstalk sur les données corrigées
  **avec le même estimateur** ;
- `reciprocity` mesure l'accord entre HV et VH, que Quegan n'optimise pas ;
- `corner_reflector_crosstalk` mesure la fuite au pic d'un trièdre, dont les
  canaux croisés sont nuls par construction (`measure_reflectors.py`).

.. warning::
   Le résiduel est **auto-jugé** : l'estimateur évalue sa propre correction. Il
   ne permet **pas** de comparer des méthodes entre elles, et pas davantage
   d'annoncer une performance absolue. Une méthode qui ne corrige rien
   (`theta = 0`) se relit, ne trouve aucun crosstalk et obtient la note
   parfaite ; une méthode qui connaîtrait le vrai `theta` se relirait avec ce
   même `theta` et semblerait n'avoir rien réduit. Mesuré sur un patch
   synthétique à `theta` connu : sans correction, -inf dB ; correction parfaite,
   -23,9 dB ; Quegan, -47,8 dB — l'ordre inverse de la réalité.

   Le résiduel ne sert donc qu'à contrôler la cohérence interne d'une méthode.
   Comparer des méthodes demande des juges extérieurs à elles : la réciprocité
   et les réflecteurs en coin. Ces derniers donnent aussi la performance
   absolue — 12,4 dB de réduction sur la pile Rosamond, si bien que l'écart avec
   les 2,8 dB de Babu et al. (table 4) ne tient pas à la circularité du résiduel.

   Ces deux juges sont nécessaires, mais pas suffisants : aucun ne voit
   `alpha`, ni la direction `(u, v, w, z) = z (alpha, -alpha, -1, 1)`, et la
   réciprocité n'est pas indépendante d'une méthode qui lit le désaccord entre
   HV et VH, comme Ainsworth. Voir le README, « Ce que les niveaux 2 et 3 ne
   voient pas ».
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from .parameters import CROSSTALK_NAMES, CrosstalkParameters
from .physics import POL_ORDER

if TYPE_CHECKING:  # pragma: no cover
    from .estimators.estimator import CrosstalkEstimator


def parameter_errors(estimate: CrosstalkParameters, reference: CrosstalkParameters) -> dict[str, float]:
    """
    Écarts entre un jeu estimé et la vérité terrain.

    N'a de sens que sur données synthétiques, `reference` étant le `theta`
    effectivement appliqué par `physics.apply_distortion`.

    Args:
        estimate: jeu estimé.
        reference: jeu vrai.

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
    for name in CROSSTALK_NAMES:
        estime = getattr(estimate, name)
        vrai = getattr(reference, name)
        absolue = abs(estime - vrai)
        resultats[f"abs_error_{name}"] = float(absolue)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = float(absolue / abs(vrai)) if vrai != 0 else float("inf")
        resultats[f"rel_error_{name}"] = relative
        relatives.append(relative)
    resultats["max_rel_error"] = max(relatives) #max_rel_error prend le pire des quatre, pas la moyenne — un seul paramètre mal estimé suffit à fausser la correction, donc on ne veut pas qu'il soit noyé.
    
    #Pour alpha, on ne rend que l'écart des modules en dB
    resultats["alpha_error_db"] = estimate.alpha_db() - reference.alpha_db()
    return resultats


def reciprocity(patch: np.ndarray) -> float:
    """
    Cohérence entre les canaux croisés : `|<HV VH*>| / sqrt(<|HV|^2><|VH|^2>)`.

    Validation indépendante de Quegan, contrairement au résiduel.
    Un radar monostatique émet et reçoit sur la même antenne, donc la réciprocité S_HV = S_VH est une propriété du capteur et non du modèle de Quegan.
    Le crosstalk la viole, une correction qui la rétablit démontre quelque chose que Quegan n'a pas optimisé.

    C'est le contrôle du §4.4 de Babu, dont les figures 7a et 8a tracent HV contre VH avant et après calibration : un nuage dispersé devient une droite.

    Angle mort : si `u = alpha z` et `v = alpha w`, les lignes HV et VH de D
    donnent `O_VH = alpha O_HV` pixel par pixel, et la cohérence vaut exactement
    1 quel que soit le crosstalk. Elle ignore aussi `alpha` lui-même. Elle ne
    voit donc que `u - alpha z` et `v - alpha w`. Elle n'est pas indépendante
    d'Ainsworth, dont le second membre est fait de `C31 - C21` et `C34 - C24`.

    Args : patch: `(4, H, W)` complexe, en `physics.POL_ORDER`.

    Returns:
        Un réel dans `[0, 1]`, `1` pour une réciprocité parfaite. Rend `nan` si
        l'un des deux canaux croisés est éteint.
    """
    hv = patch[POL_ORDER.index("HV")].ravel()
    vh = patch[POL_ORDER.index("VH")].ravel()
    denominateur = np.sqrt(np.mean(np.abs(hv) ** 2) * np.mean(np.abs(vh) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.abs(np.mean(hv * np.conj(vh))) / denominateur)


def corner_reflector_crosstalk(hh: complex, hv: complex, vh: complex, vv: complex) -> float:
    """
    Fuite co-pol vers cross-pol mesurée à un trièdre, en dB.

    La seule vérité terrain disponible sur ces données. Un trièdre a pour
    matrice de diffusion `S = sqrt(sigma) * I` (Fore et al., 2015, équation 5) :
    ses canaux croisés sont **nuls par construction**. Tout ce qu'on y mesure
    est donc du crosstalk, sans aucune hypothèse de modèle — contrairement au
    résiduel de `residual_crosstalk`, que le même estimateur juge.

    C'est la mesure de Babu et al. (2022, §4.6 et figure 9).

    Au premier ordre en `(u, v, w, z)`, appliquer `D` à `S = I` donne
    `O_HV / O_HH = z + w` et `O_VH / O_VV = u + v` : la fuite rendue ici vaut
    donc `max(|z + w|, |u + v|)`. Ce n'est **pas** le `max 20 log10 |u,v,w,z|`
    de `CrosstalkParameters.max_crosstalk_db`, qui porte sur les paramètres
    isolés — les deux peuvent différer par compensation entre termes.
    Le trièdre est donc aveugle à `u = -v, z = -w`, et à `alpha`, qui ne
    multiplie que la composante croisée du signal vrai, nulle ici.

    Args: hh, hv, vh, vv: les quatre valeurs complexes observées au réflecteur, en ordre `physics.POL_ORDER`.

    Returns:
        20 log10( max(|hv|, |vh|) / sqrt(|hh| |vv|) ). 
        La référence co-polaire est la moyenne géométrique des deux canaux, et non `|hh|` seul, pour ne pas dépendre du déséquilibre résiduel entre HH et VV.

        Rend `-inf` sur un trièdre parfait, et une valeur non finie quand la
        référence co-polaire s'annule — `inf` si les canaux croisés ne sont pas
        nuls, nan si les canaux croisés le sont aussi. L'appelant doit donc filtrer par
        `np.isfinite`, comme le fait le garde-fou de `QueganEstimator._solve` :
        un réflecteur absent ou masqué ne doit pas produire un chiffre.
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
        initial: jeu estimé sur les données d'origine.
        residual: jeu ré-estimé sur les données corrigées.
        reciprocity_before: cohérence HV/VH avant correction.
        reciprocity_after: la même après correction. C'est la seule des quatre grandeurs que l'estimateur ne juge pas lui-même.
    """

    initial: CrosstalkParameters
    residual: CrosstalkParameters
    reciprocity_before: float = float("nan")
    reciprocity_after: float = float("nan")

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
        """
        Les six grandeurs résumées du rapport, en un dictionnaire plat : une ligne
        de résultats, dont `run_baselines.summarize` prend la médiane.
        """
        return {
            "initial_db": self.initial_db,
            "residual_db": self.residual_db,
            "reduction_db": self.reduction_db,
            "alpha_abs": float(abs(self.initial.alpha)),
            "reciprocity_before": self.reciprocity_before,
            "reciprocity_after": self.reciprocity_after,
        }


def residual_crosstalk(patch: np.ndarray, estimator: "CrosstalkEstimator") -> ResidualReport:
    """
    Estime, corrige, puis ré-estime sur les données corrigées.

    Returns: Le rapport complet : les deux jeux, et la cohérence HV/VH de part et d'autre de la correction.

    Raises:
        IllConditionedCovarianceError: si l'une des deux estimations échoue.
        SingularDmatrixError: si la correction est impossible.
    """
    patch_corrige, theta_initial = estimator.estimate_and_correct(patch) #Applique la méthode du méthode de type CrosstalkEstimator
    return ResidualReport(
        initial=theta_initial,
        residual=estimator.estimate(patch_corrige),
        reciprocity_before=reciprocity(patch),
        reciprocity_after=reciprocity(patch_corrige),
    )