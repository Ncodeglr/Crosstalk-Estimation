"""
Méthode de Quegan : estimation du crosstalk en forme fermée.

Domaine de validité. Les formes fermées (Babu, éq. 13 ; Fore, éq. 24) sont la
solution exacte, par la règle de Cramer, de deux systèmes 2x2 obtenus en
développant la covariance observée au premier ordre en crosstalk — **privés**
du terme en puissance croisée vraie `P_X = <|S_HV|^2>` ::

    C31 = u C11 + v C41 + P_X (e^{i arg alpha} v* + |alpha| w*) + O(theta^3)

La méthode n'est donc exacte qu'à `P_X` nul. Son biais relatif est
proportionnel à `beta = P_X / sigma_copol` et indépendant du niveau de
crosstalk ; le coefficient dépend de la direction de `theta`, de `alpha` et de
la corrélation HH-VV. Sur la pile Rosamond (`beta` médian 0,118), il est de
l'ordre du tiers du crosstalk. Voir le README, « Le biais propre de Quegan ».
"""

from __future__ import annotations
import numpy as np
from ..covariance import polarimetric_covariance
from ..exceptions import IllConditionedCovarianceError
from ..parameters import CrosstalkParameters
from ..physics import DEFAULT_TOL
from .base import CrosstalkEstimator
from .registry import register_estimator

#: Seuils **relatifs** par défaut des garde-fous.
#:
#: Les dénominateurs des formes fermées sont homogènes à des puissances : une
#: homothétie globale `g` sur les données multiplie `Delta` par `g^4`. Un seuil
#: absolu dépendrait donc de la normalisation appliquée en amont et ne voudrait
#: rien dire. Les rapports ci-dessous sont sans dimension.
#:
#: Ordres de grandeur mesurés sur 600 patchs 64x64 de la pile Rosamond :
#: `|Delta|/(C11 C44)` descend à 0,080 et `|X|/sqrt(C22 C33)` à 0,027. 
#: Les valeurs par défaut sont très en deçà : elles ne se déclenchent que sur une dégénérescence vraie (patch constant, canal éteint), pas sur une scène difficile.
DEFAULT_DELTA_TOL: float = 1e-6
DEFAULT_X_TOL: float = 1e-6


@register_estimator
class QueganEstimator(CrosstalkEstimator):
    """
    Estimation du crosstalk par les formes fermées de Quegan (1994).

    Un jeu `(u, v, w, z, alpha)` par patch, tiré de l'unique covariance du patch
    (`covariance.polarimetric_covariance`).

    Args:
        tol: seuil de singularité de D, transmis à la correction.
        delta_tol: seuil relatif sur `|Delta| / (C11 C44)`.
        x_tol: seuil relatif sur `|X| / sqrt(C22 C33)`.
    """

    name = "quegan"

    def __init__(self, tol: float = DEFAULT_TOL, delta_tol: float = DEFAULT_DELTA_TOL, x_tol: float = DEFAULT_X_TOL) -> None:
        super().__init__(tol=tol)
        self.delta_tol = float(delta_tol) 
        self.x_tol = float(x_tol)

    # -----------------------------Formes Fermées----------------------------------------------

    def _solve(self, cov: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Résout les formes fermées sur une covariance.

        Args: cov: `(4, 4)` complex128.

        Returns: (theta, rapports) où :
            -theta : `(5,)` dans l'ordre `(u, v, w, z, alpha)`, `nan` là où un dénominateur s'est annulé ;
            -rapports : les critères de conditionnement, sans dimension, que `estimate` compare aux seuils.
        """
        c11, c13, c14 = cov[0, 0], cov[0, 2], cov[0, 3]
        c21, c22, c23, c24 = cov[1, 0], cov[1, 1], cov[1, 2], cov[1, 3]
        c31, c33, c34 = cov[2, 0], cov[2, 2], cov[2, 3]
        c41, c43, c44 = cov[3, 0], cov[3, 2], cov[3, 3]

        with np.errstate(divide="ignore", invalid="ignore"):
            #np.errstate(divide="ignore") permet de supprimer les avertissements de division par zéro / invalid="ignore" permet de supprimer les avertissements de calculs invalides, comme la division par zéro ou la racine carrée d'un nombre négatif
            
            # ---------------------------------Crosstalk--------------------------------------------------
            # Babu éq. 14, Fore éq. 24. C étant hermitienne, |C14|^2 = C14 C41 : les
            # deux écritures sont équivalentes, la forme en module garantit un Delta réel.
            delta = c11 * c44 - np.abs(c14) ** 2
            rapport_delta = np.abs(delta) / np.abs(c11 * c44)
            u = (c44 * c31 - c41 * c34) / delta
            v = (c11 * c34 - c31 * c14) / delta
            z = (c44 * c21 - c41 * c24) / delta
            w = (c11 * c24 - c21 * c14) / delta

            # ------------------- Déséquilibre croisé - (Babu, équations 15 à 18 ; Fore, équations 25-26) -------------------
            x = c23 - z * c13 - w * c43
            rapport_x = np.abs(x) / np.sqrt(np.abs(c22 * c33))
            alpha1 = (c33 - u * c13 - v * c43) / x

            denominateur_2 = c22 - np.conj(z) * c21 - np.conj(w) * c24
            rapport_denom2 = np.abs(denominateur_2) / np.abs(c22)
            alpha2 = np.conj(x) / denominateur_2

            # Le module suit Babu éq. 15 (`4|alpha2|^2` sous la racine, là où le
            # texte de Fore porte la faute `4 + |alpha2|^2`). La phase suit Fore
            # éq. 25, qui multiplie par `alpha1 / |alpha1|` : Babu omet ce facteur
            # et rend un alpha réel, ce qui laisserait après correction un
            # déphasage `arg(alpha)` entre HV et VH. `exp(i angle)` plutôt
            # qu'une division : `angle(0)` vaut 0, sans produire de `nan`.
            produit = np.abs(alpha1 * alpha2)
            module = (
                produit - 1.0 + np.sqrt((produit - 1.0) ** 2 + 4.0 * np.abs(alpha2) ** 2)
            ) / (2.0 * np.abs(alpha2))
            alpha = module * np.exp(1j * np.angle(alpha1))

        rapports = {
            "Delta / (C11 C44)": float(rapport_delta),
            "X / sqrt(C22 C33)": float(rapport_x),
            "(C22 - z* C21 - w* C24) / C22": float(rapport_denom2),
        }
        return np.array([u, v, w, z, alpha]), rapports

    # -----------------------------Interface-------------------------------------------
    #On override la méthode estimate de la classe mère CrosstalkEstimator pour implémenter l'estimation du crosstalk selon Quegan. Cette méthode prend en entrée un patch d'observation et renvoie les paramètres de crosstalk estimés.
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """
        Estime un jeu `(u, v, w, z, alpha)` pour ce patch.

        Raises:
            IllConditionedCovarianceError: si la covariance est dégénérée. Le
                message nomme le critère le plus dégradé.
        """
        self.check_patch(patch)
        theta, rapports = self._solve(polarimetric_covariance(patch))

        seuils = {
            "Delta / (C11 C44)": self.delta_tol,
            "X / sqrt(C22 C33)": self.x_tol,
            "(C22 - z* C21 - w* C24) / C22": self.x_tol,
        }
        # `isfinite` n'est pas redondant avec les seuils : `nan > tol` est faux,
        # mais `inf > tol` est vrai, et un critère infini n'en est pas moins un
        # dénominateur annulé.
        defaillants = [nom for nom, valeur in rapports.items() if not (np.isfinite(valeur) and valeur > seuils[nom])]
        if defaillants:
            # Un critère non fini — canal éteint, patch constant — est le plus
            # grave. Mais `nan` perd toute comparaison (`nan < x` et `x < nan`
            # sont faux) : un `min` naïf ne le choisirait jamais et nommerait un
            # critère sain dans le message. Il compte donc pour `-inf`.
            nom = min(defaillants, key=lambda cle: rapports[cle] if np.isfinite(rapports[cle]) else -np.inf)
            raise IllConditionedCovarianceError(nom, rapports[nom], seuils[nom])
        return CrosstalkParameters.from_vector(theta)
