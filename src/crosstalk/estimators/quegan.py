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
la corrélation HH-VV. Sur la pile Rosamond (`beta` médian 0,119), il est de
l'ordre de 30 % du crosstalk.

Aucune validation de la covariance : sur un patch dégénéré (patch constant ou
nul, canal éteint, HH et VV parfaitement corrélés), les dénominateurs `Delta`
ou `X` s'annulent et le jeu rendu contient des `nan`, ou des valeurs finies
dépourvues de sens. Aucun des 73 765 patchs de test de la pile Rosamond n'est
dans ce cas.
"""

from __future__ import annotations
from typing import override
import numpy as np
from ..covariance import polarimetric_covariance
from ..parameters import CrosstalkParameters
from .estimator import CrosstalkEstimator
from .registry import register_estimator


@register_estimator
class QueganEstimator(CrosstalkEstimator):
    """
    Estimation du crosstalk par les formes fermées de Quegan (1994).
    Un jeu `(u, v, w, z, alpha)` par patch, tiré de l'unique covariance du patch (`covariance.polarimetric_covariance`).
    """

    name = "quegan"

    # -----------------------------Formes Fermées----------------------------------------------

    def solve(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Résout les formes fermées sur une covariance.

        Args: cov_matrix: `(4, 4)` complex128.
        Returns: theta, `(5,)` dans l'ordre `(u, v, w, z, alpha)`, `nan` là où un dénominateur s'est annulé.
        """
        c11, c13, c14 = cov_matrix[0, 0], cov_matrix[0, 2], cov_matrix[0, 3]
        c21, c22, c23, c24 = cov_matrix[1, 0], cov_matrix[1, 1], cov_matrix[1, 2], cov_matrix[1, 3]
        c31, c33, c34 = cov_matrix[2, 0], cov_matrix[2, 2], cov_matrix[2, 3]
        c41, c43, c44 = cov_matrix[3, 0], cov_matrix[3, 2], cov_matrix[3, 3]

        # ---------------------------------Crosstalk--------------------------------------------------
        #Babu éq. 14, Fore éq. 24. C étant hermitienne, |C14|^2 = C14 C41 : les deux écritures sont équivalentes, la forme en module garantit un Delta réel.
        delta = c11 * c44 - np.abs(c14) ** 2
        u =(c44 * c31 - c41 * c34) / delta
        v = (c11 * c34 - c31 * c14) / delta
        z = (c44 * c21 - c41 * c24) / delta
        w = (c11 * c24 - c21 * c14) / delta

        # ------------------- Déséquilibre croisé - (Babu, équations 15 à 18 ; Fore, équations 25-26) -------------------
        x = c23 - z * c13 - w * c43
        alpha1 = (c33 - u * c13 - v * c43) / x

        denominateur_2 = c22 - np.conj(z) * c21 - np.conj(w) * c24
        alpha2 =np.conj(x) / denominateur_2

        # Le module suit Babu éq. 15 (`4|alpha2|^2` sous la racine, là où le
        # texte de Fore porte la faute `4 + |alpha2|^2`). La phase suit Fore
        # éq. 25, qui multiplie par `alpha1 / |alpha1|` : Babu omet ce facteur
        # et rend un alpha réel, ce qui laisserait après correction un
        # déphasage `arg(alpha)` entre HV et VH. `exp(i angle)` plutôt
        # qu'une division : `angle(0)` vaut 0, sans produire de `nan`.
        produit = np.abs(alpha1 * alpha2)
        module = (produit - 1.0 + np.sqrt((produit - 1.0) ** 2 + 4.0 * np.abs(alpha2) ** 2)) / (2.0 * np.abs(alpha2))
        alpha = module * np.exp(1j * np.angle(alpha1))
        
        return np.array([u, v, w, z, alpha])


    @override
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """
        Estime un jeu `(u, v, w, z, alpha)` pour ce patch.
        """
        self.check_patch(patch)
        return CrosstalkParameters.from_vector(self.solve(polarimetric_covariance(patch)))
