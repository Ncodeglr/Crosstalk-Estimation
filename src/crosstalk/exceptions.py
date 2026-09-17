"""Exceptions du paquet `crosstalk`.

Les estimateurs de crosstalk manipulent des quotients dont les dénominateurs
s'annulent sur des configurations dégénérées (scène à un seul mécanisme de
diffusion, patch homogène, covariance de rang déficient). Plutôt que de laisser
NumPy produire des `inf` ou des `nan` qui se propagent silencieusement jusqu'aux
métriques, chaque garde-fou lève l'une de ces exceptions en rapportant la valeur
fautive.
"""

from __future__ import annotations


class CrosstalkError(Exception):
    """Classe de base de toutes les erreurs du paquet."""


class DegenerateDistortionError(CrosstalkError):
    """La matrice de distorsion D n'est pas inversible numériquement.

    D est inversible si et seulement si `(uw - 1)(vz - 1) != 0` (Fore et al.,2015, équation 15).
    Le crosstalk réel étant de l'ordre de -20 dB, le produit `uw` vaut ~1e-2 et le déterminant reste proche de 1 : atteindre cette exception signale une estimation aberrante, pas une limite du modèle.
    """

    def __init__(self, determinant: complex, tol: float) -> None:
        self.determinant = determinant
        self.tol = tol
        super().__init__(
            f"matrice de distorsion singulière : |(uw-1)(vz-1)| = "
            f"{abs(determinant):.3e} < tol = {tol:.3e}"
        )


class IllConditionedCovarianceError(CrosstalkError):
    """Une quantité dérivée de la covariance observée s'annule.

    Concerne les dénominateurs des formes fermées de Quegan : `Delta` (Fore et
    al., 2015, après l'équation 24) et `X` (équation 26).
    """

    def __init__(self, name: str, value: complex, tol: float) -> None:
        self.name = name
        self.value = value
        self.tol = tol
        super().__init__(
            f"covariance mal conditionnée : |{name}| = {abs(value):.3e} "
            f"< tol = {tol:.3e}"
        )
