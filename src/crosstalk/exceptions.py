"""
La correction inverse la matrice de distorsion D, dont le déterminant peut s'annuler.

Plutôt que de laisser NumPy produire des inf qui se propagent silencieusement jusqu'aux
métriques, le garde-fou lève une exception qui rapporte la valeur fautive.
Les formes fermées de Quegan, elles, ne sont pas gardées : voir la docstring de `estimators.quegan`.
"""

from __future__ import annotations

class CrosstalkError(Exception):
    """Classe de base de toutes les erreurs"""

class SingularDmatrixError(CrosstalkError):
    """
    D est inversible si et seulement si (uw - 1)(vz - 1) != 0 : 
    c'est le dénominateur de Sigma (Babu, équation 12).
    Le crosstalk réel étant de l'ordre de -20 dB, le produit uw vaut ~1e-2 et le déterminant reste proche de 1 : atteindre cette exception signale une estimation aberrante, pas une limite du modèle.
    """

    def __init__(self, determinant: complex, tol: float) -> None:
        self.determinant = determinant
        self.tol = tol
        super().__init__(
            f"matrice de distorsion singulière : |(uw-1)(vz-1)| = "f"{abs(determinant):.3e} < tol = {tol:.3e}"
            )