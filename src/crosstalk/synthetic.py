"""
Signal vrai synthétique conforme aux hypothèses de Quegan : un outil de test, hors chaîne.

Rôle
----
Vérifier les formules d'un estimateur à theta connu : on part d'un signal vrai S synthétique, on lui
applique un theta connu avec `matrix.apply_distortion`, puis on compare l'estimation à ce theta. Ce
module ne sert qu'aux tests : il ne produit ni données d'entraînement ni données d'évaluation. Il ne
fabrique que S : il ne sait rien du crosstalk.

Modèle
------
S suit la covariance vraie C_Q supposée par Quegan, Fore 2015, éq. 17 (en POL_ORDER) :

              | sigma_hhhh    0      0     sigma_hhvv |
        C_Q = |     0        beta   beta       0      |
              |     0        beta   beta       0      |
              | sigma_hhvv*   0      0     sigma_vvvv |

C'est en 4x4 la matrice 3x3 de Quegan 1994, équation 3, qui découle de ses deux hypothèses sur la cible :
- hypothèse 2, réciprocité : S_HV = S_VH, d'où le bloc croisé constant égal à beta ;
- hypothèse 3, symétrie de réflexion : <S_HH S_HV*> = <S_VV S_HV*> = 0, d'où les blocs nuls. Sur les
  données observées O, ces termes ne sont plus nuls : ils valent le crosstalk, et c'est ce que les
  formes fermées y lisent.
  
La loi du signal est gaussienne circulaire complexe, le modèle classique du chatoiement d'une scène
homogène (Quegan 1994, §III : « multivariate gaussian model »).

Contenu
-------
- `ScatteringParameters` : une scène, les 4 paramètres de C_Q, validés à la création ; `to_covariance()`
  rend C_Q, `correlation_hhvv` la corrélation HH-VV.
- `sample_scattering` : un patch S de covariance attendue C_Q. C_Q étant de rang 3 (HV et VH sont
  confondus), la factorisation de Cholesky échoue : le tirage est construit bloc par bloc, ce qui rend
  la réciprocité exacte au bit près.

Usage
-----
    rng = np.random.default_rng(graine)
    s = sample_scattering(ScatteringParameters(...), (64, 64), rng)   # signal vrai
    o = apply_distortion(s, theta)                                    # patch observé, de theta connu

Conventions
-----------
- Ordre des canaux : POL_ORDER = (HH, HV, VH, VV), redéclaré ici par les indices IDX_* pour que ce module
  reste indépendant du modèle qu'il sert à tester ; tests/test_synthetic.py verrouille leur cohérence.
- Covariance : C_ij = <S_i S_j*>.
- Reproductibilité : le générateur aléatoire est toujours passé explicitement, jamais créé en interne.
- Calcul en complex128, conversion au type demandé en sortie.

Limites
-------
Ce générateur produit exactement le cadre de Quegan, et rien de plus. Il ne modélise pas :
- le bruit thermique, qui pèse surtout sur les canaux croisés, plus faibles (Quegan 1994, éq. 31-35) ;
- une texture non gaussienne (modèle K) ni les cibles brillantes ;
- l'hétérogénéité : tout le patch suit une seule C_Q ;
- les violations des hypothèses : corrélation co-pol/cross-pol non nulle, réciprocité imparfaite.
Sur ces données, Quegan n'est limité que par son approximation au premier ordre et par le bruit
d'estimation : un résultat obtenu ici ne se transpose pas tel quel aux données réelles.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

IDX_HH, IDX_HV, IDX_VH, IDX_VV = 0, 1, 2, 3


@dataclass(frozen=True, slots=True)
class ScatteringParameters:
    """
    Paramètres de la covariance vraie C_Q supposée par Quegan (Fore 2015, éq. 17 ; Quegan 1994, éq. 3).
    Jeu immuable, validé à la création : un jeu qui ne définit pas une covariance utilisable ne peut pas exister.
    
    frozen=True : immuable, hashable, utilisable comme clé de dictionnaire.
    slots=True : pas d'attributs arbitraires, moins de mémoire.
    """

    sigma_hhhh: float = 1.0 #Puissance du canal HH, <|S_HH|^2> (sigma_11 chez Quegan).
    sigma_vvvv: float = 1.0 #Puissance du canal VV, <|S_VV|^2> (sigma_22 chez Quegan).
    sigma_hhvv: complex = 0.6 #Covariance complexe entre HH et VV, <S_HH S_VV*> (rho chez Quegan).
    beta: float = 0.05 #Puissance des canaux croisés, <|S_HV|^2> = <|S_VH|^2> (sigma_21 chez Quegan).

    def __post_init__(self) -> None:
        # Types normalisés, comme dans CrosstalkParameters : un scalaire numpy devient un float ou un complex.
        object.__setattr__(self, "sigma_hhhh", float(self.sigma_hhhh))
        object.__setattr__(self, "sigma_vvvv", float(self.sigma_vvvv))
        object.__setattr__(self, "sigma_hhvv", complex(self.sigma_hhvv))
        object.__setattr__(self, "beta", float(self.beta))
        
        if min(self.sigma_hhhh, self.sigma_vvvv, self.beta) < 0.0:
            raise ValueError("les puissances doivent être positives")
        if abs(self.sigma_hhvv) ** 2 >= self.sigma_hhhh * self.sigma_vvvv:
            raise ValueError(
                f"|sigma_hhvv|^2 = {abs(self.sigma_hhvv) ** 2:.3g} doit être < sigma_hhhh * sigma_vvvv = "
                f"{self.sigma_hhhh * self.sigma_vvvv:.3g} "
                f"(au-delà : Cauchy-Schwarz violée, pas de covariance ; à l'égalité : Delta = C11 C44 - |C14|^2 = 0 (only if the copolarized channels are perfectly correlated), Quegan indéfini)"
            )

    @property #Property : méthode sans argument, accessible comme un attribut. Ici : params.correlation_hhvv
    def correlation_hhvv(self) -> complex:
        """
        Coefficient de corrélation complexe entre HH et VV : sigma_hhvv / sqrt(sigma_hhhh sigma_vvvv).

        Son module est < 1 (garanti par la validation) et sa phase est la différence de phase moyenne
        entre HH et VV. Il est quasi invariant par crosstalk : invariant au premier ordre (Quegan 1994,
        éq. 40), il se mesure donc sur les données observées, |C14| / sqrt(C11 C44). L'écart, du second
        ordre, dépend de la scène : à -20 dB, de 0,1 % à 1,3 % du module et environ 1,5° de phase sur les
        scènes testées ; dix fois moins à -30 dB.

        La phase renseigne sur le mécanisme de diffusion dominant : environ 0° pour une réflexion simple
        (surface, trièdre), environ 180° pour un double rebond (dièdre). Fore 2015, légende de la fig. 5 :
        « the nearly 180° phase offset between the HH and VV channels, which makes the uncalibrated
        signatures look like those of a dihedral rather than a trihedral ». Babu 2022 (§4.3) associe de
        même le trièdre à la réflexion simple (T11) et le double rebond au dièdre (T22).
        """
        return self.sigma_hhvv / np.sqrt(self.sigma_hhhh * self.sigma_vvvv)

    def to_covariance(self) -> np.ndarray:
        """
        Construit la covariance vraie C_Q de Fore 2015, éq. 17, en POL_ORDER.

        Paramètres utilisés, les attributs de l'objet :
            sigma_hhhh: puissance du canal HH, <|S_HH|^2>.
            sigma_vvvv: puissance du canal VV, <|S_VV|^2>.
            sigma_hhvv: covariance complexe entre HH et VV, <S_HH S_VV*>.
            beta: puissance des canaux croisés, <|S_HV|^2> = <|S_VH|^2>.

        Returns: (4, 4) complex128, en POL_ORDER.
        """
        cov = np.zeros((4, 4), dtype=np.complex128)
        cov[IDX_HH, IDX_HH] = self.sigma_hhhh
        cov[IDX_VV, IDX_VV] = self.sigma_vvvv
        cov[IDX_HH, IDX_VV] = self.sigma_hhvv
        cov[IDX_VV, IDX_HH] = np.conj(self.sigma_hhvv)
        for i in (IDX_VH, IDX_HV):
            for j in (IDX_VH, IDX_HV):
                cov[i, j] = self.beta
        return cov


def sample_scattering(params: ScatteringParameters,shape: tuple[int, ...],rng: np.random.Generator,dtype: np.dtype | type = np.complex128,) -> np.ndarray:
    """
    Tire un patch de signal vrai S, gaussien circulaire, de covariance attendue params.to_covariance().

    La covariance mesurée sur un patch fini fluctue autour de C_Q, d'environ 1/sqrt(N) en relatif pour
    N pixels (~1,6 % sur 64x64) : c'est le bruit d'estimation réel, que subissent tous les estimateurs.

    Args:
        params: paramètres de la scène, déjà validés à leur création.
        shape: forme spatiale, p. ex. (64, 64).
        rng: générateur aléatoire, obligatoire : sa graine fixe le tirage, qui est donc reproductible.
        dtype: le calcul se fait en complex128.

    Returns:
        (4, *shape) en POL_ORDER, avec S_HV = S_VH exactement (réciprocité, Quegan hypothèse 2) et une
        corrélation co-pol/cross-pol nulle en espérance (symétrie de réflexion, Quegan hypothèse 3).
    """
    # Tirage circulaire : chaque canal est un bruit blanc gaussien complexe circulaire, de puissance 1.
    def circulaire() -> np.ndarray:
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0) #renvoie donc une image complexe de taille shape. 

    
    # Bloc co-polaire : HH et VV de puissance 1, corrélés par rho = sigma_hhvv / sqrt(sigma_hhhh sigma_vvvv),
    # puis mis à l'échelle plus bas : <S_HH S_VV*> vaut alors rho sqrt(sigma_hhhh sigma_vvvv) = sigma_hhvv.
    # La conjugaison est imposée par la convention C_ij = <S'_i S'_j*> : pour obtenir <hh vv*> = rho,
    # il faut construire VV avec conj(rho).
    rho = params.correlation_hhvv
    hh = circulaire()
    vv = np.conj(rho) * hh + np.sqrt(1.0 - abs(rho) ** 2) * circulaire()

    # Bloc croisé : un unique tirage, partagé par VH et HV (réciprocité stricte).
    croise = circulaire()

    signal = np.empty((4, *shape), dtype=np.complex128)
    signal[IDX_HH] = np.sqrt(params.sigma_hhhh) * hh
    signal[IDX_VV] = np.sqrt(params.sigma_vvvv) * vv
    signal[IDX_VH] = np.sqrt(params.beta) * croise
    signal[IDX_HV] = signal[IDX_VH]
    return signal.astype(dtype, copy=False)
