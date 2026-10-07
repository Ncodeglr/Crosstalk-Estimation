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
la corrélation HH-VV. Sur la pile Rosamond (`beta` médian 0,134), il est de
l'ordre de 30 % du crosstalk. Voir le README, « Le biais propre de Quegan ».
"""

from __future__ import annotations
import logging
import numpy as np
from ..covariance import WindowSpec, polarimetric_covariance
from ..exceptions import IllConditionedCovarianceError
from ..parameters import CrosstalkParameters
from ..physics import DEFAULT_TOL
from .base import CrosstalkEstimator
from .registry import register_estimator

logger = logging.getLogger(__name__)

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

    Args:
        window: 
        -"full" : le patch entier est une unique fenêtre d'estimation, un jeu par patch ; c'est le mode de la comparaison réseau contre baselines. Ou un couple `(kh, kw)` 
        -"fenêtre 5x5" : un jeu par fenêtre puis moyenne reproduit Babu.
        
        tol: seuil de singularité de D, transmis à la correction.
        delta_tol: seuil relatif sur `|Delta| / (C11 C44)`.
        x_tol: seuil relatif sur `|X| / sqrt(C22 C33)`.

    Raises: ValueError: si `window` est invalide.
    """

    name = "quegan"

    def __init__(self,window: WindowSpec = "full",tol: float = DEFAULT_TOL,delta_tol: float = DEFAULT_DELTA_TOL,x_tol: float = DEFAULT_X_TOL) -> None:
        super().__init__(tol=tol)
        if window != "full":
            window = tuple(window)
            if len(window) != 2:
                raise ValueError(f"window = {window!r} : 'full' ou (kh, kw) attendu")
        self.window = window
        self.delta_tol = float(delta_tol) 
        self.x_tol = float(x_tol)

    # -----------------------------Formes Fermées----------------------------------------------

    def _solve(self, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Résout les formes fermées, vectorisé sur les dimensions de tête.

        Args: cov: `(..., 4, 4)` complex128.

        Returns: (thetas, valides, rapports) où :
            -thetas : (..., 5) dans l'ordre `(u, v, w, z, alpha)` avec `NaN` là où un garde-fou a joué,
            -valides : tableau de booléens : une case par fenêtre d'estimation, qui dit si son theta est exploitable.
            -rapports : le dictionnaire des critères de conditionnement, pour le diagnostic.
        """
        c11, c13, c14 = cov[..., 0, 0], cov[..., 0, 2], cov[..., 0, 3]
        c21, c22, c23, c24 = (
            cov[..., 1, 0],
            cov[..., 1, 1],
            cov[..., 1, 2],
            cov[..., 1, 3],
        )
        c31, c33, c34 = cov[..., 2, 0], cov[..., 2, 2], cov[..., 2, 3]
        c41, c43, c44 = cov[..., 3, 0], cov[..., 3, 2], cov[..., 3, 3]

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
            "Delta / (C11 C44)": rapport_delta,
            "X / sqrt(C22 C33)": rapport_x,
            "(C22 - z* C21 - w* C24) / C22": rapport_denom2,
        }
        # `isfinite` n'est pas redondant avec les seuils. Un `nan` est bien
        # rejeté, puisque `nan > tol` est faux ; un `inf` ne l'est pas, car
        # `inf > tol` est vrai. Or une fenêtre exactement nulle en produit : la
        # table de sommes cumulées y laisse un résidu d'annulation de ~1e-18 au
        # lieu de zéro, si bien que `|X| / sqrt(C22 C33)` vaut `inf` et que la
        # fenêtre serait déclarée exploitable sur du bruit numérique pur. Les
        # trois rapports étant bornés par construction — `rapport_delta` vaut
        # `1 - |rho_HHVV|^2`, donc entre 0 et 1 — ce filtre ne peut écarter
        # aucune valeur légitime.
        valides = (
            np.isfinite(rapport_delta) & (rapport_delta > self.delta_tol)
            & np.isfinite(rapport_x) & (rapport_x > self.x_tol)
            & np.isfinite(rapport_denom2) & (rapport_denom2 > self.x_tol)
        )

        thetas = np.stack([u, v, w, z, alpha], axis=-1)
        thetas = np.where(valides[..., None], thetas, np.nan)
        return thetas, np.asarray(valides), rapports 

    @staticmethod
    def _pire(valeurs: np.ndarray) -> float:
        """
        Valeur la plus dégradée d'un critère, nan comptant pour la pire.

        Un critère vaut `nan` quand son dénominateur s'est annulé : canal éteint, patch constant. 
        C'est donc le cas le plus grave, mais nan perd toute comparaison (nan < x et x < nan sont tous deux faux),
        si bien qu'un `min` naïf ne le sélectionne jamais et nomme un critère sain dans le message d'erreur.
        """
        valeurs = np.asarray(valeurs)
        if not np.isfinite(valeurs).all():
            return float("-inf")
        return float(valeurs.min())

    def _raise_ill_conditioned(self, rapports: dict) -> None:
        """Lève en nommant le critère le plus dégradé."""
        nom = min(rapports, key=lambda cle: self._pire(rapports[cle]))
        valeur = np.asarray(rapports[nom])
        valeur = float(valeur[np.isfinite(valeur)].min()) if np.isfinite(valeur).any() else float("nan")
        seuil = self.delta_tol if nom.startswith("Delta") else self.x_tol
        raise IllConditionedCovarianceError(nom, valeur, seuil)

    # -----------------------------Interface-------------------------------------------
    #On override la méthode estimate de la classe mère CrosstalkEstimator pour implémenter l'estimation du crosstalk selon Quegan. Cette méthode prend en entrée un patch d'observation et renvoie les paramètres de crosstalk estimés.
    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """
        Estime un jeu `(u, v, w, z, alpha)` pour ce patch.

        En mode "full" : les formes fermées sont appliquées à l'unique covariance du patch. 
        En mode "5x5" : elles le sont à chaque fenêtre et les theta obtenus sont moyennés sur **toutes** les fenêtres — les fenêtres mal conditionnées étant écartées.

        La moyenne des (u, v, w, z) est prise sur les nombres complexes, ce qui
        est nettement le bon choix : mesuré sur 150 patchs de la pile Rosamond
        contre l'estimation plein patch, moyenner les complexes donne -1,1 dB
        de biais, moyenner les modules +6,6 dB.

        `alpha` ne se moyenne pas ainsi (`CrosstalkParameters.mean`). Sa phase varie d'environ 23° d'une
        fenêtre 5x5 à l'autre, si bien que la moyenne complexe fait s'annuler une
        partie du module. Mesuré sur 40 patchs 256x256 contre l'estimation plein
        patch (`|alpha|` = 1,205) : le module de la moyenne complexe le sous-estime
        de 5,3 %, la moyenne des modules le surestime de 3,5 % (inégalité de
        Jensen sur une quantité positive bruitée). On retient donc la **moyenne
        des modules**, qui est aussi celle de Babu puisque son `alpha` est réel,
        et la **phase de la moyenne complexe**, que Fore ajoute (éq. 25). Le
        module rendu est ainsi inchangé par l'ajout de la phase. Le mode "full"
        n'est concerné par aucun de ces biais.

        Note:
            Cette moyenne porte sur toutes les fenêtres du patch, distance comprise. 
            Ce n'est donc pas le protocole de Babu (§3.3.1), qui moyenne le long d'une seule ligne de distance : voir estimate_range_lines.

        Raises:
            IllConditionedCovarianceError: en mode `"full"` si la covariance est
                dégénérée ; en mode 5x5 si aucune fenêtre n'est exploitable.
        """
        self.check_patch(patch)
        cov = polarimetric_covariance(patch, self.window)
        thetas, valides, rapports = self._solve(cov)

        if self.window == "full":
            if not valides:
                self._raise_ill_conditioned(rapports)
            return CrosstalkParameters.from_vector(thetas)

        retenues = int(np.count_nonzero(valides))
        if retenues == 0:
            self._raise_ill_conditioned(rapports)
        if retenues < valides.size:
            logger.warning(
                "%d fenêtres sur %d écartées pour mauvais conditionnement",
                valides.size - retenues,
                valides.size,
            )
        return CrosstalkParameters.mean(thetas[valides])

    def estimate_range_lines(self, patch: np.ndarray) -> list[CrosstalkParameters]:
        """
        Un jeu par ligne de distance, à la manière de Babu.
        
        Le crosstalk varie lentement en distance et pas du tout en azimut. 
        Babu (§3.3.1) estiment donc un `theta` par fenêtre 5x5 glissante, moyennent ces `theta` **le long d'une ligne de distance**, puis corrigent toute la ligne avec ce jeu unique.
        Une ligne de distance est une colonne du patch : les `.slc` UAVSAR rangent l'azimut en lignes et la distance en colonnes.

        Args: patch: (4, H, W) complexe, en ordre POL_ORDER.

        Returns: Liste de W - kw + 1 jeux, un par position de fenêtre en distance.

        Raises:
            ValueError: si l'estimateur est en mode `"full"`.
            IllConditionedCovarianceError: si une ligne n'a aucune fenêtre exploitable.
        """
        if self.window == "full":
            raise ValueError(
                "estimate_range_lines demande une fenêtre glissante ; "
                "instanciez QueganEstimator(window=(5, 5))"
            )
        self.check_patch(patch)
        cov = polarimetric_covariance(patch, self.window)
        thetas, valides, rapports = self._solve(cov)

        lignes: list[CrosstalkParameters] = []
        for colonne in range(thetas.shape[1]):
            masque = valides[:, colonne]
            if not masque.any():
                self._raise_ill_conditioned(
                    {f"{nom} (ligne de distance {colonne})": val[:, colonne]
                     for nom, val in rapports.items()}
                )
            # Même règle que `estimate` : alpha en module moyen et phase de la
            # moyenne complexe. La moyenne complexe directe sous-estimerait
            # |alpha| de 8 % sur le patch de contrôle du README.
            lignes.append(CrosstalkParameters.mean(thetas[:, colonne][masque]))
        return lignes

    def __repr__(self) -> str:
        return f"QueganEstimator(window={self.window!r}, tol={self.tol:g})"
