"""Méthode de Quegan : estimation du crosstalk en forme fermée.

Les quatre termes de crosstalk se lisent directement dans la covariance du
signal observé — Fore et al. (2015, équation 24) réindexée, ce qui coïncide
avec l'équation 13 de Babu et al. (2022), la seule qu'il ait lui-même
réindexée — avec `Delta = C11 C44 - |C14|^2` ::

    u = (C44 C31 - C41 C34) / Delta        v = (C11 C34 - C31 C14) / Delta
    z = (C44 C21 - C41 C24) / Delta        w = (C11 C24 - C21 C14) / Delta

puis le déséquilibre croisé (Fore et al., équations 25 et 26, réindexées ;
les équations 16 à 18 de Babu et al. sont celles de Fore **non** réindexées et
contredisent donc sa propre équation 13) ::

    X      = C23 - z C13 - w C43
    alpha1 = (C33 - u C13 - v C43) / X
    alpha2 = X* / (C22 - z* C21 - w* C24)

             |alpha1 alpha2| - 1 + sqrt( (|alpha1 alpha2| - 1)^2 + 4 |alpha2|^2 )
    alpha = -------------------------------------------------------------------
                                     2 |alpha2|

Le terme `4 |alpha2|^2` sous la racine vient de l'équation 15 de Babu et al. ;
le texte de Fore imprime `4 + |alpha2|^2`. Sur un `alpha` vrai de 1,3000, la
première forme rend 1,2990 et la seconde 1,2199 — c'est le seul endroit où
Babu corrige Fore. L'identité `x - 1/x = |alpha1| - 1/|alpha2|`, vérifiée par
`test_l_estimateur_de_alpha_resout_l_identite_attendue`, verrouille ce choix.

Indexation dans l'ordre `POL_ORDER = [HH, HV, VH, VV]`, indices 1 à 4 des
articles ; `C_ij = <O_i O_j*>`. Attention : ces expressions ne sont **pas**
celles de Fore et al. (2015, équations 24 à 26), qui indexent `[HH, VH, HV, VV]`.
Appliquer les leurs à des données rangées ainsi rend `(z, w, v, u)` au lieu de
`(u, v, w, z)`, sans lever la moindre erreur.

`alpha` est réel positif : l'équation 15 de Babu s'arrête au module. Fore et al.
(équation 25) le multiplient en plus par `alpha1 / |alpha1|` ; sur la pile Rosamond de ce
projet cette phase vaut 1,6° en médiane (étendue -3,7° à +15,2° sur 40 patchs
256x256), le produit JPL 2016 étant déjà étalonné en phase. Elle n'entre dans
aucune des métriques du paquet, qui ne portent que sur `(u, v, w, z)`.

Domaine de validité. Ces expressions sont dérivées en négligeant les termes
d'ordre deux et plus en `(u, v, w, z)` (Fore et al., 2015, après l'équation 24,
énonce explicitement cette condition). L'estimateur est exact au premier ordre
seulement : même sans bruit et avec un nombre infini d'échantillons, il subsiste
un biais en `O(|theta|^2)`, soit de l'ordre du pour-cent à -20 dB.
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
#: `|Delta|/(C11 C44)` descend à 0,080 et `|X|/sqrt(C22 C33)` à 0,027. Les
#: valeurs par défaut sont très en deçà : elles ne se déclenchent que sur une
#: dégénérescence vraie (patch constant, canal éteint), pas sur une scène
#: difficile.
DEFAULT_DELTA_TOL: float = 1e-6
DEFAULT_X_TOL: float = 1e-6


@register_estimator
class QueganEstimator(CrosstalkEstimator):
    """Estimation du crosstalk par les formes fermées de Quegan (1994).

    Args:
        window: `"full"` — le patch entier est une unique fenêtre d'estimation,
            un jeu par patch ; c'est le mode de la comparaison réseau contre
            baselines. Ou un couple `(kh, kw)` — fenêtre glissante, un jeu par
            fenêtre puis moyenne ; `(5, 5)` reproduit Babu et al. (2022).
        tol: seuil de singularité de D, transmis à la correction.
        delta_tol: seuil relatif sur `|Delta| / (C11 C44)`.
        x_tol: seuil relatif sur `|X| / sqrt(C22 C33)`.

    Raises:
        ValueError: si `window` est invalide.
    """

    name = "quegan"

    def __init__(
        self,
        window: WindowSpec = "full",
        tol: float = DEFAULT_TOL,
        delta_tol: float = DEFAULT_DELTA_TOL,
        x_tol: float = DEFAULT_X_TOL,
    ) -> None:
        super().__init__(tol=tol)
        if window != "full":
            window = tuple(window)
            if len(window) != 2:
                raise ValueError(f"window = {window!r} : 'full' ou (kh, kw) attendu")
        self.window = window
        self.delta_tol = float(delta_tol)
        self.x_tol = float(x_tol)

    # ------------------------------------------------------------ formes fermées

    def _solve(self, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Résout les formes fermées, vectorisé sur les dimensions de tête.

        Args:
            cov: `(..., 4, 4)` complex128.

        Returns:
            `(thetas, valides, rapports)` où `thetas` est `(..., 5)` dans
            l'ordre `(u, v, w, z, alpha)` avec `NaN` là où un garde-fou a joué,
            `valides` est le masque booléen `(...)`, et `rapports` le
            dictionnaire des critères de conditionnement, pour le diagnostic.
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
            # --- crosstalk, équation 13 -------------------------------------
            delta = c11 * c44 - np.abs(c14) ** 2
            rapport_delta = np.abs(delta) / np.abs(c11 * c44)
            u = (c44 * c31 - c41 * c34) / delta
            v = (c11 * c34 - c31 * c14) / delta
            z = (c44 * c21 - c41 * c24) / delta
            w = (c11 * c24 - c21 * c14) / delta

            # --- déséquilibre croisé, équations 15 à 18 ---------------------
            x = c23 - z * c13 - w * c43
            rapport_x = np.abs(x) / np.sqrt(np.abs(c22 * c33))
            alpha1 = (c33 - u * c13 - v * c43) / x

            denominateur_2 = c22 - np.conj(z) * c21 - np.conj(w) * c24
            rapport_denom2 = np.abs(denominateur_2) / np.abs(c22)
            alpha2 = np.conj(x) / denominateur_2

            produit = np.abs(alpha1 * alpha2)
            alpha = ((
                produit - 1.0 + np.sqrt((produit - 1.0) ** 2 + 4.0 * np.abs(alpha2) ** 2)
            ) / (2.0 * np.abs(alpha2))).astype(np.complex128)

        rapports = {
            "Delta / (C11 C44)": rapport_delta,
            "X / sqrt(C22 C33)": rapport_x,
            "(C22 - z* C21 - w* C24) / C22": rapport_denom2,
        }
        valides = (
            (rapport_delta > self.delta_tol)
            & (rapport_x > self.x_tol)
            & (rapport_denom2 > self.x_tol)
            # `|alpha2| > 0` découle des deux critères précédents, et `alpha1`
            # n'apparaît que dans le produit `|alpha1 alpha2|` : aucun critère
            # supplémentaire n'est nécessaire.
        )

        thetas = np.stack([u, v, w, z, alpha], axis=-1)
        thetas = np.where(valides[..., None], thetas, np.nan)
        return thetas, np.asarray(valides), rapports

    def _raise_ill_conditioned(self, rapports: dict) -> None:
        """Lève en nommant le critère le plus dégradé."""
        nom, valeur = min(
            ((nom, float(np.min(val))) for nom, val in rapports.items()),
            key=lambda couple: couple[1],
        )
        seuil = self.delta_tol if nom.startswith("Delta") else self.x_tol
        raise IllConditionedCovarianceError(nom, valeur, seuil)

    # ---------------------------------------------------------------- interface

    def estimate(self, patch: np.ndarray) -> CrosstalkParameters:
        """Estime un jeu `(u, v, w, z, alpha)` pour ce patch.

        En mode `"full"`, les formes fermées sont appliquées à l'unique
        covariance du patch. En mode glissant, elles le sont à chaque fenêtre et
        les `theta` obtenus sont moyennés sur **toutes** les fenêtres — les
        fenêtres mal conditionnées étant écartées.

        La moyenne des `(u, v, w, z)` est prise sur les nombres complexes, ce qui
        est nettement le bon choix : mesuré sur 150 patchs de la pile Rosamond
        contre l'estimation plein patch, moyenner les complexes donne -1,1 dB
        de biais, moyenner les modules +6,6 dB. `alpha` étant réel positif,
        sa moyenne est arithmétique ; à 25 échantillons par fenêtre elle
        surestime `|alpha|` de **2,4 %** environ par rapport au plein patch
        (même mesure, 150 patchs 64x64), l'inégalité de Jensen jouant sur une
        quantité positive bruitée. Le mode `"full"` n'en souffre pas.

        Note:
            Cette moyenne porte sur **toutes** les fenêtres du patch, distance
            comprise. Ce n'est donc pas le protocole de Babu et al. (2022,
            §3.3.1), qui moyenne le long d'une seule ligne de distance : voir
            `estimate_range_lines`.

        Raises:
            IllConditionedCovarianceError: en mode `"full"` si la covariance est
                dégénérée ; en mode glissant si aucune fenêtre n'est exploitable.
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
        return CrosstalkParameters.from_vector(
            thetas[valides].mean(axis=0)
        )

    def estimate_range_lines(self, patch: np.ndarray) -> list[CrosstalkParameters]:
        """Un jeu par ligne de distance, à la manière de Babu et al. (2022).

        Le crosstalk varie lentement en distance et pas du tout en azimut. Babu
        et al. (2022, §3.3.1) estiment donc un `theta` par fenêtre 5x5 glissante,
        moyennent ces `theta` **le long d'une ligne de distance**, puis corrigent
        toute la ligne avec ce jeu unique.

        Une ligne de distance est une colonne du patch : les `.slc` UAVSAR
        rangent l'azimut en lignes et la distance en colonnes.

        Args:
            patch: `(4, H, W)` complexe, en ordre `POL_ORDER`.

        Returns:
            Liste de `W - kw + 1` jeux, un par position de fenêtre en distance.

        Raises:
            ValueError: si l'estimateur est en mode `"full"`.
            IllConditionedCovarianceError: si une ligne n'a aucune fenêtre
                exploitable.
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
            lignes.append(
                CrosstalkParameters.from_vector(
                    thetas[:, colonne][masque].mean(axis=0)
                )
            )
        return lignes

    def __repr__(self) -> str:
        return f"QueganEstimator(window={self.window!r}, tol={self.tol:g})"
