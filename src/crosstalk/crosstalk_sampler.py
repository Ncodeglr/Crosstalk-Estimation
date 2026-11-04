"""
Loi d'injection du banc d'essai : le crosstalk connu theta_k qu'on ajoute à des patchs réels.

Rôle
----
Les données réelles n'ont pas de ground truth : leur crosstalk instrumental theta_0 est inconnu. Le banc d'essai en
crée une sans fabriquer de scène : on applique à un patch réel O = D(theta_0) S une distorsion connue D(theta_k), puis
on demande aux estimateurs de retrouver theta_k. Le patch reste réel (texture, corrélation spatiale, bruit, violations
des hypothèses de Quegan) ; seule la cible est connue, exactement.

Ce que la cible représente
--------------------------
Le patch injecté vaut D(theta_k) D(theta_0) S : theta_k est le crosstalk AJOUTÉ par-dessus celui de l'instrument, pas
le crosstalk total. Composer deux distorsions redonne une distorsion de même forme, au second ordre près, ce qui rend
la cible bien définie. Un estimateur entraîné à retrouver theta_k mesure donc un écart au crosstalk typique des
données : sur un patch brut, il doit rendre environ 0. Il ne calibre pas les données à lui seul.


L'injection elle-même est matrix.apply_distortion(patch, theta_k).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .parameters import CrosstalkParameters

#: theta_k neutre : aucun crosstalk ajouté, aucun déséquilibre. D(NEUTRE) est l'identité.
NEUTRE: CrosstalkParameters = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CrosstalkSampler:
    """
    Loi de theta_k, et son tirage (generate). Chaque tirage est, avec la probabilité no_injection_rate, le theta_k
    NEUTRE ; sinon :
        niveau commun L : uniforme en dB dans [crosstalk_db[0] + term_spread_db, crosstalk_db[1] - term_spread_db] ;
        |u|, |v|, |w|, |z| : L + un écart uniforme dans [-term_spread_db, +term_spread_db] dB, tiré pour chaque
            terme séparément, si bien que les quatre termes restent dans l'enveloppe crosstalk_db ;
        phases de u, v, w, z : uniformes sur [0, 360°[ ;
        |alpha| : uniforme en dB dans alpha_module_db ;
        arg(alpha) : uniforme en degrés dans alpha_phase_deg.

    Valeurs par défaut. « Nos mesures » désignent le theta estimé par résolution exacte des contraintes de Quegan
    sur la covariance moyenne de chaque colonne de patchs et de chaque date de la pile Rosamond (patchs 128 x 128,
    380 couples, partition d'entraînement, mesure du 2026-11-04). Ce sont des valeurs apparentes, pas une vérité terrain.

        -crosstalk_db = (-35, -10) dB, l'enveloppe des quatre termes.
            Le haut couvre le crosstalk de l'instrument : au plus -15 dB environ chez Fore et al. (2015, fig. 7) sur UAVSAR ;
            rarement plus de 0,1 (-20 dB) sur MAESTRO chez Quegan (1994, après les éq. 22 à 25) ; moins de -20 dB
            pour les systèmes modernes selon Shi et al. (2020, annexe A), qui le pose sans le mesurer ; de -15,4 à
            -24,3 dB selon la date et le réflecteur à Rosamond chez Babu et al. (2022, §4.6), mesure au trièdre et non
            sur u, v, w, z. Nos mesures : le terme le plus fort va de -20,0 à -10,3 dB (médiane -16,3 dB).
            Le bas est un choix pratique, là où le banc ne mesure plus grand-chose : le MLP atteint -35,8 dB sur la
            tranche -35/-30 (validation, patchs 128 x 128, mesure du 2026-11-04) ; entre -40 et -30 dB, le MLP et le
            ResNet ne font qu'environ 3 dB de mieux que le prédicteur nul (loi (-40, -5), patchs 64 x 64).
        -term_spread_db = 4 dB, l'écart de chaque terme au niveau commun.
            Les quatre termes d'un instrument sont du même ordre de grandeur (Fore et al., 2015, fig. 7). Nos mesures :
            chaque terme s'écarte du niveau commun de -3,6 à +3,1 dB (quantiles 5 et 95 %), et de -4,9 à +4,5 dB aux
            extrêmes. Des termes indépendants dans toute l'enveloppe pourraient s'écarter de 25 dB, ce qui n'est jamais
            observé.
        -alpha_module_db = (-3, 3) dB, soit |alpha| de 0,71 à 1,41.
            C'est la plage d'injection de Shi et al. (2020, annexe A) sur des données réelles calibrées, et non une
            mesure de alpha. Elle couvre |alpha| ≈ 1 après la calibration du JPL (Fore et al., 2015, légende de la
            fig. 7), le déséquilibre croisé g = 1,128 mesuré sur UAVSAR par Babu et al. (2022, tab. 3), et nos
            mesures : |alpha| de -0,9 à +1,5 dB.
        -alpha_phase_deg = (-30, 30) degrés.
            Les données sont calibrées (Polarimetrically Calibrated = Yes) : la phase de alpha y est quasi nulle
            (Fore et al., 2015, légende de la fig. 7), le biais de phase vaut -1,14° chez Babu et al. (2022,
            tab. 3), et nos mesures vont de -5° à +18°. Nous ne reprenons donc pas les +-180° de l'injection de Shi et
            al. (2020, annexe A) : selon nous, une phase quelconque ne se justifie que pour un système non calibré,
            comme les 42,5° en bande L, 102,5° en bande C et -70° en bande P sur MAESTRO chez Quegan (1994).
        -no_injection_rate = 0 : choix d'apprentissage, pas une valeur physique. Des patchs laissés bruts mettraient
            dans la loi un pic à zéro, que le réseau apprend : sur un patch brut, le ResNet rend -58,7 dB avec un taux
            de 0,1, contre -40,3 dB avec un taux de 0, sans rien gagner sur les patchs injectés (0,03 à 0,14 dB de mieux
            avec un taux de 0, loi (-40, -5), mesure du 2026-11-04). Les patchs bruts s'évaluent donc à part.
    """

    crosstalk_db: tuple[float, float] = (-35.0, -10.0)
    term_spread_db: float = 4.0
    alpha_module_db: tuple[float, float] = (-3.0, 3.0)
    alpha_phase_deg: tuple[float, float] = (-30.0, 30.0)
    no_injection_rate: float = 0.0

    def __post_init__(self) -> None:
        for nom in ("crosstalk_db", "alpha_module_db", "alpha_phase_deg"):
            valeurs = getattr(self, nom)
            if np.ndim(valeurs) != 1 or len(valeurs) != 2:
                raise ValueError(f"{nom} = {valeurs!r} : deux bornes (borne_min, borne_max) attendues")
            
            borne_min, borne_max = (float(valeur) for valeur in valeurs)
            if not (np.isfinite(borne_min) and np.isfinite(borne_max)):
                raise ValueError(f"{nom} = ({borne_min}, {borne_max}) : les bornes doivent être finies")
            if not borne_min < borne_max:
                raise ValueError(f"{nom} = ({borne_min}, {borne_max}) : borne_min doit être < à borne_max")
            object.__setattr__(self, nom, (borne_min, borne_max))
        
        object.__setattr__(self, "term_spread_db", float(self.term_spread_db))
        object.__setattr__(self, "no_injection_rate", float(self.no_injection_rate))
        
        if self.crosstalk_db[1] >= 0.0:
            raise ValueError(f"crosstalk_db atteint {self.crosstalk_db[1]} dB : il doit rester < 0 dB")
        largeur = self.crosstalk_db[1] - self.crosstalk_db[0]
        if not 0.0 <= self.term_spread_db < largeur / 2:
            raise ValueError(f"term_spread_db = {self.term_spread_db} dB : il doit être dans [0, {largeur / 2}[ pour "
                             f"laisser une place au niveau commun dans crosstalk_db = {self.crosstalk_db}")
        if self.alpha_phase_deg[0] < -180.0 or self.alpha_phase_deg[1] > 180.0:
            raise ValueError("alpha_phase_deg doit rester dans [-180, 180]")
        if not 0.0 <= self.no_injection_rate < 1.0:
            raise ValueError(f"no_injection_rate = {self.no_injection_rate} : il doit être dans [0, 1[")

    def generate(self, rng: np.random.Generator) -> CrosstalkParameters:
        """Tire un theta_k selon la loi, permet d'avoir des valeurs pas trop extrêmes, et laisse la proportion no_injection_rate de patchs sans injection."""
        if rng.random() < self.no_injection_rate:
            return NEUTRE
        ecart = self.term_spread_db
        niveau_L = rng.uniform(self.crosstalk_db[0] + ecart, self.crosstalk_db[1] - ecart) #niveau commun L : uniforme dans l'enveloppe, mais en laissant la place à l'écart de chaque terme
        modules = 10.0 ** ((niveau_L + rng.uniform(-ecart, ecart, size=4)) / 20.0)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=4)

        u, v, w, z = modules * np.exp(1j * phases)
        module_alpha = 10.0 ** (rng.uniform(*self.alpha_module_db) / 20.0)
        alpha = module_alpha * np.exp(1j * np.deg2rad(rng.uniform(*self.alpha_phase_deg)))
        return CrosstalkParameters(u, v, w, z, alpha)
