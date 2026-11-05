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
le crosstalk total. Composer deux distorsions redonne exactement une distorsion de même forme :
    D(theta_k) D(theta_0) = g D(theta_tot) K, avec K = diag(1, racine(kappa), racine(kappa), kappa).
Le scalaire g n'a pas d'effet après la normalisation par la trace, et K passe du côté de la scène sans y changer la
réciprocité ni la symétrie de réflexion : la cible est donc bien définie. En revanche, theta_tot n'est pas
theta_0 + theta_k. Un estimateur entraîné à retrouver theta_k mesure donc un écart au crosstalk typique des données :
sur un patch brut, il doit rendre environ 0. Il ne calibre pas les données à lui seul.


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
