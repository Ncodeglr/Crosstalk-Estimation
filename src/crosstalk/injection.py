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
import hashlib
from dataclasses import dataclass
import numpy as np
from .parameters import CrosstalkParameters

#: theta_k neutre : aucun crosstalk ajouté, aucun déséquilibre. D(NEUTRE) est l'identité.
NEUTRE: CrosstalkParameters = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class InjectionLaw:
    """
    Loi de theta_k. Chaque tirage est, avec la probabilité no_injection_rate, le theta_k NEUTRE ; sinon :
        niveau commun L : uniforme en dB dans [crosstalk_db[0] + term_spread_db, crosstalk_db[1] - term_spread_db] ;
        |u|, |v|, |w|, |z| : L + un écart uniforme dans [-term_spread_db, +term_spread_db] dB, tiré pour chaque
            terme séparément, si bien que les quatre termes restent dans l'enveloppe crosstalk_db ;
        phases de u, v, w, z : uniformes sur [0, 360°[ ;
        |alpha| : uniforme en dB dans alpha_module_db ;
        arg(alpha) : uniforme en degrés dans alpha_phase_deg.

    Valeurs par défaut. « Nos mesures » désignent le theta estimé par résolution exacte des contraintes de Quegan
    sur chaque colonne de patchs et chaque date de la pile Rosamond (765 couples, partition d'entraînement).

        -crosstalk_db = (-40, -5) dB, l'enveloppe des quatre termes.
            Le crosstalk de l'instrument : au plus -15 dB environ chez Fore et al. (2015, fig. 7) sur UAVSAR ;
            rarement plus de 0,1 (-20 dB) sur MAESTRO chez Quegan (1994, après les éq. 22 à 25) ; moins de -20 dB
            pour les systèmes modernes selon Shi et al. (2020) ; de -18,6 à -23,4 dB aux réflecteurs de Rosamond
            chez Babu et al. (2022, §4.6). Nos mesures : le terme le plus fort va de -20,7 à -10,7 dB (médiane
            -16,7 dB).
        -term_spread_db = 4 dB, l'écart de chaque terme au niveau commun.
            Les quatre termes d'un instrument sont du même ordre de grandeur (Fore et al., 2015, fig. 7) ; Shi et
            al. (2020) les injectent même avec une amplitude identique. Nos mesures : chaque terme s'écarte du
            niveau commun de -3,9 à +3,3 dB (quantiles 5 et 95 %). Des termes indépendants dans toute l'enveloppe
            pourraient s'écarter de 35 dB, ce qui n'est jamais observé.
        -alpha_module_db = (-3, 3) dB, soit |alpha| de 0,71 à 1,41.
            C'est la plage d'injection de Shi et al. (2020) sur des données réelles calibrées. Elle couvre
            |alpha| ≈ 1 après la calibration du JPL (Fore et al., 2015, légende de la fig. 7), le déséquilibre
            croisé g = 1,128 mesuré sur UAVSAR par Babu et al. (2022, tab. 3), et nos mesures : |alpha| de -0,9 à
            +1,5 dB.
        -alpha_phase_deg = (-30, 30) degrés.
            Les données sont calibrées (Polarimetrically Calibrated = Yes) : la phase de alpha y est quasi nulle
            (Fore et al., 2015, légende de la fig. 7), le biais de phase vaut -1,14° chez Babu et al. (2022,
            tab. 3), et nos mesures vont de -7° à +18°. Une phase quelconque, comme les +-180° de Shi et al. (2020),
            ne vaut que pour un système non calibré : 42,5° en bande L, 102,5° en bande C et -70° en bande P sur
            MAESTRO chez Quegan (1994).
        -no_injection_rate = 0,2 : choix d'apprentissage, pas une valeur physique. Un patch sur cinq reste brut, pour
            que l'estimateur apprenne qu'un patch sans injection n'a pas de crosstalk ajouté.
    """

    crosstalk_db: tuple[float, float] = (-40.0, -5.0)
    term_spread_db: float = 4.0
    alpha_module_db: tuple[float, float] = (-3.0, 3.0)
    alpha_phase_deg: tuple[float, float] = (-30.0, 30.0)
    no_injection_rate: float = 0.2

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


def draw_injection(law: InjectionLaw, rng: np.random.Generator) -> CrosstalkParameters:
    """Tire un theta_k selon law, permet d'avoir des valeurs pas trop extrêmes, et de laisser un patch sur cinq sans injection pour l'apprentissage."""
    if rng.random() < law.no_injection_rate:
        return NEUTRE
    ecart = law.term_spread_db
    niveau_L = rng.uniform(law.crosstalk_db[0] + ecart, law.crosstalk_db[1] - ecart) #niveau commun L : uniforme dans l'enveloppe, mais en laissant la place à l'écart de chaque terme
    modules = 10.0 ** ((niveau_L + rng.uniform(-ecart, ecart, size=4)) / 20.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    
    u, v, w, z = modules * np.exp(1j * phases)
    module_alpha = 10.0 ** (rng.uniform(*law.alpha_module_db) / 20.0)
    alpha = module_alpha * np.exp(1j * np.deg2rad(rng.uniform(*law.alpha_phase_deg)))
    return CrosstalkParameters(u, v, w, z, alpha)


def patch_rng(patch: np.ndarray, seed: int) -> np.random.Generator:
    """
    Générateur aléatoire propre à un patch : sa graine combine seed et une empreinte du contenu du patch.

    Le même patch donne toujours le même générateur, donc le même theta_k, quel que soit le processus ou l'époque.
    L'empreinte porte sur les octets, la forme et le type du tableau : un même patch en complex64 et en complex128
    donne deux générateurs différents. Le patch doit donc toujours être passé tel qu'il est lu.

    Args:
        patch: le patch, tel qu'il est lu.
        seed: entier positif ou nul. Le changer change toutes les injections : c'est une autre expérience.
    """
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"patch de type {type(patch).__name__} : np.ndarray attendu")
    # Pour un tableau d'objets Python, tobytes() écrit des adresses mémoire et non des valeurs : l'empreinte
    # changerait d'un processus à l'autre. Seuls les types numériques donnent une empreinte déterministe.
    if patch.dtype.kind not in "biufc":
        raise TypeError(f"patch de type d'éléments {patch.dtype} : un type numérique est attendu (complex64 pour les .slc)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed de type {type(seed).__name__} : entier attendu")
    if seed < 0:
        raise ValueError(f"seed = {seed} : il doit être positif ou nul")
    
    #hashlib.blake2b est une fonction de hachage cryptographique qui produit un haché de 64 octets par défaut. 
    empreinte = hashlib.blake2b(digest_size=8) #Empreinte du patch, pour que le même patch donne toujours le même générateur, donc le même theta_k, quel que soit le processus ou l'époch.
    empreinte.update(f"{patch.dtype.str}{patch.shape}".encode()) #patch.dtype.str = '<c8' pour complex64 ou '<c16' pour complex128,  patch.shape = (4, H, W). On encode en bytes pour que hashlib puisse traiter la chaîne.
    empreinte.update(patch.tobytes(order='C')) #tobytes() écrit les octets dans l'ordre des index (ordre C), quelle que soit la disposition du tableau en mémoire : 8 octets par complexe en complex64, soit 8 x (Nb_channels x H_patch x W_patch) octets. On met à jour l'empreinte avec le contenu du patch.
    return np.random.default_rng([int(seed), int.from_bytes(empreinte.digest(), "little")])
