"""
La transformation du banc d'essai par réinjection, enregistrée dans cvnn sous le nom « crosstalk_injection ».

Rôle
----
cvnn lit un patch réel (4, H, W) ; cette transformation en fait le couple (entrée, cible) du réseau :
    1. un patch contenant un zéro ou une valeur non finie (bord de fauchée) est écarté : entrée nulle, cible nan ;
       la perte et l'évaluation ignorent les cibles nan ;
    2. theta_k = draw_injection(loi, patch_rng(patch, seed)), le même pour un patch donné à chaque époque ;
    3. C = covariance du patch, puis C_inj = D(theta_k) C D(theta_k)^H. Injecter le crosstalk dans les pixels puis
       calculer la covariance revient exactement au même, D agissant linéairement sur chaque pixel ;
    4. C_inj / trace(C_inj) : le crosstalk ne dépend pas de l'échelle du patch ;
    5. entrée : les 10 entrées distinctes de C_inj normalisée (covariance.covariance_entries), rangées comme les
       10 canaux d'une image 1 x 1, soit (10, 1, 1) en complex64 ; cible : (u, v, w, z, log alpha) de theta_k, (5,) en
       complex64. Le réseau prédit log alpha : alpha = exp(sortie) ne peut pas s'annuler.
Le DataLoader de cvnn regroupe ces couples en lots (B, 10, 1, 1) et (B, 5), que sa boucle d'entraînement reçoit sous
la forme (inputs, targets).

Pourquoi (10, 1, 1) et non (10,) : à la construction des chargeurs, cvnn lit un lot pour en déduire le nombre de
canaux et la taille d'entrée, et il n'accepte que des images (B, C, H, W) (cvnn/src/cvnn/data.py,
infer_channels_from_dataloader). Rangées en 10 canaux d'une image 1 x 1, les entrées passent sans modifier cvnn, qui
en déduit 10 canaux ; le réseau n'a qu'à aplatir son entrée.

Piège de cvnn
-------------
cvnn n'échoue pas quand une transformation du YAML ne peut pas être construite (loi invalide, nom inconnu, module du
projet non importé) : il émet un avertissement, « Failed to add transform », et continue sans elle
(cvnn/src/cvnn/transform_registry.py). Le réseau s'entraînerait alors sur des patchs bruts, sans erreur.
check_pipeline vérifie qu'un lot a bien la forme que produit cette transformation.
"""

from __future__ import annotations
import numpy as np
import torch
from ..covariance import (
    HERMITIAN_ENTRIES,
    check_patch,
    covariance_entries,
    normalize_by_trace,
    polarimetric_covariance,
)
from ..injection import InjectionLaw, draw_injection, patch_rng
from ..matrix import distortion_matrix

#: Nombre de valeurs complexes de l'entrée (les entrées distinctes de la covariance) et de la cible (u, v, w, z, log alpha).
N_ENTREES: int = len(HERMITIAN_ENTRIES)
N_CIBLES: int = 5


class CrosstalkInjectionTransform:
    """
    Patch réel (4, H, W) -> (entrée (10, 1, 1), cible (5,)), en complex64.

    Args:
        seed: graine des injections (injection.patch_rng), entier positif ou nul. La changer change toutes les
            injections : c'est une autre expérience. Ce n'est pas la graine globale de cvnn.
        **loi: les champs de injection.InjectionLaw ; ceux qui manquent prennent leur valeur par défaut.

    Raises:
        TypeError: si seed n'est pas un entier, ou si loi contient un champ inconnu d'InjectionLaw.
        ValueError: si seed est négatif, ou si la loi est invalide.
    """

    def __init__(self, seed: int = 0, **loi) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed de type {type(seed).__name__} : entier attendu")
        if seed < 0:
            raise ValueError(f"seed = {seed} : il doit être positif ou nul")
        self.seed = int(seed)
        self.law = InjectionLaw(**loi)

    def __call__(self, patch: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applique la transformation d'injection de crosstalk au patch.

        Args: patch: le patch à transformer, soit un tenseur PyTorch, soit un tableau NumPy.
        Returns: Un tuple (entrée, cible) où l'entrée est un tenseur PyTorch de forme (10, 1, 1) et la cible est un tenseur PyTorch de forme (5,).
        """
        patch = patch.detach().cpu().numpy() if isinstance(patch, torch.Tensor) else patch #Convertit en np.ndarray si c'est un tenseur, pour être compatible avec patch_rng et polarimetric_covariance
        check_patch(patch)#Lève TypeError ou ValueError si le patch est mal formé : c'est une erreur de programmation (YAML, ordre des transformations), l'entraînement s'arrête.
        
        #1er cas : patch contenant un zéro ou une valeur non finie (bord de fauchée) : on écarte le patch, entrée nulle, cible nan.
        if np.any(patch == 0) or not np.all(np.isfinite(patch)):
            return (torch.zeros((N_ENTREES, 1, 1), dtype=torch.complex64),
                    torch.full((N_CIBLES,), complex(np.nan, np.nan), dtype=torch.complex64))


        #2ème cas : patch valide : on tire theta_k selon la loi et la graine, on injecte le crosstalk dans la covariance, on normalise par la trace, et on renvoie les entrées et la cible.
        theta = draw_injection(self.law, patch_rng(patch, self.seed)) #Détermine les paramètres d'injection selon la loi et la graine.
        D = distortion_matrix(theta)
        covariance = D @ polarimetric_covariance(patch) @ D.conj().T
        covariance = (covariance + covariance.conj().T) / 2 #Les arrondis de D C D^H laissent des parties imaginaires de l'ordre de 1e-19 sur la diagonale : on rend la matrice exactement hermitienne, pour que les 4 puissances soient exactement réelles.
        covariance = normalize_by_trace(covariance)
        u, v, w, z, alpha = theta.to_vector()
        cible = np.array([u, v, w, z, np.log(alpha)])
        return (torch.from_numpy(covariance_entries(covariance).astype(np.complex64)).reshape(N_ENTREES, 1, 1),
                torch.from_numpy(cible.astype(np.complex64)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, {self.law})"


def check_pipeline(loader) -> None:
    """
    Vérifie qu'un lot du chargeur a la forme que produit CrosstalkInjectionTransform.

    Parade au piège de cvnn décrit dans la docstring du module : une transformation que cvnn n'a pas pu construire
    ne laisse qu'un avertissement, et les lots contiennent alors des patchs bruts. À appeler avant tout entraînement
    ou toute évaluation, sur chaque chargeur.

    Raises:
        RuntimeError: si le premier lot n'est pas un couple (entrées (B, 10, 1, 1), cibles (B, 5)) de tenseurs
            complex64.
    """
    aide = ("la transformation « crosstalk_injection » n'a pas été appliquée. cvnn ignore une transformation qu'il ne "
            "peut pas construire (avertissement « Failed to add transform » dans le journal) : vérifier la loi "
            "d'injection dans le YAML, et que src.crosstalk.deep_learning est importé avant de construire les chargeurs")
    lot = next(iter(loader))
    if not (isinstance(lot, (list, tuple)) and len(lot) == 2):
        raise RuntimeError(f"lot de type {type(lot).__name__}, couple (entrées, cibles) attendu : {aide}")
    for nom, tenseur, forme_attendue in (("entrées", lot[0], (N_ENTREES, 1, 1)), ("cibles", lot[1], (N_CIBLES,))):
        if not (isinstance(tenseur, torch.Tensor) and tenseur.dtype == torch.complex64 and tuple(tenseur.shape[1:]) == forme_attendue):
            forme = tuple(tenseur.shape) if isinstance(tenseur, torch.Tensor) else type(tenseur).__name__
            raise RuntimeError(f"{nom} de forme {forme}, (B, {', '.join(map(str, forme_attendue))}) complex64 attendu : {aide}")
    if lot[0].shape[0] != lot[1].shape[0]:
        raise RuntimeError(f"{lot[0].shape[0]} entrées pour {lot[1].shape[0]} cibles dans un même lot")
