"""
Les transformations du banc d'essai par réinjection, enregistrées dans cvnn sous les noms « crosstalk_injection_mlp »
et « crosstalk_injection_resnet ».

Rôle
----
cvnn lit un patch réel (4, H, W) ; ces transformations en font le couple (entrée, cible) du réseau. L'injection leur
est commune (CrosstalkInjector.inject) :
    1. un patch contenant un zéro ou une valeur non finie (bord de fauchée) est écarté : entrée nulle, cible nan ;
       la perte et l'évaluation ignorent les cibles nan ;
    2. theta_k = loi.generate(patch_rng(patch, seed)), le même pour un patch donné à chaque époque ;
    3. le crosstalk est injecté dans les pixels : le vecteur k de chaque pixel devient D(theta_k) k
       (matrix.apply_distortion) ;
    4. le patch injecté est divisé par la racine de la trace de sa covariance, qui vaut alors 1 : le crosstalk ne
       dépend pas de l'échelle du patch.
Chaque transformation en tire l'entrée de son réseau, en complex64 :
    - « crosstalk_injection_mlp » (MLPInjectionTransform) : les 10 entrées distinctes de la covariance du patch injecté
      (covariance.covariance_entries), rangées comme les 10 canaux d'une image 1 x 1, soit (10, 1, 1) ;
    - « crosstalk_injection_resnet » (ResNetInjectionTransform) : le patch injecté lui-même, (4, H, W).
La cible est la même pour les deux : (u, v, w, z, log alpha) de theta_k, (5,) en complex64. Le réseau prédit
log alpha : alpha = exp(sortie) ne peut pas s'annuler. Pour un même patch, le MLP et le ResNet voient donc le même
theta_k, et la covariance de l'entrée du ResNet est l'entrée du MLP : leurs erreurs se comparent patch par patch.
Injecter dans les pixels puis calculer la covariance revient exactement à injecter dans la covariance,
C_inj = D C D^H, D agissant linéairement sur chaque pixel.
Le DataLoader de cvnn regroupe ces couples en lots ((B, 10, 1, 1) ou (B, 4, H, W), et (B, 5)), que sa boucle
d'entraînement reçoit sous la forme (inputs, targets).

Les noms suivent la règle du projet : le nom d'une brique propre à un modèle contient ce modèle (MLPInjectionTransform,
ResNetInjectionTransform) ; celui d'une brique commune n'en contient aucun (CrosstalkInjector, check_pipeline).

Pourquoi (10, 1, 1) et non (10,) pour le MLP : à la construction des chargeurs, cvnn lit un lot pour en déduire le
nombre de canaux et la taille d'entrée, et il n'accepte que des images (B, C, H, W) (cvnn/src/cvnn/data.py,
infer_channels_from_dataloader). Rangées en 10 canaux d'une image 1 x 1, les entrées passent sans modifier cvnn, qui
en déduit 10 canaux ; le réseau n'a qu'à aplatir son entrée.

Piège de cvnn
-------------
cvnn n'échoue pas quand une transformation du YAML ne peut pas être construite (loi invalide, nom inconnu, module du
projet non importé) : il émet un avertissement, « Failed to add transform », et continue sans elle
(cvnn/src/cvnn/transform_registry.py). Le réseau s'entraînerait alors sur des patchs bruts, sans erreur.
check_pipeline vérifie qu'un lot a bien la forme que produit la transformation attendue.
"""

from __future__ import annotations
import numpy as np
import torch
from ..covariance import (
    HERMITIAN_ENTRIES,
    check_patch,
    covariance_entries,
    polarimetric_covariance,
)
from ..crosstalk_sampler import CrosstalkSampler
from ..patch_rng import patch_rng
from ..matrix import apply_distortion

#: Nombre de valeurs complexes de l'entrée (les entrées distinctes de la covariance) et de la cible (u, v, w, z, log alpha).
N_ENTREES: int = len(HERMITIAN_ENTRIES)
CARD_THETA: int = 5


def _couple_ecarte(forme_entree: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Le couple d'un patch écarté : entrée nulle de la forme demandée, cible nan, que la perte et l'évaluation ignorent."""
    return (torch.zeros(forme_entree, dtype=torch.complex64),
            torch.full((CARD_THETA,), complex(np.nan, np.nan), dtype=torch.complex64))


class CrosstalkInjector:
    """
    L'injection commune au MLP et au ResNet : la graine, la loi, et inject (voir la docstring du module).

    Ce n'est pas une transformation de cvnn : MLPInjectionTransform et ResNetInjectionTransform en héritent, et chacune
    n'ajoute que la représentation du patch injecté qu'attend son réseau.

    Args:
        seed: graine des injections (src/crosstalk/patch_rng.py), entier positif ou nul. La changer change toutes les
            injections : c'est une autre expérience. Ce n'est pas la graine globale de cvnn.
        **loi: les champs de crosstalk_sampler.CrosstalkSampler ; ceux qui manquent prennent leur valeur par défaut.

    Raises:
        TypeError: si seed n'est pas un entier, ou si loi contient un champ inconnu de CrosstalkSampler.
        ValueError: si seed est négatif, ou si la loi est invalide.
    """

    def __init__(self, seed: int = 0, **loi) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed de type {type(seed).__name__} : entier attendu")
        if seed < 0:
            raise ValueError(f"seed = {seed} : il doit être positif ou nul")
        self.seed = int(seed)
        self.sampler = CrosstalkSampler(**loi)

    def inject(self, patch: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """
        theta_k injecté dans les pixels du patch, puis normalisation.

        Args: patch: le patch (4, H, W) complexe, en POL_ORDER, soit un tenseur PyTorch, soit un tableau NumPy.
        Returns:
            (patch injecté (4, H, W), cible (u, v, w, z, log alpha) (5,)), en complex128 ; la covariance du patch
            injecté est de trace 1. None si le patch contient un zéro ou une valeur non finie (bord de fauchée).
        Raises: TypeError, ValueError : patch mal formé (covariance.check_patch).
        """
        patch = patch.detach().cpu().numpy() if isinstance(patch, torch.Tensor) else patch #Convertit en np.ndarray si c'est un tenseur, pour être compatible avec patch_rng et polarimetric_covariance
        check_patch(patch)#Lève TypeError ou ValueError si le patch est mal formé : c'est une erreur de programmation (YAML, ordre des transformations), l'entraînement s'arrête.

        #1er cas : patch contenant un zéro ou une valeur non finie (bord de fauchée) : on écarte le patch.
        if np.any(patch == 0) or not np.all(np.isfinite(patch)):
            return None

        #2ème cas : patch valide : on tire theta_k selon la loi et la graine, on l'injecte dans les pixels, on normalise.
        theta = self.sampler.generate(patch_rng(patch, self.seed)) #Détermine les paramètres d'injection selon la loi et la graine.
        injecte = apply_distortion(patch.astype(np.complex128), theta)   # complex128 : apply_distortion garde le type
        injecte /= np.sqrt(np.trace(polarimetric_covariance(injecte)).real)
        u, v, w, z, alpha = theta.to_vector()
        return injecte, np.array([u, v, w, z, np.log(alpha)])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, {self.sampler})"


class MLPInjectionTransform(CrosstalkInjector):
    """
    Patch réel (4, H, W) -> (entrée du MLP (10, 1, 1), cible (5,)), en complex64 : les 10 entrées distinctes de la
    covariance du patch injecté. Mêmes arguments que CrosstalkInjector.
    """

    def __call__(self, patch: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args: patch: le patch à transformer, soit un tenseur PyTorch, soit un tableau NumPy.
        Returns: (entrée (10, 1, 1), cible (5,)), en complex64 ; pour un patch écarté, entrée nulle et cible nan.
        """
        injection = self.inject(patch)
        if injection is None:
            return _couple_ecarte((N_ENTREES, 1, 1))
        injecte, cible = injection
        covariance = polarimetric_covariance(injecte)
        covariance = (covariance + covariance.conj().T) / 2 #Les arrondis du produit matriciel peuvent laisser des parties imaginaires de l'ordre de 1e-19 sur la diagonale : on rend la matrice exactement hermitienne, pour que les 4 puissances soient exactement réelles.
        return (torch.from_numpy(covariance_entries(covariance).astype(np.complex64)).reshape(N_ENTREES, 1, 1),
                torch.from_numpy(cible.astype(np.complex64)))


class ResNetInjectionTransform(CrosstalkInjector):
    """
    Patch réel (4, H, W) -> (entrée du ResNet (4, H, W), cible (5,)), en complex64 : le patch injecté lui-même.
    Mêmes arguments que CrosstalkInjector. Pour un même patch, la covariance de l'entrée est l'entrée du MLP.
    """

    def __call__(self, patch: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args: patch: le patch à transformer, soit un tenseur PyTorch, soit un tableau NumPy.
        Returns: (entrée (4, H, W), cible (5,)), en complex64 ; pour un patch écarté, entrée nulle et cible nan.
        """
        injection = self.inject(patch)
        if injection is None:
            return _couple_ecarte(tuple(patch.shape))
        injecte, cible = injection
        return torch.from_numpy(injecte.astype(np.complex64)), torch.from_numpy(cible.astype(np.complex64))


def check_pipeline(loader, forme_entree: tuple[int, ...] = (N_ENTREES, 1, 1)) -> None:
    """
    Vérifie qu'un lot du chargeur a la forme que produit la transformation attendue.

    Parade au piège de cvnn décrit dans la docstring du module : une transformation que cvnn n'a pas pu construire
    ne laisse qu'un avertissement, et les lots contiennent alors des patchs bruts. À appeler avant tout entraînement
    ou toute évaluation, sur chaque chargeur.

    Args:
        loader: le chargeur à vérifier.
        forme_entree: la forme d'une entrée sans l'axe du lot : (N_ENTREES, 1, 1) pour le MLP (par défaut), (4, H, W)
            pour le ResNet.

    Raises:
        RuntimeError: si le premier lot n'est pas un couple (entrées (B, *forme_entree), cibles (B, 5)) de tenseurs
            complex64.
    """
    aide = ("la transformation d'injection (« crosstalk_injection_mlp » ou « crosstalk_injection_resnet ») n'a pas été "
            "appliquée. cvnn ignore une transformation qu'il ne peut pas construire (avertissement « Failed to add "
            "transform » dans le journal) : vérifier la loi d'injection dans le YAML, et que src.crosstalk.deep_learning "
            "est importé avant de construire les chargeurs")
    lot = next(iter(loader))
    if not (isinstance(lot, (list, tuple)) and len(lot) == 2):
        raise RuntimeError(f"lot de type {type(lot).__name__}, couple (entrées, cibles) attendu : {aide}")
    for nom, tenseur, forme_attendue in (("entrées", lot[0], tuple(forme_entree)), ("cibles", lot[1], (CARD_THETA,))):
        if not (isinstance(tenseur, torch.Tensor) and tenseur.dtype == torch.complex64 and tuple(tenseur.shape[1:]) == forme_attendue):
            forme = tuple(tenseur.shape) if isinstance(tenseur, torch.Tensor) else type(tenseur).__name__
            raise RuntimeError(f"{nom} de forme {forme}, (B, {', '.join(map(str, forme_attendue))}) complex64 attendu : {aide}")
    if lot[0].shape[0] != lot[1].shape[0]:
        raise RuntimeError(f"{lot[0].shape[0]} entrées pour {lot[1].shape[0]} cibles dans un même lot")
