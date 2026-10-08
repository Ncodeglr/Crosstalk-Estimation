"""
Patchs de test servis par cvnn, avec leur position dans la pile.

Le seul module du projet qui dépende de cvnn et de torchcvnn : il construit le
chargeur de la partition de test, puis déplie ses lots en couples (position, patch).
La position — acquisition, ligne et colonne du coin haut-gauche — se recalcule à
partir des index du `Subset` et de la grille de `SLCFile` ; c'est la clé qui
permet de comparer deux méthodes patch à patch.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
import numpy as np

# cvnn n'est pas installé comme paquet, il est versionné dans le dépôt, sous cvnn/src.
RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "cvnn" / "src"))


@dataclass(frozen=True, slots=True)
class PositionPatch:
    """Emplacement d'un patch dans la pile.

    Attributes:
        acquisition: clé torchcvnn de l'acquisition, p. ex.
            `Rosamd_35012_16002_001_160211_L_090_04_BC`. On garde la clé et non
            un rang : `SLCDataset` range les acquisitions dans l'ordre de
            `glob`, pas dans l'ordre chronologique.
        ligne: coin haut-gauche, en pixels 1x1 le long de l'azimut.
        colonne: coin haut-gauche, en pixels 1x1 le long de la distance — la
            « ligne de distance » de Babu et de Fore.
    """

    acquisition: str
    ligne: int
    colonne: int


#Données
def build_dataloader(config: dict[str, Any]):
    """
    Construit les loader via cvnn. L'objet rendu est un `torch.utils.data.DataLoader`, qui ne contient aucune donnée. 
    Itéré — ce que fait `iter_patches` — il produit des lots`torch.Tensor` de forme `(batch_size, 4, H, W)` en `complex64`.
    """
    from cvnn.data import get_dataloaders, validate_and_correct_config
    from cvnn.utils import set_seed

    set_seed(int(config.get("seed", 0)))
    _, _, test_loader = get_dataloaders(validate_and_correct_config(config), use_cuda=False)
    return test_loader #Retourne un DataLoader 


#Positions
def position_du_patch(index: int, comptes: Mapping[str, int], par_ligne: int, pas: tuple[int, int]) -> PositionPatch:
    """
    Index global de `SLCDataset` -> position du patch dans la pile.

    Refait le calcul de `SLCDataset.__getitem__` — retrancher les comptes des
    acquisitions, dans l'ordre de `comptes` — puis celui de `SLCFile.__getitem__`
    — ligne et colonne de la grille, multipliées par le pas.

    Args:
        index: index global, tel que le portent `Subset.indices`.
        comptes: `SLCDataset.patch_counts`, nombre de patchs par acquisition.
        par_ligne: `SLCFile.nsamples_per_cols`, patchs par ligne de la grille.
        pas: `SLCFile.patch_stride`, pas de la grille en pixels (azimut, distance).

    Raises:
        IndexError: si l'index dépasse la pile.
    """
    reste = index
    for acquisition, compte in comptes.items():
        if reste < compte:
            return PositionPatch(acquisition, (reste // par_ligne) * pas[0], (reste % par_ligne) * pas[1])
        reste -= compte
    raise IndexError(f"index {index} hors de la pile ({sum(comptes.values())} patchs)")


def positions_de_test(loader) -> list[PositionPatch]:
    """
    Positions des patchs que sert `loader`, dans son ordre.

    Le chargeur de test de `cvnn` enveloppe un `SLCDataset` dans un `Subset` :
    `.indices` donne l'index global de chaque patch. Ce `SLCDataset` est une
    instance distincte de celle du découpage ; on lit donc les comptes et la
    grille sur celle-ci, la seule qui serve réellement les patchs.

    Raises:
        ValueError: si les fichiers de la pile n'ont pas tous la même grille.
    """
    sous_ensemble = loader.dataset
    pile = sous_ensemble.dataset
    grilles = {
        (fichier.nsamples_per_cols, tuple(fichier.patch_stride))
        for polarisations in pile.slc_polarizations.values()
        for fichier in polarisations.values()
    }
    if len(grilles) != 1:
        raise ValueError(f"grilles de patchs différentes selon les fichiers : {sorted(grilles)}")
    (par_ligne, pas), = grilles
    return [position_du_patch(index, pile.patch_counts, par_ligne, pas) for index in sous_ensemble.indices]


def iter_patches(loader) -> Iterator[tuple[PositionPatch, np.ndarray]]:
    """
    Déplie les lots en couples `(position, patch)`, le patch `(4, H, W)` numpy, dans l'ordre du pipeline.
    Générateur, et non liste : les patchs sont lus sur disque puis abandonnés au fil de l'eau.

    Les positions suivent l'ordre de `Subset.indices`, que seul un chargeur non
    mélangé respecte : un chargeur mélangé est refusé. 
    Le zip strict lève une erreur si patchs et positions ne sont pas en même nombre.
    """
    from torch.utils.data import SequentialSampler

    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("chargeur mélangé : les patchs ne suivraient plus l'ordre de leurs positions")

    def patchs() -> Iterator[np.ndarray]:
        for lot in loader:
            # UavsarSLC n'a pas d'étiquettes : `lot` est un Tensor, pas un (x, y).
            # La branche `else` s'applique donc, et `tenseur` vaut le lot entier, de forme (batch_size, 4, patch_size, patch_size).
            tenseur = lot[0] if isinstance(lot, (list, tuple)) else lot
            yield from tenseur.detach().cpu().numpy()  # un patch de forme (4, patch_size, patch_size) à la fois

    yield from zip(positions_de_test(loader), patchs(), strict=True)
