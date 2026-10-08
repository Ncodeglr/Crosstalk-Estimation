"""
Patchs de test servis par cvnn.

Le seul module du projet qui dépende de cvnn et de torchcvnn : il construit le
chargeur de la partition de test, puis déplie ses lots en patchs.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Iterator
import numpy as np

# cvnn n'est pas installé comme paquet, il est versionné dans le dépôt, sous cvnn/src.
RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "cvnn" / "src"))


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


def iter_patches(loader) -> Iterator[np.ndarray]:
    """
    Déplie les lots en patchs `(4, H, W)` numpy, un à la fois.
    Générateur, et non liste : les patchs sont lus sur disque puis abandonnés au fil de l'eau.
    """
    for lot in loader:
        # UavsarSLC n'a pas d'étiquettes : `lot` est un Tensor, pas un (x, y).
        # La branche `else` s'applique donc, et `tenseur` vaut le lot entier, de forme (batch_size, 4, patch_size, patch_size).
        tenseur = lot[0] if isinstance(lot, (list, tuple)) else lot
        yield from tenseur.detach().cpu().numpy()  # un patch de forme (4, patch_size, patch_size) à la fois
