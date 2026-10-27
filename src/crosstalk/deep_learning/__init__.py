"""
Ce qui relie le banc d'essai par réinjection à cvnn.

Importer ce paquet enregistre ses briques dans les registres de cvnn, comme cvnn le fait pour les siennes : la
transformation « crosstalk_injection ». Il doit donc être importé avant que cvnn ne construise les chargeurs.
"""

import sys
from pathlib import Path

# cvnn n'est pas installé comme paquet : il est versionné dans le dépôt, sous cvnn/src (même convention que src/patchs.py).
_CVNN = str(Path(__file__).resolve().parents[3] / "cvnn" / "src")
if _CVNN not in sys.path:
    sys.path.insert(0, _CVNN)

from cvnn.transform_registry import register_transform  # noqa: E402

from .injection_transform import CrosstalkInjectionTransform, check_pipeline  # noqa: E402

# Sous « complex » : cvnn y cherche les transformations de toute donnée polarimétrique, quel que soit le mode du réseau.
register_transform("complex", "crosstalk_injection", CrosstalkInjectionTransform)

__all__ = ["CrosstalkInjectionTransform", "check_pipeline"]
