"""
Ce qui relie le banc d'essai par réinjection à cvnn.

Importer ce paquet enregistre ses briques dans les registres de cvnn, comme cvnn le fait pour les siennes : les
transformations « crosstalk_injection_mlp » et « crosstalk_injection_resnet », et le réseau « MLP ». Il doit donc être
importé avant que cvnn ne construise les chargeurs et le modèle.
"""

import sys
from pathlib import Path

# cvnn n'est pas installé comme paquet : il est versionné dans le dépôt, sous cvnn/src (même convention que src/patchs.py).
_CVNN = str(Path(__file__).resolve().parents[3] / "cvnn" / "src")
if _CVNN not in sys.path:
    sys.path.insert(0, _CVNN)

from cvnn.transform_registry import register_transform  # noqa: E402

from .cache import cached_loaders  # noqa: E402
from .crosstalk_injection import (  # noqa: E402
    CrosstalkInjector,
    MLPInjectionTransform,
    ResNetInjectionTransform,
    check_pipeline,
)
from .loss import CrosstalkLoss  # noqa: E402  (cvnn n'a pas de registre de pertes : src/train_mlp.py la construit lui-même)
from .mlp_model import MLP  # noqa: E402  (l'import l'enregistre dans le registre de modèles de cvnn)

# Sous « complex » : cvnn y cherche les transformations de toute donnée polarimétrique, quel que soit le mode du réseau.
register_transform("complex", "crosstalk_injection_mlp", MLPInjectionTransform)
register_transform("complex", "crosstalk_injection_resnet", ResNetInjectionTransform)

__all__ = ["CrosstalkInjector", "CrosstalkLoss", "MLP", "MLPInjectionTransform", "ResNetInjectionTransform",
           "cached_loaders", "check_pipeline"]
