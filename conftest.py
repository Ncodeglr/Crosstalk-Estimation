"""Rend le dépôt importable depuis les tests (`import src.crosstalk`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
