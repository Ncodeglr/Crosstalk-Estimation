"""
Rend le dépôt importable depuis les tests (`import src.crosstalk`), ainsi que les outils de test rangés dans tests/
(`from synthetic import ...`).

Les outils de tests/ s'importent par leur propre nom, et non par `tests.<module>` : une dépendance de
l'environnement (iterative-stratification) installe son propre paquet `tests` dans site-packages, qui pourrait
prendre la place du nôtre.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).parent.resolve()
for chemin in (RACINE / "tests", RACINE):
    if str(chemin) not in sys.path:
        sys.path.insert(0, str(chemin))
