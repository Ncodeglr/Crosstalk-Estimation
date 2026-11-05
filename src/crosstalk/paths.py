"""
Les chemins du dépôt, calculés à un seul endroit pour toute la bibliothèque.

Les scripts de src/scripts/ gardent leur propre calcul de la racine : ils en ont besoin pour trouver ce paquet.
"""

from pathlib import Path

#: Racine du dépôt : src/crosstalk/paths.py -> src/crosstalk -> src -> racine.
RACINE: Path = Path(__file__).resolve().parents[2]

#: Sources de cvnn, qui n'est pas installé comme paquet : il est versionné dans le dépôt.
CVNN_SRC: Path = RACINE / "cvnn" / "src"

#: Dossier des caches, à la racine du dépôt (ignoré par git).
DOSSIER_CACHE: Path = RACINE / "cache"
