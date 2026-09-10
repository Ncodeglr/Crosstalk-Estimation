"""Lecture et inspection des produits UAVSAR *Stack SLC* (JPL).

Ce module regroupe toute la logique réutilisable de *data engineering* pour le
projet d'estimation supervisée du crosstalk polarimétrique :

* découverte et indexation des fichiers d'une pile (``discover_slc_files``) ;
* parsing robuste des fichiers d'annotation ``.ann`` (``parse_ann``) ;
* extraction des métadonnées géométriques utiles (``read_slc_geometry``) ;
* ouverture paresseuse des ``.slc`` en ``numpy.memmap`` (``open_slc_memmap``) ;
* lectures **bornées** : tuile, version décimée, patch de diffusion
  (``read_window``, ``read_decimated``, ``read_scattering_patch``) ;
* aide à la visualisation (``amplitude_db``, ``plot_amplitude_db``).

Faits de format (documentation JPL « UAVSAR Stack SLC », confirmés par les
en-têtes ``.ann`` de la pile ``Rosamd_35012_04``) :

* ``.slc`` : binaire brut **sans en-tête**, little-endian, 8 octets/pixel
  (float32 partie réelle puis float32 partie imaginaire) → ``numpy.complex64``.
  Amplitude **linéaire** (jamais en dB), géométrie *slant range*,
  forme ``(n_rows, n_cols)`` = (azimut, range).
* ``.llh`` / ``.lkv`` : binaire float32 little-endian, 12 octets/pixel
  (3 canaux : lat/lon/hauteur, ou est/nord/haut). Ils sont produits en **2x8**
  alors que les ``.slc`` exploités ici sont en **1x1** : les grilles diffèrent,
  l'appariement fin (trièdre → pixel) est hors périmètre de cette brique.

Sécurité mémoire : un ``.slc`` 1x1 de cette pile pèse ~4,5 Gio et la pile
complète ~90 Go. Aucune fonction de ce module ne matérialise un fichier entier :
tout passe par ``numpy.memmap`` et les lectures denses sont plafonnées par
``max_bytes`` (voir ``DEFAULT_MAX_DENSE_BYTES``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "POL_ORDER_ONERA",
    "POL_ORDER_ALPHABETICAL",
    "POL_COLORS",
    "POL_LINESTYLES",
    "CO_POL",
    "CROSS_POL",
    "SLC_DTYPE",
    "SLC_BYTES_PER_PIXEL",
    "LLH_LKV_DTYPE",
    "LLH_LKV_N_CHANNELS",
    "DEFAULT_MAX_DENSE_BYTES",
    "AnnEntry",
    "SlcGeometry",
    "UavsarStack",
    "repo_root",
    "parse_slc_filename",
    "discover_slc_files",
    "parse_ann",
    "parse_ann_entries",
    "find_ann_keys",
    "read_slc_geometry",
    "open_slc_memmap",
    "open_vector_memmap",
    "read_window",
    "decimation_stride",
    "read_decimated",
    "read_scattering_patch",
    "amplitude_db",
    "plot_amplitude_db",
    "plot_amplitude_histogram",
    "load_corner_reflectors",
]


# --------------------------------------------------------------------------- #
# Constantes de format
# --------------------------------------------------------------------------- #

#: Ordre des canaux du **vecteur de diffusion** retenu pour tout le projet.
#:
#: Le protocole suit la convention ONERA ``s = [S_Hh, S_Vh, S_Hv, S_Vv]``, où le
#: premier indice est la polarisation de **réception** et le second celle
#: d'**émission**. Cela place VH en position 2 et HV en position 3, c'est-à-dire
#: les positions 2 et 3 permutées par rapport à l'ordre alphabétique usuel
#: (HH, HV, VH, VV). Cette permutation n'est pas cosmétique : elle conditionne
#: le remapping u↔z / v↔w des paramètres de crosstalk dans le modèle de
#: distorsion, donc les matrices de distorsion écrites plus tard (baselines
#: Quegan/Ainsworth, injection synthétique, CVNN) supposent cet ordre.
#: Toute pile ``(4, H, W)`` du projet DOIT le respecter.
POL_ORDER_ONERA: tuple[str, str, str, str] = ("HH", "VH", "HV", "VV")

#: Ordre alphabétique — sert uniquement à l'affichage / au tri des fichiers.
POL_ORDER_ALPHABETICAL: tuple[str, str, str, str] = ("HH", "HV", "VH", "VV")

#: dtype des ``.slc`` : complexe 64 bits, little-endian, explicite (portabilité).
SLC_DTYPE: np.dtype = np.dtype("<c8")

#: Octets par pixel d'un ``.slc`` (4 octets Re + 4 octets Im).
SLC_BYTES_PER_PIXEL: int = SLC_DTYPE.itemsize  # == 8

#: dtype des ``.llh`` / ``.lkv`` : float32 little-endian.
LLH_LKV_DTYPE: np.dtype = np.dtype("<f4")

#: Nombre de canaux d'un ``.llh`` / ``.lkv`` (12 octets/pixel = 3 x float32).
LLH_LKV_N_CHANNELS: int = 3

#: Couleurs de tracé par polarisation (palette Okabe-Ito, sûre pour les
#: déficiences de vision des couleurs ; validée : bande de clarté, plancher de
#: chroma et plancher en vision normale OK, séparation CVD dans la bande 6–8).
#: Cette bande n'est admissible qu'avec un **encodage secondaire** : on utilise
#: :data:`POL_LINESTYLES` (trait plein pour la co-polarisation, tireté pour la
#: cross-polarisation), qui porte en plus une information physique utile.
POL_COLORS: dict[str, str] = {
    "HH": "#0072B2",  # bleu
    "VH": "#D55E00",  # vermillon
    "HV": "#009E73",  # vert
    "VV": "#CC79A7",  # rose
}

#: Style de trait par polarisation : co-pol pleine, cross-pol tiretée.
POL_LINESTYLES: dict[str, str] = {"HH": "-", "VH": "--", "HV": "--", "VV": "-"}

#: Polarisations co-polaires et cross-polaires (l'écart entre HV et VH porte
#: l'essentiel du signal de crosstalk que le projet cherche à estimer).
CO_POL: tuple[str, str] = ("HH", "VV")
CROSS_POL: tuple[str, str] = ("VH", "HV")

#: Plafond par défaut (octets) pour toute lecture **dense** en RAM.
#: 256 Mio : très en dessous de la contrainte projet de ~1 Gio, et largement
#: suffisant pour une tuile 4096x4096 complexe (128 Mio) ou un aperçu décimé.
DEFAULT_MAX_DENSE_BYTES: int = 256 * 1024**2

#: Nom du produit : ``<site>_<ligne>_<flightID>_<dataTake>_<date>_L090<POL>_<stack>_BC[...]``
_SLC_NAME_RE = re.compile(
    r"^(?P<site>[A-Za-z]+)_(?P<line_id>\d+)_(?P<flight_id>\d+)_(?P<data_take>\d+)_"
    r"(?P<date>\d{6})_L(?P<band>\d{3})(?P<pol>HH|HV|VH|VV)_(?P<stack_version>\d+)_BC"
    r"(?:_s(?P<segment>\d+)_(?P<looks>\d+x\d+))?"
    r"\.(?P<ext>slc|ann)$",
    re.IGNORECASE,
)

#: Ligne d'annotation ``clé (unité) = valeur ; commentaire`` (unité optionnelle).
_ANN_KEY_UNIT_RE = re.compile(r"^(?P<key>.*?)\s*(?:\((?P<unit>[^()]*)\)\s*)?$")


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

def repo_root(start: Path | str | None = None) -> Path:
    """Retourne la racine du dépôt, trouvée en remontant jusqu'au marqueur ``.git``.

    Permet aux notebooks de n'utiliser que des chemins **relatifs** à la racine,
    quel que soit le répertoire de travail du kernel.

    Args:
        start: Répertoire de départ ; par défaut le dossier de ce fichier.

    Returns:
        Le chemin absolu de la racine du dépôt.

    Raises:
        FileNotFoundError: si aucun ``.git`` n'est trouvé en remontant.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(
        f"Racine du dépôt introuvable (aucun '.git' en remontant depuis {here})."
    )


# --------------------------------------------------------------------------- #
# Découverte / indexation des fichiers
# --------------------------------------------------------------------------- #

def parse_slc_filename(path: Path | str) -> dict[str, Any] | None:
    """Décompose un nom de fichier UAVSAR en ses champs.

    Args:
        path: Chemin ou nom de fichier ``.slc`` / ``.ann``.

    Returns:
        Un dictionnaire ``{site, line_id, flight_id, data_take, date, band, pol,
        stack_version, segment, looks, ext}`` (``segment``/``looks`` peuvent être
        ``None`` pour un ``.ann``), ou ``None`` si le nom ne suit pas la
        convention UAVSAR.
    """
    match = _SLC_NAME_RE.match(Path(path).name)
    if match is None:
        return None
    fields = match.groupdict()
    fields["pol"] = fields["pol"].upper()
    fields["ext"] = fields["ext"].lower()
    fields["segment"] = int(fields["segment"]) if fields["segment"] else None
    return fields


def discover_slc_files(
    data_dir: Path | str,
    looks: str = "1x1",
) -> pd.DataFrame:
    """Indexe les couples ``.slc`` / ``.ann`` d'un répertoire de pile UAVSAR.

    Le numéro de track (31..35 pour ``Rosamd_35012_04``) n'apparaît pas dans le
    nom de fichier : il est lu dans le ``.ann`` associé (clé ``Stack Line
    Number``), ce qui évite tout mapping codé en dur.

    Args:
        data_dir: Répertoire contenant les fichiers de la pile.
        looks: Multilook des ``.slc`` à indexer (``"1x1"`` pour le SLC pleine
            résolution ; ``"1x4"`` ou ``"2x8"`` s'ils sont présents).

    Returns:
        Un ``DataFrame`` trié par (track, ordre ONERA des polarisations), avec
        les colonnes ``track, pol, pol_index, flight_id, data_take, date,
        acquisition_date, segment, slc_path, ann_path, slc_bytes, ann_exists``.
        Les chemins sont des ``pathlib.Path`` absolus.

    Raises:
        FileNotFoundError: si ``data_dir`` n'existe pas.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Répertoire de données introuvable : {directory}")

    rows: list[dict[str, Any]] = []
    for slc_path in sorted(directory.glob(f"*_{looks}.slc")):
        fields = parse_slc_filename(slc_path)
        if fields is None:
            continue
        # Le .ann est commun aux multilooks : même préfixe, sans le suffixe _sN_LxL.
        ann_name = re.sub(r"_s\d+_\d+x\d+\.slc$", ".ann", slc_path.name)
        ann_path = directory / ann_name
        ann = parse_ann(ann_path) if ann_path.is_file() else {}
        track = ann.get("Stack Line Number")
        rows.append(
            {
                "track": int(track) if isinstance(track, (int, float)) else None,
                "pol": fields["pol"],
                "pol_index": (
                    POL_ORDER_ONERA.index(fields["pol"])
                    if fields["pol"] in POL_ORDER_ONERA
                    else -1
                ),
                "flight_id": fields["flight_id"],
                "data_take": fields["data_take"],
                "date": fields["date"],
                "acquisition_date": pd.to_datetime(fields["date"], format="%y%m%d"),
                "segment": fields["segment"],
                "slc_path": slc_path.resolve(),
                "ann_path": ann_path.resolve(),
                "slc_bytes": slc_path.stat().st_size,
                "ann_exists": ann_path.is_file(),
            }
        )

    index = pd.DataFrame(
        rows,
        columns=[
            "track", "pol", "pol_index", "flight_id", "data_take", "date",
            "acquisition_date", "segment", "slc_path", "ann_path",
            "slc_bytes", "ann_exists",
        ],
    )
    if index.empty:
        return index
    return (
        index.sort_values(["track", "pol_index"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Parsing des annotations .ann
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AnnEntry:
    """Une ligne ``clé = valeur`` d'un fichier d'annotation UAVSAR.

    Attributes:
        key: Clé nettoyée, unité retirée (ex. ``"slc_1_1x1 Rows"``).
        unit: Unité déclarée entre parenthèses (``"pixels"``, ``"m"``, ...).
            ``None`` si absente ; ``"&"`` et ``"-"`` (chaîne / sans unité dans la
            convention JPL) sont conservés tels quels.
        value: Valeur convertie en ``int``/``float`` si possible, sinon ``str``.
        raw: Valeur brute, avant conversion et sans le commentaire.
        comment: Commentaire de fin de ligne (après ``;``), ou ``None``.
    """

    key: str
    unit: str | None
    value: Any
    raw: str
    comment: str | None


def _coerce(text: str) -> Any:
    """Convertit une valeur d'annotation en ``int`` ou ``float`` si possible."""
    if not text:
        return ""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_ann_entries(ann_path: Path | str) -> dict[str, AnnEntry]:
    """Parse un fichier ``.ann`` en conservant unités, valeurs brutes et commentaires.

    Le parseur ne suppose **aucun numéro de ligne fixe** (l'en-tête JPL invite
    d'ailleurs explicitement à chercher les paramètres par leur nom). Il :

    * ignore les lignes vides et les commentaires (``;`` en début **ou** milieu
      de ligne) ;
    * découpe sur le **premier** ``=`` seulement (des valeurs comme l'URL du
      produit contiennent elles-mêmes des ``=``) ;
    * détache l'unité entre parenthèses située juste avant le ``=`` ;
    * tente une conversion numérique de la valeur.

    Args:
        ann_path: Chemin du fichier d'annotation.

    Returns:
        Un dictionnaire ``clé -> AnnEntry``, dans l'ordre du fichier.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    path = Path(ann_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier d'annotation introuvable : {path}")

    entries: dict[str, AnnEntry] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "=" not in line:
                continue
            body, _, comment_text = line.partition(";")
            if "=" not in body:  # la ligne était entièrement un commentaire
                continue
            left, _, right = body.partition("=")  # premier '=' uniquement
            key_match = _ANN_KEY_UNIT_RE.match(left.strip())
            key = (key_match.group("key") or "").strip() if key_match else left.strip()
            unit = key_match.group("unit") if key_match else None
            if not key:
                continue
            raw = right.strip()
            entries[key] = AnnEntry(
                key=key,
                unit=unit.strip() if unit is not None else None,
                value=_coerce(raw),
                raw=raw,
                comment=comment_text.strip() or None,
            )
    return entries


def parse_ann(ann_path: Path | str) -> dict[str, Any]:
    """Parse un ``.ann`` et renvoie un simple dictionnaire ``clé -> valeur``.

    Version condensée de :func:`parse_ann_entries` (unités et commentaires
    écartés), pratique pour les recherches directes.

    Args:
        ann_path: Chemin du fichier d'annotation.

    Returns:
        Un dictionnaire ``clé -> valeur`` (valeurs numériques converties).
    """
    return {key: entry.value for key, entry in parse_ann_entries(ann_path).items()}


def find_ann_keys(
    ann: Mapping[str, Any],
    pattern: str,
    flags: int = re.IGNORECASE,
) -> dict[str, Any]:
    """Filtre les clés d'une annotation par expression régulière.

    Utile pour retrouver une clé dont le libellé exact varierait d'un produit à
    l'autre (``"Rows"``, ``"Columns"``, ``"Pixel Spacing"``, ...) sans jamais la
    coder en dur.

    Args:
        ann: Dictionnaire issu de :func:`parse_ann` ou :func:`parse_ann_entries`.
        pattern: Motif de recherche appliqué à la clé.
        flags: Drapeaux ``re`` (insensible à la casse par défaut).

    Returns:
        Le sous-dictionnaire des paires dont la clé contient le motif.
    """
    regex = re.compile(pattern, flags)
    return {key: value for key, value in ann.items() if regex.search(key)}


# --------------------------------------------------------------------------- #
# Géométrie SLC
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SlcGeometry:
    """Métadonnées géométriques d'un ``.slc``, lues dans son ``.ann``.

    Attributes:
        n_rows: Nombre de lignes = échantillons **azimut**.
        n_cols: Nombre de colonnes = échantillons **range** (slant range).
        looks: Multilook du produit (``"1x1"``, ``"1x4"``, ``"2x8"``).
        range_pixel_spacing_m: Pas pixel en range (m).
        azimuth_pixel_spacing_m: Pas pixel en azimut (m).
        starting_slant_range_km: Slant range du premier échantillon (km).
        min_look_angle_deg: Angle de visée minimal (deg).
        max_look_angle_deg: Angle de visée maximal (deg).
        center_wavelength_cm: Longueur d'onde centrale (cm).
        n_range_looks: Nombre de vues en range (1 en 1x1).
        n_azimuth_looks: Nombre de vues en azimut (1 en 1x1).
        polarization: Polarisation déclarée dans le ``.ann`` (``"HH"``, ...).
        stack_line_number: Numéro de track dans la pile.
        acquisition_start: Date/heure de début d'acquisition (chaîne brute).
        bytes_per_pixel: Octets par pixel déclarés (attendu : 8).
        ann_path: Fichier d'annotation d'origine.
    """

    n_rows: int
    n_cols: int
    looks: str
    range_pixel_spacing_m: float | None = None
    azimuth_pixel_spacing_m: float | None = None
    starting_slant_range_km: float | None = None
    min_look_angle_deg: float | None = None
    max_look_angle_deg: float | None = None
    center_wavelength_cm: float | None = None
    n_range_looks: int | None = None
    n_azimuth_looks: int | None = None
    polarization: str | None = None
    stack_line_number: int | None = None
    acquisition_start: str | None = None
    bytes_per_pixel: int = SLC_BYTES_PER_PIXEL
    ann_path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Forme ``(n_rows, n_cols)`` = (azimut, range)."""
        return (self.n_rows, self.n_cols)

    @property
    def n_pixels(self) -> int:
        """Nombre total de pixels."""
        return self.n_rows * self.n_cols

    @property
    def expected_bytes(self) -> int:
        """Taille attendue du ``.slc`` : ``n_rows * n_cols * bytes_per_pixel``."""
        return self.n_pixels * self.bytes_per_pixel

    def extent_m(self) -> tuple[float, float] | None:
        """Étendue physique approximative ``(azimut_m, range_m)``, si les pas sont connus."""
        if self.azimuth_pixel_spacing_m is None or self.range_pixel_spacing_m is None:
            return None
        return (
            self.n_rows * self.azimuth_pixel_spacing_m,
            self.n_cols * self.range_pixel_spacing_m,
        )

    def summary(self) -> dict[str, Any]:
        """Résumé plat des champs, pour affichage ou construction d'un ``DataFrame``."""
        return {
            "track": self.stack_line_number,
            "pol": self.polarization,
            "looks": self.looks,
            "n_rows (azimut)": self.n_rows,
            "n_cols (range)": self.n_cols,
            "range_pixel_spacing_m": self.range_pixel_spacing_m,
            "azimuth_pixel_spacing_m": self.azimuth_pixel_spacing_m,
            "starting_slant_range_km": self.starting_slant_range_km,
            "min_look_angle_deg": self.min_look_angle_deg,
            "max_look_angle_deg": self.max_look_angle_deg,
            "center_wavelength_cm": self.center_wavelength_cm,
            "n_range_looks": self.n_range_looks,
            "n_azimuth_looks": self.n_azimuth_looks,
            "acquisition_start": self.acquisition_start,
            "expected_bytes": self.expected_bytes,
        }


def _first_int(ann: Mapping[str, Any], pattern: str) -> int | None:
    """Première valeur entière dont la clé correspond au motif, sinon ``None``."""
    for value in find_ann_keys(ann, pattern).values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _get_number(ann: Mapping[str, Any], *keys: str) -> float | None:
    """Première clé exacte présente et numérique parmi ``keys``, sinon ``None``."""
    for key in keys:
        value = ann.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def read_slc_geometry(
    ann_path: Path | str,
    looks: str = "1x1",
    segment: int = 1,
) -> SlcGeometry:
    """Lit dans un ``.ann`` la géométrie du ``.slc`` correspondant.

    Les dimensions sont cherchées via les clés ``slc_<segment>_<looks> Rows`` et
    ``... Columns``. Si ces clés exactes sont absentes (variante de produit), on
    retombe sur une recherche par expression régulière restreinte au bon
    multilook, puis sur la première clé ``Rows``/``Columns`` trouvée — aucune
    dimension n'est jamais codée en dur.

    Args:
        ann_path: Chemin du fichier d'annotation.
        looks: Multilook visé (``"1x1"``, ``"1x4"``, ``"2x8"``).
        segment: Numéro de segment (1 pour cette pile mono-segment).

    Returns:
        La :class:`SlcGeometry` correspondante.

    Raises:
        FileNotFoundError: si le ``.ann`` n'existe pas.
        KeyError: si les dimensions restent introuvables.
    """
    ann = parse_ann(ann_path)
    prefix = f"slc_{segment}_{looks}"

    n_rows = ann.get(f"{prefix} Rows")
    n_cols = ann.get(f"{prefix} Columns")
    if not isinstance(n_rows, int):
        n_rows = _first_int(ann, rf"{re.escape(prefix)}.*Rows") or _first_int(ann, r"\bRows\b")
    if not isinstance(n_cols, int):
        n_cols = _first_int(ann, rf"{re.escape(prefix)}.*Columns") or _first_int(ann, r"\bColumns\b")
    if n_rows is None or n_cols is None:
        available = sorted(find_ann_keys(ann, r"Rows|Columns"))
        raise KeyError(
            f"Dimensions introuvables pour '{prefix}' dans {ann_path}. "
            f"Clés candidates présentes : {available}"
        )

    bytes_per_pixel = ann.get("SLC Bytes Per Pixel")
    return SlcGeometry(
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        looks=looks,
        range_pixel_spacing_m=_get_number(ann, f"{looks} SLC Range Pixel Spacing"),
        azimuth_pixel_spacing_m=_get_number(ann, f"{looks} SLC Azimuth Pixel Spacing"),
        starting_slant_range_km=_get_number(ann, "Image Starting Slant Range"),
        min_look_angle_deg=_get_number(ann, "Minimum Look Angle"),
        max_look_angle_deg=_get_number(ann, "Maximum Look Angle"),
        center_wavelength_cm=_get_number(ann, "Center Wavelength"),
        n_range_looks=_first_int(ann, rf"Number of Range Looks in {re.escape(looks)} SLC"),
        n_azimuth_looks=_first_int(ann, rf"Number of Azimuth Looks in {re.escape(looks)} SLC"),
        polarization=str(ann["Polarization"]).strip() if "Polarization" in ann else None,
        stack_line_number=(
            int(ann["Stack Line Number"])
            if isinstance(ann.get("Stack Line Number"), (int, float))
            else None
        ),
        acquisition_start=(
            str(ann["Start Time of Acquisition"]).strip()
            if "Start Time of Acquisition" in ann
            else None
        ),
        bytes_per_pixel=(
            int(bytes_per_pixel) if isinstance(bytes_per_pixel, int) else SLC_BYTES_PER_PIXEL
        ),
        ann_path=Path(ann_path).resolve(),
    )


# --------------------------------------------------------------------------- #
# Ouverture memmap (lecture paresseuse)
# --------------------------------------------------------------------------- #

def open_slc_memmap(
    slc_path: Path | str,
    geometry: SlcGeometry | tuple[int, int],
    strict_size: bool = True,
) -> np.memmap:
    """Ouvre un ``.slc`` en lecture seule via ``numpy.memmap``.

    **Rien n'est chargé en RAM** : l'objet retourné est une vue paresseuse sur le
    fichier ; seules les pages effectivement lues transitent par le cache disque.
    C'est le seul point d'entrée autorisé pour les ``.slc`` du projet (~4,5 Gio
    par fichier, ~90 Go pour la pile).

    Args:
        slc_path: Chemin du fichier ``.slc``.
        geometry: :class:`SlcGeometry` ou couple ``(n_rows, n_cols)``.
        strict_size: Si vrai, vérifie que la taille du fichier vaut exactement
            ``n_rows * n_cols * 8`` et lève une erreur explicite sinon.

    Returns:
        Un ``numpy.memmap`` de dtype ``complex64`` et de forme ``(n_rows, n_cols)``.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si la taille du fichier contredit les dimensions annoncées.
    """
    path = Path(slc_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier SLC introuvable : {path}")

    shape = geometry.shape if isinstance(geometry, SlcGeometry) else tuple(geometry)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    expected = n_rows * n_cols * SLC_BYTES_PER_PIXEL
    actual = path.stat().st_size
    if strict_size and actual != expected:
        raise ValueError(
            f"Taille incohérente pour {path.name} : {actual} octets lus, "
            f"{expected} attendus ({n_rows} lignes x {n_cols} colonnes x "
            f"{SLC_BYTES_PER_PIXEL} octets). Vérifier le multilook et le segment "
            f"utilisés pour lire le .ann, ou l'intégrité du téléchargement."
        )
    return np.memmap(path, dtype=SLC_DTYPE, mode="r", shape=(n_rows, n_cols))


def open_vector_memmap(
    path: Path | str,
    shape: tuple[int, int],
    n_channels: int = LLH_LKV_N_CHANNELS,
) -> np.memmap:
    """Ouvre un ``.llh`` ou ``.lkv`` en ``numpy.memmap`` de forme ``(rows, cols, 3)``.

    Ces fichiers sont des float32 little-endian entrelacés par pixel
    (``lat1 lon1 hgt1 lat2 ...`` pour ``.llh``, ``east north up`` pour ``.lkv``).
    Attention : ils sont fournis en **2x8** pour cette pile alors que les
    ``.slc`` exploités sont en **1x1** — les grilles ne coïncident pas.

    Args:
        path: Chemin du ``.llh`` ou ``.lkv``.
        shape: ``(n_rows, n_cols)`` de la grille 2x8, lues dans le ``.ann``
            (clés ``llh_1_2x8 Rows`` / ``Columns``).
        n_channels: Nombre de canaux par pixel (3).

    Returns:
        Un ``numpy.memmap`` float32 de forme ``(n_rows, n_cols, n_channels)``.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si la taille du fichier contredit la forme annoncée.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Fichier vecteur (llh/lkv) introuvable : {file_path}")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    expected = n_rows * n_cols * n_channels * LLH_LKV_DTYPE.itemsize
    actual = file_path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"Taille incohérente pour {file_path.name} : {actual} octets lus, "
            f"{expected} attendus ({n_rows} x {n_cols} x {n_channels} x "
            f"{LLH_LKV_DTYPE.itemsize} octets)."
        )
    return np.memmap(
        file_path, dtype=LLH_LKV_DTYPE, mode="r", shape=(n_rows, n_cols, n_channels)
    )


# --------------------------------------------------------------------------- #
# Lectures bornées : fenêtre, décimation, patch
# --------------------------------------------------------------------------- #

def _check_dense_budget(n_elements: int, itemsize: int, max_bytes: int, what: str) -> None:
    """Lève une ``MemoryError`` explicite si une lecture dense dépasse le budget."""
    n_bytes = n_elements * itemsize
    if n_bytes > max_bytes:
        raise MemoryError(
            f"{what} matérialiserait {n_bytes / 1024**2:.1f} Mio en RAM, au-delà du "
            f"budget de {max_bytes / 1024**2:.1f} Mio. Réduire la taille de la "
            f"fenêtre, augmenter le pas de décimation, ou relever `max_bytes` "
            f"en connaissance de cause."
        )


def read_window(
    source: np.memmap | np.ndarray,
    row0: int,
    col0: int,
    n_rows: int,
    n_cols: int,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> np.ndarray:
    """Lit une tuile dense et **bornée** depuis un memmap.

    Args:
        source: Memmap (ou tableau) source, de forme ``(H, W)``.
        row0: Ligne (azimut) du coin haut-gauche.
        col0: Colonne (range) du coin haut-gauche.
        n_rows: Hauteur de la fenêtre en pixels.
        n_cols: Largeur de la fenêtre en pixels.
        max_bytes: Plafond mémoire de la lecture.

    Returns:
        Un ``numpy.ndarray`` dense de forme ``(n_rows, n_cols)``, copié en RAM.

    Raises:
        ValueError: si la fenêtre déborde du tableau ou a une taille non positive.
        MemoryError: si la fenêtre dépasse ``max_bytes``.
    """
    height, width = source.shape[:2]
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"Fenêtre de taille non positive : {n_rows} x {n_cols}.")
    if row0 < 0 or col0 < 0 or row0 + n_rows > height or col0 + n_cols > width:
        raise ValueError(
            f"Fenêtre [{row0}:{row0 + n_rows}, {col0}:{col0 + n_cols}] hors de "
            f"l'image de forme {(height, width)}."
        )
    _check_dense_budget(n_rows * n_cols, source.dtype.itemsize, max_bytes, "La fenêtre")
    return np.asarray(source[row0 : row0 + n_rows, col0 : col0 + n_cols])


def decimation_stride(shape: Sequence[int], target_max_size: int = 2000) -> int:
    """Calcule le pas de décimation pour ramener le grand axe sous une cible.

    Args:
        shape: Forme ``(n_rows, n_cols)`` de l'image pleine résolution.
        target_max_size: Taille maximale visée sur le plus grand axe (px).

    Returns:
        Un pas entier ``>= 1`` tel que ``max(shape) / stride <= target_max_size``.
    """
    if target_max_size <= 0:
        raise ValueError("`target_max_size` doit être strictement positif.")
    return max(1, math.ceil(max(shape[:2]) / target_max_size))


def read_decimated(
    source: np.memmap | np.ndarray,
    stride: int | None = None,
    target_max_size: int = 2000,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> tuple[np.ndarray, int]:
    """Lit une version **sous-échantillonnée** (stride) de l'image entière.

    Aucune interpolation ni moyennage : on prélève un pixel sur ``stride`` dans
    chaque direction (``source[::stride, ::stride]``). C'est volontaire :

    * la moyenne locale d'un SLC complexe n'a pas de sens sans multilook propre,
      et lisserait le chatoiement que l'on veut justement observer ;
    * la décimation en **lignes** divise d'autant le volume réellement lu sur
      disque (une ligne de la pile fait ~79 Kio, on n'en touche qu'une sur
      ``stride``), ce qui rend l'aperçu praticable sur un fichier de 4,5 Gio ;
    * le tableau dense résultant reste minuscule (~1979 x 319 complexes, soit
      ~5 Mio, pour la pile Rosamond avec ``target_max_size=2000``).

    Args:
        source: Memmap source de forme ``(H, W)``.
        stride: Pas imposé ; si ``None``, calculé via :func:`decimation_stride`.
        target_max_size: Cible sur le grand axe quand ``stride`` est ``None``.
        max_bytes: Plafond mémoire de la lecture.

    Returns:
        Le couple ``(image_décimée, stride_utilisé)``.

    Raises:
        MemoryError: si le résultat dépasse ``max_bytes``.
    """
    step = int(stride) if stride is not None else decimation_stride(source.shape, target_max_size)
    if step < 1:
        raise ValueError(f"Le pas de décimation doit valoir au moins 1 (reçu {step}).")
    view = source[::step, ::step]
    _check_dense_budget(
        int(np.prod(view.shape)), source.dtype.itemsize, max_bytes, "L'aperçu décimé"
    )
    return np.asarray(view), step


def read_scattering_patch(
    memmaps: Mapping[str, np.memmap],
    row: int,
    col: int,
    size: int = 64,
    pol_order: Sequence[str] = POL_ORDER_ONERA,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> np.ndarray:
    """Extrait un patch du **vecteur de diffusion** sur les 4 polarisations.

    Le tenseur retourné suit la convention ONERA ``s = [S_Hh, S_Vh, S_Hv, S_Vv]``
    (cf. :data:`POL_ORDER_ONERA`). Les 4 canaux sont conservés distincts : même
    si la réciprocité physique fait attendre ``S_Hv ≈ S_Vh``, c'est justement
    l'écart entre eux qui porte l'information de crosstalk.

    Args:
        memmaps: Dictionnaire ``polarisation -> memmap`` (les 4 polarisations
            d'une même track, de dimensions identiques).
        row: Ligne (azimut) du coin haut-gauche du patch.
        col: Colonne (range) du coin haut-gauche du patch.
        size: Côté du patch carré, en pixels.
        pol_order: Ordre des canaux ; ne pas changer sans raison.
        max_bytes: Plafond mémoire de la lecture (les 4 canaux cumulés).

    Returns:
        Un ``numpy.ndarray`` ``complex64`` de forme ``(4, size, size)``.

    Raises:
        KeyError: si une polarisation manque dans ``memmaps``.
        ValueError: si les memmaps n'ont pas tous la même forme, ou si le patch
            déborde de l'image.
        MemoryError: si le patch dépasse ``max_bytes``.
    """
    missing = [pol for pol in pol_order if pol not in memmaps]
    if missing:
        raise KeyError(
            f"Polarisation(s) absente(s) de `memmaps` : {missing}. "
            f"Disponibles : {sorted(memmaps)}"
        )
    shapes = {pol: tuple(memmaps[pol].shape) for pol in pol_order}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Formes incohérentes entre polarisations : {shapes}")

    _check_dense_budget(
        len(pol_order) * size * size, SLC_DTYPE.itemsize, max_bytes, "Le patch"
    )
    patch = np.empty((len(pol_order), size, size), dtype=SLC_DTYPE)
    for channel, pol in enumerate(pol_order):
        patch[channel] = read_window(memmaps[pol], row, col, size, size, max_bytes=max_bytes)
    return patch


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def amplitude_db(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convertit une amplitude complexe **linéaire** en décibels.

    Les ``.slc`` UAVSAR sont en amplitude linéaire (``SLC Data Units = linear
    amplitude and phase``) : l'affichage passe donc par
    ``20 * log10(|x| + eps)``, l'``eps`` évitant ``-inf`` sur les pixels nuls
    (bords de fauchée, zones non imagées).

    Args:
        values: Tableau complexe (ou réel) d'amplitudes linéaires.
        eps: Terme de régularisation ajouté au module.

    Returns:
        Un tableau ``float32`` d'amplitudes en dB, de même forme que l'entrée.
    """
    return (20.0 * np.log10(np.abs(values).astype(np.float64) + eps)).astype(np.float32)


def plot_amplitude_db(
    ax: "Any",
    image_db: np.ndarray,
    title: str = "",
    stride: int = 1,
    row_offset: int = 0,
    col_offset: int = 0,
    geometry: SlcGeometry | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
    add_colorbar: bool = True,
) -> "Any":
    """Affiche une image d'amplitude en dB avec des axes explicites.

    Les axes sont libellés « range (colonnes) » et « azimut (lignes) », et
    gradués en pixels **pleine résolution** (le pas de décimation est réintégré),
    afin que les coordonnées lues sur la figure servent directement à
    :func:`read_scattering_patch`.

    Args:
        ax: ``matplotlib.axes.Axes`` cible.
        image_db: Image en dB (typiquement décimée).
        title: Titre de l'axe.
        stride: Pas de décimation appliqué pour obtenir ``image_db``.
        row_offset: Ligne (azimut) pleine résolution du coin haut-gauche, si
            ``image_db`` est une tuile extraite ailleurs que l'origine.
        col_offset: Colonne (range) pleine résolution du coin haut-gauche.
        geometry: Géométrie du produit ; si fournie, les pas pixel sont rappelés
            dans les libellés d'axes.
        percentiles: Bornes de contraste (percentiles des valeurs finies), utilisées
            seulement si ``vmin``/``vmax`` ne sont pas fournis.
        vmin: Borne basse imposée (dB). À renseigner pour comparer plusieurs
            panneaux sous une **même** échelle de couleur.
        vmax: Borne haute imposée (dB).
        cmap: Colormap matplotlib. L'amplitude est une grandeur de magnitude :
            on reste sur une rampe monotone à une seule teinte (niveaux de gris,
            usage standard en SAR), jamais un arc-en-ciel.
        add_colorbar: Ajoute une barre de couleur en dB.

    Returns:
        L'objet ``AxesImage`` créé.
    """
    if vmin is None or vmax is None:
        finite = image_db[np.isfinite(image_db)]
        auto_min, auto_max = (
            np.percentile(finite, percentiles) if finite.size else (None, None)
        )
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    n_rows, n_cols = image_db.shape[:2]
    mappable = ax.imshow(
        image_db,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        aspect="auto",
        extent=(
            col_offset,
            col_offset + n_cols * stride,
            row_offset + n_rows * stride,
            row_offset,
        ),
        interpolation="nearest",
    )
    range_spacing = geometry.range_pixel_spacing_m if geometry else None
    azimuth_spacing = geometry.azimuth_pixel_spacing_m if geometry else None
    ax.set_xlabel(
        "range — colonnes (px)"
        + (f", pas {range_spacing:.3f} m" if range_spacing else "")
    )
    ax.set_ylabel(
        "azimut — lignes (px)"
        + (f", pas {azimuth_spacing:.3f} m" if azimuth_spacing else "")
    )
    if title:
        ax.set_title(title)
    if add_colorbar:
        figure = ax.get_figure()
        figure.colorbar(mappable, ax=ax, label="amplitude (dB)", fraction=0.046, pad=0.04)
    return mappable


def plot_amplitude_histogram(
    ax: "Any",
    images_db: Mapping[str, np.ndarray],
    bins: int = 160,
    range_db: tuple[float, float] | None = None,
    pol_order: Sequence[str] = POL_ORDER_ONERA,
) -> "Any":
    """Superpose les histogrammes d'amplitude (dB) des polarisations d'une track.

    L'identité des courbes n'est jamais portée par la seule couleur : la légende
    est toujours présente, et le style de trait distingue co-polarisation (plein)
    et cross-polarisation (tireté) — un encodage secondaire qui double
    l'information de couleur tout en restant physiquement parlant.

    Args:
        ax: ``matplotlib.axes.Axes`` cible.
        images_db: Dictionnaire ``polarisation -> image en dB`` (typiquement des
            aperçus décimés).
        bins: Nombre de classes de l'histogramme.
        range_db: Bornes ``(min, max)`` en dB ; par défaut les percentiles 0,5 %
            et 99,5 % de l'ensemble des canaux.
        pol_order: Ordre de tracé des polarisations.

    Returns:
        L'``Axes`` utilisé.
    """
    present = [pol for pol in pol_order if pol in images_db]
    if not present:
        raise ValueError("`images_db` ne contient aucune des polarisations attendues.")
    if range_db is None:
        pooled = np.concatenate(
            [images_db[pol][np.isfinite(images_db[pol])].ravel() for pol in present]
        )
        range_db = tuple(np.percentile(pooled, [0.5, 99.5]))

    for pol in present:
        values = images_db[pol]
        ax.hist(
            values[np.isfinite(values)].ravel(),
            bins=bins,
            range=range_db,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=POL_COLORS.get(pol),
            linestyle=POL_LINESTYLES.get(pol, "-"),
            label=f"{pol} ({'co' if pol in CO_POL else 'cross'}-pol)",
        )
    ax.set_xlabel("amplitude (dB) — 20·log10(|s| + eps)")
    ax.set_ylabel("densité de probabilité")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


# --------------------------------------------------------------------------- #
# Trièdres (réflecteurs de calibration) — optionnel
# --------------------------------------------------------------------------- #

def load_corner_reflectors(
    csv_path: Path | str,
    ids: Iterable[int] | None = None,
) -> pd.DataFrame | None:
    """Charge le CSV des trièdres de calibration, s'il est disponible.

    Le fichier attendu est celui publié par la page calibration du JPL pour le
    réseau de Rosamond (colonnes : identifiant, latitude, longitude, hauteur,
    azimut, élévation, longueur de côté ; les libellés exacts varient selon la
    date d'export, on ne les impose donc pas).

    Args:
        csv_path: Chemin du CSV.
        ids: Identifiants à conserver ; ``None`` pour tout garder.

    Returns:
        Le ``DataFrame`` correspondant, ou ``None`` si le fichier est absent
        (cas normal à ce stade du projet : le CSV est à récupérer plus tard).
    """
    path = Path(csv_path)
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if ids is None:
        return frame
    wanted = list(ids)
    id_columns = [
        column
        for column in frame.columns
        if re.search(r"\bid\b|corner|reflector", str(column), re.IGNORECASE)
    ]
    if not id_columns:
        return frame
    key = id_columns[0]
    return frame[pd.to_numeric(frame[key], errors="coerce").isin(wanted)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Façade : une pile UAVSAR
# --------------------------------------------------------------------------- #

@dataclass
class UavsarStack:
    """Façade d'accès à une pile UAVSAR coregistrée (un répertoire de données).

    Construit l'index des fichiers, met en cache les géométries lues dans les
    ``.ann`` et ouvre les ``.slc`` en ``numpy.memmap`` **à la demande**. Aucun
    ``.slc`` n'est lu tant qu'on ne demande pas explicitement une tuile, un
    aperçu décimé ou un patch.

    Attributes:
        data_dir: Répertoire des données.
        looks: Multilook des ``.slc`` indexés.
        index: ``DataFrame`` produit par :func:`discover_slc_files`.
    """

    data_dir: Path
    looks: str = "1x1"
    index: pd.DataFrame = field(default_factory=pd.DataFrame)
    _geometries: dict[tuple[int, str], SlcGeometry] = field(default_factory=dict, repr=False)
    _memmaps: dict[tuple[int, str], np.memmap] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, data_dir: Path | str, looks: str = "1x1") -> "UavsarStack":
        """Indexe un répertoire de pile et retourne la façade correspondante.

        Args:
            data_dir: Répertoire contenant les ``.slc``/``.ann``.
            looks: Multilook des ``.slc`` à indexer.

        Returns:
            Une :class:`UavsarStack` prête à l'emploi.

        Raises:
            FileNotFoundError: si le répertoire ou tout ``.slc`` est absent.
        """
        directory = Path(data_dir)
        index = discover_slc_files(directory, looks=looks)
        if index.empty:
            raise FileNotFoundError(
                f"Aucun fichier '*_{looks}.slc' trouvé dans {directory}. "
                f"Vérifier le chemin des données et le multilook demandé."
            )
        return cls(data_dir=directory, looks=looks, index=index)

    # -- introspection ----------------------------------------------------- #

    @property
    def tracks(self) -> list[int]:
        """Numéros de track présents, triés."""
        return sorted(int(t) for t in self.index["track"].dropna().unique())

    def polarizations(self, track: int) -> list[str]:
        """Polarisations disponibles pour une track, dans l'ordre ONERA."""
        available = set(self.index.loc[self.index["track"] == track, "pol"])
        return [pol for pol in POL_ORDER_ONERA if pol in available]

    def row(self, track: int, pol: str) -> pd.Series:
        """Ligne de l'index pour un couple (track, polarisation).

        Raises:
            KeyError: si le couple est absent de l'index.
        """
        selection = self.index[(self.index["track"] == track) & (self.index["pol"] == pol)]
        if selection.empty:
            raise KeyError(
                f"Couple (track={track}, pol={pol}) absent de l'index. "
                f"Tracks connues : {self.tracks} ; polarisations pour cette track : "
                f"{self.polarizations(track)}"
            )
        return selection.iloc[0]

    # -- métadonnées ------------------------------------------------------- #

    def geometry(self, track: int, pol: str = "HH") -> SlcGeometry:
        """Géométrie d'un ``.slc`` (lue une fois, puis mise en cache)."""
        key = (track, pol)
        if key not in self._geometries:
            self._geometries[key] = read_slc_geometry(
                self.row(track, pol)["ann_path"], looks=self.looks
            )
        return self._geometries[key]

    def geometry_table(self, track: int | None = None) -> pd.DataFrame:
        """Tableau récapitulatif des géométries (toutes tracks, ou une seule).

        Args:
            track: Track à décrire ; ``None`` pour toutes.

        Returns:
            Un ``DataFrame`` d'une ligne par couple (track, polarisation).
        """
        tracks = self.tracks if track is None else [track]
        records = [
            self.geometry(current, pol).summary()
            for current in tracks
            for pol in self.polarizations(current)
        ]
        return pd.DataFrame(records)

    # -- accès aux données ------------------------------------------------- #

    def memmap(self, track: int, pol: str) -> np.memmap:
        """Memmap ``complex64`` d'un ``.slc`` (ouvert une fois, puis mis en cache)."""
        key = (track, pol)
        if key not in self._memmaps:
            self._memmaps[key] = open_slc_memmap(
                self.row(track, pol)["slc_path"], self.geometry(track, pol)
            )
        return self._memmaps[key]

    def memmaps(self, track: int) -> dict[str, np.memmap]:
        """Memmaps des 4 polarisations d'une track, clés = polarisations."""
        return {pol: self.memmap(track, pol) for pol in self.polarizations(track)}

    def read_window(
        self,
        track: int,
        pol: str,
        row0: int,
        col0: int,
        n_rows: int,
        n_cols: int,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> np.ndarray:
        """Lit une tuile bornée d'un ``.slc`` (cf. :func:`read_window`)."""
        return read_window(
            self.memmap(track, pol), row0, col0, n_rows, n_cols, max_bytes=max_bytes
        )

    def read_decimated(
        self,
        track: int,
        pol: str,
        stride: int | None = None,
        target_max_size: int = 2000,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> tuple[np.ndarray, int]:
        """Lit un aperçu décimé d'un ``.slc`` (cf. :func:`read_decimated`)."""
        return read_decimated(
            self.memmap(track, pol),
            stride=stride,
            target_max_size=target_max_size,
            max_bytes=max_bytes,
        )

    def read_scattering_patch(
        self,
        track: int,
        row: int,
        col: int,
        size: int = 64,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> np.ndarray:
        """Extrait un patch ``(4, size, size)`` du vecteur de diffusion d'une track.

        Ordre des canaux : :data:`POL_ORDER_ONERA` = ``[HH, VH, HV, VV]``.
        C'est l'ossature de l'extraction de patchs du futur pipeline
        (baselines statistiques puis CVNN).

        Args:
            track: Numéro de track.
            row: Ligne (azimut) du coin haut-gauche.
            col: Colonne (range) du coin haut-gauche.
            size: Côté du patch carré, en pixels.
            max_bytes: Plafond mémoire de la lecture.

        Returns:
            Un ``numpy.ndarray`` ``complex64`` de forme ``(4, size, size)``.

        Raises:
            KeyError: si les 4 polarisations ne sont pas toutes disponibles.
            ValueError: si le patch déborde de l'image.
        """
        available = self.polarizations(track)
        if len(available) != len(POL_ORDER_ONERA):
            raise KeyError(
                f"Track {track} : {len(available)} polarisation(s) disponible(s) "
                f"({available}), les 4 canaux {list(POL_ORDER_ONERA)} sont requis "
                f"pour former un vecteur de diffusion."
            )
        return read_scattering_patch(
            self.memmaps(track), row, col, size=size, max_bytes=max_bytes
        )
