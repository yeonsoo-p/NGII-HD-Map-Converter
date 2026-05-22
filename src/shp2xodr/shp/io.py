"""SHP loaders that normalize column-name drift across NGII data-model versions."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_COLUMN_ALIASES: dict[str, str] = {
    "L_LinKID": "L_LinkID",
}


def _looks_like_cp949_mojibake(gdf: gpd.GeoDataFrame) -> bool:
    """CP949 bytes decoded as UTF-8 land in the half-width katakana block.

    NGII datasets routinely omit (or lie in) the .cpg sidecar, so pyogrio
    defaults to UTF-8 and silently mojibakes Korean DBF fields. Half-width
    katakana in any object column is the giveaway since real NGII text is
    Hangul + ASCII.
    """
    for col in gdf.select_dtypes(include=["object", "str"]):
        if col == "geometry":
            continue
        for v in gdf[col].dropna().astype(str):
            if any("｡" <= ch <= "ﾟ" for ch in v):
                return True
    return False


def _load(shp_path: Path) -> gpd.GeoDataFrame:
    if not shp_path.is_file():
        raise FileNotFoundError(shp_path)
    try:
        gdf = gpd.read_file(shp_path)
    except UnicodeDecodeError:
        log.debug("retrying %s with cp949 after UTF-8 decode failure", shp_path.name)
        gdf = gpd.read_file(shp_path, encoding="cp949")
    else:
        if _looks_like_cp949_mojibake(gdf):
            log.debug("retrying %s with cp949 after mojibake heuristic", shp_path.name)
            gdf = gpd.read_file(shp_path, encoding="cp949")
    rename = {src: dst for src, dst in _COLUMN_ALIASES.items() if src in gdf.columns}
    if rename:
        log.debug("normalized columns in %s: %s", shp_path.name, rename)
        gdf = gdf.rename(columns=rename)
    return gdf


def load_a1_nodes(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A1_NODE.shp")


def load_a2_links(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A2_LINK.shp")


def a1_data(shp_dir: Path) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Load A1_NODE and flatten into ``(ids, points)`` arrays.

    ``ids`` has shape ``(N,)``; ``points`` has shape ``(N, 3)`` with XYZ in
    source CRS units.
    """
    a1 = load_a1_nodes(shp_dir)
    ids = a1["ID"].astype(str).to_numpy()
    pts = np.array([(g.x, g.y, g.z) for g in a1.geometry], dtype=np.float64)
    return ids, pts


def a2_data(shp_dir: Path) -> tuple[NDArray[np.str_], list[NDArray[np.float64]]]:
    """Load A2_LINK and return ``(ids, polylines)``.

    ``ids`` has shape ``(M,)``; ``polylines`` is a list of ``(N_i, 3)`` arrays,
    one per link, with XYZ vertices in source CRS units.
    """
    a2 = load_a2_links(shp_dir)
    ids = a2["ID"].astype(str).to_numpy()
    polylines = [np.asarray(g.coords, dtype=np.float64) for g in a2.geometry]
    return ids, polylines


def load_a3_sections(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A3_DRIVEWAYSECTION.shp")


def load_a4_sections(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A4_SUBSIDIARYSECTION.shp")


def _outer_rings(gdf: gpd.GeoDataFrame) -> list[NDArray[np.float64]]:
    return [np.asarray(g.exterior.coords, dtype=np.float64) for g in gdf.geometry]


def a3_data(
    shp_dir: Path,
) -> tuple[NDArray[np.str_], list[NDArray[np.float64]], NDArray[np.str_], NDArray[np.str_]]:
    """Load A3_DRIVEWAYSECTION and return ``(ids, polygons, kinds, road_types)``.

    ``polygons`` is a list of outer-ring ``(N_i, 3)`` arrays (interior rings
    aren't present in NGII sample data and are dropped if encountered).
    ``kinds`` and ``road_types`` are the raw varchar codes — ``Kind`` ∈
    ``{'1' 주행구간, '7' 보호구역}``, ``RoadType`` ∈ ``{'1' 일반, '2' 터널,
    '3' 교량, '4' 지하차도, '5' 고가차도}``.
    """
    a3 = load_a3_sections(shp_dir)
    ids = a3["ID"].astype(str).to_numpy()
    rings = _outer_rings(a3)
    kinds = a3["Kind"].astype(str).to_numpy()
    road_types = a3["RoadType"].astype(str).to_numpy()
    return ids, rings, kinds, road_types


def a4_data(
    shp_dir: Path,
) -> tuple[NDArray[np.str_], list[NDArray[np.float64]], NDArray[np.str_], NDArray[np.str_]]:
    """Load A4_SUBSIDIARYSECTION and return ``(ids, polygons, subtypes, names)``."""
    a4 = load_a4_sections(shp_dir)
    ids = a4["ID"].astype(str).to_numpy()
    rings = _outer_rings(a4)
    subtypes = a4["SubType"].astype(str).to_numpy()
    names = a4["Name"].astype(str).to_numpy()
    return ids, rings, subtypes, names


def load_b2_lines(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "B2_SURFACELINEMARK.shp")


def load_c3_lines(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "C3_VEHICLEPROTECTIONSAFETY.shp")


def b2_data(
    shp_dir: Path,
) -> tuple[
    NDArray[np.str_],
    list[NDArray[np.float64]],
    NDArray[np.str_],
    NDArray[np.str_],
    NDArray[np.str_],
    NDArray[np.str_],
]:
    """Load B2_SURFACELINEMARK and return ``(ids, polylines, types, kinds, r_ids, l_ids)``.

    ``polylines`` is a list of ``(N_i, 3)`` XYZ arrays in source CRS units.
    ``Type`` is the line-style varchar code (solid/broken/double); ``Kind`` is
    the semantic class (중앙선, 차선, 정지선, 유턴선, …). ``r_ids`` and
    ``l_ids`` are A2_LINK IDs of the lanes on the right / left of the line in
    that lane's driving direction; either side may be the empty string when
    the line bounds the road on only one side (e.g. shoulder edge) or when
    its A2 reference falls outside the loaded section.
    """
    b2 = load_b2_lines(shp_dir)
    ids = b2["ID"].astype(str).to_numpy()
    polylines = [np.asarray(g.coords, dtype=np.float64) for g in b2.geometry]
    types = b2["Type"].astype(str).to_numpy()
    kinds = b2["Kind"].astype(str).to_numpy()
    r_ids = b2["R_LinkID"].fillna("").astype(str).to_numpy()
    l_ids = b2["L_LinkID"].fillna("").astype(str).to_numpy()
    return ids, polylines, types, kinds, r_ids, l_ids


def c3_data(
    shp_dir: Path,
) -> tuple[
    NDArray[np.str_],
    list[NDArray[np.float64]],
    NDArray[np.str_],
    NDArray[np.str_],
    NDArray[np.str_],
    NDArray[np.str_],
]:
    """Load C3_VEHICLEPROTECTIONSAFETY and return
    ``(ids, polylines, types, is_central, low_high, ref_ids)``.

    ``polylines`` is a list of ``(N_i, 3)`` XYZ arrays. ``Type`` is the
    facility-class varchar (가드레일 / 콘크리트연석 / 무단횡단방지시설 …);
    ``IsCentral`` is ``"0"`` road-edge or ``"1"`` central median; ``LowHigh``
    is ``"1"`` 상단 (top) or ``"2"`` 하단 (base); ``Ref_ID`` pairs the
    upper / lower polylines of one physical barrier.

    The NGII manual table 9.60 spells the column ``isCentral``, but real
    NGII SHPs ship it as ``IsCentral`` — we read the SHP name as-is rather
    than aliasing capitalization differences (the existing column-alias
    table is reserved for actual mojibake typos).
    """
    c3 = load_c3_lines(shp_dir)
    ids = c3["ID"].astype(str).to_numpy()
    polylines = [np.asarray(g.coords, dtype=np.float64) for g in c3.geometry]
    types = c3["Type"].astype(str).to_numpy()
    is_central = c3["IsCentral"].astype(str).to_numpy()
    low_high = c3["LowHigh"].astype(str).to_numpy()
    ref_ids = c3["Ref_ID"].fillna("").astype(str).to_numpy()
    return ids, polylines, types, is_central, low_high, ref_ids
