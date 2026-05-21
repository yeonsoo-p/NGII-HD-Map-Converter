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


def _load(shp_path: Path) -> gpd.GeoDataFrame:
    if not shp_path.is_file():
        raise FileNotFoundError(shp_path)
    try:
        gdf = gpd.read_file(shp_path)
    except UnicodeDecodeError:
        # NGII sample data ships CPG files that claim UTF-8 but the DBF is
        # actually CP949 — retry with the legacy encoding before giving up.
        log.debug("retrying %s with cp949 after UTF-8 decode failure", shp_path.name)
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
