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
    gdf = gpd.read_file(shp_path)
    rename = {src: dst for src, dst in _COLUMN_ALIASES.items() if src in gdf.columns}
    if rename:
        log.debug("normalized columns in %s: %s", shp_path.name, rename)
        gdf = gdf.rename(columns=rename)
    return gdf


def load_a1_nodes(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A1_NODE.shp")


def load_a2_links(shp_dir: Path) -> gpd.GeoDataFrame:
    return _load(shp_dir / "A2_LINK.shp")


def a1_data(a1: gpd.GeoDataFrame) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Flatten A1_NODE into ``(ids, points)`` arrays.

    ``ids`` has shape ``(N,)``; ``points`` has shape ``(N, 3)`` with XYZ in
    source CRS units.
    """
    ids = a1["ID"].astype(str).to_numpy()
    pts = np.array([(g.x, g.y, g.z) for g in a1.geometry], dtype=np.float64)
    return ids, pts


def a2_data(a2: gpd.GeoDataFrame) -> tuple[NDArray[np.str_], list[NDArray[np.float64]]]:
    """Return ``(ids, polylines)`` for A2_LINK.

    ``ids`` has shape ``(M,)``; ``polylines`` is a list of ``(N_i, 3)`` arrays,
    one per link, with XYZ vertices in source CRS units.
    """
    ids = a2["ID"].astype(str).to_numpy()
    polylines = [np.asarray(g.coords, dtype=np.float64) for g in a2.geometry]
    return ids, polylines
