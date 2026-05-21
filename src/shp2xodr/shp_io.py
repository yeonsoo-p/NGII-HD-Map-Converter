"""SHP loaders that normalize column-name drift across NGII data-model versions."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

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
