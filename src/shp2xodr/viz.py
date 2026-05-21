"""Raw 3D visualization of NGII A1 nodes and A2 links — no modifications."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyvista as pv

log = logging.getLogger(__name__)


def _a2_polydata(shp_dir: Path) -> pv.PolyData:
    """Build one PolyData containing every A2_LINK polyline."""
    path = shp_dir / "A2_LINK.shp"
    a2 = gpd.read_file(path)
    log.info("loaded %d A2 links from %s", len(a2), path)

    point_chunks: list[np.ndarray] = []
    line_cells: list[int] = []
    offset = 0
    for geom in a2.geometry:
        pts = np.asarray(geom.coords, dtype=np.float64)
        n_pts = len(pts)
        # pyvista line encoding: [n, idx0, idx1, ..., idx_{n-1}] per polyline
        line_cells.append(n_pts)
        line_cells.extend(range(offset, offset + n_pts))
        point_chunks.append(pts)
        offset += n_pts

    points = np.vstack(point_chunks)
    return pv.PolyData(points, lines=np.asarray(line_cells, dtype=np.int64))


def _a1_polydata(shp_dir: Path) -> pv.PolyData:
    """Build one PolyData containing every A1_NODE point."""
    path = shp_dir / "A1_NODE.shp"
    a1 = gpd.read_file(path)
    log.info("loaded %d A1 nodes from %s", len(a1), path)
    pts = np.array([(g.x, g.y, g.z) for g in a1.geometry], dtype=np.float64)
    return pv.PolyData(pts)


def show_raw(shp_dir: Path) -> None:
    """Open an interactive pyvista window with A1 (points) and A2 (lines)."""
    plotter = pv.Plotter()
    plotter.background_color = "white"
    plotter.add_mesh(_a2_polydata(shp_dir), color="black", line_width=2.5)
    plotter.add_mesh(
        _a1_polydata(shp_dir),
        color="crimson",
        point_size=8.0,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.show()
