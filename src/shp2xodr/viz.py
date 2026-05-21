"""Raw 3D visualization of NGII A1 nodes and A2 links — no modifications."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk

from shp2xodr.shp_io import load_a1_nodes, load_a2_links

log = logging.getLogger(__name__)


def _a2_polydata(shp_dir: Path) -> pv.PolyData:
    """Build one PolyData containing every A2_LINK polyline; tag cells with link ID."""
    a2 = load_a2_links(shp_dir)
    log.info("loaded %d A2 links", len(a2))

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
    poly = pv.PolyData(points, lines=np.asarray(line_cells, dtype=np.int64))
    poly.cell_data["link_id"] = a2["ID"].astype(str).to_numpy()
    return poly


def _a1_polydata(shp_dir: Path) -> pv.PolyData:
    """Build one PolyData containing every A1_NODE point."""
    a1 = load_a1_nodes(shp_dir)
    log.info("loaded %d A1 nodes", len(a1))
    pts = np.array([(g.x, g.y, g.z) for g in a1.geometry], dtype=np.float64)
    return pv.PolyData(pts)


def show_raw(shp_dir: Path) -> None:
    """Open an interactive pyvista window with A1 (points) and A2 (lines).

    Keys:
        2 — top-down orthographic (2D feel)
        3 — perspective (default 3D)
    Shift + left-click on a link to display its ID.
    """
    a2_poly = _a2_polydata(shp_dir)
    a1_poly = _a1_polydata(shp_dir)

    cell_colors = np.tile(np.array([0, 0, 0], dtype=np.uint8), (a2_poly.n_cells, 1))
    a2_poly.cell_data["rgb"] = cell_colors

    plotter = pv.Plotter()
    plotter.background_color = "white"
    a2_actor = plotter.add_mesh(
        a2_poly,
        scalars="rgb",
        rgb=True,
        line_width=2.5,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        a1_poly,
        color="crimson",
        point_size=8.0,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.add_text(
        "[2] top-down  [3] 3D  [shift+click] link id",
        position="lower_left",
        font_size=10,
    )

    def _view_2d() -> None:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    def _view_3d() -> None:
        plotter.disable_parallel_projection()
        plotter.view_isometric()

    plotter.add_key_event("2", _view_2d)
    plotter.add_key_event("3", _view_3d)

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)
    picker.PickFromListOn()
    picker.AddPickList(a2_actor)

    def _on_left_press(_obj: Any, _event: str) -> None:
        iren = plotter.iren.interactor
        if not iren.GetShiftKey():
            return
        x, y = iren.GetEventPosition()
        picker.Pick(x, y, 0, plotter.renderer)
        cell_id = picker.GetCellId()
        if cell_id < 0:
            return
        link_id = a2_poly.cell_data["link_id"][cell_id]
        log.info("picked link id=%s", link_id)
        plotter.add_text(
            f"link id: {link_id}",
            position="upper_right",
            name="link_id_text",
            font_size=12,
        )
        cell_colors[:] = [0, 0, 0]
        cell_colors[cell_id] = [255, 255, 0]
        a2_poly.cell_data["rgb"] = cell_colors
        plotter.render()

    plotter.iren.add_observer("LeftButtonPressEvent", _on_left_press)

    plotter.show()
