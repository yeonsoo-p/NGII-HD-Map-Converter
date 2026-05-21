"""3D visualization of NGII A1 nodes and A2 links.

Two viz subclasses share one ``show()`` implementation and override hooks for
colors, overlay extras, and pick info:

    :class:`RawViz` — every A2_LINK in black, A1 nodes as crimson dots.
    :class:`SegmentsViz` — A2_LINKs colored by segmentation bundle id (one
        color per future OpenDRIVE ``<road>``).

Either way, shift + left-click a link to log + display its id (and bundle id
where available) and highlight it in yellow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyvista as pv
import vtk
from numpy.typing import NDArray

from shp2xodr.segmentation import bundle_links
from shp2xodr.shp_io import a1_data, a2_data, load_a1_nodes, load_a2_links

log = logging.getLogger(__name__)


def _a1_polydata(a1: gpd.GeoDataFrame) -> pv.PolyData:
    _ids, pts = a1_data(a1)
    return pv.PolyData(pts)


def _a2_polydata(a2: gpd.GeoDataFrame) -> pv.PolyData:
    ids, polylines = a2_data(a2)
    line_cells: list[int] = []
    offset = 0
    for pts in polylines:
        n = len(pts)
        line_cells.append(n)
        line_cells.extend(range(offset, offset + n))
        offset += n
    poly = pv.PolyData(np.vstack(polylines), lines=np.asarray(line_cells, dtype=np.int64))
    poly.cell_data["link_id"] = ids
    return poly


class A2Viz:
    """Base 3D viz of A2_LINK polylines; subclasses customize colors and pick info.

    Keys (any subclass):
        2 — top-down orthographic
        3 — perspective
    Shift + left-click a link to highlight it in yellow and log/display its info.
    """

    def __init__(self, shp_dir: Path) -> None:
        self.a1 = load_a1_nodes(shp_dir)
        self.a2 = load_a2_links(shp_dir)
        log.info("loaded %d A1 nodes, %d A2 links", len(self.a1), len(self.a2))
        self.a2_poly = _a2_polydata(self.a2)
        self.plotter = pv.Plotter()
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.picker.PickFromListOn()

    # ---- subclass hooks ------------------------------------------------------

    def _cell_colors(self) -> NDArray[np.uint8]:
        """Initial RGB per A2_LINK cell."""
        return np.zeros((self.a2_poly.n_cells, 3), dtype=np.uint8)

    def _add_extras(self) -> None:
        """Add overlays beyond the A2 mesh (e.g. A1 dots). Default: nothing."""

    def _pick_text(self, cell_id: int) -> str:
        """Single-line info string used for both the on-screen overlay and the log."""
        return f"link id: {self.a2_poly.cell_data['link_id'][cell_id]}"

    # ---- helpers -------------------------------------------------------------

    def _add_view_keys(self) -> None:
        def _view_2d() -> None:
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()

        def _view_3d() -> None:
            self.plotter.disable_parallel_projection()
            self.plotter.view_isometric()

        self.plotter.add_key_event("2", _view_2d)
        self.plotter.add_key_event("3", _view_3d)

    # ---- entry point ---------------------------------------------------------

    def show(self) -> None:
        initial_rgb = self._cell_colors()
        self.a2_poly.cell_data["rgb"] = initial_rgb.copy()

        self.plotter.background_color = "white"
        a2_actor = self.plotter.add_mesh(
            self.a2_poly,
            scalars="rgb",
            rgb=True,
            line_width=2.5,
            show_scalar_bar=False,
        )
        self._add_extras()
        self.plotter.add_axes()
        self.plotter.add_text(
            "[2] top-down  [3] 3D  [shift+click] link id",
            position="lower_left",
            font_size=10,
        )
        self._add_view_keys()

        self.picker.AddPickList(a2_actor)

        def _on_left_press(_obj: Any, _event: str) -> None:
            iren = self.plotter.iren.interactor
            if not iren.GetShiftKey():
                return
            x, y = iren.GetEventPosition()
            self.picker.Pick(x, y, 0, self.plotter.renderer)
            cell_id = self.picker.GetCellId()
            if cell_id < 0:
                return
            text = self._pick_text(cell_id)
            log.info("picked %s", text)
            self.plotter.add_text(text, position="upper_right", name="link_id_text", font_size=12)
            rgb = self.a2_poly.cell_data["rgb"]
            rgb[:] = initial_rgb
            rgb[cell_id] = [255, 255, 0]
            self.a2_poly.cell_data["rgb"] = rgb
            self.plotter.render()

        self.plotter.iren.add_observer("LeftButtonPressEvent", _on_left_press)
        self.plotter.show()


class RawViz(A2Viz):
    """A2 lines in black, A1 nodes in crimson."""

    def _add_extras(self) -> None:
        self.plotter.add_mesh(
            _a1_polydata(self.a1),
            color="crimson",
            point_size=8.0,
            render_points_as_spheres=True,
        )


class SegmentsViz(A2Viz):
    """A2 lines colored by segmentation bundle id."""

    def __init__(self, shp_dir: Path) -> None:
        super().__init__(shp_dir)
        self.bundle_ids = bundle_links(self.a1, self.a2)
        self.palette = self._bundle_colors(int(self.bundle_ids.max()) + 1)
        self.a2_poly.cell_data["bundle_id"] = self.bundle_ids

    @staticmethod
    def _bundle_colors(n_bundles: int, seed: int = 42) -> NDArray[np.uint8]:
        """Deterministic, saturated RGB per bundle id — kept clear of pure black/white."""
        rng = np.random.default_rng(seed)
        return rng.integers(60, 240, size=(n_bundles, 3), dtype=np.uint8)

    def _cell_colors(self) -> NDArray[np.uint8]:
        return self.palette[self.bundle_ids]

    def _pick_text(self, cell_id: int) -> str:
        link_id = self.a2_poly.cell_data["link_id"][cell_id]
        bid = int(self.a2_poly.cell_data["bundle_id"][cell_id])
        return f"link id: {link_id}  bundle: {bid}"
