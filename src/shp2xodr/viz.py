"""3D visualization of NGII A1 nodes and A2 links.

Two viz subclasses share one ``show()`` implementation and override hooks for
colors, overlay extras, and pick info:

    :class:`RawViz` — every A2_LINK in black, A1 nodes as small black dots.
    :class:`SegmentsViz` — mainline A2_LINKs colored by bundle id (one color
        per future OpenDRIVE ``<road>``), interior A2_LINKs colored by
        junction id (shared across all connecting lanes of one future
        ``<junction>``). A1 nodes are drawn as small black dots for spatial
        reference only.

Either way, shift + left-click a link to log + display its id (and bundle id
where available) and highlight it in yellow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from numpy.typing import NDArray

from shp2xodr.segmentation import Segmentation, segment_links
from shp2xodr.shp_io import a1_data, a2_data

_NODE_POINT_SIZE = 3.0


def _random_palette(n: int, seed: int) -> NDArray[np.uint8]:
    """Deterministic, saturated RGB rows — kept clear of pure black/white."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(60, 240, size=(n, 3), dtype=np.uint8)


log = logging.getLogger(__name__)


def _a1_polydata(shp_dir: Path) -> pv.PolyData:
    _ids, pts = a1_data(shp_dir)
    return pv.PolyData(pts)


def _a2_polydata(shp_dir: Path) -> pv.PolyData:
    ids, polylines = a2_data(shp_dir)
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
        self.shp_dir = shp_dir
        self.a2_poly = _a2_polydata(shp_dir)
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


def _add_a1_dots(plotter: pv.Plotter, shp_dir: Path) -> None:
    plotter.add_mesh(
        _a1_polydata(shp_dir),
        color="black",
        point_size=_NODE_POINT_SIZE,
        render_points_as_spheres=True,
    )


class RawViz(A2Viz):
    """A2 lines in black, A1 nodes as small black dots."""

    def _add_extras(self) -> None:
        _add_a1_dots(self.plotter, self.shp_dir)


class SegmentsViz(A2Viz):
    """A2 lines colored by junction (interior) or bundle (mainline).

    Interior links share one color per future ``<junction>`` so every
    connecting lane inside one intersection is visually unified. Mainline
    links get one color per future ``<road>`` (bundle). A1 nodes are drawn
    as small black dots for spatial reference only.
    """

    def __init__(self, shp_dir: Path) -> None:
        super().__init__(shp_dir)
        self.segmentation: Segmentation = segment_links(shp_dir)
        self.bundle_palette = _random_palette(int(self.segmentation.bundle_id.max()) + 1, seed=42)
        n_junctions = int(self.segmentation.node_junction_id.max()) + 1
        self.junction_palette = _random_palette(n_junctions, seed=137)
        self.a2_poly.cell_data["bundle_id"] = self.segmentation.bundle_id
        self.a2_poly.cell_data["junction_id"] = self.segmentation.junction_id

    def _cell_colors(self) -> NDArray[np.uint8]:
        bundle_rgb = self.bundle_palette[self.segmentation.bundle_id]
        is_interior = self.segmentation.junction_id >= 0
        rgb = np.asarray(bundle_rgb).copy()
        if is_interior.any():
            rgb[is_interior] = self.junction_palette[self.segmentation.junction_id[is_interior]]
        return rgb

    def _add_extras(self) -> None:
        _add_a1_dots(self.plotter, self.shp_dir)

    def _pick_text(self, cell_id: int) -> str:
        link_id = self.a2_poly.cell_data["link_id"][cell_id]
        bid = int(self.a2_poly.cell_data["bundle_id"][cell_id])
        jid = int(self.a2_poly.cell_data["junction_id"][cell_id])
        jstr = str(jid) if jid >= 0 else "-"
        return f"link {link_id}  bundle {bid}  junction {jstr}"
