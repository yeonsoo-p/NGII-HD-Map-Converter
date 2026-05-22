"""3D scene builder for NGII HD-map (정밀도로지도) layers.

:class:`HdMapViz` builds the VTK actors for every layer (A1 / A2 / A3 / A4
/ B2 / C3) and binds shift-click pickers. The class is plotter-agnostic:
pass any ``pyvista.BasePlotter`` subclass — a vanilla ``pv.Plotter`` for
standalone use or a ``pyvistaqt.QtInteractor`` for the Qt inspector window.

Pick events are delivered through ``on_pick(kind, idx)`` — the scene does
not render pick info on-screen. The Qt window translates picks into
structured tab fields; standalone mode just logs them.

Layer visibility and the 3-level abstraction selector are exposed as plain
methods so the GUI can wire them to dock widgets:
:meth:`set_layer_visible`, :meth:`set_abstraction_level`.

Abstraction levels (passed to :meth:`set_abstraction_level`):

* ``1`` — None. A2 uniform neutral; B2 by paint color; everything else
  by natural NGII codes.
* ``2`` — Group. A2 colored per SHP group (lateral lane cluster within a
  road segment); B2 inherits its bound group's color (paint code ignored).
* ``3`` — Road & Junction. A2 mainline rows colored per OpenDRIVE road;
  A2 junction-interior rows colored per junction; B2 inherits the
  road/junction color of its bound side.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import shapely
import vtk
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon

from shp2xodr.shp.data import (
    A1Data,
    A2Data,
    A3Data,
    A4Data,
    B2Data,
    C3Data,
    LineLayerData,
    PointLayerData,
    PolygonLayerData,
)
from shp2xodr.shp.segmentation import Segmentation

log = logging.getLogger(__name__)


# ---- Rendering constants -------------------------------------------------------

_NODE_POINT_SIZE = 3.0
_POLY_OPACITY = 0.85
# Polygon offset for A3/A4 mappers: pushes coincident-Z polygons back so
# A2 lines render in front. Values chosen empirically — large enough to
# beat translucent-pass draw order, small enough to avoid visible gaps.
_POLY_DEPTH_OFFSET: tuple[float, float] = (2.0, 2.0)

# Line widths per layer (A2 thickest so the group/junction coloring stays
# visually dominant over markings + barriers).
_LW_A2 = 2.5
_LW_B2 = 1.4
_LW_C3 = 1.8
# Highlight overlay - fatter than every base layer so it always reads as
# "the picked line" even when multiple base cells share XY (단선 중앙선).
_LW_HIGHLIGHT = 4.5

# Picker tolerances. A1 dots are small (point picker); A2 has tight tol since
# the lines are thick. B2 / C3 are slimmer lines that pass over A3 / A4
# polygons whose own picker is point-in-polygon (tol=0). A tight tolerance on
# the line pickers would let the cursor "slip off" thin lines and have the
# polygon picker win; ~1.2% of screen makes line grabbing reliable without
# bleeding into wrong cells (B2 / C3 are still several screen-px apart at
# typical zoom).
_PICKER_TOL_A1 = 0.01
_PICKER_TOL_A2 = 0.005
_PICKER_TOL_THIN = 0.012
_PICKER_TOL_POLY = 0.0

# Slight cool off-white background so pure-white B2 lines (백색 차선 /
# 유도선 / 정지선) have contrast — they're invisible on a true #ffffff
# background. Dark enough to give white paint a visible edge, light enough
# that the A3 road fill (200,200,200) still reads as "darker than empty".
_BG_COLOR: tuple[float, float, float] = (0.86, 0.88, 0.90)

# Highlight overlay color. The overlay sits on top of every base layer at
# _LW_HIGHLIGHT thickness, so this color is what the user reads as "picked".
# Magenta wins against every base palette (B2 primaries, C3 neutrals, A3/A4
# cool fills, A2 random saturated range).
_HIGHLIGHT_RGB: tuple[int, int, int] = (255, 30, 200)

# Uniform color used for A2 (and B2 fallback) at abstraction level 1 — the
# "no segmentation overlay" view that lets the user read raw geometry.
_A2_UNIFORM_RGB: tuple[int, int, int] = (110, 110, 120)

# Abstraction levels — keep symbolic; the GUI radio buttons map to these
# integers, and viz.set_abstraction_level asserts 1 <= level <= 3.
_LEVEL_RAW = 1
_LEVEL_GROUP = 2
_LEVEL_ROAD_JUNCTION = 3
_DEFAULT_LEVEL = _LEVEL_ROAD_JUNCTION

# Enable VTK's coincident-topology resolution mode globally; per-mapper
# relative offsets above only take effect once this is on.
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

# The highlight overlay starts empty (no picks yet). PyVista 0.43+ refuses
# empty meshes by default; flipping this global theme bit lets us attach the
# actor up-front and swap geometry in on the first pick.
pv.global_theme.allow_empty_mesh = True


# ---- A3 / A4 polygon palettes --------------------------------------------------

# A3.RoadType → fill color (NGII codes from manual table 9.23).
_A3_ROAD_TYPE_RGB: dict[str, tuple[int, int, int]] = {
    "1": (200, 200, 200),  # 일반도로
    "2": (80, 80, 80),  # 터널
    "3": (140, 180, 220),  # 교량
    "4": (60, 40, 140),  # 지하차도
    "5": (200, 170, 120),  # 고가차도
}
_A3_PROTECTED_RGB: tuple[int, int, int] = (220, 80, 80)  # Kind=7 overrides RoadType color

# A4.SubType → fill color. SubType code list per data.A4Data.SUBTYPE_LABEL.
_A4_SUBTYPE_RGB: dict[str, tuple[int, int, int]] = {
    "1": (120, 200, 120),
    "2": (180, 220, 140),
    "3": (220, 200, 160),
    "4": (240, 180, 80),
    "5": (180, 140, 220),
}
_A4_FALLBACK_RGB: tuple[int, int, int] = (180, 180, 180)


# ---- B2 line palette -----------------------------------------------------------

# B2.Type is a 3-digit code; the first digit is the paint color:
# 1 황색, 2 백색, 3 청색, 9 기타 (see data.B2Data.TYPE_COLOR_LABEL).
_B2_PAINT_RGB: dict[str, tuple[int, int, int]] = {
    "1": (245, 235, 0),
    "2": (255, 255, 255),
    "3": (20, 90, 230),
}
_B2_PAINT_FALLBACK_RGB: tuple[int, int, int] = (130, 130, 130)


# ---- C3 line palette -----------------------------------------------------------

# C3.Type code (NGII manual table 9.61) → line color. C3 covers guardrails,
# concrete barriers, kerbs, pedestrian-crossing fences, and walls — real
# materials, so the palette stays in the neutral / warm-gray band that
# matches concrete + metal, well clear of B2's pure primaries and A3/A4's
# saturated polygon fills.
_C3_TYPE_RGB: dict[str, tuple[int, int, int]] = {
    "2": (140, 140, 150),
    "3": (210, 210, 200),
    "4": (190, 175, 140),
    "5": (50, 50, 50),
    "6": (170, 170, 170),
    "7": (220, 130, 30),
    "8": (90, 90, 90),
}
_C3_TYPE_FALLBACK_RGB: tuple[int, int, int] = (140, 140, 140)


# ---- Pick callback type --------------------------------------------------------

# (layer_name, row_idx). layer_name ∈ {"A1", "A2", "A3", "A4", "B2", "C3"}.
PickCallback = Callable[[str, int], None]


# ---- Generic helpers -----------------------------------------------------------


def _random_palette(n: int, seed: int) -> NDArray[np.uint8]:
    """Deterministic, saturated RGB rows — kept clear of pure black/white."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(60, 240, size=(n, 3), dtype=np.uint8)


def _polyline_polydata(polylines: list[NDArray[np.float64]]) -> pv.PolyData:
    """Stack a list of ``(N_i, 3)`` polylines into one ``pv.PolyData`` with
    one line cell per source polyline.
    """
    if not polylines:
        return pv.PolyData()
    line_cells: list[int] = []
    offset = 0
    for pts in polylines:
        n = len(pts)
        line_cells.append(n)
        line_cells.extend(range(offset, offset + n))
        offset += n
    return pv.PolyData(np.vstack(polylines), lines=np.asarray(line_cells, dtype=np.int64))


def _polygon_polydata(rings: list[NDArray[np.float64]]) -> pv.PolyData:
    """Triangulate each ring with shapely's constrained Delaunay.

    VTK's built-in triangulator (``vtkTriangleFilter`` / ``PolyData.triangulate``)
    fan-triangulates from one vertex, which produces triangles spilling outside
    the polygon for thin or non-convex shapes (sidewalks, curved bridges).
    Shapely's constrained Delaunay keeps every triangle inside the boundary
    and only uses input vertices, so we can look Z up by (x, y).

    ``cell_data['poly_idx']`` maps each triangle to its source ring so picking
    still resolves to the original polygon.
    """
    if not rings:
        return pv.PolyData()

    all_pts: list[NDArray[np.float64]] = []
    face_cells: list[int] = []
    tri_poly_idx: list[int] = []
    vert_offset = 0

    for poly_idx, pts in enumerate(rings):
        # xy → z lookup via key rounded to mm; NGII coords are in meters.
        z_by_xy: dict[tuple[int, int], float] = {
            (round(x * 1000), round(y * 1000)): z for x, y, z in pts
        }
        ring_xy = pts[:, :2]
        tris = shapely.constrained_delaunay_triangles(ShapelyPolygon(ring_xy))
        for tri in tris.geoms:
            tri_xy = np.asarray(tri.exterior.coords[:3], dtype=np.float64)
            tri_z = np.array([z_by_xy[(round(x * 1000), round(y * 1000))] for x, y in tri_xy])
            tri_xyz = np.column_stack([tri_xy, tri_z])
            all_pts.append(tri_xyz)
            face_cells.extend([3, vert_offset, vert_offset + 1, vert_offset + 2])
            vert_offset += 3
            tri_poly_idx.append(poly_idx)

    poly = pv.PolyData(np.vstack(all_pts), faces=np.asarray(face_cells, dtype=np.int64))
    poly.cell_data["poly_idx"] = np.asarray(tri_poly_idx, dtype=np.int32)
    return poly


# ---- Pickable layer wrappers ---------------------------------------------------


@dataclass(slots=True)
class _PointLayer:
    """One pickable point layer (A1 today; B1 / C1 later) wrapping a
    :class:`PointLayerData` record plus its render-time picker / actor.
    """

    name: str
    data: PointLayerData
    point_size: float
    picker_tolerance: float
    poly: pv.PolyData = field(init=False)
    actor: vtk.vtkActor | None = field(default=None, init=False)
    picker: vtk.vtkPointPicker = field(init=False)

    def __post_init__(self) -> None:
        self.poly = pv.PolyData(self.data.points)
        self.picker = vtk.vtkPointPicker()
        self.picker.SetTolerance(self.picker_tolerance)
        self.picker.PickFromListOn()

    def attach(self, plotter: pv.Plotter) -> None:
        if self.poly.n_points == 0:
            return
        self.actor = plotter.add_mesh(
            self.poly,
            color="black",
            point_size=self.point_size,
            render_points_as_spheres=True,
        )
        self.picker.AddPickList(self.actor)

    def set_visible(self, on: bool) -> None:
        if self.actor is not None:
            self.actor.SetVisibility(int(on))


@dataclass(slots=True)
class _LineLayer:
    """One pickable line layer (A2 / B2 / C3) wrapping a
    :class:`LineLayerData` record plus its render-time picker / actor.

    ``color_fn`` produces the per-cell baseline colors used at attach time.
    Highlight on pick is delivered via a dedicated overlay on :class:`HdMapViz`,
    not by mutating this layer's per-cell RGB, so the same highlight is
    visible even when two base cells share geometry (단선 중앙선).
    """

    name: str
    data: LineLayerData
    color_fn: Callable[[], NDArray[np.uint8]]
    line_width: float
    picker_tolerance: float
    poly: pv.PolyData = field(init=False)
    actor: vtk.vtkActor | None = field(default=None, init=False)
    picker: vtk.vtkCellPicker = field(init=False)

    def __post_init__(self) -> None:
        self.poly = _polyline_polydata(self.data.polylines)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(self.picker_tolerance)
        self.picker.PickFromListOn()

    def attach(self, plotter: pv.Plotter) -> None:
        if self.poly.n_cells == 0:
            return
        self.poly.cell_data["rgb"] = self.color_fn()
        self.actor = plotter.add_mesh(
            self.poly,
            scalars="rgb",
            rgb=True,
            line_width=self.line_width,
            show_scalar_bar=False,
        )
        self.picker.AddPickList(self.actor)

    def set_visible(self, on: bool) -> None:
        if self.actor is not None:
            self.actor.SetVisibility(int(on))


@dataclass(slots=True)
class _PolygonLayer:
    """One pickable polygon layer (A3 / A4) wrapping a
    :class:`PolygonLayerData` record. The polygon picker is shared across
    all polygon layers; dispatch back to the layer is by actor identity.
    """

    name: str
    data: PolygonLayerData
    face_rgb_fn: Callable[[], NDArray[np.uint8]]
    poly: pv.PolyData = field(init=False)
    actor: vtk.vtkActor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.poly = _polygon_polydata(self.data.rings)

    def attach(self, plotter: pv.Plotter, picker: vtk.vtkCellPicker) -> None:
        if self.poly.n_cells == 0:
            return
        face_rgb = self.face_rgb_fn()
        per_tri_rgb = face_rgb[self.poly.cell_data["poly_idx"]]
        self.poly.cell_data["rgb"] = per_tri_rgb
        self.actor = plotter.add_mesh(
            self.poly,
            scalars="rgb",
            rgb=True,
            opacity=_POLY_OPACITY,
            show_scalar_bar=False,
            show_edges=False,
            lighting=False,
        )
        self.actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(
            *_POLY_DEPTH_OFFSET
        )
        picker.AddPickList(self.actor)

    def set_visible(self, on: bool) -> None:
        if self.actor is not None:
            self.actor.SetVisibility(int(on))


# ---- Main viz class ------------------------------------------------------------


class HdMapViz:
    """3D scene of every NGII layer.

    Pass an external ``plotter`` (e.g. ``QtInteractor``) for embedded use,
    or omit it for a standalone window. Pass ``on_pick`` to receive
    shift-click events as ``(layer_name, row_index)`` callbacks; the scene
    never draws pick text on-screen.
    """

    def __init__(
        self,
        shp_dir: Path,
        junction_merge_dist_m: float = 0.0,
        bidirectional_merge_max_separation_m: float = 15.0,
        plotter: pv.Plotter | None = None,
        on_pick: PickCallback | None = None,
    ) -> None:
        self.shp_dir = shp_dir
        self.plotter = plotter if plotter is not None else pv.Plotter()
        self.on_pick = on_pick

        # Typed data records — one per NGII layer, geometry kind enforced by base.
        self.a1 = A1Data(shp_dir)
        self.a2 = A2Data(shp_dir)
        self.a3 = A3Data(shp_dir)
        self.a4 = A4Data(shp_dir)
        self.b2 = B2Data(shp_dir)
        self.c3 = C3Data(shp_dir)

        # Segmentation + per-group / per-junction / per-road palettes.
        self.segmentation = Segmentation.from_shp_dir(
            shp_dir,
            junction_merge_dist_m=junction_merge_dist_m,
            bidirectional_merge_max_separation_m=bidirectional_merge_max_separation_m,
        )
        self.group_palette = _random_palette(int(self.segmentation.group_id.max()) + 1, seed=42)
        self.junction_palette = _random_palette(
            int(self.segmentation.node_junction_id.max()) + 1, seed=137
        )
        self.road_palette = _random_palette(len(self.segmentation.roads), seed=1729)

        # Pickable layer wrappers. Order in the line-layer tuple is also the
        # shift-click priority order (first match wins).
        self.a1_layer = _PointLayer(
            "A1",
            self.a1,
            point_size=_NODE_POINT_SIZE,
            picker_tolerance=_PICKER_TOL_A1,
        )
        self.line_layers: tuple[_LineLayer, ...] = (
            _LineLayer(
                "A2",
                self.a2,
                self._a2_cell_colors,
                line_width=_LW_A2,
                picker_tolerance=_PICKER_TOL_A2,
            ),
            _LineLayer(
                "B2",
                self.b2,
                self._b2_cell_colors,
                line_width=_LW_B2,
                picker_tolerance=_PICKER_TOL_THIN,
            ),
            _LineLayer(
                "C3",
                self.c3,
                self._c3_cell_colors,
                line_width=_LW_C3,
                picker_tolerance=_PICKER_TOL_THIN,
            ),
        )

        # A3 / A4 share one polygon picker — dispatch is by actor identity.
        self.poly_picker = vtk.vtkCellPicker()
        self.poly_picker.SetTolerance(_PICKER_TOL_POLY)
        self.poly_picker.PickFromListOn()
        self.polygon_layers: tuple[_PolygonLayer, ...] = (
            _PolygonLayer("A3", self.a3, self._a3_face_rgb),
            _PolygonLayer("A4", self.a4, self._a4_face_rgb),
        )

        # Highlight overlay — starts empty so PyVista's allow_empty_mesh
        # covers it; geometry is swapped in on the first pick.
        self.highlight_poly = pv.PolyData()
        self.highlight_actor: vtk.vtkActor | None = None

        # Current abstraction level — default to level 3 so the user lands
        # on the OpenDRIVE-aligned road/junction view.
        self._abstraction_level: int = _DEFAULT_LEVEL

        # Teardown handles - populated in attach(), consumed by detach().
        self._left_press_tag: int | None = None
        self._key_events_bound: tuple[str, ...] = ()

    # ---- Per-layer color compute ---------------------------------------------

    def _a2_cell_colors_at(self, level: int) -> NDArray[np.uint8]:
        """RGB per A2_LINK at the requested abstraction level.

        * Level 1 — uniform neutral.
        * Level 2 — group palette across every row (mainline + interior
          alike), so the user reads the SHP-level group structure.
        * Level 3 — road palette on mainline rows, junction palette on
          interior rows; mirrors the OpenDRIVE entity each link belongs to.
        """
        seg = self.segmentation
        if level == _LEVEL_RAW:
            return np.tile(np.asarray(_A2_UNIFORM_RGB, dtype=np.uint8), (len(self.a2.ids), 1))
        if level == _LEVEL_GROUP:
            return np.asarray(self.group_palette[seg.group_id], dtype=np.uint8).copy()
        if level == _LEVEL_ROAD_JUNCTION:
            n = len(self.a2.ids)
            rgb = np.zeros((n, 3), dtype=np.uint8)
            mainline = seg.road_id_per_link >= 0
            if mainline.any():
                rgb[mainline] = self.road_palette[seg.road_id_per_link[mainline]]
            interior = seg.junction_id >= 0
            if interior.any():
                rgb[interior] = self.junction_palette[seg.junction_id[interior]]
            return rgb
        raise ValueError(level)

    def _b2_cell_colors_at(self, level: int) -> NDArray[np.uint8]:
        """RGB per B2 row at the requested abstraction level.

        * Level 1 — paint code (current B2 palette).
        * Level 2 — bound group's color; R side wins, falls back to L; both
          unbound rows fall back to ``_A2_UNIFORM_RGB``.
        * Level 3 — bound road's color (mainline), else bound junction's
          color (interior), else fallback. A centerline between two
          bidirectionally-merged groups paints the same color as the road,
          which is by design: the visual merge signals one OpenDRIVE road.
        """
        seg = self.segmentation
        n = len(self.b2.types)
        if level == _LEVEL_RAW:
            rgb = np.zeros((n, 3), dtype=np.uint8)
            for i, t in enumerate(self.b2.types):
                rgb[i] = _B2_PAINT_RGB.get(t[:1], _B2_PAINT_FALLBACK_RGB)
            return rgb
        if level == _LEVEL_GROUP:
            rgb = np.tile(np.asarray(_A2_UNIFORM_RGB, dtype=np.uint8), (n, 1))
            r_bound = seg.b2_r_group >= 0
            rgb[r_bound] = self.group_palette[seg.b2_r_group[r_bound]]
            # L-side as fallback for rows where R is unbound but L is bound.
            l_only = (seg.b2_r_group < 0) & (seg.b2_l_group >= 0)
            rgb[l_only] = self.group_palette[seg.b2_l_group[l_only]]
            return rgb
        if level == _LEVEL_ROAD_JUNCTION:
            rgb = np.tile(np.asarray(_A2_UNIFORM_RGB, dtype=np.uint8), (n, 1))
            r_road = seg.b2_r_road >= 0
            rgb[r_road] = self.road_palette[seg.b2_r_road[r_road]]
            r_junction = (~r_road) & (seg.b2_r_junction >= 0)
            rgb[r_junction] = self.junction_palette[seg.b2_r_junction[r_junction]]
            covered = r_road | r_junction
            l_road = (~covered) & (seg.b2_l_road >= 0)
            rgb[l_road] = self.road_palette[seg.b2_l_road[l_road]]
            l_junction = (~covered) & (~l_road) & (seg.b2_l_junction >= 0)
            rgb[l_junction] = self.junction_palette[seg.b2_l_junction[l_junction]]
            return rgb
        raise ValueError(level)

    def _a2_cell_colors(self) -> NDArray[np.uint8]:
        """Initial-attach hook: A2 colors at the current abstraction level."""
        return self._a2_cell_colors_at(self._abstraction_level)

    def _b2_cell_colors(self) -> NDArray[np.uint8]:
        """Initial-attach hook: B2 colors at the current abstraction level."""
        return self._b2_cell_colors_at(self._abstraction_level)

    def _c3_cell_colors(self) -> NDArray[np.uint8]:
        """RGB per C3 row keyed off Type (facility class)."""
        n = len(self.c3.types)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, t in enumerate(self.c3.types):
            rgb[i] = _C3_TYPE_RGB.get(t, _C3_TYPE_FALLBACK_RGB)
        return rgb

    def _a3_face_rgb(self) -> NDArray[np.uint8]:
        n = len(self.a3.ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, (kind, rt) in enumerate(zip(self.a3.kinds, self.a3.road_types, strict=True)):
            base = _A3_PROTECTED_RGB if kind == "7" else _A3_ROAD_TYPE_RGB.get(rt, _A4_FALLBACK_RGB)
            rgb[i] = base
        return rgb

    def _a4_face_rgb(self) -> NDArray[np.uint8]:
        n = len(self.a4.ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, st in enumerate(self.a4.subtypes):
            rgb[i] = _A4_SUBTYPE_RGB.get(st, _A4_FALLBACK_RGB)
        return rgb

    # ---- View keys -----------------------------------------------------------

    def _add_view_keys(self) -> None:
        def _view_2d() -> None:
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()

        self.plotter.add_key_event("2", _view_2d)
        self._key_events_bound = ("2",)

    # ---- Scene attach / standalone entry -------------------------------------

    def attach(self) -> None:
        """Build all actors, bind pickers, and register key events.

        Idempotent in spirit but not strictly idempotent — call once per
        plotter instance. Standalone callers use :meth:`show` instead.
        """
        self.plotter.background_color = _BG_COLOR

        # Drawing order: polygons first (A3 / A4), then lines in pick-priority
        # reverse (C3 → B2 → A2 so A2 paints over B2 paints over C3), then A1
        # dots on top. Pick priority is independent of draw order — each
        # picker has its own pick list.
        for poly_layer in self.polygon_layers:
            poly_layer.attach(self.plotter, self.poly_picker)
        for line_layer in reversed(self.line_layers):
            line_layer.attach(self.plotter)
        self.a1_layer.attach(self.plotter)

        # A2 cell_data extras let pick consumers read group/junction/road off
        # the PolyData if they prefer that over the segmentation arrays.
        a2_poly = self.line_layers[0].poly
        a2_poly.cell_data["link_id"] = self.a2.ids
        a2_poly.cell_data["group_id"] = self.segmentation.group_id
        a2_poly.cell_data["junction_id"] = self.segmentation.junction_id
        a2_poly.cell_data["road_id"] = self.segmentation.road_id_per_link

        # Highlight overlay — added last so it draws on top of every base
        # mesh. Initial PolyData is empty, so nothing renders until a pick
        # populates it. pickable=False keeps the overlay out of the pickers.
        self.highlight_actor = self.plotter.add_mesh(
            self.highlight_poly,
            color=_HIGHLIGHT_RGB,
            line_width=_LW_HIGHLIGHT,
            show_scalar_bar=False,
            pickable=False,
        )

        self.plotter.add_axes()
        self._add_view_keys()
        self._left_press_tag = self.plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_left_press
        )

    def show(self) -> None:
        """Standalone entry: build the scene and open a window."""
        self.attach()
        self.plotter.show()

    def detach(self) -> None:
        """Remove every actor, observer, and key event this viz added.

        Leaves the underlying plotter (including camera, parallel-projection
        setting, and background color) usable so the GUI can attach a fresh
        viz against the same ``QtInteractor``.
        """
        for actor in (
            self.a1_layer.actor,
            *(line.actor for line in self.line_layers),
            *(poly.actor for poly in self.polygon_layers),
            self.highlight_actor,
        ):
            if actor is not None:
                self.plotter.remove_actor(actor)
        if self._left_press_tag is not None:
            self.plotter.iren.remove_observer(self._left_press_tag)
            self._left_press_tag = None
        for key in self._key_events_bound:
            self.plotter.clear_events_for_key(key)
        self._key_events_bound = ()
        self.plotter.hide_axes()

    # ---- GUI hooks -----------------------------------------------------------

    def set_layer_visible(self, name: str, on: bool) -> None:
        """Toggle visibility of a base layer (``A1`` / ``A2`` / ``A3`` /
        ``A4`` / ``B2`` / ``C3``). Unknown names are ignored.
        """
        if name == self.a1_layer.name:
            self.a1_layer.set_visible(on)
        else:
            for line_layer in self.line_layers:
                if line_layer.name == name:
                    line_layer.set_visible(on)
                    break
            else:
                for poly_layer in self.polygon_layers:
                    if poly_layer.name == name:
                        poly_layer.set_visible(on)
                        break
        self.plotter.render()

    def set_abstraction_level(self, level: int) -> None:
        """Switch the A2 + B2 cell coloring to one of the three abstraction
        levels (1 None / 2 Group / 3 Road & Junction).

        Idempotent: re-setting the current level is a no-op. C3 / A3 / A4 /
        A1 layers stay on their natural NGII codes at every level.
        """
        if not _LEVEL_RAW <= level <= _LEVEL_ROAD_JUNCTION:
            raise ValueError(level)
        if level == self._abstraction_level:
            return
        self._abstraction_level = level
        a2_poly = self.line_layers[0].poly
        a2_poly.cell_data["rgb"] = self._a2_cell_colors_at(level)
        a2_poly.Modified()
        b2_poly = self.line_layers[1].poly
        b2_poly.cell_data["rgb"] = self._b2_cell_colors_at(level)
        b2_poly.Modified()
        self.plotter.render()

    # ---- Shift-click dispatch ------------------------------------------------

    def _on_left_press(self, _obj: Any, _event: str) -> None:
        iren = self.plotter.iren.interactor
        if not iren.GetShiftKey():
            return
        x, y = iren.GetEventPosition()
        renderer = self.plotter.renderer
        if self._try_point_pick(self.a1_layer, x, y, renderer):
            return
        for layer in self.line_layers:
            if self._try_line_pick(layer, x, y, renderer):
                return
        self._try_polygon_pick(x, y, renderer)

    def _try_point_pick(self, layer: _PointLayer, x: int, y: int, renderer: Any) -> bool:
        if layer.actor is None or not layer.picker.Pick(x, y, 0, renderer):
            return False
        pid = layer.picker.GetPointId()
        if pid < 0:
            return False
        self._emit_pick(layer.name, pid)
        return True

    def _try_line_pick(self, layer: _LineLayer, x: int, y: int, renderer: Any) -> bool:
        """Hit-test ``layer.picker``; on hit, swap the highlight overlay's
        geometry to the picked cell's polyline.

        The overlay is a separate PolyData rendered last with a fat stroke,
        which means the highlight stays visible even when multiple base
        cells share the same XY (단선 중앙선: two B2 rows draw on top of
        each other per NGII manual §9.4.7).
        """
        if layer.actor is None or not layer.picker.Pick(x, y, 0, renderer):
            return False
        cid = layer.picker.GetCellId()
        if cid < 0:
            return False
        self._set_highlight(layer.data.polylines[cid])
        self._emit_pick(layer.name, cid)
        return True

    def _try_polygon_pick(self, x: int, y: int, renderer: Any) -> None:
        if not self.poly_picker.Pick(x, y, 0, renderer):
            return
        cid = self.poly_picker.GetCellId()
        if cid < 0:
            return
        actor = self.poly_picker.GetActor()
        for layer in self.polygon_layers:
            if layer.actor is actor:
                pi = int(layer.poly.cell_data["poly_idx"][cid])
                self._emit_pick(layer.name, pi)
                return

    def _set_highlight(self, polyline: NDArray[np.float64]) -> None:
        """Swap the highlight overlay's geometry to one polyline in place.

        Uses the same ``self.highlight_poly`` reference the actor was bound
        to in :meth:`attach`, so VTK keeps the existing mapper.
        """
        n = len(polyline)
        cells = np.concatenate(([n], np.arange(n, dtype=np.int64)))
        new_poly = pv.PolyData(polyline, lines=cells)
        self.highlight_poly.points = new_poly.points
        self.highlight_poly.lines = new_poly.lines
        self.plotter.render()

    def _emit_pick(self, kind: str, idx: int) -> None:
        if self.on_pick is not None:
            self.on_pick(kind, idx)
        else:
            log.info("picked %s[%d]", kind, idx)
