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
from shp2xodr.shp.segmentation import Segmentation, SegmentationConfig

log = logging.getLogger(__name__)


# Abstraction levels — contract constants matching the GUI radio-button IDs in
# gui.py. Not tunable; the rest of the scene-coloring code reads these literals.
_LEVEL_RAW = 1
_LEVEL_GROUP = 2
_LEVEL_ROAD_JUNCTION = 3

# Enable VTK's coincident-topology resolution mode globally; per-mapper
# relative offsets only take effect once this is on.
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

# The highlight overlay starts empty (no picks yet). PyVista 0.43+ refuses
# empty meshes by default; flipping this global theme bit lets us attach the
# actor up-front and swap geometry in on the first pick.
pv.global_theme.allow_empty_mesh = True


# ---- Pick callback type --------------------------------------------------------

# (layer_name, row_idx). layer_name ∈ {"A1", "A2", "A3", "A4", "B2", "C3"}.
PickCallback = Callable[[str, int], None]


# ---- Hydra-managed config ------------------------------------------------------


@dataclass(slots=True, frozen=True)
class VizConfig:
    """Tuning knobs for :class:`HdMapViz`. Values live in ``conf/config.yaml``
    under ``viz:``; the entry point builds an instance and hands it to the
    window / viz constructors. Tests construct directly.

    The NGII code-list color tables (``a3_road_type_rgb``, ``a4_subtype_rgb``,
    ``b2_paint_rgb``, ``c3_type_rgb``) map data-model codes — first digit of
    B2.Type, A3.RoadType, A4.SubType, C3.Type — to RGB triples.

    ``background_color`` is a float-RGB in ``[0, 1]`` (VTK convention for
    renderer backgrounds); every other RGB is uint8 in ``[0, 255]``.
    """

    # Rendering geometry
    node_point_size: float
    poly_opacity: float
    poly_depth_offset_factor: float
    poly_depth_offset_units: float
    line_width_a2: float
    line_width_b2: float
    line_width_c3: float
    line_width_highlight: float
    # Picker tolerances (fraction of screen)
    picker_tol_a1: float
    picker_tol_a2: float
    picker_tol_thin: float
    picker_tol_poly: float
    # Scene colors
    background_color: tuple[float, float, float]
    highlight_rgb: tuple[int, int, int]
    a2_uniform_rgb: tuple[int, int, int]
    # Random-palette seeds
    group_palette_seed: int
    junction_palette_seed: int
    road_palette_seed: int
    # Default abstraction level (1 None / 2 Group / 3 Road & Junction)
    default_abstraction_level: int
    # NGII code-list color tables
    a3_road_type_rgb: dict[str, tuple[int, int, int]]
    a3_protected_rgb: tuple[int, int, int]
    a3_fallback_rgb: tuple[int, int, int]
    a4_subtype_rgb: dict[str, tuple[int, int, int]]
    a4_fallback_rgb: tuple[int, int, int]
    b2_paint_rgb: dict[str, tuple[int, int, int]]
    b2_paint_fallback_rgb: tuple[int, int, int]
    c3_type_rgb: dict[str, tuple[int, int, int]]
    c3_type_fallback_rgb: tuple[int, int, int]


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
    opacity: float
    depth_offset_factor: float
    depth_offset_units: float
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
            opacity=self.opacity,
            show_scalar_bar=False,
            show_edges=False,
            lighting=False,
        )
        self.actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(
            self.depth_offset_factor, self.depth_offset_units
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
        seg_cfg: SegmentationConfig,
        viz_cfg: VizConfig,
        plotter: pv.Plotter | None = None,
        on_pick: PickCallback | None = None,
    ) -> None:
        self.shp_dir = shp_dir
        self.viz_cfg = viz_cfg
        self.plotter = plotter if plotter is not None else pv.Plotter()
        self.on_pick = on_pick

        # Typed data records — one per NGII layer, geometry kind enforced by
        # base. A1 / A2 / B2 are required for the segmentation pipeline; A3 /
        # A4 / C3 are optional layers a section is allowed to ship without.
        self.a1 = A1Data(shp_dir)
        self.a2 = A2Data(shp_dir)
        self.b2 = B2Data(shp_dir)
        self.a3: A3Data | None = A3Data.try_load(shp_dir)
        self.a4: A4Data | None = A4Data.try_load(shp_dir)
        self.c3: C3Data | None = C3Data.try_load(shp_dir)

        # Segmentation + per-group / per-junction / per-road palettes.
        self.segmentation = Segmentation.from_shp_dir(shp_dir, seg_cfg)
        self.group_palette = _random_palette(
            int(self.segmentation.group_id.max()) + 1, seed=viz_cfg.group_palette_seed
        )
        self.junction_palette = _random_palette(
            int(self.segmentation.node_junction_id.max()) + 1,
            seed=viz_cfg.junction_palette_seed,
        )
        self.road_palette = _random_palette(
            len(self.segmentation.roads), seed=viz_cfg.road_palette_seed
        )

        # Pickable layer wrappers. Order in the line-layer tuple is also the
        # shift-click priority order (first match wins).
        self.a1_layer = _PointLayer(
            "A1",
            self.a1,
            point_size=viz_cfg.node_point_size,
            picker_tolerance=viz_cfg.picker_tol_a1,
        )
        line_specs: list[_LineLayer] = [
            _LineLayer(
                "A2",
                self.a2,
                self._a2_cell_colors,
                line_width=viz_cfg.line_width_a2,
                picker_tolerance=viz_cfg.picker_tol_a2,
            ),
            _LineLayer(
                "B2",
                self.b2,
                self._b2_cell_colors,
                line_width=viz_cfg.line_width_b2,
                picker_tolerance=viz_cfg.picker_tol_thin,
            ),
        ]
        if self.c3 is not None:
            line_specs.append(
                _LineLayer(
                    "C3",
                    self.c3,
                    self._c3_cell_colors,
                    line_width=viz_cfg.line_width_c3,
                    picker_tolerance=viz_cfg.picker_tol_thin,
                )
            )
        self.line_layers: tuple[_LineLayer, ...] = tuple(line_specs)

        # A3 / A4 share one polygon picker — dispatch is by actor identity.
        self.poly_picker = vtk.vtkCellPicker()
        self.poly_picker.SetTolerance(viz_cfg.picker_tol_poly)
        self.poly_picker.PickFromListOn()
        poly_specs: list[_PolygonLayer] = []
        if self.a3 is not None:
            poly_specs.append(
                _PolygonLayer(
                    "A3",
                    self.a3,
                    self._a3_face_rgb,
                    opacity=viz_cfg.poly_opacity,
                    depth_offset_factor=viz_cfg.poly_depth_offset_factor,
                    depth_offset_units=viz_cfg.poly_depth_offset_units,
                )
            )
        if self.a4 is not None:
            poly_specs.append(
                _PolygonLayer(
                    "A4",
                    self.a4,
                    self._a4_face_rgb,
                    opacity=viz_cfg.poly_opacity,
                    depth_offset_factor=viz_cfg.poly_depth_offset_factor,
                    depth_offset_units=viz_cfg.poly_depth_offset_units,
                )
            )
        self.polygon_layers: tuple[_PolygonLayer, ...] = tuple(poly_specs)

        # Highlight overlay — starts empty so PyVista's allow_empty_mesh
        # covers it; geometry is swapped in on the first pick.
        self.highlight_poly = pv.PolyData()
        self.highlight_actor: vtk.vtkActor | None = None

        # Current abstraction level — default per viz config so the user
        # lands on whatever level the config picked.
        self._abstraction_level: int = viz_cfg.default_abstraction_level

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
            return np.tile(
                np.asarray(self.viz_cfg.a2_uniform_rgb, dtype=np.uint8), (len(self.a2.ids), 1)
            )
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
          unbound rows fall back to ``viz_cfg.a2_uniform_rgb``.
        * Level 3 — bound road's color (mainline), else bound junction's
          color (interior), else fallback. A centerline between two
          bidirectionally-merged groups paints the same color as the road,
          which is by design: the visual merge signals one OpenDRIVE road.
        """
        seg = self.segmentation
        cfg = self.viz_cfg
        n = len(self.b2.types)
        if level == _LEVEL_RAW:
            rgb = np.zeros((n, 3), dtype=np.uint8)
            for i, t in enumerate(self.b2.types):
                rgb[i] = cfg.b2_paint_rgb.get(t[:1], cfg.b2_paint_fallback_rgb)
            return rgb
        if level == _LEVEL_GROUP:
            rgb = np.tile(np.asarray(cfg.a2_uniform_rgb, dtype=np.uint8), (n, 1))
            r_bound = seg.b2_r_group >= 0
            rgb[r_bound] = self.group_palette[seg.b2_r_group[r_bound]]
            # L-side as fallback for rows where R is unbound but L is bound.
            l_only = (seg.b2_r_group < 0) & (seg.b2_l_group >= 0)
            rgb[l_only] = self.group_palette[seg.b2_l_group[l_only]]
            return rgb
        if level == _LEVEL_ROAD_JUNCTION:
            rgb = np.tile(np.asarray(cfg.a2_uniform_rgb, dtype=np.uint8), (n, 1))
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
        """RGB per C3 row keyed off Type (facility class).

        Invariant: only invoked when ``self.c3`` is the data of an existing
        ``_LineLayer`` in :attr:`line_layers` — so it's never None here.
        """
        if self.c3 is None:
            msg = "C3 callback invoked but layer was not loaded"
            raise RuntimeError(msg)
        cfg = self.viz_cfg
        n = len(self.c3.types)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, t in enumerate(self.c3.types):
            rgb[i] = cfg.c3_type_rgb.get(t, cfg.c3_type_fallback_rgb)
        return rgb

    def _a3_face_rgb(self) -> NDArray[np.uint8]:
        if self.a3 is None:
            msg = "A3 callback invoked but layer was not loaded"
            raise RuntimeError(msg)
        cfg = self.viz_cfg
        n = len(self.a3.ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, (kind, rt) in enumerate(zip(self.a3.kinds, self.a3.road_types, strict=True)):
            base = (
                cfg.a3_protected_rgb
                if kind == "7"
                else cfg.a3_road_type_rgb.get(rt, cfg.a3_fallback_rgb)
            )
            rgb[i] = base
        return rgb

    def _a4_face_rgb(self) -> NDArray[np.uint8]:
        if self.a4 is None:
            msg = "A4 callback invoked but layer was not loaded"
            raise RuntimeError(msg)
        cfg = self.viz_cfg
        n = len(self.a4.ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, st in enumerate(self.a4.subtypes):
            rgb[i] = cfg.a4_subtype_rgb.get(st, cfg.a4_fallback_rgb)
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
        self.plotter.background_color = self.viz_cfg.background_color

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
            color=self.viz_cfg.highlight_rgb,
            line_width=self.viz_cfg.line_width_highlight,
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
