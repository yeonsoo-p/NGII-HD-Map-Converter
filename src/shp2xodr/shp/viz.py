"""3D visualization of NGII HD-map (정밀도로지도) layers.

:class:`HdMapViz` shows every layer at once: A2_LINKs colored by segmentation
(mainline links get one color per future OpenDRIVE ``<road>`` bundle;
interior links share one color per future ``<junction>``), B2 surface line
marks rendered in their actual paint colors (yellow / white / blue),
C3 vehicle-protection facilities (guardrails / kerbs / barriers / walls)
colored by facility type, A1 nodes as small black dots, and A3 / A4 polygon
footprints colored by their NGII codes.

Shift + left-click any link, line, or polygon to log + display its
attributes. Picking an A2 / B2 / C3 cell sets a dedicated overlay actor to
that cell's geometry, drawn last with a fat stroke so the highlight stays
visible even when multiple base cells share the same XY (which the NGII
manual mandates for 단선 중앙선 - centerlines are drawn twice, one per
direction, geometrically overlapping).

Korean labels for NGII codes (B2.Kind / C3.Type / A3.Kind / A3.RoadType /
A4.SubType / A2.LinkType / A2.RoadRank / A2.RoadType / A1.NodeType) live on
the data classes in :mod:`shp2xodr.shp.io` as ``ClassVar`` dicts.
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

# Line widths per layer (A2 thickest so the bundle/junction coloring stays
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

_HUD_TEXT = "[2] top-down A1 / A2 / B2 / C3 / A3 / A4"

# Highlight overlay color. The overlay sits on top of every base layer at
# _LW_HIGHLIGHT thickness, so this color is what the user reads as "picked".
# Magenta wins against every base palette (B2 primaries, C3 neutrals, A3/A4
# cool fills, A2 random saturated range).
_HIGHLIGHT_RGB: tuple[int, int, int] = (255, 30, 200)

# Enable VTK's coincident-topology resolution mode globally; per-mapper
# relative offsets above only take effect once this is on.
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

# The highlight overlay starts as an empty PolyData (no picks yet). PyVista
# 0.43+ refuses empty meshes by default; flipping this global theme bit lets
# us attach the actor up-front and swap geometry in on each pick.
pv.global_theme.allow_empty_mesh = True


# ---- CJK font discovery --------------------------------------------------------


def _find_cjk_font() -> str | None:
    """First well-known CJK font on disk, else ``None``.

    VTK's default font has no Hangul glyphs, so Korean ``Name`` fields render
    as blanks in pick text without overriding the font.
    """
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    return None


_CJK_FONT_FILE: str | None = _find_cjk_font()


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

# A4.SubType → fill color. SubType code list per io.A4Data.SUBTYPE_LABEL.
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
# 1 황색, 2 백색, 3 청색, 9 기타 (see io.B2Data.TYPE_COLOR_LABEL).
# Colors are tuned to stay distinct from the A4 fills — A4 SubType=4 is
# orange (240, 180, 80) and SubType=3 is tan (220, 200, 160), so the yellow
# here pushes toward pure lemon to avoid being read as "an A4 polygon".
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
    text_fn: Callable[[int], str]
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
    text_fn: Callable[[int], str]
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


@dataclass(slots=True)
class _PolygonLayer:
    """One pickable polygon layer (A3 / A4) wrapping a
    :class:`PolygonLayerData` record. The polygon picker is shared across
    all polygon layers; dispatch back to the layer is by actor identity.
    """

    name: str
    data: PolygonLayerData
    face_rgb_fn: Callable[[], NDArray[np.uint8]]
    text_fn: Callable[[int], str]
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


# ---- Main viz class ------------------------------------------------------------


class HdMapViz:
    """3D viz of every NGII layer at once.

    Keys:
        2 — top-down orthographic
        3 — perspective
    Shift + left-click a link, line, or polygon to highlight / display its info.
    """

    def __init__(self, shp_dir: Path, junction_merge_dist_m: float = 0.0) -> None:
        self.shp_dir = shp_dir
        self.plotter = pv.Plotter()

        # Typed data records — one per NGII layer, geometry kind enforced by base.
        self.a1 = A1Data(shp_dir)
        self.a2 = A2Data(shp_dir)
        self.a3 = A3Data(shp_dir)
        self.a4 = A4Data(shp_dir)
        self.b2 = B2Data(shp_dir)
        self.c3 = C3Data(shp_dir)

        # Segmentation + per-bundle / per-junction palettes.
        self.segmentation = Segmentation.from_shp_dir(
            shp_dir, junction_merge_dist_m=junction_merge_dist_m
        )
        self.bundle_palette = _random_palette(int(self.segmentation.bundle_id.max()) + 1, seed=42)
        self.junction_palette = _random_palette(
            int(self.segmentation.node_junction_id.max()) + 1, seed=137
        )

        # Pickable layer wrappers. Order in the line-layer tuple is also the
        # shift-click priority order (first match wins).
        self.a1_layer = _PointLayer(
            "A1",
            self.a1,
            self._a1_pick_text,
            point_size=_NODE_POINT_SIZE,
            picker_tolerance=_PICKER_TOL_A1,
        )
        self.line_layers: tuple[_LineLayer, ...] = (
            _LineLayer(
                "A2",
                self.a2,
                self._a2_cell_colors,
                self._a2_pick_text,
                line_width=_LW_A2,
                picker_tolerance=_PICKER_TOL_A2,
            ),
            _LineLayer(
                "B2",
                self.b2,
                self._b2_cell_colors,
                self._b2_pick_text,
                line_width=_LW_B2,
                picker_tolerance=_PICKER_TOL_THIN,
            ),
            _LineLayer(
                "C3",
                self.c3,
                self._c3_cell_colors,
                self._c3_pick_text,
                line_width=_LW_C3,
                picker_tolerance=_PICKER_TOL_THIN,
            ),
        )
        # A2's pick text reads bundle/junction off the PolyData's cell_data
        # so the lookup survives any future re-layout of the per-cell arrays.
        a2_poly = self.line_layers[0].poly
        a2_poly.cell_data["link_id"] = self.a2.ids
        a2_poly.cell_data["bundle_id"] = self.segmentation.bundle_id
        a2_poly.cell_data["junction_id"] = self.segmentation.junction_id

        # A3 / A4 share one polygon picker — dispatch is by actor identity.
        self.poly_picker = vtk.vtkCellPicker()
        self.poly_picker.SetTolerance(_PICKER_TOL_POLY)
        self.poly_picker.PickFromListOn()
        self.polygon_layers: tuple[_PolygonLayer, ...] = (
            _PolygonLayer("A3", self.a3, self._a3_face_rgb, self._a3_pick_text),
            _PolygonLayer("A4", self.a4, self._a4_face_rgb, self._a4_pick_text),
        )

        # Highlight overlay - a separate PolyData rendered last with a fat
        # stroke. On pick, we swap its points + cells to match the picked
        # cell's polyline; the actor stays bound to the same PolyData
        # reference so we don't need to re-add the mesh.
        self.highlight_poly = pv.PolyData()
        self.highlight_actor: vtk.vtkActor | None = None

    # ---- Per-layer color compute ---------------------------------------------

    def _a2_cell_colors(self) -> NDArray[np.uint8]:
        """RGB per A2_LINK: bundle color on mainline, junction color on interior."""
        bundle_rgb = self.bundle_palette[self.segmentation.bundle_id]
        is_interior = self.segmentation.junction_id >= 0
        rgb = np.asarray(bundle_rgb).copy()
        if is_interior.any():
            rgb[is_interior] = self.junction_palette[self.segmentation.junction_id[is_interior]]
        return rgb

    def _b2_cell_colors(self) -> NDArray[np.uint8]:
        """RGB per B2 row keyed off the first digit of ``Type`` (paint color)."""
        n = len(self.b2.types)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, t in enumerate(self.b2.types):
            rgb[i] = _B2_PAINT_RGB.get(t[:1], _B2_PAINT_FALLBACK_RGB)
        return rgb

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

    # ---- Per-layer pick-text -------------------------------------------------

    def _a1_pick_text(self, point_idx: int) -> str:
        ntype = self.a1.node_types[point_idx]
        label = A1Data.NODE_TYPE_LABEL.get(ntype, ntype)
        return f"A1 {self.a1.ids[point_idx]}  NodeType={label}"

    def _a2_pick_text(self, cell_id: int) -> str:
        a2_poly = self.line_layers[0].poly
        link_id = a2_poly.cell_data["link_id"][cell_id]
        bid = int(a2_poly.cell_data["bundle_id"][cell_id])
        jid = int(a2_poly.cell_data["junction_id"][cell_id])
        jstr = str(jid) if jid >= 0 else "-"
        lt = self.a2.link_types[cell_id]
        lt_label = A2Data.LINK_TYPE_LABEL.get(lt, lt)
        return f"link {link_id}  LinkType={lt_label}  bundle {bid}  junction {jstr}"

    def _b2_pick_text(self, cell_id: int) -> str:
        b2_id = self.b2.ids[cell_id]
        type_code = self.b2.types[cell_id]
        kind_code = self.b2.kinds[cell_id]
        kind = B2Data.KIND_LABEL.get(kind_code, kind_code)
        r_b = int(self.segmentation.b2_r_bundle[cell_id])
        l_b = int(self.segmentation.b2_l_bundle[cell_id])
        r_str = str(r_b) if r_b >= 0 else "-"
        l_str = str(l_b) if l_b >= 0 else "-"
        return f"B2 {b2_id}  Type={type_code}  Kind={kind}  R-bundle={r_str}  L-bundle={l_str}"

    def _c3_pick_text(self, cell_id: int) -> str:
        c3_id = self.c3.ids[cell_id]
        type_code = self.c3.types[cell_id]
        type_label = C3Data.TYPE_LABEL.get(type_code, type_code)
        ic_label = C3Data.IS_CENTRAL_LABEL.get(
            self.c3.is_central[cell_id], self.c3.is_central[cell_id]
        )
        lh_label = C3Data.LOW_HIGH_LABEL.get(self.c3.low_high[cell_id], self.c3.low_high[cell_id])
        return f"C3 {c3_id}  Type={type_label}  IsCentral={ic_label}  LowHigh={lh_label}"

    def _a3_pick_text(self, poly_idx: int) -> str:
        kind = self.a3.kinds[poly_idx]
        road_type = self.a3.road_types[poly_idx]
        return (
            f"A3 {self.a3.ids[poly_idx]}  "
            f"Kind={A3Data.KIND_LABEL.get(kind, kind)}  "
            f"RoadType={A3Data.ROAD_TYPE_LABEL.get(road_type, road_type)}"
        )

    def _a4_pick_text(self, poly_idx: int) -> str:
        st = self.a4.subtypes[poly_idx]
        return (
            f"A4 {self.a4.ids[poly_idx]}  "
            f"SubType={A4Data.SUBTYPE_LABEL.get(st, st)}  "
            f"Name={self.a4.names[poly_idx]}"
        )

    # ---- View keys -----------------------------------------------------------

    def _add_view_keys(self) -> None:
        def _view_2d() -> None:
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()

        self.plotter.add_key_event("2", _view_2d)

    # ---- Entry point ---------------------------------------------------------

    def show(self) -> None:
        self.plotter.background_color = _BG_COLOR

        # Drawing order: polygons first (A3/A4), then lines in pick-priority
        # reverse (C3 → B2 → A2 so A2 paints over B2 paints over C3), then A1
        # dots on top. Pick priority is independent of draw order — each
        # picker has its own pick list.
        for poly_layer in self.polygon_layers:
            poly_layer.attach(self.plotter, self.poly_picker)
        for line_layer in reversed(self.line_layers):
            line_layer.attach(self.plotter)
        self.a1_layer.attach(self.plotter)

        # Highlight overlay - added last so it draws on top of every base
        # mesh. Initial PolyData is empty, so nothing renders until a pick
        # populates it. pickable=False keeps the overlay out of the pickers
        # (no self-pick loops).
        self.highlight_actor = self.plotter.add_mesh(
            self.highlight_poly,
            color=_HIGHLIGHT_RGB,
            line_width=_LW_HIGHLIGHT,
            show_scalar_bar=False,
            pickable=False,
        )

        self.plotter.add_axes()
        self.plotter.add_text(_HUD_TEXT, position="lower_left", font_size=10)
        self._add_view_keys()

        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_press)
        self.plotter.show()

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
        self._show_pick(layer.text_fn(pid))
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
        self._show_pick(layer.text_fn(cid))
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
                self._show_pick(layer.text_fn(pi))
                return

    def _set_highlight(self, polyline: NDArray[np.float64]) -> None:
        """Swap the highlight overlay's geometry to one polyline in place.

        Uses the same ``self.highlight_poly`` reference the actor was bound
        to in :meth:`show`, so VTK keeps the existing mapper.
        """
        n = len(polyline)
        cells = np.concatenate(([n], np.arange(n, dtype=np.int64)))
        new_poly = pv.PolyData(polyline, lines=cells)
        self.highlight_poly.points = new_poly.points
        self.highlight_poly.lines = new_poly.lines
        self.plotter.render()

    def _show_pick(self, text: str) -> None:
        log.info("picked %s", text)
        text_kwargs: dict[str, Any] = {
            "position": "upper_right",
            "name": "pick_text",
            "font_size": 12,
        }
        if _CJK_FONT_FILE is not None:
            text_kwargs["font_file"] = _CJK_FONT_FILE
        self.plotter.add_text(text, **text_kwargs)
        self.plotter.render()
