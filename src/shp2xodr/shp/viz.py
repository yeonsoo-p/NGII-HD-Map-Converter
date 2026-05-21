"""3D visualization of NGII HD-map (정밀도로지도) layers.

:class:`HdMapViz` shows every layer at once: A2_LINKs colored by segmentation
(mainline links get one color per future OpenDRIVE ``<road>`` bundle;
interior links share one color per future ``<junction>``), A1 nodes as small
black dots, and A3 / A4 polygon footprints colored by their NGII codes.

Shift + left-click any link or polygon to log + display its attributes (and
highlight A2 links in yellow).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import shapely
import vtk
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon

from shp2xodr.shp.io import a1_data, a2_data, a3_data, a4_data
from shp2xodr.shp.segmentation import Segmentation, segment_links

_NODE_POINT_SIZE = 3.0
_POLY_OPACITY = 0.85
# Polygon offset for A3/A4 mappers: pushes coincident-Z polygons back so
# A2 lines render in front. Values chosen empirically — large enough to
# beat translucent-pass draw order, small enough to avoid visible gaps.
_POLY_DEPTH_OFFSET: tuple[float, float] = (2.0, 2.0)

# Enable VTK's coincident-topology resolution mode globally; per-mapper
# relative offsets below only take effect once this is on.
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()


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

# A3.RoadType → fill color (NGII codes from manual table 9.23)
_A3_ROAD_TYPE_RGB: dict[str, tuple[int, int, int]] = {
    "1": (200, 200, 200),  # 일반도로
    "2": (80, 80, 80),  # 터널
    "3": (140, 180, 220),  # 교량
    "4": (60, 40, 140),  # 지하차도
    "5": (200, 170, 120),  # 고가차도
}
_A3_KIND_LABEL: dict[str, str] = {"1": "주행구간", "7": "보호구역"}
_A3_ROAD_TYPE_LABEL: dict[str, str] = {
    "1": "일반도로",
    "2": "터널",
    "3": "교량",
    "4": "지하차도",
    "5": "고가차도",
}
_A3_PROTECTED_RGB: tuple[int, int, int] = (220, 80, 80)  # Kind=7 overrides RoadType color

# A4.SubType → fill color. NGII manual lists 휴게소/졸음쉼터/보도/자전거도로/과적검문소
# but the explicit code table isn't in the chapter; '3' confirmed=보도 from sample data.
# Trust the Name field for the on-screen label; SubType only drives color.
_A4_SUBTYPE_RGB: dict[str, tuple[int, int, int]] = {
    "1": (120, 200, 120),
    "2": (180, 220, 140),
    "3": (220, 200, 160),
    "4": (240, 180, 80),
    "5": (180, 140, 220),
}
_A4_FALLBACK_RGB: tuple[int, int, int] = (180, 180, 180)


def _random_palette(n: int, seed: int) -> NDArray[np.uint8]:
    """Deterministic, saturated RGB rows — kept clear of pure black/white."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(60, 240, size=(n, 3), dtype=np.uint8)


log = logging.getLogger(__name__)


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


def _a3_polydata(
    shp_dir: Path,
) -> tuple[pv.PolyData, NDArray[np.str_], NDArray[np.str_], NDArray[np.str_]]:
    ids, rings, kinds, road_types = a3_data(shp_dir)
    poly = _polygon_polydata(rings)
    return poly, ids, kinds, road_types


def _a4_polydata(
    shp_dir: Path,
) -> tuple[pv.PolyData, NDArray[np.str_], NDArray[np.str_], NDArray[np.str_]]:
    ids, rings, subtypes, names = a4_data(shp_dir)
    poly = _polygon_polydata(rings)
    return poly, ids, subtypes, names


class HdMapViz:
    """3D viz of every NGII layer at once.

    Keys:
        2 — top-down orthographic
        3 — perspective
    Shift + left-click a link or polygon to highlight / display its info.
    """

    def __init__(self, shp_dir: Path, junction_merge_dist_m: float = 0.0) -> None:
        self.shp_dir = shp_dir
        self.a1_ids, a1_pts = a1_data(shp_dir)
        self.a1_poly = pv.PolyData(a1_pts)
        self.a2_poly = _a2_polydata(shp_dir)
        self.a3_poly, self.a3_ids, self.a3_kinds, self.a3_road_types = _a3_polydata(shp_dir)
        self.a4_poly, self.a4_ids, self.a4_subtypes, self.a4_names = _a4_polydata(shp_dir)
        self.segmentation: Segmentation = segment_links(
            shp_dir, junction_merge_dist_m=junction_merge_dist_m
        )
        self.bundle_palette = _random_palette(int(self.segmentation.bundle_id.max()) + 1, seed=42)
        n_junctions = int(self.segmentation.node_junction_id.max()) + 1
        self.junction_palette = _random_palette(n_junctions, seed=137)
        self.a2_poly.cell_data["bundle_id"] = self.segmentation.bundle_id
        self.a2_poly.cell_data["junction_id"] = self.segmentation.junction_id
        self.plotter = pv.Plotter()
        # One picker per layer so A2 stays clickable through A3/A4 polygons —
        # a single picker returns the front-most hit and A3 covers A2 visually.
        # Tried in priority order: A1 (dots) > A2 (lines) > A3 / A4 (polygons).
        self.a1_picker = vtk.vtkPointPicker()
        self.a1_picker.SetTolerance(0.01)
        self.a1_picker.PickFromListOn()
        self.a2_picker = vtk.vtkCellPicker()
        self.a2_picker.SetTolerance(0.005)
        self.a2_picker.PickFromListOn()
        self.poly_picker = vtk.vtkCellPicker()
        self.poly_picker.SetTolerance(0.0)
        self.poly_picker.PickFromListOn()

    # ---- A1 node pick --------------------------------------------------------

    def _a1_pick_text(self, point_idx: int) -> str:
        return f"A1 node {self.a1_ids[point_idx]}"

    # ---- A2 link coloring / pick ---------------------------------------------

    def _a2_cell_colors(self) -> NDArray[np.uint8]:
        """RGB per A2_LINK: bundle color on mainline, junction color on interior."""
        bundle_rgb = self.bundle_palette[self.segmentation.bundle_id]
        is_interior = self.segmentation.junction_id >= 0
        rgb = np.asarray(bundle_rgb).copy()
        if is_interior.any():
            rgb[is_interior] = self.junction_palette[self.segmentation.junction_id[is_interior]]
        return rgb

    def _a2_pick_text(self, cell_id: int) -> str:
        link_id = self.a2_poly.cell_data["link_id"][cell_id]
        bid = int(self.a2_poly.cell_data["bundle_id"][cell_id])
        jid = int(self.a2_poly.cell_data["junction_id"][cell_id])
        jstr = str(jid) if jid >= 0 else "-"
        return f"link {link_id}  bundle {bid}  junction {jstr}"

    # ---- A3 / A4 polygon coloring / pick -------------------------------------

    def _a3_face_rgb(self) -> NDArray[np.uint8]:
        n = len(self.a3_ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, (kind, rt) in enumerate(zip(self.a3_kinds, self.a3_road_types, strict=True)):
            base = _A3_PROTECTED_RGB if kind == "7" else _A3_ROAD_TYPE_RGB.get(rt, _A4_FALLBACK_RGB)
            rgb[i] = base
        return rgb

    def _a4_face_rgb(self) -> NDArray[np.uint8]:
        n = len(self.a4_ids)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i, st in enumerate(self.a4_subtypes):
            rgb[i] = _A4_SUBTYPE_RGB.get(st, _A4_FALLBACK_RGB)
        return rgb

    def _a3_pick_text(self, poly_idx: int) -> str:
        kind = self.a3_kinds[poly_idx]
        road_type = self.a3_road_types[poly_idx]
        return (
            f"A3 {self.a3_ids[poly_idx]}  "
            f"Kind={_A3_KIND_LABEL.get(kind, kind)}  "
            f"RoadType={_A3_ROAD_TYPE_LABEL.get(road_type, road_type)}"
        )

    def _a4_pick_text(self, poly_idx: int) -> str:
        return (
            f"A4 {self.a4_ids[poly_idx]}  "
            f"SubType={self.a4_subtypes[poly_idx]}  "
            f"Name={self.a4_names[poly_idx]}"
        )

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
        initial_rgb = self._a2_cell_colors()
        self.a2_poly.cell_data["rgb"] = initial_rgb.copy()

        self.plotter.background_color = "white"

        # A3/A4 polygon footprints (drawn first so links/dots sit on top).
        a3_actor = self._add_polygon_layer(self.a3_poly, self._a3_face_rgb())
        a4_actor = self._add_polygon_layer(self.a4_poly, self._a4_face_rgb())

        a2_actor = self.plotter.add_mesh(
            self.a2_poly,
            scalars="rgb",
            rgb=True,
            line_width=2.5,
            show_scalar_bar=False,
        )
        a1_actor = self.plotter.add_mesh(
            self.a1_poly,
            color="black",
            point_size=_NODE_POINT_SIZE,
            render_points_as_spheres=True,
        )
        self.plotter.add_axes()
        self.plotter.add_text(
            "[2] top-down  [3] 3D  [shift+click] A1 / A2 / A3 / A4",
            position="lower_left",
            font_size=10,
        )
        self._add_view_keys()

        self.a1_picker.AddPickList(a1_actor)
        self.a2_picker.AddPickList(a2_actor)
        if a3_actor is not None:
            self.poly_picker.AddPickList(a3_actor)
        if a4_actor is not None:
            self.poly_picker.AddPickList(a4_actor)

        self.plotter.iren.add_observer(
            "LeftButtonPressEvent",
            self._make_pick_handler(initial_rgb, a3_actor, a4_actor),
        )
        self.plotter.show()

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

    def _make_pick_handler(
        self,
        initial_rgb: NDArray[np.uint8],
        a3_actor: vtk.vtkActor | None,
        a4_actor: vtk.vtkActor | None,
    ) -> Any:
        def _on_left_press(_obj: Any, _event: str) -> None:
            iren = self.plotter.iren.interactor
            if not iren.GetShiftKey():
                return
            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer
            if self.a1_picker.Pick(x, y, 0, renderer):
                pid = self.a1_picker.GetPointId()
                if pid >= 0:
                    self._show_pick(self._a1_pick_text(pid))
                    return
            if self.a2_picker.Pick(x, y, 0, renderer):
                cid = self.a2_picker.GetCellId()
                if cid >= 0:
                    rgb = self.a2_poly.cell_data["rgb"]
                    rgb[:] = initial_rgb
                    rgb[cid] = [255, 255, 0]
                    self.a2_poly.cell_data["rgb"] = rgb
                    self._show_pick(self._a2_pick_text(cid))
                    return
            if self.poly_picker.Pick(x, y, 0, renderer):
                cid = self.poly_picker.GetCellId()
                if cid < 0:
                    return
                actor = self.poly_picker.GetActor()
                if actor is a3_actor:
                    pi = int(self.a3_poly.cell_data["poly_idx"][cid])
                    self._show_pick(self._a3_pick_text(pi))
                elif actor is a4_actor:
                    pi = int(self.a4_poly.cell_data["poly_idx"][cid])
                    self._show_pick(self._a4_pick_text(pi))

        return _on_left_press

    def _add_polygon_layer(
        self, poly: pv.PolyData, face_rgb: NDArray[np.uint8]
    ) -> vtk.vtkActor | None:
        """Add a translucent polygon layer; ``None`` when the layer is empty.

        Lighting is disabled so the fill is a flat color: polygons here are
        abstract footprints, not 3D surfaces, and VTK's Phong shading darkens
        triangles whose normals tilt away from the camera. That produced
        visible wedge-shaped shading artifacts on sloped sidewalks where the
        constrained-Delaunay triangulation picks triangles spanning a few
        meters of elevation.
        """
        if poly.n_cells == 0:
            return None
        n_tris = poly.n_cells
        # Broadcast per-original-polygon colors to triangulated cells.
        per_tri_rgb = face_rgb[poly.cell_data["poly_idx"]]
        poly.cell_data["rgb"] = per_tri_rgb
        assert per_tri_rgb.shape == (n_tris, 3)
        actor = self.plotter.add_mesh(
            poly,
            scalars="rgb",
            rgb=True,
            opacity=_POLY_OPACITY,
            show_scalar_bar=False,
            show_edges=False,
            lighting=False,
        )
        actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(*_POLY_DEPTH_OFFSET)
        return actor
