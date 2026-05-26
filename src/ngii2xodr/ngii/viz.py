"""Generic 3D visualization for canonical NGII datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
import shapely
import shapely.ops
import vtk
from numpy.typing import NDArray
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon

from ngii2xodr.ngii.app import FeatureRef, LoadedMap
from ngii2xodr.ngii.data import NGIIConfig, load_ngii
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import (
    LineFeature,
    NGIIFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
)
from ngii2xodr.ngii.data.manual_2023 import LAYER_SPECS, SPECS_BY_LAYER_NAME
from ngii2xodr.ngii.segmentation import Segmentation, SegmentationConfig

log = logging.getLogger(__name__)

vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

SelectCallback = Callable[[FeatureRef], None]
GeometryKind = Literal["point", "line", "polygon"]
LayerGeometryKind = Literal["point", "line", "polygon", "mixed"]


@dataclass(slots=True, frozen=True)
class VizLayerConfig:
    visible: bool
    rgb: tuple[int, int, int]
    point_size: float
    line_width: float
    opacity: float


@dataclass(slots=True, frozen=True)
class VizCameraFocusConfig:
    padding_m: float
    min_scale_m: float
    max_scale_m: float


@dataclass(slots=True, frozen=True)
class VizConfig:
    background_color: tuple[float, float, float]
    highlight_rgb: tuple[int, int, int]
    selector_tol_point: float
    selector_tol_line: float
    selector_tol_poly: float
    point_hit_radius_px: float
    poly_depth_offset_factor: float
    poly_depth_offset_units: float
    segmentation_seed: int
    camera_focus: VizCameraFocusConfig
    layers: dict[str, VizLayerConfig]


@dataclass(slots=True)
class RenderRegistry:
    layers: dict[str, RenderLayer]

    def selectable_layers(self) -> tuple[RenderLayer, ...]:
        return tuple(self.layers.values())

    def layer_for_ref(self, ref: FeatureRef) -> RenderLayer | None:
        return self.layers.get(ref.layer_attr)

    def render_layers(self, kind: GeometryKind | None = None) -> tuple[RenderLayer, ...]:
        layers: list[RenderLayer] = []
        for layer in self.layers.values():
            layers.extend(layer.render_layers(kind))
        return tuple(layers)


@dataclass(slots=True)
class RenderLayer:
    layer_attr: str
    store: LayerStore[Any]
    kind: LayerGeometryKind
    config: VizLayerConfig
    selected_rgb: tuple[int, int, int]
    selector_tolerance: float
    poly_depth_offset_factor: float
    poly_depth_offset_units: float
    color_fn: Callable[[], NDArray[np.uint8]]
    feature_indices: tuple[int, ...] | None = None
    actor: vtk.vtkActor | None = field(default=None, init=False)
    selected_index: int | None = field(default=None, init=False)

    @property
    def layer_name(self) -> str:
        return self.store.layer_name

    @property
    def count(self) -> int:
        return len(self.store)

    @property
    def geometry_label(self) -> str:
        return self.kind

    def feature_ref_at(self, index: int) -> FeatureRef:
        feature_index = self._feature_index_at(index)
        return FeatureRef(self.layer_attr, self.store.features[feature_index].id)

    def index_for_ref(self, ref: FeatureRef) -> int | None:
        if ref.layer_attr != self.layer_attr:
            return None
        return self.store.id_to_index.get(ref.feature_id)

    def _feature_index_at(self, render_index: int) -> int:
        if self.feature_indices is None:
            return render_index
        return self.feature_indices[render_index]

    def _render_index_for_selected(self) -> int | None:
        if self.selected_index is None:
            return None
        if self.feature_indices is None:
            return self.selected_index
        try:
            return self.feature_indices.index(self.selected_index)
        except ValueError:
            return None

    def render_layers(self, kind: GeometryKind | None = None) -> tuple[RenderLayer, ...]:
        if self.kind == "mixed":
            return ()
        if kind is None or self.kind == kind:
            return (self,)
        return ()

    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        raise NotImplementedError

    def set_visible(self, on: bool) -> None:
        if self.actor is not None:
            self.actor.SetVisibility(int(on))

    def set_selected_ref(self, ref: FeatureRef | None) -> None:
        self.selected_index = None if ref is None else self.index_for_ref(ref)
        self.refresh_colors()

    def refresh_colors(self) -> None:
        raise NotImplementedError

    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        raise NotImplementedError


@dataclass(slots=True)
class PointRenderLayer(RenderLayer):
    poly: pv.PolyData = field(init=False)
    selector: vtk.vtkPointPicker = field(init=False)

    def __post_init__(self) -> None:
        points: list[NDArray[np.float64]] = []
        for index in self._iter_feature_indices():
            point = _point_feature_xyz(self.store.features[index])
            if point is not None:
                points.append(point)
        xyz = np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)
        self.poly = pv.PolyData(xyz)
        self.selector = vtk.vtkPointPicker()
        self.selector.SetTolerance(self.selector_tolerance)
        self.selector.PickFromListOn()

    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        del polygon_selector
        if self.poly.n_points == 0:
            return
        self.refresh_colors()
        self.actor = plotter.add_mesh(
            self.poly,
            scalars="rgb",
            rgb=True,
            point_size=self.config.point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        self.actor.SetVisibility(int(self.config.visible))
        self.selector.AddPickList(self.actor)

    def refresh_colors(self) -> None:
        if self.poly.n_points == 0:
            return
        rgb = self._render_colors()
        render_selected = self._render_index_for_selected()
        if render_selected is not None and 0 <= render_selected < len(rgb):
            rgb[render_selected] = self.selected_rgb
        self.poly.point_data["rgb"] = rgb
        self.poly.Modified()

    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        if self.actor is None or not self.selector.Pick(x, y, 0, renderer):
            return None
        index = self.selector.GetPointId()
        return self.feature_ref_at(index) if 0 <= index < self.poly.n_points else None

    def nearest_ref_in_display_radius(
        self, x: int, y: int, renderer: Any, radius_px: float
    ) -> tuple[FeatureRef, float] | None:
        if self.actor is None or not self.actor.GetVisibility() or self.poly.n_points == 0:
            return None
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        best_index: int | None = None
        best_dist2 = radius_px * radius_px
        for render_index in range(self.poly.n_points):
            point = self.poly.points[render_index]
            coordinate.SetValue(float(point[0]), float(point[1]), float(point[2]))
            display = coordinate.GetComputedDoubleDisplayValue(renderer)
            dx = float(display[0] - x)
            dy = float(display[1] - y)
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist2:
                best_dist2 = dist2
                best_index = render_index
        if best_index is None:
            return None
        return self.feature_ref_at(best_index), best_dist2

    def _iter_feature_indices(self) -> tuple[int, ...]:
        if self.feature_indices is not None:
            return self.feature_indices
        return tuple(range(len(self.store.features)))

    def _render_colors(self) -> NDArray[np.uint8]:
        colors = np.asarray(self.color_fn(), dtype=np.uint8)
        if self.feature_indices is None:
            return np.array(colors, dtype=np.uint8, copy=True)
        return np.array(colors[list(self.feature_indices)], dtype=np.uint8, copy=True)


@dataclass(slots=True)
class LineRenderLayer(RenderLayer):
    poly: pv.PolyData = field(init=False)
    selector: vtk.vtkCellPicker = field(init=False)

    def __post_init__(self) -> None:
        polylines: list[NDArray[np.float64]] = []
        for index in self._iter_feature_indices():
            feature = self.store.features[index]
            if isinstance(feature, LineFeature):
                polylines.append(feature.polyline)
        self.poly = _polyline_polydata(polylines)
        self.selector = vtk.vtkCellPicker()
        self.selector.SetTolerance(self.selector_tolerance)
        self.selector.PickFromListOn()

    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        del polygon_selector
        if self.poly.n_cells == 0:
            return
        self.refresh_colors()
        self.actor = plotter.add_mesh(
            self.poly,
            scalars="rgb",
            rgb=True,
            line_width=self.config.line_width,
            show_scalar_bar=False,
        )
        self.actor.SetVisibility(int(self.config.visible))
        self.selector.AddPickList(self.actor)

    def refresh_colors(self) -> None:
        if self.poly.n_cells == 0:
            return
        rgb = self._render_colors()
        render_selected = self._render_index_for_selected()
        if render_selected is not None and 0 <= render_selected < len(rgb):
            rgb[render_selected] = self.selected_rgb
        self.poly.cell_data["rgb"] = rgb
        self.poly.Modified()

    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        if self.actor is None or not self.selector.Pick(x, y, 0, renderer):
            return None
        index = self.selector.GetCellId()
        return self.feature_ref_at(index) if 0 <= index < self.poly.n_cells else None

    def _iter_feature_indices(self) -> tuple[int, ...]:
        if self.feature_indices is not None:
            return self.feature_indices
        return tuple(range(len(self.store.features)))

    def _render_colors(self) -> NDArray[np.uint8]:
        colors = np.asarray(self.color_fn(), dtype=np.uint8)
        if self.feature_indices is None:
            return np.array(colors, dtype=np.uint8, copy=True)
        return np.array(colors[list(self.feature_indices)], dtype=np.uint8, copy=True)


@dataclass(slots=True)
class PolygonRenderLayer(RenderLayer):
    poly: pv.PolyData = field(init=False)

    def __post_init__(self) -> None:
        feature_indices = self.feature_indices or tuple(range(len(self.store.features)))
        self.poly = _polygon_polydata(self.layer_name, self.store.features, feature_indices)

    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        if self.poly.n_cells == 0:
            return
        self.refresh_colors()
        self.actor = plotter.add_mesh(
            self.poly,
            scalars="rgb",
            rgb=True,
            opacity=self.config.opacity,
            show_scalar_bar=False,
            show_edges=False,
            lighting=False,
        )
        self.actor.SetVisibility(int(self.config.visible))
        self.actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(
            self.poly_depth_offset_factor, self.poly_depth_offset_units
        )
        polygon_selector.AddPickList(self.actor)

    def refresh_colors(self) -> None:
        if self.poly.n_cells == 0:
            return
        face_rgb = self.color_fn()
        feature_idx = np.asarray(self.poly.cell_data["feature_idx"])
        rgb = np.array(face_rgb[feature_idx], dtype=np.uint8, copy=True)
        if self.selected_index is not None:
            rgb[feature_idx == self.selected_index] = self.selected_rgb
        self.poly.cell_data["rgb"] = rgb
        self.poly.Modified()

    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        del x, y, renderer
        return None


@dataclass(slots=True)
class CompositeRenderLayer(RenderLayer):
    sublayers: tuple[RenderLayer, ...] = ()

    @property
    def geometry_label(self) -> str:
        return "+".join(layer.kind for layer in self.sublayers if layer.kind != "mixed")

    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        for layer in self.sublayers:
            layer.attach(plotter, polygon_selector)

    def set_visible(self, on: bool) -> None:
        for layer in self.sublayers:
            layer.set_visible(on)

    def set_selected_ref(self, ref: FeatureRef | None) -> None:
        self.selected_index = None if ref is None else self.index_for_ref(ref)
        for layer in self.sublayers:
            layer.set_selected_ref(ref)

    def refresh_colors(self) -> None:
        for layer in self.sublayers:
            layer.refresh_colors()

    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        for layer in self.sublayers:
            ref = layer.try_select(x, y, renderer)
            if ref is not None:
                return ref
        return None

    def render_layers(self, kind: GeometryKind | None = None) -> tuple[RenderLayer, ...]:
        layers: list[RenderLayer] = []
        for layer in self.sublayers:
            layers.extend(layer.render_layers(kind))
        return tuple(layers)


class HdMapViz:
    """Dataset-native scene. Selection speaks only in :class:`FeatureRef`."""

    def __init__(
        self,
        ngii_dir: Path,
        coordinate: str,
        ngii_cfg: NGIIConfig,
        seg_cfg: SegmentationConfig,
        viz_cfg: VizConfig,
        plotter: pv.Plotter | None = None,
        on_select: SelectCallback | None = None,
    ) -> None:
        self.ngii_dir = ngii_dir
        self.viz_cfg = viz_cfg
        self.plotter = plotter if plotter is not None else pv.Plotter()
        self.on_select = on_select
        self.dataset: NGIIDataset = load_ngii(ngii_dir, coordinate, ngii_cfg)
        self.segmentation = Segmentation.from_dataset(self.dataset, seg_cfg)
        self._segmentation_level = 0
        self._selected_ref: FeatureRef | None = None
        self._palette_by_stage = {
            result.stage_id: _random_palette(len(result.entities), viz_cfg.segmentation_seed + i)
            for i, result in enumerate(self.segmentation.stage_results)
        }
        self.polygon_selector = vtk.vtkCellPicker()
        self.polygon_selector.SetTolerance(viz_cfg.selector_tol_poly)
        self.polygon_selector.PickFromListOn()
        self.registry = self._build_registry()
        self.loaded_map = LoadedMap(
            dataset=self.dataset,
            segmentation=self.segmentation,
            render_registry=self.registry,
            load_profile=self.dataset.load_profile,
            segmentation_profile=self.segmentation.profile,
        )
        self._left_press_tag: int | None = None
        self._key_events_bound: tuple[str, ...] = ()

    def _build_registry(self) -> RenderRegistry:
        layers: dict[str, RenderLayer] = {}
        for attr, store in _dataset_layer_items(self.dataset):
            kinds = _geometry_kinds(store)
            if not kinds:
                continue
            config = self.viz_cfg.layers.get(attr, _default_layer_config(attr))
            color_fn = self._color_fn(attr, store, config)
            if len(kinds) == 1:
                layers[attr] = self._make_render_layer(
                    attr, store, kinds[0], config, color_fn, None
                )
                continue
            sublayers = tuple(
                self._make_render_layer(
                    attr,
                    store,
                    kind,
                    config,
                    color_fn,
                    _feature_indices_for_kind(store, kind),
                )
                for kind in kinds
            )
            layers[attr] = CompositeRenderLayer(
                layer_attr=attr,
                store=store,
                kind="mixed",
                config=config,
                selected_rgb=self.viz_cfg.highlight_rgb,
                selector_tolerance=0.0,
                poly_depth_offset_factor=self.viz_cfg.poly_depth_offset_factor,
                poly_depth_offset_units=self.viz_cfg.poly_depth_offset_units,
                color_fn=color_fn,
                sublayers=sublayers,
            )
        return RenderRegistry(layers)

    def _make_render_layer(
        self,
        attr: str,
        store: LayerStore[Any],
        kind: GeometryKind,
        config: VizLayerConfig,
        color_fn: Callable[[], NDArray[np.uint8]],
        feature_indices: tuple[int, ...] | None,
    ) -> RenderLayer:
        layer_cls: type[RenderLayer]
        if kind == "point":
            layer_cls = PointRenderLayer
        elif kind == "line":
            layer_cls = LineRenderLayer
        else:
            layer_cls = PolygonRenderLayer
        return layer_cls(
            layer_attr=attr,
            store=store,
            kind=kind,
            config=config,
            selected_rgb=self.viz_cfg.highlight_rgb,
            selector_tolerance=self._selector_tolerance(kind),
            poly_depth_offset_factor=self.viz_cfg.poly_depth_offset_factor,
            poly_depth_offset_units=self.viz_cfg.poly_depth_offset_units,
            color_fn=color_fn,
            feature_indices=feature_indices,
        )

    def _selector_tolerance(self, kind: GeometryKind) -> float:
        if kind == "point":
            return self.viz_cfg.selector_tol_point
        if kind == "line":
            return self.viz_cfg.selector_tol_line
        return self.viz_cfg.selector_tol_poly

    def _color_fn(
        self, attr: str, store: LayerStore[Any], config: VizLayerConfig
    ) -> Callable[[], NDArray[np.uint8]]:
        if attr == "a2_link":
            return lambda: self._a2_colors(store, config)
        return lambda: np.tile(np.asarray(config.rgb, dtype=np.uint8), (len(store), 1))

    def _a2_colors(self, store: LayerStore[Any], config: VizLayerConfig) -> NDArray[np.uint8]:
        rgb = np.tile(np.asarray(config.rgb, dtype=np.uint8), (len(store), 1))
        colored_refs: set[FeatureRef] = set()
        for result in self.segmentation.active_results(self._segmentation_level):
            palette = self._palette_by_stage[result.stage_id]
            for ref, entity_id in result.entity_id_by_ref.items():
                if ref.layer_attr != "a2_link" or ref in colored_refs:
                    continue
                idx = store.id_to_index.get(ref.feature_id)
                if idx is not None and 0 <= entity_id < len(palette):
                    rgb[idx] = palette[entity_id]
                    colored_refs.add(ref)
        return rgb

    def attach(self) -> None:
        self.plotter.background_color = self.viz_cfg.background_color
        for layer in self.registry.selectable_layers():
            layer.attach(self.plotter, self.polygon_selector)
        self.plotter.add_axes()
        self._add_view_keys()
        self._left_press_tag = self.plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_left_press
        )

    def show(self) -> None:
        self.attach()
        self.plotter.show()

    def detach(self) -> None:
        for layer in self.registry.render_layers():
            if layer.actor is not None:
                self.plotter.remove_actor(layer.actor)
        if self._left_press_tag is not None:
            self.plotter.iren.remove_observer(self._left_press_tag)
            self._left_press_tag = None
        for key in self._key_events_bound:
            self.plotter.clear_events_for_key(key)
        self._key_events_bound = ()
        self.plotter.hide_axes()

    def set_layer_visible(self, layer_attr: str, on: bool) -> None:
        layer = self.registry.layers.get(layer_attr)
        if layer is not None:
            layer.set_visible(on)
            self.plotter.render()

    def set_segmentation_level(self, level: int) -> None:
        if level < 0 or level > len(self.segmentation.stage_results):
            raise ValueError(level)
        if level == self._segmentation_level:
            return
        self._segmentation_level = level
        layer = self.registry.layers.get("a2_link")
        if layer is not None:
            layer.refresh_colors()
        self.plotter.render()

    def select_feature(self, ref: FeatureRef | None, *, emit: bool = True) -> None:
        self._selected_ref = ref
        for layer in self.registry.selectable_layers():
            layer.set_selected_ref(ref)
        self.plotter.render()
        if emit and ref is not None and self.on_select is not None:
            self.on_select(ref)

    def focus_feature(self, ref: FeatureRef) -> None:
        feature = self.dataset.store_for_attr(ref.layer_attr).get(ref.feature_id)
        if feature is None:
            return
        points = _feature_points(feature)
        if points is None or len(points) == 0:
            return
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        center = (mins + maxs) / 2.0
        span_xy = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]))
        cfg = self.viz_cfg.camera_focus
        scale = min(max(span_xy * 0.75 + cfg.padding_m, cfg.min_scale_m), cfg.max_scale_m)
        camera_height = max(scale * 4.0, 1000.0)
        self.plotter.enable_parallel_projection()
        self.plotter.camera_position = [
            (float(center[0]), float(center[1]), float(center[2] + camera_height)),
            (float(center[0]), float(center[1]), float(center[2])),
            (0.0, 1.0, 0.0),
        ]
        self.plotter.camera.parallel_scale = scale
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _add_view_keys(self) -> None:
        def _view_2d() -> None:
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()

        self.plotter.add_key_event("2", _view_2d)
        self._key_events_bound = ("2",)

    def _on_left_press(self, _obj: Any, _event: str) -> None:
        iren = self.plotter.iren.interactor
        if not iren.GetShiftKey():
            return
        x, y = iren.GetEventPosition()
        renderer = self.plotter.renderer
        point_ref = self._try_point_priority_select(x, y, renderer)
        if point_ref is not None:
            self.select_feature(point_ref)
            return
        for layer in self.registry.render_layers("line"):
            ref = layer.try_select(x, y, renderer)
            if ref is not None:
                self.select_feature(ref)
                return
        self._try_polygon_select(x, y, renderer)

    def _try_point_priority_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        best_ref: FeatureRef | None = None
        best_priority = 1_000_000
        best_dist2 = float("inf")
        for layer in self.registry.render_layers("point"):
            if not isinstance(layer, PointRenderLayer):
                continue
            candidate = layer.nearest_ref_in_display_radius(
                x, y, renderer, self.viz_cfg.point_hit_radius_px
            )
            if candidate is None:
                continue
            ref, dist2 = candidate
            priority = _point_layer_priority(ref.layer_attr)
            if priority < best_priority or (priority == best_priority and dist2 < best_dist2):
                best_ref = ref
                best_priority = priority
                best_dist2 = dist2
        return best_ref

    def _try_polygon_select(self, x: int, y: int, renderer: Any) -> None:
        if not self.polygon_selector.Pick(x, y, 0, renderer):
            return
        cid = self.polygon_selector.GetCellId()
        if cid < 0:
            return
        actor = self.polygon_selector.GetActor()
        for layer in self.registry.render_layers("polygon"):
            if not isinstance(layer, PolygonRenderLayer) or layer.actor is not actor:
                continue
            feature_idx = int(layer.poly.cell_data["feature_idx"][cid])
            self.select_feature(FeatureRef(layer.layer_attr, layer.store.features[feature_idx].id))
            return


def _dataset_layer_items(dataset: NGIIDataset) -> tuple[tuple[str, LayerStore[Any]], ...]:
    return tuple(
        (spec.python_attr, dataset.store_for_attr(spec.python_attr)) for spec in LAYER_SPECS
    )


def _geometry_kinds(store: LayerStore[Any]) -> tuple[GeometryKind, ...]:
    if not store.features:
        spec = SPECS_BY_LAYER_NAME.get(store.layer_name)
        return () if spec is None else (spec.geometry_kind,)
    kinds = {_feature_geometry_kind(feature) for feature in store.features}
    return tuple(kind for kind in ("point", "line", "polygon") if kind in kinds)


def _feature_indices_for_kind(store: LayerStore[Any], kind: GeometryKind) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature in enumerate(store.features)
        if _feature_geometry_kind(feature) == kind
    )


def _feature_geometry_kind(feature: NGIIFeature) -> GeometryKind:
    if isinstance(feature, PointFeature):
        return "point"
    if isinstance(feature, LineFeature):
        return "line"
    if isinstance(feature, PolygonFeature):
        return "polygon"
    if isinstance(feature, PointOrPolygonFeature):
        return feature.geometry_kind
    msg = f"{feature.layer_name} {feature.id!r} has no renderable geometry"
    raise TypeError(msg)


def _default_layer_config(attr: str) -> VizLayerConfig:
    rgb_by_attr = {
        "a1_node": (40, 40, 40),
        "a2_link": (110, 110, 120),
        "a3_drivewaysection": (180, 180, 180),
        "a4_subsidiarysection": (120, 200, 120),
        "a5_parkinglot": (160, 160, 220),
        "b1_safetysign": (200, 80, 80),
        "b2_surfacelinemark": (255, 255, 255),
        "b3_surfacemark": (255, 180, 60),
        "c1_trafficlight": (30, 180, 60),
        "c2_kilopost": (80, 120, 220),
        "c3_vehicleprotectionsafety": (140, 140, 140),
        "c4_speedbump": (200, 120, 40),
        "c5_heightbarrier": (160, 80, 200),
        "c6_postpoint": (80, 80, 80),
    }
    return VizLayerConfig(
        visible=True,
        rgb=rgb_by_attr.get(attr, (160, 160, 160)),
        point_size=4.0,
        line_width=1.8,
        opacity=0.85,
    )


def _random_palette(n: int, seed: int) -> NDArray[np.uint8]:
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(60, 240, size=(n, 3), dtype=np.uint8)


def _polyline_polydata(polylines: list[NDArray[np.float64]]) -> pv.PolyData:
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


def _polygon_polydata(
    layer_name: str, features: list[NGIIFeature], feature_indices: tuple[int, ...]
) -> pv.PolyData:
    if not features:
        return pv.PolyData()
    all_pts: list[NDArray[np.float64]] = []
    face_cells: list[int] = []
    tri_feature_idx: list[int] = []
    vert_offset = 0
    for feature_idx in feature_indices:
        feature = features[feature_idx]
        ring = _polygon_feature_ring(feature)
        if ring is None:
            continue
        for tri_xy in _triangulated_ring_xy(layer_name, feature.id, ring):
            tri_z = np.array([_nearest_ring_z(ring, x, y) for x, y in tri_xy])
            all_pts.append(np.column_stack([tri_xy, tri_z]))
            face_cells.extend([3, vert_offset, vert_offset + 1, vert_offset + 2])
            vert_offset += 3
            tri_feature_idx.append(feature_idx)
    if not all_pts:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(all_pts), faces=np.asarray(face_cells, dtype=np.int64))
    poly.cell_data["feature_idx"] = np.asarray(tri_feature_idx, dtype=np.int32)
    return poly


def _triangulated_ring_xy(
    layer_name: str, feature_id: str, ring: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    polygon = ShapelyPolygon(ring[:, :2])
    triangles = _constrained_triangle_xys(polygon)
    if triangles:
        return triangles

    repaired_triangles: list[NDArray[np.float64]] = []
    for repaired in _polygonal_repair_candidates(polygon):
        repaired_triangles.extend(_constrained_triangle_xys(repaired))
        if not repaired_triangles:
            repaired_triangles.extend(_unconstrained_triangle_xys(repaired))
    if repaired_triangles:
        log.warning(
            "render polygon repaired for display: %s %s (%s)",
            layer_name,
            feature_id,
            shapely.is_valid_reason(polygon),
        )
        return repaired_triangles

    log.warning(
        "render polygon skipped: %s %s cannot be triangulated (%s)",
        layer_name,
        feature_id,
        shapely.is_valid_reason(polygon),
    )
    return []


def _constrained_triangle_xys(polygon: ShapelyPolygon) -> list[NDArray[np.float64]]:
    if polygon.is_empty or polygon.area <= 0.0:
        return []
    try:
        tris = shapely.constrained_delaunay_triangles(polygon)
    except (GEOSException, ValueError):
        return []
    return [
        np.asarray(tri.exterior.coords[:3], dtype=np.float64)
        for tri in tris.geoms
        if isinstance(tri, ShapelyPolygon) and tri.area > 0.0
    ]


def _unconstrained_triangle_xys(polygon: ShapelyPolygon) -> list[NDArray[np.float64]]:
    if polygon.is_empty or polygon.area <= 0.0:
        return []
    triangles = []
    for tri in shapely.ops.triangulate(polygon):
        if polygon.covers(tri.representative_point()) and tri.area > 0.0:
            triangles.append(np.asarray(tri.exterior.coords[:3], dtype=np.float64))
    return triangles


def _polygonal_repair_candidates(polygon: ShapelyPolygon) -> list[ShapelyPolygon]:
    candidates: list[ShapelyPolygon] = []
    for repaired in (shapely.make_valid(polygon), polygon.buffer(0)):
        candidates.extend(_polygonal_parts(repaired))
    return candidates


def _polygonal_parts(geometry: object) -> list[ShapelyPolygon]:
    if isinstance(geometry, ShapelyPolygon):
        return [geometry] if geometry.area > 0.0 else []
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if part.area > 0.0]
    if isinstance(geometry, GeometryCollection):
        parts: list[ShapelyPolygon] = []
        for part in geometry.geoms:
            parts.extend(_polygonal_parts(part))
        return parts
    return []


def _nearest_ring_z(ring: NDArray[np.float64], x: float, y: float) -> float:
    xy = ring[:, :2]
    diff = xy - np.asarray((x, y), dtype=np.float64)
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    return float(ring[idx, 2])


def _point_feature_xyz(feature: NGIIFeature) -> NDArray[np.float64] | None:
    if isinstance(feature, PointFeature):
        return feature.point
    if isinstance(feature, PointOrPolygonFeature):
        return feature.point
    return None


def _polygon_feature_ring(feature: NGIIFeature) -> NDArray[np.float64] | None:
    if isinstance(feature, PolygonFeature):
        return feature.ring
    if isinstance(feature, PointOrPolygonFeature):
        return feature.ring
    return None


def _feature_points(feature: NGIIFeature) -> NDArray[np.float64] | None:
    point = _point_feature_xyz(feature)
    if point is not None:
        return point.reshape(1, 3)
    if isinstance(feature, LineFeature):
        return feature.polyline
    ring = _polygon_feature_ring(feature)
    if ring is not None:
        return ring
    return None


def _point_layer_priority(layer_attr: str) -> int:
    priority = {
        "a1_node": 0,
        "c1_trafficlight": 1,
        "c6_postpoint": 1,
        "c2_kilopost": 2,
        "b1_safetysign": 2,
    }
    return priority.get(layer_attr, 10)
