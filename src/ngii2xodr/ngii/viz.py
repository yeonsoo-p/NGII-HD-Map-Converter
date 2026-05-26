"""Generic 3D visualization for canonical NGII datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, override

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
from ngii2xodr.ngii.segmentation import Segmentation, SegmentationConfig
from ngii2xodr.profile import (
    PerformanceProfile,
    ProfileTimer,
    ViewportInteractionProfiler,
    ViewportProfilingConfig,
)

log = logging.getLogger(__name__)

vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

SelectCallback = Callable[[FeatureRef], None]
GeometryKind = Literal["point", "line", "polygon"]
LayerGeometryKind = Literal["point", "line", "polygon", "mixed"]


@dataclass(slots=True, frozen=True)
class _PolygonBuildResult:
    poly: pv.PolyData
    fallback_point_poly: pv.PolyData
    fallback_feature_indices: tuple[int, ...]


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
class VizPointConfig:
    render_as_spheres: bool


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
    points: VizPointConfig
    profiling: ViewportProfilingConfig
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
class RenderLayer(ABC):
    layer_attr: str
    store: LayerStore[Any]
    kind: LayerGeometryKind
    config: VizLayerConfig
    selected_rgb: tuple[int, int, int]
    selector_tolerance: float
    poly_depth_offset_factor: float
    poly_depth_offset_units: float
    render_points_as_spheres: bool
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

    @abstractmethod
    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None: ...

    def set_visible(self, on: bool) -> None:
        if self.actor is not None:
            self.actor.SetVisibility(int(on))

    def set_selected_ref(self, ref: FeatureRef | None) -> None:
        self.selected_index = None if ref is None else self.index_for_ref(ref)
        self.refresh_colors()

    @abstractmethod
    def refresh_colors(self) -> None: ...

    @abstractmethod
    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None: ...

    def try_point_select(
        self, x: int, y: int, renderer: Any, radius_px: float
    ) -> tuple[FeatureRef, float] | None:
        del x, y, renderer, radius_px
        return None


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

    @override
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
            render_points_as_spheres=self.render_points_as_spheres,
            show_scalar_bar=False,
        )
        self.actor.SetVisibility(int(self.config.visible))
        self.selector.AddPickList(self.actor)

    @override
    def refresh_colors(self) -> None:
        if self.poly.n_points == 0:
            return
        rgb = self._render_colors()
        render_selected = self._render_index_for_selected()
        if render_selected is not None and 0 <= render_selected < len(rgb):
            rgb[render_selected] = self.selected_rgb
        self.poly.point_data["rgb"] = rgb
        self.poly.Modified()

    @override
    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        if self.actor is None or not self.selector.Pick(x, y, 0, renderer):
            return None
        index = self.selector.GetPointId()
        return self.feature_ref_at(index) if 0 <= index < self.poly.n_points else None

    @override
    def try_point_select(
        self, x: int, y: int, renderer: Any, radius_px: float
    ) -> tuple[FeatureRef, float] | None:
        if self.actor is None or not self.actor.GetVisibility() or self.poly.n_points == 0:
            return None
        if not self.selector.Pick(x, y, 0, renderer):
            return None
        index = self.selector.GetPointId()
        if not 0 <= index < self.poly.n_points:
            return None
        dist2 = _display_distance2(self.poly.points[index], x, y, renderer)
        if dist2 > radius_px * radius_px:
            return None
        return self.feature_ref_at(index), dist2

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

    @override
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

    @override
    def refresh_colors(self) -> None:
        if self.poly.n_cells == 0:
            return
        rgb = self._render_colors()
        render_selected = self._render_index_for_selected()
        if render_selected is not None and 0 <= render_selected < len(rgb):
            rgb[render_selected] = self.selected_rgb
        self.poly.cell_data["rgb"] = rgb
        self.poly.Modified()

    @override
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
    fallback_point_poly: pv.PolyData = field(init=False)
    fallback_feature_indices: tuple[int, ...] = field(init=False)
    fallback_actor: vtk.vtkActor | None = field(default=None, init=False)
    fallback_selector: vtk.vtkPointPicker = field(init=False)

    def __post_init__(self) -> None:
        feature_indices = self.feature_indices or tuple(range(len(self.store.features)))
        built = _polygon_polydata(self.layer_name, self.store.features, feature_indices)
        self.poly = built.poly
        self.fallback_point_poly = built.fallback_point_poly
        self.fallback_feature_indices = built.fallback_feature_indices
        self.fallback_selector = vtk.vtkPointPicker()
        self.fallback_selector.SetTolerance(self.selector_tolerance)
        self.fallback_selector.PickFromListOn()

    @property
    @override
    def geometry_label(self) -> str:
        if self.fallback_point_poly.n_points > 0:
            return "polygon+point"
        return self.kind

    @override
    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        self.refresh_colors()
        if self.poly.n_cells > 0:
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
        if self.fallback_point_poly.n_points > 0:
            self.fallback_actor = plotter.add_mesh(
                self.fallback_point_poly,
                scalars="rgb",
                rgb=True,
                point_size=self.config.point_size,
                render_points_as_spheres=self.render_points_as_spheres,
                show_scalar_bar=False,
            )
            self.fallback_actor.SetVisibility(int(self.config.visible))
            self.fallback_selector.AddPickList(self.fallback_actor)

    @override
    def set_visible(self, on: bool) -> None:
        RenderLayer.set_visible(self, on)
        if self.fallback_actor is not None:
            self.fallback_actor.SetVisibility(int(on))

    @override
    def refresh_colors(self) -> None:
        face_rgb = self.color_fn()
        if self.poly.n_cells > 0:
            feature_idx = np.asarray(self.poly.cell_data["feature_idx"])
            rgb = np.array(face_rgb[feature_idx], dtype=np.uint8, copy=True)
            if self.selected_index is not None:
                rgb[feature_idx == self.selected_index] = self.selected_rgb
            self.poly.cell_data["rgb"] = rgb
            self.poly.Modified()
        if self.fallback_point_poly.n_points > 0:
            indices = np.asarray(self.fallback_feature_indices, dtype=np.int32)
            rgb = np.array(face_rgb[indices], dtype=np.uint8, copy=True)
            if self.selected_index is not None:
                rgb[indices == self.selected_index] = self.selected_rgb
            self.fallback_point_poly.point_data["rgb"] = rgb
            self.fallback_point_poly.Modified()

    @override
    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        del x, y, renderer
        return None

    @override
    def try_point_select(
        self, x: int, y: int, renderer: Any, radius_px: float
    ) -> tuple[FeatureRef, float] | None:
        if (
            self.fallback_actor is None
            or not self.fallback_actor.GetVisibility()
            or self.fallback_point_poly.n_points == 0
        ):
            return None
        if not self.fallback_selector.Pick(x, y, 0, renderer):
            return None
        index = self.fallback_selector.GetPointId()
        if not 0 <= index < self.fallback_point_poly.n_points:
            return None
        dist2 = _display_distance2(self.fallback_point_poly.points[index], x, y, renderer)
        if dist2 > radius_px * radius_px:
            return None
        feature_idx = self.fallback_feature_indices[index]
        return FeatureRef(self.layer_attr, self.store.features[feature_idx].id), dist2

    @override
    def render_layers(self, kind: GeometryKind | None = None) -> tuple[RenderLayer, ...]:
        if kind == "point" and self.fallback_point_poly.n_points > 0:
            return (self,)
        return RenderLayer.render_layers(self, kind)


@dataclass(slots=True)
class CompositeRenderLayer(RenderLayer):
    sublayers: tuple[RenderLayer, ...] = ()

    @property
    @override
    def geometry_label(self) -> str:
        return "+".join(layer.kind for layer in self.sublayers if layer.kind != "mixed")

    @override
    def attach(self, plotter: pv.Plotter, polygon_selector: vtk.vtkCellPicker) -> None:
        for layer in self.sublayers:
            layer.attach(plotter, polygon_selector)

    @override
    def set_visible(self, on: bool) -> None:
        for layer in self.sublayers:
            layer.set_visible(on)

    @override
    def set_selected_ref(self, ref: FeatureRef | None) -> None:
        self.selected_index = None if ref is None else self.index_for_ref(ref)
        for layer in self.sublayers:
            layer.set_selected_ref(ref)

    @override
    def refresh_colors(self) -> None:
        for layer in self.sublayers:
            layer.refresh_colors()

    @override
    def try_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        for layer in self.sublayers:
            ref = layer.try_select(x, y, renderer)
            if ref is not None:
                return ref
        return None

    @override
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
        self.viewport_profile = PerformanceProfile()
        self._viewport_profiler = ViewportInteractionProfiler(
            profile=self.viewport_profile,
            config=viz_cfg.profiling,
            logger=log,
        )
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
            viewport_profile=self.viewport_profile,
        )
        self._left_press_tag: int | None = None
        self._vtk_observer_tags: list[tuple[Any, int]] = []
        self._key_events_bound: tuple[str, ...] = ()
        self._attached = False

    def _build_registry(self) -> RenderRegistry:
        layers: dict[str, RenderLayer] = {}
        for attr, store in _dataset_layer_items(self.dataset):
            kinds = _geometry_kinds(self.dataset, store)
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
                render_points_as_spheres=self.viz_cfg.points.render_as_spheres,
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
            render_points_as_spheres=self.viz_cfg.points.render_as_spheres,
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
        if attr in self.dataset.schema.attrs_for_role("link"):
            return lambda: self._link_colors(attr, store, config)
        return lambda: np.tile(np.asarray(config.rgb, dtype=np.uint8), (len(store), 1))

    def _link_colors(
        self, layer_attr: str, store: LayerStore[Any], config: VizLayerConfig
    ) -> NDArray[np.uint8]:
        rgb = np.tile(np.asarray(config.rgb, dtype=np.uint8), (len(store), 1))
        colored_refs: set[FeatureRef] = set()
        for result in self.segmentation.active_results(self._segmentation_level):
            palette = self._palette_by_stage[result.stage_id]
            for ref, entity_id in result.entity_id_by_ref.items():
                if ref.layer_attr != layer_attr or ref in colored_refs:
                    continue
                idx = store.id_to_index.get(ref.feature_id)
                if idx is not None and 0 <= entity_id < len(palette):
                    rgb[idx] = palette[entity_id]
                    colored_refs.add(ref)
        return rgb

    def attach(self) -> None:
        if self._attached:
            return
        attached = False
        try:
            self.plotter.background_color = self.viz_cfg.background_color
            for layer in self.registry.selectable_layers():
                layer.attach(self.plotter, self.polygon_selector)
            self.plotter.add_axes()
            self._add_view_keys()
            self._left_press_tag = self.plotter.iren.add_observer(
                "LeftButtonPressEvent", self._on_left_press
            )
            self._attach_viewport_profile_observers()
            attached = True
        finally:
            if attached:
                self._attached = True
            else:
                self.detach()

    @contextmanager
    def attached(self) -> Iterator[None]:
        already_attached = self._attached
        self.attach()
        try:
            yield
        finally:
            if not already_attached:
                self.detach()

    def show(self) -> None:
        with self.attached():
            self.plotter.show()

    def detach(self) -> None:
        for layer in self.registry.render_layers():
            if layer.actor is not None:
                self.plotter.remove_actor(layer.actor)
                layer.actor = None
            if isinstance(layer, PolygonRenderLayer) and layer.fallback_actor is not None:
                self.plotter.remove_actor(layer.fallback_actor)
                layer.fallback_actor = None
        if self._left_press_tag is not None:
            self.plotter.iren.remove_observer(self._left_press_tag)
            self._left_press_tag = None
        self._remove_vtk_observers()
        for key in self._key_events_bound:
            self.plotter.clear_events_for_key(key)
        self._key_events_bound = ()
        self.plotter.hide_axes()
        self._attached = False

    def _attach_viewport_profile_observers(self) -> None:
        if not self.viz_cfg.profiling.enabled:
            return
        iren = self.plotter.iren
        self._vtk_observer_tags.extend(
            (
                (iren, iren.add_observer("StartInteractionEvent", self._on_interaction_start)),
                (
                    iren,
                    iren.add_observer("InteractionEvent", self._on_interaction_event),
                ),
                (iren, iren.add_observer("EndInteractionEvent", self._on_interaction_end)),
            )
        )
        render_window = getattr(self.plotter, "ren_win", None)
        if render_window is None:
            render_window = getattr(self.plotter, "render_window", None)
        if render_window is not None and hasattr(render_window, "AddObserver"):
            self._vtk_observer_tags.extend(
                (
                    (
                        render_window,
                        int(render_window.AddObserver("StartEvent", self._on_render_start)),
                    ),
                    (
                        render_window,
                        int(render_window.AddObserver("EndEvent", self._on_render_end)),
                    ),
                )
            )

    def _remove_vtk_observers(self) -> None:
        for observed, tag in self._vtk_observer_tags:
            if hasattr(observed, "remove_observer"):
                observed.remove_observer(tag)
            elif hasattr(observed, "RemoveObserver"):
                observed.RemoveObserver(tag)
        self._vtk_observer_tags = []

    def _on_interaction_start(self, _obj: Any, event: str) -> None:
        self._viewport_profiler.start_interaction(event)

    def _on_interaction_event(self, _obj: Any, _event: str) -> None:
        self._viewport_profiler.note_interaction_event()

    def _on_interaction_end(self, _obj: Any, event: str) -> None:
        self._viewport_profiler.end_interaction(event)

    def _on_render_start(self, _obj: Any, event: str) -> None:
        self._viewport_profiler.start_render(event)

    def _on_render_end(self, _obj: Any, event: str) -> None:
        self._viewport_profiler.end_render(event)

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
        for attr in self.dataset.schema.attrs_for_role("link"):
            layer = self.registry.layers.get(attr)
            if layer is not None:
                layer.refresh_colors()
        self.plotter.render()

    def select_feature(self, ref: FeatureRef | None, *, emit: bool = True) -> None:
        with self.viewport_profile.timed("selection_highlight") as timer:
            previous_ref = self._selected_ref
            self._selected_ref = ref
            touched_attrs = {
                selected_ref.layer_attr
                for selected_ref in (previous_ref, ref)
                if selected_ref is not None
            }
            for attr in touched_attrs:
                layer = self.registry.layers.get(attr)
                if layer is not None:
                    layer.set_selected_ref(ref)
            timer.detail = _selection_detail(ref, extra=f"layers={len(touched_attrs)}")
        self._log_profile_step(timer)
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
        with self.viewport_profile.timed("selection_pick_points") as timer:
            point_ref = self._try_point_priority_select(x, y, renderer)
            timer.detail = f"hit={point_ref is not None}"
        self._log_profile_step(timer)
        if point_ref is not None:
            self.select_feature(point_ref)
            return
        line_ref: FeatureRef | None = None
        with self.viewport_profile.timed("selection_pick_lines") as timer:
            for layer in self.registry.render_layers("line"):
                line_ref = layer.try_select(x, y, renderer)
                if line_ref is not None:
                    break
            timer.detail = f"hit={line_ref is not None}"
        self._log_profile_step(timer)
        if line_ref is not None:
            self.select_feature(line_ref)
            return
        with self.viewport_profile.timed("selection_pick_polygons") as timer:
            polygon_ref = self._try_polygon_select(x, y, renderer)
            timer.detail = f"hit={polygon_ref is not None}"
        self._log_profile_step(timer)
        if polygon_ref is not None:
            self.select_feature(polygon_ref)

    def _try_point_priority_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        best_ref: FeatureRef | None = None
        best_priority = 1_000_000
        best_dist2 = float("inf")
        for layer in self.registry.render_layers("point"):
            candidate = layer.try_point_select(x, y, renderer, self.viz_cfg.point_hit_radius_px)
            if candidate is None:
                continue
            ref, dist2 = candidate
            priority = _point_layer_priority(ref.layer_attr)
            if priority < best_priority or (priority == best_priority and dist2 < best_dist2):
                best_ref = ref
                best_priority = priority
                best_dist2 = dist2
        return best_ref

    def _try_polygon_select(self, x: int, y: int, renderer: Any) -> FeatureRef | None:
        if not self.polygon_selector.Pick(x, y, 0, renderer):
            return None
        cid = self.polygon_selector.GetCellId()
        if cid < 0:
            return None
        actor = self.polygon_selector.GetActor()
        for layer in self.registry.render_layers("polygon"):
            if not isinstance(layer, PolygonRenderLayer) or layer.actor is not actor:
                continue
            feature_idx = int(layer.poly.cell_data["feature_idx"][cid])
            return FeatureRef(layer.layer_attr, layer.store.features[feature_idx].id)
        return None

    def _log_profile_step(self, timer: ProfileTimer) -> None:
        if self.viz_cfg.profiling.enabled:
            log.info(
                "viewport %s: %.1fms%s",
                timer.name,
                timer.duration_s * 1000.0,
                _detail_suffix(timer.detail),
            )


def _dataset_layer_items(dataset: NGIIDataset) -> tuple[tuple[str, LayerStore[Any]], ...]:
    return tuple(
        (spec.python_attr, dataset.store_for_attr(spec.python_attr))
        for spec in dataset.schema.layer_specs
    )


def _geometry_kinds(dataset: NGIIDataset, store: LayerStore[Any]) -> tuple[GeometryKind, ...]:
    if not store.features:
        spec = dataset.schema.spec_for_layer_name(store.layer_name)
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
        "nt1_node": (40, 40, 40),
        "nt2_link": (110, 110, 120),
        "rs1_roadborder": (120, 120, 120),
        "rs2_roadstructure": (180, 180, 180),
        "rs3_subsidiarysection": (120, 200, 120),
        "pw1_pathway": (95, 165, 120),
        "rm1_laneline": (255, 255, 255),
        "rm2_roadmarking": (255, 180, 60),
        "rm3_parkinglot": (160, 160, 220),
        "sf1_barrier": (140, 140, 140),
        "sf2_trafficsign": (200, 80, 80),
        "sf3_trafficlight": (30, 180, 60),
        "sf4_supportpost": (80, 80, 80),
        "sf5_speedbump": (200, 120, 40),
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
) -> _PolygonBuildResult:
    if not features:
        return _PolygonBuildResult(pv.PolyData(), pv.PolyData(), ())
    all_pts: list[NDArray[np.float64]] = []
    face_cells: list[int] = []
    tri_feature_idx: list[int] = []
    fallback_pts: list[NDArray[np.float64]] = []
    fallback_feature_idx: list[int] = []
    outcome_counts = {"direct": 0, "repaired": 0, "fallback_point": 0, "skipped": 0}
    outcome_examples: dict[str, list[str]] = {
        "repaired": [],
        "fallback_point": [],
        "skipped": [],
    }
    skipped_reasons: dict[str, int] = {}
    vert_offset = 0
    for feature_idx in feature_indices:
        feature = features[feature_idx]
        ring = _polygon_feature_ring(feature)
        if ring is None:
            continue
        tri_result = _triangulated_ring_xy(ring)
        if tri_result[0]:
            outcome_counts[tri_result[1]] += 1
        elif layer_name == "B1_SAFETYSIGN":
            fallback_point = _fallback_point_from_ring(ring)
            if fallback_point is not None:
                fallback_pts.append(fallback_point)
                fallback_feature_idx.append(feature_idx)
                outcome_counts["fallback_point"] += 1
                _append_example(outcome_examples["fallback_point"], feature.id)
            else:
                outcome_counts["skipped"] += 1
                _append_example(outcome_examples["skipped"], feature.id)
                skipped_reasons[tri_result[2]] = skipped_reasons.get(tri_result[2], 0) + 1
            continue
        else:
            outcome_counts["skipped"] += 1
            _append_example(outcome_examples["skipped"], feature.id)
            skipped_reasons[tri_result[2]] = skipped_reasons.get(tri_result[2], 0) + 1
            continue

        if tri_result[1] == "repaired":
            _append_example(outcome_examples["repaired"], feature.id)
        for tri_xy in tri_result[0]:
            tri_z = np.array([_nearest_ring_z(ring, x, y) for x, y in tri_xy])
            all_pts.append(np.column_stack([tri_xy, tri_z]))
            face_cells.extend([3, vert_offset, vert_offset + 1, vert_offset + 2])
            vert_offset += 3
            tri_feature_idx.append(feature_idx)
    _log_polygon_render_summary(layer_name, outcome_counts, outcome_examples, skipped_reasons)
    fallback_poly = (
        pv.PolyData(np.asarray(fallback_pts, dtype=np.float64)) if fallback_pts else pv.PolyData()
    )
    if not all_pts:
        return _PolygonBuildResult(
            pv.PolyData(),
            fallback_poly,
            tuple(fallback_feature_idx),
        )
    poly = pv.PolyData(np.vstack(all_pts), faces=np.asarray(face_cells, dtype=np.int64))
    poly.cell_data["feature_idx"] = np.asarray(tri_feature_idx, dtype=np.int32)
    return _PolygonBuildResult(poly, fallback_poly, tuple(fallback_feature_idx))


def _triangulated_ring_xy(
    ring: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], Literal["direct", "repaired", "skipped"], str]:
    polygon = ShapelyPolygon(ring[:, :2])
    reason = shapely.is_valid_reason(polygon)
    triangles = _constrained_triangle_xys(polygon)
    if triangles:
        return triangles, "direct", reason

    repaired_triangles: list[NDArray[np.float64]] = []
    for repaired in _polygonal_repair_candidates(polygon):
        constrained = _constrained_triangle_xys(repaired)
        if constrained:
            repaired_triangles.extend(constrained)
        else:
            repaired_triangles.extend(_unconstrained_triangle_xys(repaired))
    if repaired_triangles:
        return repaired_triangles, "repaired", reason
    return [], "skipped", reason


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


def _fallback_point_from_ring(ring: NDArray[np.float64]) -> NDArray[np.float64] | None:
    finite = ring[np.isfinite(ring).all(axis=1)]
    if len(finite) == 0:
        return None
    if len(finite) > 1 and np.allclose(finite[0, :2], finite[-1, :2]):
        finite = finite[:-1]
    if len(finite) == 0:
        return None
    _, unique_indices = np.unique(finite[:, :2], axis=0, return_index=True)
    unique = finite[np.sort(unique_indices)]
    if len(unique) == 0:
        return None
    return np.asarray(
        (
            float(np.mean(unique[:, 0])),
            float(np.mean(unique[:, 1])),
            float(np.median(unique[:, 2])),
        ),
        dtype=np.float64,
    )


def _append_example(examples: list[str], feature_id: str) -> None:
    if len(examples) < 8:
        examples.append(feature_id)


def _log_polygon_render_summary(
    layer_name: str,
    outcome_counts: dict[str, int],
    outcome_examples: dict[str, list[str]],
    skipped_reasons: dict[str, int],
) -> None:
    noteworthy = (
        outcome_counts["repaired"] or outcome_counts["fallback_point"] or outcome_counts["skipped"]
    )
    if not noteworthy:
        return
    detail = (
        f"direct={outcome_counts['direct']}, repaired={outcome_counts['repaired']}, "
        f"fallback_point={outcome_counts['fallback_point']}, skipped={outcome_counts['skipped']}"
    )
    for outcome in ("repaired", "fallback_point", "skipped"):
        examples = outcome_examples[outcome]
        if examples:
            detail += f"; {outcome}_examples={', '.join(examples)}"
    if skipped_reasons:
        reasons = ", ".join(f"{reason}: {count}" for reason, count in skipped_reasons.items())
        detail += f"; skipped_reasons={reasons}"
    if outcome_counts["skipped"]:
        log.warning("render polygon summary: %s %s", layer_name, detail)
    else:
        log.info("render polygon summary: %s %s", layer_name, detail)


def _display_distance2(point: NDArray[np.float64], x: int, y: int, renderer: Any) -> float:
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    coordinate.SetValue(float(point[0]), float(point[1]), float(point[2]))
    display = coordinate.GetComputedDoubleDisplayValue(renderer)
    dx = float(display[0] - x)
    dy = float(display[1] - y)
    return dx * dx + dy * dy


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
        "nt1_node": 0,
        "c1_trafficlight": 1,
        "c6_postpoint": 1,
        "sf3_trafficlight": 1,
        "sf4_supportpost": 1,
        "c2_kilopost": 2,
        "b1_safetysign": 2,
        "sf2_trafficsign": 2,
    }
    return priority.get(layer_attr, 10)


def _selection_detail(ref: FeatureRef | None, *, extra: str = "") -> str:
    detail = "ref=None" if ref is None else f"ref={ref.layer_attr}:{ref.feature_id}"
    if extra:
        return f"{detail}; {extra}"
    return detail


def _detail_suffix(detail: str) -> str:
    return f" ({detail})" if detail else ""
