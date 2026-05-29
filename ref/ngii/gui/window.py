from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import vtk
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ngii.geometry import (
    Geometry,
    GeometryName,
    geometry_bounds,
    geometry_name,
    is_geometry,
    translated_geometry,
)
from ngii.geometry.rendering import (
    build_line_polydata,
    build_point_polydata,
    build_polygon_polydata,
)
from ngii.v2023.dataset import DataSet
from ngii.v2023.io.loader import load_dataset
from ngii.v2023.layers import (
    A1_Layer,
    A2_Layer,
    A3_Layer,
    A4_Layer,
    A5_Layer,
    B1_Layer,
    B2_Data,
    B2_Layer,
    B3_Layer,
    BaseData,
    BaseLayer,
    C1_Layer,
    C2_Layer,
    C3_Layer,
    C4_Layer,
    C5_Layer,
    C6_Layer,
    UnresolvedLayer,
)
from ngii.v2023.section import Section

logger = logging.getLogger(__name__)

type Color = tuple[float, float, float]

_POLYGON_OFFSET_FACTOR = 1.0
_POLYGON_OFFSET_UNITS = 1.0


@dataclass(slots=True, frozen=True)
class RenderStyle:
    color: Color
    opacity: float = 1.0
    point_radius: float = 0.8
    line_width: float = 1.5


@dataclass(slots=True, frozen=True)
class LayerRenderSpec:
    layer_name: str
    korean_name: str
    accessor: Callable[[Section], BaseLayer[Any] | UnresolvedLayer | None]
    default_visible: bool
    style_for: Callable[[BaseData], RenderStyle]


class _GuiLogEmitter(QObject):
    message = Signal(str, str)


class _GuiLogHandler(logging.Handler):
    def __init__(self, emitter: _GuiLogEmitter) -> None:
        super().__init__(level=logging.INFO)
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        self._emitter.message.emit(record.levelname, self.format(record))


@dataclass(slots=True)
class LoadedMap:
    dataset_root: Path
    coordinate: str
    dataset: DataSet
    render_origin: NDArray[np.float64]
    bounds: tuple[float, float, float, float, float, float] | None


@dataclass(slots=True)
class RenderActors:
    by_layer: dict[str, list[Any]] = field(default_factory=dict)

    def clear(self) -> None:
        self.by_layer.clear()

    def add(self, layer_name: str, actor: Any) -> None:
        self.by_layer.setdefault(layer_name, []).append(actor)

    def set_visible(self, layer_name: str, visible: bool) -> None:
        for actor in self.by_layer.get(layer_name, []):
            actor.SetVisibility(visible)


class _LoadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_root: Path, coordinate: str) -> None:
        super().__init__()
        self._dataset_root = dataset_root
        self._coordinate = coordinate

    @Slot()
    def run(self) -> None:
        try:
            dataset = load_dataset(self._dataset_root, self._coordinate)
        except Exception as exc:
            logger.exception("Failed to load dataset")
            self.failed.emit(str(exc))
            return
        self.finished.emit(dataset)


class NgiiViewerWindow(QMainWindow):
    def __init__(self, coordinate: str) -> None:
        super().__init__()
        self._coordinate = coordinate
        self._plotter: Any = QtInteractor(self)
        self._actors = RenderActors()
        self._loaded_map: LoadedMap | None = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._log_text = QTextEdit()
        self._log_dock: QDockWidget | None = None
        self._show_log_action: QAction | None = None
        self._log_emitter = _GuiLogEmitter()
        self._log_handler = _GuiLogHandler(self._log_emitter)
        self._log_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        self._log_emitter.message.connect(self._append_log)
        logging.getLogger("ngii").addHandler(self._log_handler)
        logging.getLogger("ngii").setLevel(logging.INFO)

        self._status_label = QLabel("Open a 2023 NGII dataset root.")
        self._status_label.setWordWrap(True)
        self._layer_container = QWidget()
        self._layer_layout = QVBoxLayout(self._layer_container)
        self._layer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._setup_window()

    def closeEvent(self, event: QCloseEvent) -> None:
        logging.getLogger("ngii").removeHandler(self._log_handler)
        super().closeEvent(event)

    def _setup_window(self) -> None:
        self.setWindowTitle("NGII 2023 Viewer")
        self.resize(1400, 900)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._plotter, stretch=1)
        root_layout.addWidget(self._build_side_panel())
        self.setCentralWidget(root)

        self._setup_menus()
        self._setup_log_dock()
        self._plotter.set_background("#202124")

    def _setup_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Dataset Folder", self)
        open_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_action)

        view_menu = self.menuBar().addMenu("View")
        top_action = QAction("Top View", self)
        top_action.triggered.connect(self._top_view)
        reset_action = QAction("Reset 3D View", self)
        reset_action.triggered.connect(self._reset_3d_view)
        show_log_action = QAction("Show Log", self)
        show_log_action.setCheckable(True)
        show_log_action.setChecked(True)
        show_log_action.toggled.connect(self._set_log_visible)
        view_menu.addAction(top_action)
        view_menu.addAction(reset_action)
        view_menu.addSeparator()
        view_menu.addAction(show_log_action)
        self._show_log_action = show_log_action

    def _setup_log_dock(self) -> None:
        self._log_text.setReadOnly(True)
        dock = QDockWidget("Log", self)
        dock.setWidget(self._log_text)
        dock.visibilityChanged.connect(self._sync_log_action)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._log_dock = dock

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._layer_container)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(scroll, stretch=1)
        return panel

    @Slot(bool)
    def _set_log_visible(self, visible: bool) -> None:
        if self._log_dock is not None:
            self._log_dock.setVisible(visible)

    @Slot(bool)
    def _sync_log_action(self, visible: bool) -> None:
        if self._show_log_action is not None:
            self._show_log_action.setChecked(visible)

    @Slot()
    def _open_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open NGII dataset root",
            str(Path.cwd()),
        )
        if not directory:
            return
        self.load_dataset(Path(directory))

    def load_dataset(self, dataset_root: Path) -> None:
        if self._load_thread is not None:
            self._status_label.setText("A dataset is already loading.")
            return

        self._status_label.setText(f"Loading {dataset_root.name}...")
        self._log_text.clear()
        self._append_log("INFO", f"Loading {dataset_root}")
        self._set_layer_controls_enabled(False)

        thread = QThread(self)
        worker = _LoadWorker(dataset_root, self._coordinate)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_load_finished)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_load_thread_finished)

        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @Slot(object)
    def _on_load_finished(self, result: object) -> None:
        dataset = cast(DataSet, result)
        loaded_map = _build_loaded_map(dataset)
        self._loaded_map = loaded_map
        self._plotter.clear()
        self._actors.clear()
        self._plotter.set_background("#202124")
        self._build_scene(loaded_map)
        self._populate_layer_controls(loaded_map)
        self._reset_3d_view()
        self._status_label.setText(_status_text(loaded_map))

    @Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self._status_label.setText(f"Failed to load dataset: {message}")
        self._append_log("ERROR", message)
        self._set_layer_controls_enabled(True)

    @Slot()
    def _on_load_thread_finished(self) -> None:
        self._load_thread = None
        self._load_worker = None
        self._set_layer_controls_enabled(True)

    def _build_scene(self, loaded_map: LoadedMap) -> None:
        for spec in LAYER_RENDER_SPECS:
            buckets = _collect_render_buckets(
                loaded_map.dataset,
                spec,
                loaded_map.render_origin,
            )
            for (bucket_geometry_name, style), geometries in buckets.items():
                actor = _build_actor(bucket_geometry_name, geometries, style)
                if actor is None:
                    continue
                actor.SetVisibility(spec.default_visible)
                self._plotter.add_actor(actor)
                self._actors.add(spec.layer_name, actor)
        self._plotter.render()

    def _populate_layer_controls(self, loaded_map: LoadedMap) -> None:
        _clear_layout(self._layer_layout)
        self._checkboxes.clear()

        for spec in LAYER_RENDER_SPECS:
            count = _feature_count(loaded_map.dataset, spec)
            has_actors = bool(self._actors.by_layer.get(spec.layer_name))
            checkbox = QCheckBox(f"{spec.layer_name} ({count})")
            checkbox.setChecked(spec.default_visible if has_actors else False)
            checkbox.setEnabled(has_actors)
            checkbox.toggled.connect(
                lambda checked, layer_name=spec.layer_name: self._toggle_layer(layer_name, checked)
            )
            self._layer_layout.addWidget(checkbox)
            self._checkboxes[spec.layer_name] = checkbox

        self._layer_layout.addStretch(1)

    def _append_log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_text.append(f"{timestamp} [{level}] {message}")

    @Slot(str, bool)
    def _toggle_layer(self, layer_name: str, visible: bool) -> None:
        self._actors.set_visible(layer_name, visible)
        self._plotter.render()

    @Slot()
    def _top_view(self) -> None:
        if self._loaded_map is None or self._loaded_map.bounds is None:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self._loaded_map.bounds
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        diagonal = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        camera = self._plotter.renderer.GetActiveCamera()
        camera.SetPosition(xmid, ymid, zmax + diagonal * 2.0)
        camera.SetFocalPoint(xmid, ymid, (zmin + zmax) / 2.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetParallelProjection(True)
        camera.SetParallelScale(diagonal * 0.55)
        self._plotter.render()

    @Slot()
    def _reset_3d_view(self) -> None:
        if self._loaded_map is None:
            return
        camera = self._plotter.renderer.GetActiveCamera()
        camera.SetParallelProjection(False)
        self._plotter.view_isometric()
        self._plotter.reset_camera()
        self._plotter.render()

    def _set_layer_controls_enabled(self, enabled: bool) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.setEnabled(enabled and _layer_has_actors(self._actors, checkbox))


def _layer_has_actors(actors: RenderActors, checkbox: QCheckBox) -> bool:
    layer_name = checkbox.text().split(" ", maxsplit=1)[0]
    return bool(actors.by_layer.get(layer_name))


def _build_loaded_map(dataset: DataSet) -> LoadedMap:
    bounds = _dataset_bounds(dataset)
    if bounds is None:
        origin = np.zeros(3, dtype=np.float64)
        render_bounds = None
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        origin = np.asarray(
            [
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                (zmin + zmax) / 2.0,
            ],
            dtype=np.float64,
        )
        render_bounds = (
            float(bounds[0] - origin[0]),
            float(bounds[1] - origin[0]),
            float(bounds[2] - origin[1]),
            float(bounds[3] - origin[1]),
            float(bounds[4] - origin[2]),
            float(bounds[5] - origin[2]),
        )
    return LoadedMap(
        dataset_root=dataset.path,
        coordinate=dataset.coordinate,
        dataset=dataset,
        render_origin=origin,
        bounds=render_bounds,
    )


def _dataset_bounds(
    dataset: DataSet,
) -> tuple[float, float, float, float, float, float] | None:
    return geometry_bounds(_iter_geometries(dataset))


def _iter_geometries(dataset: DataSet) -> Iterator[Geometry]:
    for spec in LAYER_RENDER_SPECS:
        for feature in _iter_features(dataset, spec):
            geometry = getattr(feature, "geometry", None)
            if is_geometry(geometry):
                yield geometry


def _status_text(loaded_map: LoadedMap) -> str:
    sections = len(loaded_map.dataset.sections)
    features = sum(_feature_count(loaded_map.dataset, spec) for spec in LAYER_RENDER_SPECS)
    return (
        f"{loaded_map.dataset_root.name}\n"
        f"{loaded_map.coordinate}\n"
        f"Sections: {sections}\n"
        f"Features: {features}"
    )


def _feature_count(dataset: DataSet, spec: LayerRenderSpec) -> int:
    return sum(1 for _ in _iter_features(dataset, spec))


def _iter_features(dataset: DataSet, spec: LayerRenderSpec) -> Iterator[BaseData]:
    for section in dataset.sections:
        layer = spec.accessor(section)
        if layer is None or isinstance(layer, UnresolvedLayer):
            continue
        yield from layer.data.values()


def _collect_render_buckets(
    dataset: DataSet,
    spec: LayerRenderSpec,
    render_origin: NDArray[np.float64],
) -> dict[tuple[GeometryName, RenderStyle], list[Geometry]]:
    buckets: dict[tuple[GeometryName, RenderStyle], list[Geometry]] = {}
    for feature in _iter_features(dataset, spec):
        geometry = getattr(feature, "geometry", None)
        if not is_geometry(geometry):
            continue
        name = geometry_name(geometry)
        if name is None:
            continue
        style = spec.style_for(feature)
        buckets.setdefault((name, style), []).append(translated_geometry(geometry, render_origin))
    return buckets


def _build_actor(
    geometry_name: GeometryName,
    geometries: list[Geometry],
    style: RenderStyle,
) -> Any | None:
    if geometry_name == "point":
        return _build_point_actor(geometries, style)
    if geometry_name == "line":
        return _build_line_actor(geometries, style)
    return _build_polygon_actor(geometries, style)


def _build_point_actor(geometries: list[Geometry], style: RenderStyle) -> Any | None:
    polydata = build_point_polydata(geometries)
    if polydata is None:
        return None

    circle = vtk.vtkRegularPolygonSource()
    circle.SetNumberOfSides(24)
    circle.SetRadius(style.point_radius)
    circle.GeneratePolygonOff()

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(polydata)
    mapper.SetSourceConnection(circle.GetOutputPort())
    mapper.ScalingOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*style.color)
    actor.GetProperty().SetOpacity(style.opacity)
    return actor


def _build_line_actor(geometries: list[Geometry], style: RenderStyle) -> Any | None:
    polydata = build_line_polydata(geometries)
    if polydata is None:
        return None

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*style.color)
    actor.GetProperty().SetOpacity(style.opacity)
    actor.GetProperty().SetLineWidth(style.line_width)
    return actor


def _build_polygon_actor(geometries: list[Geometry], style: RenderStyle) -> Any | None:
    polydata = build_polygon_polydata(geometries)
    if polydata is None:
        return None

    vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
        _POLYGON_OFFSET_FACTOR,
        _POLYGON_OFFSET_UNITS,
    )

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*style.color)
    actor.GetProperty().SetOpacity(style.opacity)
    actor.GetProperty().LightingOff()
    return actor


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _constant_style(color: Color, opacity: float = 1.0) -> Callable[[BaseData], RenderStyle]:
    style = RenderStyle(color=color, opacity=opacity)
    return lambda _: style


def _b2_style(feature: BaseData) -> RenderStyle:
    if not isinstance(feature, B2_Data) or feature.b2_type is None:
        return RenderStyle(color=(0.8, 0.8, 0.8))
    code = feature.b2_type.value
    if code.startswith("1"):
        return RenderStyle(color=(1.0, 0.82, 0.15), line_width=1.8)
    if code.startswith("2"):
        return RenderStyle(color=(0.94, 0.94, 0.9), line_width=1.5)
    if code.startswith("3"):
        return RenderStyle(color=(0.25, 0.55, 1.0), line_width=1.7)
    return RenderStyle(color=(0.75, 0.75, 0.75), line_width=1.4)


LAYER_RENDER_SPECS: tuple[LayerRenderSpec, ...] = (
    LayerRenderSpec(
        A3_Layer.layer_name,
        A3_Layer.korean_name,
        lambda section: section.a3,
        True,
        _constant_style((0.26, 0.39, 0.48), 0.2),
    ),
    LayerRenderSpec(
        A4_Layer.layer_name,
        A4_Layer.korean_name,
        lambda section: section.a4,
        True,
        _constant_style((0.18, 0.48, 0.30), 0.22),
    ),
    LayerRenderSpec(
        A5_Layer.layer_name,
        A5_Layer.korean_name,
        lambda section: section.a5,
        True,
        _constant_style((0.48, 0.40, 0.20), 0.25),
    ),
    LayerRenderSpec(
        A2_Layer.layer_name,
        A2_Layer.korean_name,
        lambda section: section.a2,
        True,
        _constant_style((0.24, 0.58, 0.92)),
    ),
    LayerRenderSpec(
        B2_Layer.layer_name,
        B2_Layer.korean_name,
        lambda section: section.b2,
        True,
        _b2_style,
    ),
    LayerRenderSpec(
        C3_Layer.layer_name,
        C3_Layer.korean_name,
        lambda section: section.c3,
        True,
        _constant_style((0.92, 0.42, 0.26), 1.0),
    ),
    LayerRenderSpec(
        B3_Layer.layer_name,
        B3_Layer.korean_name,
        lambda section: section.b3,
        True,
        _constant_style((0.92, 0.84, 0.30), 0.45),
    ),
    LayerRenderSpec(
        A1_Layer.layer_name,
        A1_Layer.korean_name,
        lambda section: section.a1,
        True,
        _constant_style((1.0, 0.72, 0.20)),
    ),
    LayerRenderSpec(
        B1_Layer.layer_name,
        B1_Layer.korean_name,
        lambda section: section.b1,
        True,
        _constant_style((0.91, 0.24, 0.36), 0.9),
    ),
    LayerRenderSpec(
        C1_Layer.layer_name,
        C1_Layer.korean_name,
        lambda section: section.c1,
        True,
        _constant_style((0.3, 1.0, 0.72)),
    ),
    LayerRenderSpec(
        C2_Layer.layer_name,
        C2_Layer.korean_name,
        lambda section: section.c2,
        True,
        _constant_style((0.85, 0.65, 1.0)),
    ),
    LayerRenderSpec(
        C4_Layer.layer_name,
        C4_Layer.korean_name,
        lambda section: section.c4,
        True,
        _constant_style((1.0, 0.42, 0.16), 0.55),
    ),
    LayerRenderSpec(
        C5_Layer.layer_name,
        C5_Layer.korean_name,
        lambda section: section.c5,
        True,
        _constant_style((1.0, 0.18, 0.18)),
    ),
    LayerRenderSpec(
        C6_Layer.layer_name,
        C6_Layer.korean_name,
        lambda section: section.c6,
        True,
        _constant_style((0.58, 0.88, 1.0)),
    ),
)
