"""Qt inspector window for dataset-native NGII visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import shapely
from PySide6.QtCore import QObject, QRect, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStyle,
    QStyleOptionButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ngii2xodr.config import RuntimeConfig, validate_runtime_config, with_viz_layer_visibility
from ngii2xodr.gui.config_dialog import ConfigDialog
from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data import AmbiguousFeatureIDError, SanityReport
from ngii2xodr.ngii.data.dataset import LayerStore
from ngii2xodr.ngii.data.features import (
    LineFeature,
    NGIIFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
    is_float_value,
    is_integer_value,
)
from ngii2xodr.ngii.data.schema import FieldRule, FieldType, ReferenceRule
from ngii2xodr.ngii.segmentation import SelectedField
from ngii2xodr.ngii.viz import HdMapViz
from ngii2xodr.profile import ProfileTimer

log = logging.getLogger(__name__)

_SELECT_PLACEHOLDER = "- open a folder to begin -"
_SELECT_PROMPT = "- shift-click a feature or select an item -"
_SECTION_BG = QColor(230, 230, 235)
_TARGET_REF_ROLE = int(Qt.ItemDataRole.UserRole)
_EDIT_ROLE = _TARGET_REF_ROLE + 1
_LOG_LINE_LIMIT = 2_000


@dataclass(slots=True, frozen=True)
class _FieldEdit:
    ref: FeatureRef
    column_name: str
    attr_name: str
    field_type: FieldType
    required: bool
    max_length: int | None
    code_list: dict[str, str] | None = None
    is_reference: bool = False
    is_feature_id: bool = False


@dataclass(slots=True, frozen=True)
class _DetailRow:
    field: str
    value: str
    is_section: bool = False
    target_ref: FeatureRef | None = None
    edit: _FieldEdit | None = None
    raw_value: str = ""


class _LayerVisibilityHeader(QHeaderView):
    master_clicked = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._check_state = Qt.CheckState.Unchecked
        self._checkable = False
        self.setSectionsClickable(True)

    def set_master_state(self, state: Qt.CheckState, *, checkable: bool) -> None:
        self._check_state = state
        self._checkable = checkable
        self.viewport().update()

    @override
    def paintSection(self, painter: Any, rect: QRect, logical_index: int) -> None:
        super().paintSection(painter, rect, logical_index)
        if logical_index != 0:
            return
        option = QStyleOptionButton()
        option.rect = self._checkbox_rect(rect)
        option.state = QStyle.StateFlag.State_Active
        if self._checkable:
            option.state |= QStyle.StateFlag.State_Enabled
        if self._check_state == Qt.CheckState.Checked:
            option.state |= QStyle.StateFlag.State_On
        elif self._check_state == Qt.CheckState.PartiallyChecked:
            option.state |= QStyle.StateFlag.State_NoChange
        else:
            option.state |= QStyle.StateFlag.State_Off
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorCheckBox, option, painter, self
        )

    @override
    def mouseReleaseEvent(self, event: Any) -> None:
        if self._checkable and self.logicalIndexAt(event.position().toPoint()) == 0:
            self.master_clicked.emit()
            return
        super().mouseReleaseEvent(event)

    def _checkbox_rect(self, section_rect: QRect) -> QRect:
        indicator = self.style().pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth)
        x = section_rect.x() + (section_rect.width() - indicator) // 2
        y = section_rect.y() + (section_rect.height() - indicator) // 2
        return QRect(x, y, indicator, indicator)


class _LogEmitter(QObject):
    message = Signal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.emitter = _LogEmitter()

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except (KeyError, ValueError, TypeError):
            message = record.getMessage()
        self.emitter.message.emit(message)


class HdMapWindow(QMainWindow):
    """Main window hosting the 3D scene and canonical feature inspector."""

    def __init__(self, runtime_cfg: RuntimeConfig) -> None:
        super().__init__()
        self.setWindowTitle("ngii2xodr - (no folder)")
        self.resize(1500, 950)

        self._runtime_cfg = runtime_cfg
        self._ngii_dir: Path | None = None
        self.viz: HdMapViz | None = None

        self._layer_items: dict[str, QTreeWidgetItem] = {}
        self._item_by_ref: dict[FeatureRef, QTreeWidgetItem] = {}
        self._selected_ref: FeatureRef | None = None
        self._segmentation_level = 0
        self._segmentation_button_group: QButtonGroup | None = None
        self._memory_dirty = False
        self._populating_selected_panel = False
        self._log_handler: _QtLogHandler | None = None

        self.qt_plotter = QtInteractor(self)
        self.setCentralWidget(self.qt_plotter)
        self.qt_plotter.enable_parallel_projection()
        self.qt_plotter.view_xy()

        self._build_menu()
        self._build_dock()
        self._build_log_dock()
        self._set_dock_enabled(False)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        open_act = QAction("&Open NGII folder...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_folder)
        file_menu.addAction(open_act)
        self._recompute_act = QAction("&Recompute Current Map", self)
        self._recompute_act.setShortcut(QKeySequence("Ctrl+R"))
        self._recompute_act.setEnabled(False)
        self._recompute_act.triggered.connect(self._recompute_current_map)
        file_menu.addAction(self._recompute_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = bar.addMenu("&Edit")
        config_act = QAction("&Configuration...", self)
        config_act.setShortcut(QKeySequence("Ctrl+,"))
        config_act.triggered.connect(self._open_config_dialog)
        edit_menu.addAction(config_act)

    def _build_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self._tabs = QTabWidget()
        self._layers_tab = self._build_layers_tab()
        self._items_tab = self._build_items_tab()
        self._selected_tab = self._build_selected_tab()
        self._tabs.addTab(self._layers_tab, "Layers")
        self._tabs.addTab(self._items_tab, "Items")
        self._tabs.addTab(self._selected_tab, "Selected")
        dock.setWidget(self._tabs)
        dock.setMinimumWidth(390)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_log_dock(self) -> None:
        dock = QDockWidget("Logs", self)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_logs)
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self._copy_logs)
        controls.addWidget(clear_button)
        controls.addWidget(copy_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(_LOG_LINE_LIMIT)
        layout.addWidget(self._log_text, stretch=1)
        dock.setWidget(body)
        dock.setMinimumHeight(180)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        handler = _QtLogHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S")
        )
        handler.emitter.message.connect(self._append_log_message)
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _build_layers_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self._layers_group = QGroupBox("Renderable layers")
        self._layers_layout = QVBoxLayout(self._layers_group)
        self._layer_tree = QTreeWidget()
        self._layer_header = _LayerVisibilityHeader(self._layer_tree)
        self._layer_header.master_clicked.connect(self._on_layer_master_clicked)
        self._layer_tree.setHeader(self._layer_header)
        self._layer_tree.setHeaderLabels(("", "Layer", "Geometry", "Count"))
        self._layer_tree.setAlternatingRowColors(True)
        self._layer_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._layer_tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._layer_tree.itemChanged.connect(self._on_layer_item_changed)
        self._layer_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._layer_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._layer_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._layer_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._layers_layout.addWidget(self._layer_tree)
        layout.addWidget(self._layers_group)

        self._segmentation_group = QGroupBox("Segmentation coloring")
        self._segmentation_layout = QVBoxLayout(self._segmentation_group)
        self._segmentation_layout.addWidget(QLabel(_SELECT_PLACEHOLDER))
        layout.addWidget(self._segmentation_group)
        layout.addStretch(1)
        return tab

    def _build_items_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        self._item_search = QLineEdit()
        self._item_search.setPlaceholderText("Search layer or ID")
        self._item_search.textChanged.connect(self._filter_items)
        layout.addWidget(self._item_search)

        self._item_tree = QTreeWidget()
        self._item_tree.setHeaderLabels(("Layer / ID", "Summary"))
        self._item_tree.setAlternatingRowColors(True)
        self._item_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._item_tree.itemSelectionChanged.connect(self._on_item_selection_changed)
        self._item_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._item_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._item_tree, stretch=1)
        return tab

    def _build_selected_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        self._select_header = QLabel(_SELECT_PLACEHOLDER)
        self._select_header.setStyleSheet("font-weight: bold;")
        self._select_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._select_header)

        self._edit_state_label = QLabel("")
        self._edit_state_label.setStyleSheet("color: #8a5a00;")
        self._edit_state_label.setVisible(False)
        layout.addWidget(self._edit_state_label)

        self._select_table = QTableWidget(0, 2)
        self._select_table.setHorizontalHeaderLabels(("Field", "Value"))
        self._select_table.verticalHeader().setVisible(False)
        self._select_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._select_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._select_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._select_table.setShowGrid(True)
        self._select_table.setAlternatingRowColors(True)
        self._select_table.cellDoubleClicked.connect(self._on_detail_cell_activated)
        self._select_table.itemActivated.connect(self._on_detail_item_activated)
        self._select_table.itemChanged.connect(self._on_selected_item_changed)
        header = self._select_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._select_table, stretch=1)
        return tab

    def _set_dock_enabled(self, on: bool) -> None:
        self._layers_tab.setEnabled(on)
        self._items_tab.setEnabled(on)
        self._selected_tab.setEnabled(on)
        self._recompute_act.setEnabled(on)

    def _append_log_message(self, message: str) -> None:
        self._log_text.appendPlainText(message)

    def _clear_logs(self) -> None:
        self._log_text.clear()

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self._log_text.toPlainText())

    def _set_memory_dirty(self, dirty: bool) -> None:
        self._memory_dirty = dirty
        self._edit_state_label.setVisible(dirty)
        self._edit_state_label.setText(
            "Edited in memory - recompute pending (Ctrl+R)" if dirty else ""
        )

    def _open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open NGII folder")
        if d:
            self.load_folder(Path(d))

    def _open_config_dialog(self) -> None:
        dialog = ConfigDialog(self._runtime_cfg, self._apply_runtime_config, self)
        dialog.exec()

    def _recompute_current_map(self) -> None:
        viz = self.viz
        if viz is None:
            return
        selected_ref = self._selected_ref
        viz.recompute_from_memory(self._runtime_cfg.segmentation, self._runtime_cfg.viz)
        self._rebuild_layers_tab(viz)
        self._rebuild_items_tab(viz)
        self._set_segmentation_level(
            min(self._segmentation_level, len(viz.segmentation.stage_results))
        )
        if selected_ref is not None and _feature_exists(viz, selected_ref):
            self.select_feature(selected_ref, update_viz=False)
        else:
            self._selected_ref = None
            self._reset_select_panel(_SELECT_PROMPT)
        self._set_memory_dirty(False)
        self.statusBar().showMessage("Recomputed current in-memory map")
        log.info("recomputed current in-memory map")

    def load_folder(self, ngii_dir: Path) -> None:
        try:
            viz = self._build_viz(ngii_dir, self._runtime_cfg)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            log.warning("cannot open %s: %s", ngii_dir, e)
            QMessageBox.warning(
                self,
                "Cannot open folder",
                f"Not a valid NGII folder:\n{ngii_dir}\n\n{e}",
            )
            if self.viz is None:
                self.setWindowTitle("ngii2xodr - (no folder)")
                self._set_dock_enabled(False)
                self._reset_select_panel(_SELECT_PLACEHOLDER)
            self.qt_plotter.render()
            return

        self._install_viz(viz, ngii_dir, status_prefix="Loaded")

    def _apply_runtime_config(self, runtime_cfg: RuntimeConfig) -> bool:
        try:
            validate_runtime_config(runtime_cfg)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return False

        if self._ngii_dir is None:
            self._runtime_cfg = runtime_cfg
            self.statusBar().showMessage("Configuration updated")
            return True

        try:
            viz = self._build_viz(self._ngii_dir, runtime_cfg)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            log.warning("cannot reload %s after configuration change: %s", self._ngii_dir, e)
            QMessageBox.warning(
                self,
                "Cannot apply configuration",
                f"Could not reload the current NGII folder:\n{self._ngii_dir}\n\n{e}",
            )
            return False

        self._runtime_cfg = runtime_cfg
        self._install_viz(viz, self._ngii_dir, status_prefix="Reloaded")
        return True

    def _build_viz(self, ngii_dir: Path, runtime_cfg: RuntimeConfig) -> HdMapViz:
        return HdMapViz(
            ngii_dir,
            coordinate=runtime_cfg.coordinate,
            ngii_cfg=runtime_cfg.ngii,
            seg_cfg=runtime_cfg.segmentation,
            viz_cfg=runtime_cfg.viz,
            plotter=self.qt_plotter,
            on_select=self._on_viz_select,
        )

    def _install_viz(self, viz: HdMapViz, ngii_dir: Path, *, status_prefix: str) -> None:
        if self.viz is not None:
            self.viz.detach()
        self._selected_ref = None
        self._reset_select_panel(_SELECT_PLACEHOLDER)
        viz.attach()
        self.viz = viz
        self._ngii_dir = ngii_dir
        self._rebuild_layers_tab(viz)
        self._rebuild_items_tab(viz)
        self._set_segmentation_level(
            min(self._segmentation_level, len(viz.segmentation.stage_results))
        )

        self.qt_plotter.view_xy()
        self.qt_plotter.reset_camera()
        self.setWindowTitle(f"ngii2xodr - {ngii_dir.name}")
        self._set_dock_enabled(True)
        self._set_memory_dirty(False)
        self._reset_select_panel(_SELECT_PROMPT)
        feature_count = sum(len(store) for store in viz.dataset.layer_stores)
        self.statusBar().showMessage(f"{status_prefix} {ngii_dir.name}: {feature_count} features")
        log.info(
            "%s %s: %d warnings, %d repairs",
            status_prefix.lower(),
            ngii_dir,
            viz.sanity.warning_count,
            viz.sanity.action_count,
        )

    def _rebuild_layers_tab(self, viz: HdMapViz) -> None:
        with QSignalBlocker(self._layer_tree):
            self._layer_items.clear()
            self._layer_tree.clear()
            for attr, store in _layer_store_items(viz):
                layer = viz.registry.layers.get(attr)
                count = len(store)
                geometry = (
                    "empty/missing"
                    if count == 0
                    else layer.geometry_label
                    if layer is not None
                    else _store_geometry_label(store)
                )
                item = QTreeWidgetItem(("", store.layer_name, geometry, str(count)))
                item.setData(0, Qt.ItemDataRole.UserRole, attr)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if layer is not None and count > 0 and layer.config.visible
                    else Qt.CheckState.Unchecked,
                )
                if layer is None or count == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    for column in range(4):
                        item.setForeground(column, QBrush(QColor(145, 145, 150)))
                self._layer_tree.addTopLevelItem(item)
                self._layer_items[attr] = item
            _fit_tree_to_rows(self._layer_tree, len(viz.dataset.schema.layer_specs))
        self._sync_layer_master_checkbox()

        _clear_layout(self._segmentation_layout)
        self._segmentation_button_group = QButtonGroup(self._segmentation_group)
        for level, label in enumerate(viz.segmentation.level_labels):
            rb = QRadioButton(f"{level}  {label}")
            rb.setChecked(level == self._segmentation_level)
            self._segmentation_button_group.addButton(rb, level)
            self._segmentation_layout.addWidget(rb)
        self._segmentation_button_group.idToggled.connect(self._on_segmentation_level_changed)

    def _rebuild_items_tab(self, viz: HdMapViz) -> None:
        with QSignalBlocker(self._item_tree):
            self._item_by_ref.clear()
            self._item_tree.clear()
            for attr, store in _layer_store_items(viz):
                parent = QTreeWidgetItem((store.layer_name, f"{len(store)} items"))
                parent.setData(0, Qt.ItemDataRole.UserRole, None)
                self._item_tree.addTopLevelItem(parent)
                for feature in store.features:
                    ref = FeatureRef(attr, feature.id)
                    child = QTreeWidgetItem((feature.id, _feature_summary(feature)))
                    child.setData(0, Qt.ItemDataRole.UserRole, ref)
                    parent.addChild(child)
                    self._item_by_ref[ref] = child
                parent.setExpanded(False)
            self._item_tree.resizeColumnToContents(1)
        self._filter_items(self._item_search.text())

    def _on_layer_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self.viz is None:
            return
        attr = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(attr, str):
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.viz.set_layer_visible(attr, visible)
            self._runtime_cfg = with_viz_layer_visibility(self._runtime_cfg, attr, visible)
            self._sync_layer_master_checkbox()

    def _on_layer_master_clicked(self) -> None:
        if self.viz is None:
            return
        items = self._toggleable_layer_items()
        visible = not items or not all(
            item.checkState(0) == Qt.CheckState.Checked for item in items
        )
        with QSignalBlocker(self._layer_tree):
            for item in items:
                item.setCheckState(0, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
                attr = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(attr, str):
                    self.viz.set_layer_visible(attr, visible)
                    self._runtime_cfg = with_viz_layer_visibility(self._runtime_cfg, attr, visible)
        self._sync_layer_master_checkbox()

    def _sync_layer_master_checkbox(self) -> None:
        items = self._toggleable_layer_items()
        if not items:
            self._layer_header.set_master_state(Qt.CheckState.Unchecked, checkable=False)
        elif all(item.checkState(0) == Qt.CheckState.Checked for item in items):
            self._layer_header.set_master_state(Qt.CheckState.Checked, checkable=True)
        elif all(item.checkState(0) == Qt.CheckState.Unchecked for item in items):
            self._layer_header.set_master_state(Qt.CheckState.Unchecked, checkable=True)
        else:
            self._layer_header.set_master_state(Qt.CheckState.PartiallyChecked, checkable=True)

    def _toggleable_layer_items(self) -> tuple[QTreeWidgetItem, ...]:
        return tuple(
            item
            for item in self._layer_items.values()
            if bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        )

    def _on_segmentation_level_changed(self, level: int, checked: bool) -> None:
        if checked:
            self._set_segmentation_level(level)

    def _set_segmentation_level(self, level: int) -> None:
        self._segmentation_level = level
        if self.viz is not None:
            self.viz.set_segmentation_level(level)

    def _filter_items(self, text: str) -> None:
        query = text.casefold().strip()
        for i in range(self._item_tree.topLevelItemCount()):
            parent = self._item_tree.topLevelItem(i)
            if parent is None:
                continue
            parent_text = parent.text(0).casefold() + " " + parent.text(1).casefold()
            any_visible = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                haystack = f"{parent_text} {child.text(0).casefold()} {child.text(1).casefold()}"
                visible = query in haystack
                child.setHidden(not visible)
                any_visible = any_visible or visible
            parent.setHidden(not any_visible)
            parent.setExpanded(bool(query) and any_visible)

    def _on_item_selection_changed(self) -> None:
        item = self._item_tree.currentItem()
        ref = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(ref, FeatureRef):
            self.select_feature(ref, update_viz=True)
            self._tabs.setCurrentWidget(self._selected_tab)

    def _on_viz_select(self, ref: FeatureRef) -> None:
        self.select_feature(ref, update_viz=False)
        self._tabs.setCurrentWidget(self._selected_tab)

    def select_feature(self, ref: FeatureRef, *, update_viz: bool) -> None:
        viz = self.viz
        if viz is None:
            return
        self._selected_ref = ref
        with viz.viewport_profile.timed(
            "selection_update_viz", _ref_detail(ref, update_viz)
        ) as timer:
            if update_viz:
                viz.select_feature(ref, emit=False)
                viz.focus_feature(ref)
        _log_gui_profile(viz, timer)
        with viz.viewport_profile.timed("selection_sync_items", _ref_detail(ref, False)) as timer:
            self._sync_item_selection(ref)
        _log_gui_profile(viz, timer)
        with viz.viewport_profile.timed(
            "selection_selected_table", _ref_detail(ref, False)
        ) as timer:
            self._populate_selected_panel(viz, ref)
        _log_gui_profile(viz, timer)
        log.info("selected %s:%s", ref.layer_attr, ref.feature_id)

    def _sync_item_selection(self, ref: FeatureRef) -> None:
        item = self._item_by_ref.get(ref)
        if item is None:
            return
        with QSignalBlocker(self._item_tree):
            parent = item.parent()
            if parent is not None:
                parent.setExpanded(True)
            self._item_tree.setCurrentItem(item)
            self._item_tree.scrollToItem(item)

    def _reset_select_panel(self, header_text: str) -> None:
        self._select_header.setText(header_text)
        self._select_table.clearSpans()
        self._select_table.clearContents()
        self._select_table.setRowCount(0)

    def _populate_selected_panel(self, viz: HdMapViz, ref: FeatureRef) -> None:
        feature = viz.dataset.store_for_attr(ref.layer_attr)[ref.feature_id]
        self._select_header.setText(f"{feature.layer_name}    {feature.id}")
        rows = _detail_rows(viz, ref, feature)
        self._populating_selected_panel = True
        try:
            self._select_table.clearSpans()
            self._select_table.setRowCount(0)
            self._select_table.clearContents()
            self._select_table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                if row.is_section:
                    item = QTableWidgetItem(row.field)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QBrush(_SECTION_BG))
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._select_table.setItem(row_idx, 0, item)
                    self._select_table.setSpan(row_idx, 0, 1, 2)
                else:
                    self._populate_detail_row(row_idx, row)
        finally:
            self._populating_selected_panel = False

    def _populate_detail_row(self, row_idx: int, row: _DetailRow) -> None:
        field_item = QTableWidgetItem(row.field)
        field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        value_item = QTableWidgetItem(row.value)
        if row.target_ref is not None:
            field_item.setData(_TARGET_REF_ROLE, row.target_ref)
            field_item.setForeground(QBrush(QColor(30, 85, 180)))
            if row.edit is None:
                value_item.setData(_TARGET_REF_ROLE, row.target_ref)
                value_item.setForeground(QBrush(QColor(30, 85, 180)))
        if row.edit is None:
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        else:
            value_item.setData(_EDIT_ROLE, row.edit)
            value_item.setToolTip("Edit in memory. Press Ctrl+R to recompute derived results.")
        self._select_table.setItem(row_idx, 0, field_item)
        self._select_table.setItem(row_idx, 1, value_item)
        if row.edit is not None and row.edit.code_list is not None:
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._install_code_list_editor(row_idx, row.edit, row.raw_value)

    def _install_code_list_editor(
        self,
        row_idx: int,
        edit: _FieldEdit,
        raw_value: str,
    ) -> None:
        combo = QComboBox()
        combo.setToolTip("Edit in memory. Press Ctrl+R to recompute derived results.")
        if not edit.required:
            combo.addItem("-", "")
        elif not raw_value:
            combo.addItem("- (invalid)", "")
        code_list = edit.code_list
        if code_list is None:
            return
        for code, label in sorted(code_list.items()):
            combo.addItem(f"{code} ({label})", code)
        if raw_value and raw_value not in code_list:
            combo.addItem(f"{raw_value} (invalid)", raw_value)
        for index in range(combo.count()):
            if combo.itemData(index) == raw_value:
                combo.setCurrentIndex(index)
                break
        combo.currentIndexChanged.connect(
            lambda _index, editor=combo, spec=edit: self._on_selected_combo_changed(spec, editor)
        )
        self._select_table.setCellWidget(row_idx, 1, combo)

    def _on_detail_cell_activated(self, row: int, column: int) -> None:
        item = self._select_table.item(row, column)
        if item is not None:
            self._activate_detail_item(item)

    def _on_detail_item_activated(self, item: QTableWidgetItem) -> None:
        self._activate_detail_item(item)

    def _activate_detail_item(self, item: QTableWidgetItem) -> None:
        edit = item.data(_EDIT_ROLE)
        if isinstance(edit, _FieldEdit):
            self._select_table.editItem(item)
            return
        ref = item.data(_TARGET_REF_ROLE)
        if isinstance(ref, FeatureRef):
            self.select_feature(ref, update_viz=True)
            self._tabs.setCurrentWidget(self._selected_tab)

    def _on_selected_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating_selected_panel or item.column() != 1:
            return
        edit = item.data(_EDIT_ROLE)
        if isinstance(edit, _FieldEdit):
            self._apply_selected_edit(edit, item.text())

    def _on_selected_combo_changed(self, edit: _FieldEdit, combo: QComboBox) -> None:
        if self._populating_selected_panel:
            return
        raw = combo.currentData()
        self._apply_selected_edit(edit, "" if raw is None else str(raw))

    def _apply_selected_edit(self, edit: _FieldEdit, raw_text: str) -> None:
        viz = self.viz
        if viz is None:
            return
        try:
            value = _parse_edit_value(edit, raw_text)
            feature = viz.dataset.store_for_attr(edit.ref.layer_attr)[edit.ref.feature_id]
            new_ref = self._write_feature_value(viz, feature, edit, value)
        except (KeyError, ValueError) as e:
            QMessageBox.warning(self, "Invalid field edit", str(e))
            if self._selected_ref is not None:
                self._populate_selected_panel(viz, self._selected_ref)
            return

        if new_ref is not None:
            self._selected_ref = new_ref
            viz.select_feature(new_ref, emit=False)
        self._set_memory_dirty(True)
        self._rebuild_items_tab(viz)
        if self._selected_ref is not None:
            self._sync_item_selection(self._selected_ref)
            self._populate_selected_panel(viz, self._selected_ref)
        log.info(
            "edited in memory %s:%s %s",
            edit.ref.layer_attr,
            edit.ref.feature_id,
            edit.column_name,
        )

    def _write_feature_value(
        self,
        viz: HdMapViz,
        feature: NGIIFeature,
        edit: _FieldEdit,
        value: object,
    ) -> FeatureRef | None:
        if edit.is_feature_id:
            new_id = str(value)
            if new_id == feature.id:
                return None
            store = viz.dataset.store_for_attr(edit.ref.layer_attr)
            existing = store.get(new_id)
            if existing is not None and existing is not feature:
                msg = f"{store.layer_name} already has feature ID {new_id!r}"
                raise ValueError(msg)
            old_ref = FeatureRef(edit.ref.layer_attr, feature.id)
            feature.id = new_id
            store.rebuild_index()
            viz.dataset.bind(SanityReport(), warn_global_id_collision=False)
            self._replace_item_ref(old_ref, FeatureRef(edit.ref.layer_attr, new_id))
            return FeatureRef(edit.ref.layer_attr, new_id)
        setattr(feature, edit.attr_name, value)
        return None

    def _replace_item_ref(self, old_ref: FeatureRef, new_ref: FeatureRef) -> None:
        item = self._item_by_ref.pop(old_ref, None)
        if item is None:
            return
        item.setText(0, new_ref.feature_id)
        item.setData(0, _TARGET_REF_ROLE, new_ref)
        self._item_by_ref[new_ref] = item

    @override
    def closeEvent(self, event: object) -> None:
        if self.viz is not None:
            self.viz.detach()
            self.viz = None
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        self.qt_plotter.close()
        super().closeEvent(event)  # type: ignore[arg-type]


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _layer_store_items(viz: HdMapViz) -> tuple[tuple[str, LayerStore[Any]], ...]:
    return tuple(
        (spec.python_attr, viz.dataset.store_for_attr(spec.python_attr))
        for spec in viz.dataset.schema.layer_specs
    )


def _detail_rows(viz: HdMapViz, ref: FeatureRef, feature: NGIIFeature) -> list[_DetailRow]:
    rows: list[_DetailRow] = [_DetailRow("Feature", "", True)]
    id_edit = _id_edit_spec(viz, ref, feature)
    rows.extend(
        (
            _DetailRow("Layer", feature.layer_name),
            _DetailRow("ID", feature.id, edit=id_edit, raw_value=feature.id),
            _DetailRow("Source", f"{feature.source_path.name}:{feature.source_row}"),
        )
    )
    rows.extend(_schema_field_rows(viz, ref, feature))
    rows.extend(_geometry_rows(feature))
    rows.extend(_segmentation_rows(viz.segmentation.selected_fields_for_ref(ref)))
    return rows


def _schema_field_rows(viz: HdMapViz, ref: FeatureRef, feature: NGIIFeature) -> list[_DetailRow]:
    spec = viz.dataset.schema.spec_for_layer_name(feature.layer_name)
    if spec is None:
        return []
    references = {reference.source_attr: reference for reference in spec.references}
    seen_attrs: set[str] = set()
    rows = [_DetailRow("Fields", "", True)]
    for rule in spec.field_rules:
        if rule.name == "ID":
            continue
        attr_name = rule.attr
        seen_attrs.add(attr_name)
        value = getattr(feature, attr_name, "")
        reference = references.get(attr_name)
        edit = _field_edit_spec(ref, rule, is_reference=reference is not None)
        raw_value = _raw_value_text(value)
        if reference is None:
            rows.append(
                _DetailRow(
                    _display_name(attr_name),
                    _decoded_value(feature, attr_name, value),
                    edit=edit,
                    raw_value=raw_value,
                )
            )
        else:
            rows.append(_reference_detail_row(viz, reference, attr_name, value, edit=edit))
    for reference in spec.references:
        if reference.source_attr in seen_attrs:
            continue
        value = getattr(feature, reference.source_attr, "")
        field_rule = spec.field_rules_by_column.get(reference.column_name)
        edit = (
            _field_edit_spec(ref, field_rule, is_reference=True)
            if field_rule is not None
            else _reference_edit_spec(ref, reference)
        )
        rows.append(_reference_detail_row(viz, reference, reference.source_attr, value, edit=edit))
    return rows if len(rows) > 1 else []


def _id_edit_spec(viz: HdMapViz, ref: FeatureRef, feature: NGIIFeature) -> _FieldEdit:
    spec = viz.dataset.schema.spec_for_layer_name(feature.layer_name)
    id_rule = (
        None
        if spec is None
        else next(
            (rule for rule in spec.field_rules if rule.name == "ID"),
            None,
        )
    )
    return _FieldEdit(
        ref=ref,
        column_name="ID",
        attr_name="id",
        field_type="text",
        required=True,
        max_length=None if id_rule is None else id_rule.max_length,
        code_list=None,
        is_feature_id=True,
    )


def _field_edit_spec(ref: FeatureRef, rule: FieldRule, *, is_reference: bool) -> _FieldEdit:
    return _FieldEdit(
        ref=ref,
        column_name=rule.name,
        attr_name=rule.attr,
        field_type=rule.field_type,
        required=rule.required,
        max_length=rule.max_length,
        code_list=rule.code_list,
        is_reference=is_reference,
    )


def _reference_edit_spec(ref: FeatureRef, reference: ReferenceRule) -> _FieldEdit:
    return _FieldEdit(
        ref=ref,
        column_name=reference.column_name,
        attr_name=reference.source_attr,
        field_type="text",
        required=reference.required,
        max_length=None,
        code_list=None,
        is_reference=True,
    )


def _decoded_value(feature: NGIIFeature, field_name: str, value: object) -> str:
    if value is None or value == "":
        return "-"
    text = str(value)
    label_table = _label_table(feature, field_name)
    if label_table is None:
        return text
    label = label_table.get(text)
    return f"{text} ({label})" if label else text


def _label_table(feature: NGIIFeature, field_name: str) -> dict[str, str] | None:
    candidates = (
        f"{field_name.upper()}_LABEL",
        f"{field_name.removesuffix('_id').upper()}_LABEL",
    )
    for candidate in candidates:
        table = getattr(type(feature), candidate, None)
        if isinstance(table, dict):
            return table
    return None


def _raw_value_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_edit_value(edit: _FieldEdit, raw_text: str) -> object:
    value_text = raw_text.strip()
    if not value_text:
        if edit.required:
            msg = f"{edit.column_name} is required"
            raise ValueError(msg)
        value: object = _empty_edit_value(edit)
    else:
        _validate_text_edit_value(edit, value_text)
        if edit.field_type == "integer":
            if not is_integer_value(value_text):
                msg = f"{edit.column_name} must be an integer"
                raise ValueError(msg)
            value = int(float(value_text))
        elif edit.field_type == "float":
            if not is_float_value(value_text):
                msg = f"{edit.column_name} must be a float"
                raise ValueError(msg)
            value = float(value_text)
        else:
            value = value_text
    return value


def _empty_edit_value(edit: _FieldEdit) -> object:
    if edit.is_reference:
        return None
    if edit.field_type == "integer":
        return -1
    if edit.field_type == "float":
        return float("nan")
    return ""


def _validate_text_edit_value(edit: _FieldEdit, value_text: str) -> None:
    if edit.max_length is not None and len(value_text) > edit.max_length:
        msg = f"{edit.column_name} must be at most {edit.max_length} characters"
        raise ValueError(msg)
    if edit.code_list is not None and value_text not in edit.code_list:
        allowed = ", ".join(sorted(edit.code_list))
        msg = f"{edit.column_name} must be one of: {allowed}"
        raise ValueError(msg)


def _geometry_rows(feature: NGIIFeature) -> list[_DetailRow]:
    rows = [_DetailRow("Geometry", "", True)]
    if isinstance(feature, PointFeature):
        x, y, z = feature.point
        rows.append(_DetailRow("Point", f"{x:.3f}, {y:.3f}, {z:.3f}"))
    elif isinstance(feature, PointOrPolygonFeature):
        if feature.point is not None:
            x, y, z = feature.point
            rows.append(_DetailRow("Point", f"{x:.3f}, {y:.3f}, {z:.3f}"))
        elif feature.ring is not None:
            polygon = shapely.Polygon(feature.ring[:, :2])
            rows.extend(
                (
                    _DetailRow("Vertices", str(len(feature.ring))),
                    _DetailRow("XY area (m^2)", f"{polygon.area:.2f}"),
                )
            )
    elif isinstance(feature, LineFeature):
        line = shapely.LineString(feature.polyline[:, :2])
        rows.extend(
            (
                _DetailRow("Vertices", str(len(feature.polyline))),
                _DetailRow("XY length (m)", f"{line.length:.2f}"),
            )
        )
    elif isinstance(feature, PolygonFeature):
        polygon = shapely.Polygon(feature.ring[:, :2])
        rows.extend(
            (
                _DetailRow("Vertices", str(len(feature.ring))),
                _DetailRow("XY area (m^2)", f"{polygon.area:.2f}"),
            )
        )
    return rows


def _reference_detail_row(
    viz: HdMapViz,
    reference: ReferenceRule,
    field_name: str,
    value: object,
    *,
    edit: _FieldEdit | None = None,
) -> _DetailRow:
    if value is None or value == "":
        return _DetailRow(_display_name(field_name), "-", edit=edit, raw_value="")
    feature_id = str(value)
    target_ref = _resolve_reference_ref(viz, reference, feature_id)
    if target_ref is None:
        return _DetailRow(
            _display_name(field_name),
            feature_id if edit is not None else f"{feature_id} (unresolved)",
            edit=edit,
            raw_value=feature_id,
        )
    return _DetailRow(
        _display_name(field_name),
        feature_id,
        target_ref=target_ref,
        edit=edit,
        raw_value=feature_id,
    )


def _feature_references(viz: HdMapViz, feature: NGIIFeature) -> tuple[ReferenceRule, ...]:
    spec = viz.dataset.schema.spec_for_layer_name(feature.layer_name)
    return () if spec is None else spec.references


def _resolve_reference_ref(
    viz: HdMapViz, reference: ReferenceRule, feature_id: str
) -> FeatureRef | None:
    matches = [
        FeatureRef(attr, feature_id)
        for attr in reference.target_attrs
        if viz.dataset.store_for_attr(attr).get(feature_id) is not None
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None
    try:
        target = viz.dataset[feature_id]
    except (KeyError, AmbiguousFeatureIDError):
        return None
    attr = viz.dataset.feature_ref_attr(target)
    return None if attr is None else FeatureRef(attr, target.id)


def _feature_summary(feature: NGIIFeature) -> str:
    for name in ("node_type", "link_type", "kind", "subtype", "type", "direction"):
        if not hasattr(feature, name):
            continue
        value = getattr(feature, name)
        if value is None or value == "":
            continue
        return f"{_display_name(name)}: {_decoded_value(feature, name, value)}"
    return _feature_geometry_label(feature)


def _feature_geometry_label(feature: NGIIFeature) -> str:
    if isinstance(feature, PointFeature):
        return "Point"
    if isinstance(feature, LineFeature):
        return "Line"
    if isinstance(feature, PolygonFeature):
        return "Polygon"
    if isinstance(feature, PointOrPolygonFeature):
        return "Point" if feature.point is not None else "Polygon"
    return "-"


def _store_geometry_label(store: LayerStore[Any]) -> str:
    labels = {_feature_geometry_label(feature) for feature in store.features}
    if not labels:
        return "empty/missing"
    return "+".join(label for label in ("Point", "Line", "Polygon") if label in labels)


def _feature_exists(viz: HdMapViz, ref: FeatureRef) -> bool:
    return viz.dataset.store_for_attr(ref.layer_attr).get(ref.feature_id) is not None


def _segmentation_rows(fields_: tuple[SelectedField, ...]) -> list[_DetailRow]:
    if not fields_:
        return []
    rows = [_DetailRow("Segmentation", "", True)]
    for field in _compact_selected_fields(fields_):
        values = field.values if field.values else ("-",)
        refs = field.refs if field.refs else tuple(None for _ in values)
        for index, value in enumerate(values):
            name = field.name if index == 0 else ""
            ref = refs[index] if index < len(refs) else None
            rows.append(_DetailRow(name, value, target_ref=ref))
    return rows


def _compact_selected_fields(fields_: tuple[SelectedField, ...]) -> tuple[SelectedField, ...]:
    compacted: list[SelectedField] = []
    index_by_name: dict[str, int] = {}
    seen_values_by_name: dict[str, set[tuple[str, FeatureRef | None]]] = {}
    for field in fields_:
        output_index = index_by_name.get(field.name)
        if output_index is None:
            index_by_name[field.name] = len(compacted)
            compacted.append(field)
            seen_values_by_name[field.name] = set(zip(field.values, field.refs, strict=True))
            continue

        seen = seen_values_by_name[field.name]
        values = list(compacted[output_index].values)
        refs = list(compacted[output_index].refs)
        for value, ref in zip(field.values, field.refs, strict=True):
            key = (value, ref)
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
            refs.append(ref)
        compacted[output_index] = SelectedField(
            name=field.name,
            values=tuple(values),
            refs=tuple(refs),
        )
    return tuple(compacted)


def _log_gui_profile(viz: HdMapViz, timer: ProfileTimer) -> None:
    if viz.viz_cfg.profiling.enabled:
        log.info("gui %s: %.1fms (%s)", timer.name, timer.duration_s * 1000.0, timer.detail)


def _ref_detail(ref: FeatureRef, update_viz: bool) -> str:
    return f"ref={ref.layer_attr}:{ref.feature_id}; update_viz={update_viz}"


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


def _fit_tree_to_rows(tree: QTreeWidget, rows: int) -> None:
    row_height = max(tree.sizeHintForRow(0), 20)
    header_height = tree.header().height()
    frame = tree.frameWidth() * 2
    tree.setFixedHeight(header_height + row_height * rows + frame + 4)
