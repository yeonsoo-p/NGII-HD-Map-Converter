"""Qt inspector window for dataset-native NGII visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import shapely
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data import AmbiguousFeatureIDError, NGIIConfig
from ngii2xodr.ngii.data.dataset import LayerStore
from ngii2xodr.ngii.data.features import (
    LineFeature,
    NGIIFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
)
from ngii2xodr.ngii.data.manual_2023 import LAYER_SPECS, SPECS_BY_LAYER_NAME, RelationshipRule
from ngii2xodr.ngii.segmentation import SegmentationConfig, SelectedField
from ngii2xodr.ngii.viz import HdMapViz, VizConfig

log = logging.getLogger(__name__)

_SELECT_PLACEHOLDER = "- open a folder to begin -"
_SELECT_PROMPT = "- shift-click a feature or select an item -"
_SECTION_BG = QColor(230, 230, 235)


@dataclass(slots=True, frozen=True)
class _DetailRow:
    field: str
    value: str
    is_section: bool = False
    target_ref: FeatureRef | None = None


class HdMapWindow(QMainWindow):
    """Main window hosting the 3D scene and canonical feature inspector."""

    def __init__(
        self,
        seg_cfg: SegmentationConfig,
        viz_cfg: VizConfig,
        ngii_cfg: NGIIConfig,
        coordinate: str,
    ) -> None:
        super().__init__()
        self.setWindowTitle("ngii2xodr - (no folder)")
        self.resize(1500, 950)

        self._seg_cfg = seg_cfg
        self._viz_cfg = viz_cfg
        self._ngii_cfg = ngii_cfg
        self._coordinate = coordinate
        self.viz: HdMapViz | None = None

        self._layer_items: dict[str, QTreeWidgetItem] = {}
        self._updating_layer_items = False
        self._updating_layer_master = False
        self._item_by_ref: dict[FeatureRef, QTreeWidgetItem] = {}
        self._updating_item_selection = False
        self._selected_ref: FeatureRef | None = None
        self._segmentation_level = 0
        self._segmentation_button_group: QButtonGroup | None = None

        self.qt_plotter = QtInteractor(self)
        self.setCentralWidget(self.qt_plotter)
        self.qt_plotter.enable_parallel_projection()
        self.qt_plotter.view_xy()

        self._build_menu()
        self._build_dock()
        self._set_dock_enabled(False)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        open_act = QAction("&Open NGII folder...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_folder)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

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

    def _build_layers_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self._layers_group = QGroupBox("Renderable layers")
        self._layers_layout = QVBoxLayout(self._layers_group)
        self._layer_master_checkbox = QCheckBox("All visible")
        self._layer_master_checkbox.setTristate(True)
        self._layer_master_checkbox.stateChanged.connect(self._on_layer_master_changed)
        self._layers_layout.addWidget(self._layer_master_checkbox)
        self._layer_tree = QTreeWidget()
        self._layer_tree.setHeaderLabels(("Visible", "Layer", "Geometry", "Count"))
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

        self._select_table = QTableWidget(0, 2)
        self._select_table.setHorizontalHeaderLabels(("Field", "Value"))
        self._select_table.verticalHeader().setVisible(False)
        self._select_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._select_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._select_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._select_table.setShowGrid(True)
        self._select_table.setAlternatingRowColors(True)
        self._select_table.cellDoubleClicked.connect(self._on_detail_cell_activated)
        self._select_table.itemActivated.connect(self._on_detail_item_activated)
        header = self._select_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._select_table, stretch=1)
        return tab

    def _set_dock_enabled(self, on: bool) -> None:
        self._layers_tab.setEnabled(on)
        self._items_tab.setEnabled(on)
        self._selected_tab.setEnabled(on)

    def _open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open NGII folder")
        if d:
            self.load_folder(Path(d))

    def load_folder(self, ngii_dir: Path) -> None:
        if self.viz is not None:
            self.viz.detach()
            self.viz = None

        self._selected_ref = None
        self._reset_select_panel(_SELECT_PLACEHOLDER)
        try:
            viz = HdMapViz(
                ngii_dir,
                coordinate=self._coordinate,
                ngii_cfg=self._ngii_cfg,
                seg_cfg=self._seg_cfg,
                viz_cfg=self._viz_cfg,
                plotter=self.qt_plotter,
                on_select=self._on_viz_select,
            )
            viz.attach()
        except (FileNotFoundError, NotADirectoryError) as e:
            log.warning("cannot open %s: %s", ngii_dir, e)
            QMessageBox.warning(
                self,
                "Cannot open folder",
                f"Not a valid NGII folder:\n{ngii_dir}\n\n{e}",
            )
            self.setWindowTitle("ngii2xodr - (no folder)")
            self._set_dock_enabled(False)
            self.qt_plotter.render()
            return

        self.viz = viz
        self._rebuild_layers_tab(viz)
        self._rebuild_items_tab(viz)
        self._set_segmentation_level(
            min(self._segmentation_level, len(viz.segmentation.stage_results))
        )

        self.qt_plotter.view_xy()
        self.qt_plotter.reset_camera()
        self.setWindowTitle(f"ngii2xodr - {ngii_dir.name}")
        self._set_dock_enabled(True)
        self._reset_select_panel(_SELECT_PROMPT)
        feature_count = sum(len(store) for store in viz.dataset.layer_stores)
        self.statusBar().showMessage(f"Loaded {ngii_dir.name}: {feature_count} features")
        log.info(
            "loaded %s: %d warnings, %d repairs",
            ngii_dir,
            len(viz.dataset.sanity.warnings),
            len(viz.dataset.sanity.actions),
        )

    def _rebuild_layers_tab(self, viz: HdMapViz) -> None:
        self._updating_layer_items = True
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
        _fit_tree_to_rows(self._layer_tree, len(LAYER_SPECS))
        self._sync_layer_master_checkbox()
        self._updating_layer_items = False

        _clear_layout(self._segmentation_layout)
        self._segmentation_button_group = QButtonGroup(self._segmentation_group)
        for level, label in enumerate(viz.segmentation.level_labels):
            rb = QRadioButton(f"{level}  {label}")
            rb.setChecked(level == self._segmentation_level)
            self._segmentation_button_group.addButton(rb, level)
            self._segmentation_layout.addWidget(rb)
        self._segmentation_button_group.idToggled.connect(self._on_segmentation_level_changed)

    def _rebuild_items_tab(self, viz: HdMapViz) -> None:
        self._item_by_ref.clear()
        self._updating_item_selection = True
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
        self._updating_item_selection = False
        self._filter_items(self._item_search.text())

    def _on_layer_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_layer_items or column != 0 or self.viz is None:
            return
        attr = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(attr, str):
            self.viz.set_layer_visible(attr, item.checkState(0) == Qt.CheckState.Checked)
            self._sync_layer_master_checkbox()

    def _on_layer_master_changed(self, state: int) -> None:
        if self._updating_layer_master or self.viz is None:
            return
        visible = Qt.CheckState(state) != Qt.CheckState.Unchecked
        self._updating_layer_items = True
        for item in self._toggleable_layer_items():
            item.setCheckState(0, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            attr = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(attr, str):
                self.viz.set_layer_visible(attr, visible)
        self._updating_layer_items = False
        self._sync_layer_master_checkbox()

    def _sync_layer_master_checkbox(self) -> None:
        items = self._toggleable_layer_items()
        self._updating_layer_master = True
        self._layer_master_checkbox.setEnabled(bool(items))
        if not items:
            self._layer_master_checkbox.setCheckState(Qt.CheckState.Unchecked)
        elif all(item.checkState(0) == Qt.CheckState.Checked for item in items):
            self._layer_master_checkbox.setCheckState(Qt.CheckState.Checked)
        elif all(item.checkState(0) == Qt.CheckState.Unchecked for item in items):
            self._layer_master_checkbox.setCheckState(Qt.CheckState.Unchecked)
        else:
            self._layer_master_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self._updating_layer_master = False

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
        if self._updating_item_selection:
            return
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
        if update_viz:
            viz.select_feature(ref, emit=False)
            viz.focus_feature(ref)
        self._sync_item_selection(ref)
        self._populate_selected_panel(viz, ref)
        log.info("selected %s:%s", ref.layer_attr, ref.feature_id)

    def _sync_item_selection(self, ref: FeatureRef) -> None:
        item = self._item_by_ref.get(ref)
        if item is None:
            return
        self._updating_item_selection = True
        parent = item.parent()
        if parent is not None:
            parent.setExpanded(True)
        self._item_tree.setCurrentItem(item)
        self._item_tree.scrollToItem(item)
        self._updating_item_selection = False

    def _reset_select_panel(self, header_text: str) -> None:
        self._select_header.setText(header_text)
        self._select_table.clearSpans()
        self._select_table.clearContents()
        self._select_table.setRowCount(0)

    def _populate_selected_panel(self, viz: HdMapViz, ref: FeatureRef) -> None:
        feature = viz.dataset.store_for_attr(ref.layer_attr)[ref.feature_id]
        self._select_header.setText(f"{feature.layer_name}    {feature.id}")
        rows = _detail_rows(viz, ref, feature)
        self._select_table.clearSpans()
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
                self._select_table.setItem(row_idx, 0, item)
                self._select_table.setSpan(row_idx, 0, 1, 2)
            else:
                field_item = QTableWidgetItem(row.field)
                value_item = QTableWidgetItem(row.value)
                if row.target_ref is not None:
                    for item in (field_item, value_item):
                        item.setData(Qt.ItemDataRole.UserRole, row.target_ref)
                        item.setForeground(QBrush(QColor(30, 85, 180)))
                self._select_table.setItem(row_idx, 0, field_item)
                self._select_table.setItem(row_idx, 1, value_item)

    def _on_detail_cell_activated(self, row: int, column: int) -> None:
        item = self._select_table.item(row, column)
        if item is not None:
            self._activate_detail_item(item)

    def _on_detail_item_activated(self, item: QTableWidgetItem) -> None:
        self._activate_detail_item(item)

    def _activate_detail_item(self, item: QTableWidgetItem) -> None:
        ref = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(ref, FeatureRef):
            self.select_feature(ref, update_viz=True)
            self._tabs.setCurrentWidget(self._selected_tab)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        if self.viz is not None:
            self.viz.detach()
            self.viz = None
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
        (spec.python_attr, viz.dataset.store_for_attr(spec.python_attr)) for spec in LAYER_SPECS
    )


def _detail_rows(viz: HdMapViz, ref: FeatureRef, feature: NGIIFeature) -> list[_DetailRow]:
    rows: list[_DetailRow] = [_DetailRow("Feature", "", True)]
    rows.extend(
        (
            _DetailRow("Layer", feature.layer_name),
            _DetailRow("ID", feature.id),
            _DetailRow("Source", f"{feature.source_path.name}:{feature.source_row}"),
        )
    )
    rows.extend(_dataclass_rows(viz, feature))
    rows.extend(_geometry_rows(feature))
    rows.extend(_segmentation_rows(viz.segmentation.selected_fields_for_ref(ref)))
    return rows


def _dataclass_rows(viz: HdMapViz, feature: NGIIFeature) -> list[_DetailRow]:
    skipped = {
        "id",
        "source_path",
        "source_row",
        "_dataset",
        "point",
        "polyline",
        "ring",
    }
    relationships = {
        relationship.source_attr: relationship for relationship in _feature_relationships(feature)
    }
    rows = [_DetailRow("Fields", "", True)]
    for field in fields(feature):
        if field.name in skipped:
            continue
        value = getattr(feature, field.name)
        relationship = relationships.get(field.name)
        if relationship is None:
            rows.append(
                _DetailRow(_display_name(field.name), _decoded_value(feature, field.name, value))
            )
        else:
            rows.append(_relationship_detail_row(viz, relationship, field.name, value))
    return rows if len(rows) > 1 else []


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


def _relationship_detail_row(
    viz: HdMapViz,
    relationship: RelationshipRule,
    field_name: str,
    value: object,
) -> _DetailRow:
    if value is None or value == "":
        return _DetailRow(_display_name(field_name), "-")
    feature_id = str(value)
    target_ref = _resolve_relationship_ref(viz, relationship, feature_id)
    if target_ref is None:
        return _DetailRow(_display_name(field_name), f"{feature_id} (unresolved)")
    return _DetailRow(_display_name(field_name), feature_id, target_ref=target_ref)


def _feature_relationships(feature: NGIIFeature) -> tuple[RelationshipRule, ...]:
    spec = SPECS_BY_LAYER_NAME.get(feature.layer_name)
    return () if spec is None else spec.relationships


def _resolve_relationship_ref(
    viz: HdMapViz, relationship: RelationshipRule, feature_id: str
) -> FeatureRef | None:
    matches = [
        FeatureRef(attr, feature_id)
        for attr in relationship.target_attrs
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
    for spec in LAYER_SPECS:
        if spec.layer_name == target.layer_name:
            return FeatureRef(spec.python_attr, target.id)
    return None


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


def _segmentation_rows(fields_: tuple[SelectedField, ...]) -> list[_DetailRow]:
    if not fields_:
        return []
    rows = [_DetailRow("Segmentation", "", True)]
    for field in fields_:
        values = field.values if field.values else ("-",)
        refs = field.refs if field.refs else tuple(None for _ in values)
        for index, value in enumerate(values):
            name = field.name if index == 0 else ""
            ref = refs[index] if index < len(refs) else None
            rows.append(_DetailRow(name, value, target_ref=ref))
    return rows


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


def _fit_tree_to_rows(tree: QTreeWidget, rows: int) -> None:
    row_height = max(tree.sizeHintForRow(0), 20)
    header_height = tree.header().height()
    frame = tree.frameWidth() * 2
    tree.setFixedHeight(header_height + row_height * rows + frame + 4)
