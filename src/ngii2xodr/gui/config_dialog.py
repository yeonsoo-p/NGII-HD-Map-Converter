"""Runtime configuration dialog for the NGII viewer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ngii2xodr.config import (
    SANITY_CHECK_NAMES,
    WARNING_ONLY_SANITY_CHECKS,
    RuntimeConfig,
    make_sanity_checks_config,
    validate_runtime_config,
)
from ngii2xodr.ngii.data import (
    NGIIConfig,
    NGIIEncodingConfig,
    NGIIGeometryConfig,
    NGIISanityConfig,
    SanityMode,
)
from ngii2xodr.ngii.data.config import SanityCheckMode
from ngii2xodr.ngii.segmentation import SegmentationConfig
from ngii2xodr.ngii.viz import VizCameraFocusConfig, VizConfig, VizLayerConfig
from ngii2xodr.profile import ViewportProfilingConfig

ApplyCallback = Callable[[RuntimeConfig], bool]

_SANITY_HELP: dict[str, tuple[str, str]] = {
    "layer_required_missing": (
        "A required layer from the active NGII schema is not discovered.",
        "Not available. Missing source layers need new input data.",
    ),
    "layer_unknown": (
        "A SHP layer is present but not documented by the active NGII schema.",
        "Not available. Unknown layers are omitted.",
    ),
    "shp_sidecar_missing": (
        "A discovered SHP file is missing its DBF, SHX, or PRJ sidecar.",
        "Not available. Sidecar files must be supplied with the dataset.",
    ),
    "dbf_column_case_duplicate": (
        "A DBF has manual columns that differ only by letter case.",
        "Not available. The loader reports the ambiguous columns.",
    ),
    "manual_column_missing": (
        "A documented manual column is absent from a layer DBF.",
        "Not available. The feature uses parser defaults when possible.",
    ),
    "manual_column_unknown": (
        "A DBF contains columns outside the active schema manual.",
        "Not available. Unknown columns are ignored.",
    ),
    "manual_field_required_missing": (
        "A required manual field is empty after loading.",
        "Not available. The missing source value is reported.",
    ),
    "manual_field_length_exceeded": (
        "A text value is longer than the schema VARCHAR length.",
        "Not available. The overlong source value is reported.",
    ),
    "manual_field_type_invalid": (
        "A manual integer or float field cannot be parsed as that type.",
        "Not available. The loader falls back to the existing parser default.",
    ),
    "manual_field_code_invalid": (
        "A value is outside the schema code list for that field.",
        "Not available. The source value is preserved and reported.",
    ),
    "manual_hist_type_invalid": (
        "A HistType value does not match the schema-specific historical type range.",
        "Not available. The source value is preserved and reported.",
    ),
    "geometry_missing": (
        "A row has no supported geometry object.",
        "Not available. The feature is skipped.",
    ),
    "geometry_invalid": (
        "A geometry cannot be converted into the expected point, line, or polygon shape.",
        "Not available. The feature is skipped.",
    ),
    "text_utf8_dbf_row_mismatch": (
        "A UTF-8 DBF overlay has a different row count from the decoded SHP rows.",
        "Not available. The overlay is skipped.",
    ),
    "text_utf8_decode_replacement": (
        "UTF-8 DBF decoding produced replacement characters.",
        "Not available. The decoded text is reported.",
    ),
    "text_mojibake": (
        "Text contains common mojibake patterns after DBF decoding.",
        "Replace deterministic mojibake sequences with their intended Korean text.",
    ),
    "text_replacement_char": (
        "Text contains the Unicode replacement character.",
        "Not available. The damaged value is reported.",
    ),
    "feature_id_missing": (
        "A feature has an empty ID field.",
        "Not available. The feature is reported because stable repair is ambiguous.",
    ),
    "feature_id_duplicate_identical": (
        "Two rows in one layer share an ID and have identical content.",
        "Not available. The duplicate is reported.",
    ),
    "feature_id_duplicate_conflicting": (
        "Two rows in one layer share an ID but differ in attributes or geometry.",
        "Drop the later conflicting duplicate and keep the first loaded feature.",
    ),
    "global_id_collision": (
        "The same ID appears in multiple NGII layers, making dataset-wide lookup ambiguous.",
        "Not available. Layer-scoped references remain usable.",
    ),
    "reference_unresolved": (
        "A reference ID does not resolve to its target layer.",
        "Clear optional unresolved references when deterministic; "
        "required references remain warnings.",
    ),
    "reciprocal_reference_missing": (
        "A side-reference exists on one link but the reciprocal link does not point back.",
        "Fill the missing reciprocal side-reference when the target link resolves.",
    ),
    "reciprocal_reference_conflict": (
        "Two side-reference values disagree with each other.",
        "Not available. Conflicts are reported for manual inspection.",
    ),
    "link_too_short": (
        "A link geometry length is below the configured minimum.",
        "Remove the short link and clear references to it.",
    ),
    "link_endpoint_isolated": (
        "A link endpoint node is not used by any other link endpoint.",
        "Snap the endpoint reference to a nearby node within the configured tolerance.",
    ),
    "link_endpoint_unresolved": (
        "A link FromNodeID or ToNodeID does not resolve to the node layer.",
        "Replace the endpoint reference with a nearby node within the configured tolerance.",
    ),
    "link_endpoint_misaligned": (
        "A link endpoint node does not match either end of the link geometry.",
        "Not available. Misaligned geometry is reported.",
    ),
    "link_orientation_reversed": (
        "A link's node IDs or polyline direction disagree with topology-derived traffic flow.",
        "Swap FromNodeID/ToNodeID, reverse the polyline, or do both when "
        "evidence is deterministic.",
    ),
    "node_unreferenced": (
        "A node is not referenced by any link endpoint.",
        "Remove unreferenced nodes from the in-memory dataset.",
    ),
    "link_side_reference_longitudinal": (
        "A lateral side-reference points to a longitudinal neighbor rather than a side neighbor.",
        "Clear the invalid side-reference.",
    ),
    "link_side_reference_nonreciprocal": (
        "A lateral side-reference cannot be reconciled with reciprocal topology.",
        "Clear the invalid side-reference.",
    ),
}

_SEGMENTATION_HELP: dict[str, str] = {
    "enable_uturn": (
        "Builds U-turn entities from direct link filters or matching U-turn marker geometry. "
        "Uses z_intersection_tol_m when matching marker intersections."
    ),
    "enable_lateral_link_group": (
        "Groups ordinary lane links through valid R_LinkID/L_LinkID topology. "
        "U-turn and pocket links are tracked separately inside each group."
    ),
    "enable_lateral_node_group": (
        "Groups from/to endpoint nodes for each lateral link group. "
        "Uses node GroupID when available and falls back to endpoint topology."
    ),
    "enable_junction": (
        "Builds junction entities from junction-like link and endpoint-node candidates."
    ),
    "enable_junction_connection": (
        "Pairs lateral endpoint groups into junction connections. "
        "Uses junction_connection_node_merge_dist_m and "
        "junction_connection_opposite_direction_dot_min."
    ),
    "enable_junction_reference": (
        "Selects reference links and endpoint tangents for each junction connection. "
        "Uses endpoint_tangent_lookback_m."
    ),
    "enable_junction_edge": (
        "Builds junction edge line segments across each connection from the selected reference."
    ),
}


@dataclass(slots=True, frozen=True)
class _LayerControls:
    visible: QCheckBox
    color: _ColorButton
    point_size: QDoubleSpinBox
    line_width: QDoubleSpinBox
    opacity: QDoubleSpinBox


class _ColorButton(QPushButton):
    def __init__(self, rgb: tuple[int, int, int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rgb = rgb
        self.setFixedSize(36, 22)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip("Color")
        self.clicked.connect(self._choose_color)
        self._refresh()

    def color_rgb(self) -> tuple[int, int, int]:
        return self._rgb

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self._rgb), self, "Select color")
        if color.isValid():
            self._rgb = (color.red(), color.green(), color.blue())
            self._refresh()

    def _refresh(self) -> None:
        r, g, b = self._rgb
        self.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #777; border-radius: 3px;"
        )


class _DownwardComboBox(QComboBox):
    @override
    def showPopup(self) -> None:
        super().showPopup()
        popup = self.view().window()
        popup.move(self.mapToGlobal(QPoint(0, self.height())))


class ConfigDialog(QDialog):
    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        on_apply: ApplyCallback,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runtime_cfg = runtime_cfg
        self._on_apply = on_apply
        self._sanity_modes: dict[str, QComboBox] = {}
        self._layer_controls: dict[str, _LayerControls] = {}

        self.setWindowTitle("Configuration")
        self.resize(980, 720)
        self._build_ui(runtime_cfg)

    def _build_ui(self, cfg: RuntimeConfig) -> None:
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        self._nav = QListWidget()
        self._nav.addItems(("NGII", "Segmentation", "Visualization"))
        self._nav.setFixedWidth(170)
        body.addWidget(self._nav)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._scroll_page(self._build_ngii_page(cfg)))
        self._pages.addWidget(self._scroll_page(self._build_segmentation_page(cfg)))
        self._pages.addWidget(self._scroll_page(self._build_viz_page(cfg)))
        body.addWidget(self._pages, stretch=1)

        self._nav.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._nav.setCurrentRow(0)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        apply_button = self._button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_button.clicked.connect(self._apply_requested)
        self._button_box.accepted.connect(self._accept_requested)
        self._button_box.rejected.connect(self.reject)
        root.addWidget(self._button_box)

    def _build_ngii_page(self, cfg: RuntimeConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_ngii_general_group(cfg), stretch=0)
        layout.addWidget(self._build_sanity_group(cfg.ngii.sanity), stretch=1)
        return page

    def _build_ngii_general_group(self, cfg: RuntimeConfig) -> QGroupBox:
        group = QGroupBox("NGII")
        form = QFormLayout(group)
        self._coordinate = QLineEdit(cfg.coordinate)
        form.addRow("Coordinate", self._coordinate)
        self._multipart_snap_tolerance_m = _float_spin(
            cfg.ngii.geometry.multipart_snap_tolerance_m,
            minimum=0.0,
            maximum=1_000.0,
            decimals=3,
            step=0.05,
        )
        form.addRow("Multipart snap tolerance (m)", self._multipart_snap_tolerance_m)
        self._utf8_dbf_invalid_non_ascii_ratio_max = _float_spin(
            cfg.ngii.encoding.utf8_dbf_invalid_non_ascii_ratio_max,
            minimum=0.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
        )
        form.addRow("UTF-8 DBF invalid text ratio max", self._utf8_dbf_invalid_non_ascii_ratio_max)
        self._node_match_tolerance_m = _float_spin(
            cfg.ngii.sanity.node_match_tolerance_m,
            minimum=0.0,
            maximum=1_000.0,
            decimals=3,
            step=0.05,
        )
        form.addRow("Node match tolerance (m)", self._node_match_tolerance_m)
        self._direction_parallel_dot_min = _float_spin(
            cfg.ngii.sanity.direction_parallel_dot_min,
            minimum=-1.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
        )
        form.addRow("Direction parallel dot min", self._direction_parallel_dot_min)
        self._link_min_length_m = _float_spin(
            cfg.ngii.sanity.link_min_length_m,
            minimum=0.0,
            maximum=1_000.0,
            decimals=3,
            step=0.05,
        )
        form.addRow("Link minimum length (m)", self._link_min_length_m)
        return group

    def _build_sanity_group(self, sanity_cfg: NGIISanityConfig) -> QGroupBox:
        group = QGroupBox("Sanity checks")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(group)
        table = QTableWidget(len(SANITY_CHECK_NAMES), 3)
        table.setHorizontalHeaderLabels(("Check", "Max", "Mode"))
        table.verticalHeader().setVisible(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for row, name in enumerate(SANITY_CHECK_NAMES):
            allow_repair = name not in WARNING_ONLY_SANITY_CHECKS
            tooltip = _sanity_tooltip(name, allow_repair=allow_repair)
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(tooltip)
            table.setItem(row, 0, name_item)
            max_item = QTableWidgetItem("Repair" if allow_repair else "Warn")
            max_item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            max_item.setToolTip(tooltip)
            table.setItem(row, 1, max_item)
            combo = self._sanity_combo(
                getattr(sanity_cfg.checks, name),
                allow_repair=allow_repair,
            )
            combo.setToolTip(tooltip)
            table.setCellWidget(row, 2, combo)
            self._sanity_modes[name] = combo
        table.resizeRowsToContents()
        layout.addWidget(table, stretch=1)
        return group

    def _build_segmentation_page(self, cfg: RuntimeConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("Segmentation")
        form = QFormLayout(group)
        seg = cfg.segmentation
        self._enable_uturn = QCheckBox()
        self._enable_uturn.setChecked(seg.enable_uturn)
        _add_help_row(
            form, "Enable U-turn", self._enable_uturn, _segmentation_tooltip("enable_uturn")
        )
        self._enable_lateral_link_group = QCheckBox()
        self._enable_lateral_link_group.setChecked(seg.enable_lateral_link_group)
        _add_help_row(
            form,
            "Enable lateral link group",
            self._enable_lateral_link_group,
            _segmentation_tooltip("enable_lateral_link_group"),
        )
        self._enable_lateral_node_group = QCheckBox()
        self._enable_lateral_node_group.setChecked(seg.enable_lateral_node_group)
        _add_help_row(
            form,
            "Enable lateral node group",
            self._enable_lateral_node_group,
            _segmentation_tooltip("enable_lateral_node_group"),
        )
        self._enable_junction = QCheckBox()
        self._enable_junction.setChecked(seg.enable_junction)
        _add_help_row(
            form, "Enable junction", self._enable_junction, _segmentation_tooltip("enable_junction")
        )
        self._enable_junction_connection = QCheckBox()
        self._enable_junction_connection.setChecked(seg.enable_junction_connection)
        _add_help_row(
            form,
            "Enable junction connection",
            self._enable_junction_connection,
            _segmentation_tooltip("enable_junction_connection"),
        )
        self._enable_junction_reference = QCheckBox()
        self._enable_junction_reference.setChecked(seg.enable_junction_reference)
        _add_help_row(
            form,
            "Enable junction reference",
            self._enable_junction_reference,
            _segmentation_tooltip("enable_junction_reference"),
        )
        self._enable_junction_edge = QCheckBox()
        self._enable_junction_edge.setChecked(seg.enable_junction_edge)
        _add_help_row(
            form,
            "Enable junction edge",
            self._enable_junction_edge,
            _segmentation_tooltip("enable_junction_edge"),
        )
        self._z_intersection_tol_m = _float_spin(
            seg.z_intersection_tol_m,
            minimum=0.0,
            maximum=1_000.0,
            decimals=3,
            step=0.1,
        )
        form.addRow("Z intersection tolerance (m)", self._z_intersection_tol_m)
        self._junction_connection_node_merge_dist_m = _float_spin(
            seg.junction_connection_node_merge_dist_m,
            minimum=0.0,
            maximum=10_000.0,
            decimals=3,
            step=0.5,
        )
        form.addRow(
            "Junction connection node merge distance (m)",
            self._junction_connection_node_merge_dist_m,
        )
        self._junction_connection_opposite_direction_dot_min = _float_spin(
            seg.junction_connection_opposite_direction_dot_min,
            minimum=0.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
        )
        form.addRow(
            "Junction connection opposite direction dot min",
            self._junction_connection_opposite_direction_dot_min,
        )
        self._endpoint_tangent_lookback_m = _float_spin(
            seg.endpoint_tangent_lookback_m,
            minimum=0.001,
            maximum=10_000.0,
            decimals=3,
            step=0.1,
        )
        form.addRow("Endpoint tangent lookback (m)", self._endpoint_tangent_lookback_m)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_viz_page(self, cfg: RuntimeConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_viz_colors_group(cfg.viz))
        layout.addWidget(self._build_viz_rendering_group(cfg.viz))
        layout.addWidget(self._build_viz_camera_group(cfg.viz))
        layout.addWidget(self._build_viz_layers_group(cfg.viz), stretch=1)
        return page

    def _build_viz_colors_group(self, viz: VizConfig) -> QGroupBox:
        group = QGroupBox("Colors and overlays")
        form = _compact_form(group)
        self._background_color = _ColorButton(_rgb_float_to_int(viz.background_color))
        self._highlight_rgb = _ColorButton(viz.highlight_rgb)
        self._junction_connection_node_point_size = _float_spin(
            viz.junction_connection_node_point_size, 0.001, 1_000.0, 3, 0.5
        )
        self._junction_reference_arrow_length_m = _float_spin(
            viz.junction_reference_arrow_length_m, 0.001, 10_000.0, 3, 1.0
        )
        self._junction_reference_arrow_rgb = _ColorButton(viz.junction_reference_arrow_rgb)
        self._junction_edge_rgb = _ColorButton(viz.junction_edge_rgb)
        self._junction_edge_line_width = _float_spin(
            viz.junction_edge_line_width, 0.001, 1_000.0, 3, 0.5
        )
        form.addRow("Background", self._background_color)
        form.addRow("Highlight", self._highlight_rgb)
        form.addRow("Junction node size", self._junction_connection_node_point_size)
        form.addRow("Reference arrow length", self._junction_reference_arrow_length_m)
        form.addRow("Reference arrow color", self._junction_reference_arrow_rgb)
        form.addRow("Junction edge color", self._junction_edge_rgb)
        form.addRow("Junction edge width", self._junction_edge_line_width)
        return group

    def _build_viz_rendering_group(self, viz: VizConfig) -> QGroupBox:
        group = QGroupBox("Selection and rendering")
        form = _compact_form(group)
        self._selector_tol_point = _float_spin(viz.selector_tol_point, 0.0, 1.0, 5, 0.001)
        self._selector_tol_line = _float_spin(viz.selector_tol_line, 0.0, 1.0, 5, 0.001)
        self._selector_tol_poly = _float_spin(viz.selector_tol_poly, 0.0, 1.0, 5, 0.001)
        self._point_hit_radius_px = _float_spin(viz.point_hit_radius_px, 0.001, 1_000.0, 3, 1.0)
        self._poly_depth_offset_factor = _float_spin(
            viz.poly_depth_offset_factor, -10_000.0, 10_000.0, 3, 0.5
        )
        self._poly_depth_offset_units = _float_spin(
            viz.poly_depth_offset_units, -10_000.0, 10_000.0, 3, 0.5
        )
        self._segmentation_seed = _int_spin(viz.segmentation_seed, -1_000_000_000, 1_000_000_000)
        form.addRow("Point selector tol", self._selector_tol_point)
        form.addRow("Line selector tol", self._selector_tol_line)
        form.addRow("Polygon selector tol", self._selector_tol_poly)
        form.addRow("Point hit radius", self._point_hit_radius_px)
        form.addRow("Polygon offset factor", self._poly_depth_offset_factor)
        form.addRow("Polygon offset units", self._poly_depth_offset_units)
        form.addRow("Segmentation seed", self._segmentation_seed)
        return group

    def _build_viz_camera_group(self, viz: VizConfig) -> QGroupBox:
        group = QGroupBox("Camera and profiling")
        form = _compact_form(group)
        self._camera_padding_m = _float_spin(viz.camera_focus.padding_m, 0.0, 10_000.0, 3, 1.0)
        self._camera_min_scale_m = _float_spin(
            viz.camera_focus.min_scale_m, 0.001, 1_000_000.0, 3, 1.0
        )
        self._camera_max_scale_m = _float_spin(
            viz.camera_focus.max_scale_m, 0.001, 1_000_000.0, 3, 10.0
        )
        self._profiling_enabled = _checked(viz.profiling.enabled)
        self._profiling_slow_frame_ms = _float_spin(
            viz.profiling.slow_frame_ms, 0.001, 60_000.0, 3, 1.0
        )
        self._profiling_log_every_n_interactions = _int_spin(
            viz.profiling.log_every_n_interactions, 1, 1_000_000
        )
        form.addRow("Padding", self._camera_padding_m)
        form.addRow("Min scale", self._camera_min_scale_m)
        form.addRow("Max scale", self._camera_max_scale_m)
        form.addRow("Profiling", self._profiling_enabled)
        form.addRow("Slow frame", self._profiling_slow_frame_ms)
        form.addRow("Log every", self._profiling_log_every_n_interactions)
        return group

    def _build_viz_layers_group(self, viz: VizConfig) -> QGroupBox:
        group = QGroupBox("Layers")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group.setMinimumHeight(420)
        layout = QVBoxLayout(group)
        table = QTableWidget(len(viz.layers), 6)
        table.setMinimumHeight(340)
        table.setHorizontalHeaderLabels(("Layer", "On", "Color", "Point", "Line", "Opacity"))
        table.verticalHeader().setVisible(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, 42)
        table.setColumnWidth(2, 54)
        table.setColumnWidth(3, 88)
        table.setColumnWidth(4, 88)
        table.setColumnWidth(5, 88)
        for row, (attr, layer) in enumerate(sorted(viz.layers.items())):
            table.setItem(row, 0, QTableWidgetItem(attr))
            visible = QCheckBox()
            visible.setChecked(layer.visible)
            color = _ColorButton(layer.rgb)
            point_size = _float_spin(layer.point_size, 0.001, 1_000.0, 3, 0.5, width=80)
            line_width = _float_spin(layer.line_width, 0.001, 1_000.0, 3, 0.5, width=80)
            opacity = _float_spin(layer.opacity, 0.0, 1.0, 3, 0.05, width=80)
            table.setCellWidget(row, 1, _centered(visible))
            table.setCellWidget(row, 2, _centered(color))
            table.setCellWidget(row, 3, point_size)
            table.setCellWidget(row, 4, line_width)
            table.setCellWidget(row, 5, opacity)
            self._layer_controls[attr] = _LayerControls(
                visible=visible,
                color=color,
                point_size=point_size,
                line_width=line_width,
                opacity=opacity,
            )
        table.resizeRowsToContents()
        layout.addWidget(table, stretch=1)
        return group

    def _sanity_combo(self, current: SanityCheckMode, *, allow_repair: bool) -> QComboBox:
        combo = _DownwardComboBox()
        combo.setFixedWidth(96)
        combo.addItem("Disabled", None)
        combo.addItem("Warn", SanityMode.WARN.value)
        if allow_repair:
            combo.addItem("Repair", SanityMode.REPAIR.value)
        for index in range(combo.count()):
            if combo.itemData(index) == (None if current is None else current.value):
                combo.setCurrentIndex(index)
                break
        return combo

    def _apply_requested(self) -> None:
        self._apply()

    def _accept_requested(self) -> None:
        if self._apply():
            self.accept()

    def _apply(self) -> bool:
        try:
            candidate = self._collect_runtime_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return False
        if not self._on_apply(candidate):
            return False
        self._runtime_cfg = candidate
        return True

    def _collect_runtime_config(self) -> RuntimeConfig:
        cfg = RuntimeConfig(
            coordinate=self._coordinate.text().strip(),
            ngii=self._collect_ngii_config(),
            segmentation=self._collect_segmentation_config(),
            viz=self._collect_viz_config(),
        )
        validate_runtime_config(cfg)
        return cfg

    def _collect_ngii_config(self) -> NGIIConfig:
        return NGIIConfig(
            geometry=NGIIGeometryConfig(
                multipart_snap_tolerance_m=self._multipart_snap_tolerance_m.value()
            ),
            encoding=NGIIEncodingConfig(
                utf8_dbf_invalid_non_ascii_ratio_max=(
                    self._utf8_dbf_invalid_non_ascii_ratio_max.value()
                )
            ),
            sanity=NGIISanityConfig(
                node_match_tolerance_m=self._node_match_tolerance_m.value(),
                direction_parallel_dot_min=self._direction_parallel_dot_min.value(),
                link_min_length_m=self._link_min_length_m.value(),
                checks=make_sanity_checks_config(
                    {name: _mode_from_combo(combo) for name, combo in self._sanity_modes.items()}
                ),
            ),
        )

    def _collect_segmentation_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            z_intersection_tol_m=self._z_intersection_tol_m.value(),
            junction_connection_node_merge_dist_m=(
                self._junction_connection_node_merge_dist_m.value()
            ),
            junction_connection_opposite_direction_dot_min=(
                self._junction_connection_opposite_direction_dot_min.value()
            ),
            endpoint_tangent_lookback_m=self._endpoint_tangent_lookback_m.value(),
            enable_uturn=self._enable_uturn.isChecked(),
            enable_lateral_link_group=self._enable_lateral_link_group.isChecked(),
            enable_lateral_node_group=self._enable_lateral_node_group.isChecked(),
            enable_junction=self._enable_junction.isChecked(),
            enable_junction_connection=self._enable_junction_connection.isChecked(),
            enable_junction_reference=self._enable_junction_reference.isChecked(),
            enable_junction_edge=self._enable_junction_edge.isChecked(),
        )

    def _collect_viz_config(self) -> VizConfig:
        return VizConfig(
            background_color=_rgb_int_to_float(self._background_color.color_rgb()),
            highlight_rgb=self._highlight_rgb.color_rgb(),
            junction_connection_node_point_size=self._junction_connection_node_point_size.value(),
            junction_reference_arrow_length_m=self._junction_reference_arrow_length_m.value(),
            junction_reference_arrow_rgb=self._junction_reference_arrow_rgb.color_rgb(),
            junction_edge_rgb=self._junction_edge_rgb.color_rgb(),
            junction_edge_line_width=self._junction_edge_line_width.value(),
            selector_tol_point=self._selector_tol_point.value(),
            selector_tol_line=self._selector_tol_line.value(),
            selector_tol_poly=self._selector_tol_poly.value(),
            point_hit_radius_px=self._point_hit_radius_px.value(),
            poly_depth_offset_factor=self._poly_depth_offset_factor.value(),
            poly_depth_offset_units=self._poly_depth_offset_units.value(),
            segmentation_seed=self._segmentation_seed.value(),
            camera_focus=VizCameraFocusConfig(
                padding_m=self._camera_padding_m.value(),
                min_scale_m=self._camera_min_scale_m.value(),
                max_scale_m=self._camera_max_scale_m.value(),
            ),
            profiling=ViewportProfilingConfig(
                enabled=self._profiling_enabled.isChecked(),
                slow_frame_ms=self._profiling_slow_frame_ms.value(),
                log_every_n_interactions=self._profiling_log_every_n_interactions.value(),
            ),
            layers={
                attr: VizLayerConfig(
                    visible=controls.visible.isChecked(),
                    rgb=controls.color.color_rgb(),
                    point_size=controls.point_size.value(),
                    line_width=controls.line_width.value(),
                    opacity=controls.opacity.value(),
                )
                for attr, controls in self._layer_controls.items()
            },
        )

    def _scroll_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll


def _float_spin(
    value: float,
    minimum: float,
    maximum: float,
    decimals: int,
    step: float,
    *,
    width: int = 112,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setFixedWidth(width)
    return spin


def _int_spin(value: int, minimum: int, maximum: int, *, width: int = 112) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setFixedWidth(width)
    return spin


def _checked(value: bool) -> QCheckBox:
    checkbox = QCheckBox()
    checkbox.setChecked(value)
    return checkbox


def _add_help_row(
    form: QFormLayout,
    label_text: str,
    field: QWidget,
    tooltip: str,
) -> None:
    label = QLabel(label_text)
    label.setToolTip(tooltip)
    field.setToolTip(tooltip)
    form.addRow(label, field)


def _centered(widget: QWidget) -> QWidget:
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)
    return wrapper


def _mode_from_combo(combo: QComboBox) -> SanityCheckMode:
    raw = combo.currentData()
    if raw is None:
        return None
    return SanityMode(str(raw))


def _compact_form(parent: QWidget) -> QFormLayout:
    form = QFormLayout(parent)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return form


def _sanity_tooltip(name: str, *, allow_repair: bool) -> str:
    try:
        trigger, repair = _SANITY_HELP[name]
    except KeyError:
        msg = f"missing GUI sanity help metadata for {name}"
        raise RuntimeError(msg) from None
    if not allow_repair:
        repair = "Not available. This check can only warn."
    return f"Trigger: {trigger}\nRepair: {repair}"


def _segmentation_tooltip(name: str) -> str:
    try:
        return _SEGMENTATION_HELP[name]
    except KeyError:
        msg = f"missing GUI segmentation help metadata for {name}"
        raise RuntimeError(msg) from None


def _rgb_float_to_int(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return (
        round(rgb[0] * 255.0),
        round(rgb[1] * 255.0),
        round(rgb[2] * 255.0),
    )


def _rgb_int_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
