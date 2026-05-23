"""Qt inspector window that hosts :class:`HdMapViz`.

The window starts empty: select a section directory via **File → Open SHP
folder…** (Ctrl+O) and the scene is built against it. Opening another
folder tears down the previous viz and rebuilds. The window itself never
closes the underlying ``QtInteractor``, so the plotter persists across
reloads.

The right-side dock has three sections:

* **Data layers** — one checkbox per NGII layer (A1 / A2 / A3 / A4 / B2 /
  C3), toggling base-actor visibility.
* **Segmentation level** — raw A2 coloring plus one cumulative level per
  code-registered segmentation stage.
* **Selected** — a header line naming the selected layer + ID, and a key/value
  table populated from the data class. Only one feature can be selected at a
  time, so one panel is sufficient.

Dock widgets stay disabled until the first successful load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from shp2xodr.shp.data import A1Data, A2Data, A3Data, A4Data, B2Data, C3Data
from shp2xodr.shp.segmentation import (
    SegmentationConfig,
    SelectedField,
    SelectedSection,
    segmentation_level_labels,
)
from shp2xodr.shp.viz import HdMapViz, VizConfig

log = logging.getLogger(__name__)


# Layer code → side-dock checkbox label. Order here drives the data
# checkboxes top-down so the user sees the NGII layer codes in order.
_LAYERS: tuple[tuple[str, str], ...] = (
    ("A1", "A1  nodes"),
    ("A2", "A2  links"),
    ("A3", "A3  driveway sections"),
    ("A4", "A4  subsidiary sections"),
    ("B2", "B2  surface line marks"),
    ("C3", "C3  safety fixtures"),
)

# Layer code → header text shown above the selected-fields table.
_LAYER_TITLE: dict[str, str] = {
    "A1": "A1 Node",
    "A2": "A2 Link",
    "A3": "A3 Driveway Section",
    "A4": "A4 Subsidiary Section",
    "B2": "B2 Surface Line Mark",
    "C3": "C3 Vehicle Protection Safety",
}

_SELECT_PLACEHOLDER = "— open a folder to begin —"
_SELECT_PROMPT = "— shift-click a feature to inspect —"
_SECTION_BG = QColor(230, 230, 235)


@dataclass(slots=True, frozen=True)
class _SelectedTableRow:
    field: str
    value: str
    is_section: bool = False


def _coded(value: str, table: dict[str, str]) -> str:
    """Format a coded field as ``"<code> (<label>)"`` or ``"-"`` if empty."""
    if not value:
        return "-"
    label = table.get(value)
    return f"{value} ({label})" if label else value


def _opt(value: str) -> str:
    """Empty-string sentinel → ``"-"``."""
    return value if value else "-"


def _field(name: str, value: str | int | float) -> SelectedField:
    return SelectedField.scalar(name, value)


def _raw_section(rows: tuple[SelectedField, ...]) -> SelectedSection:
    return SelectedSection("Raw", rows)


def _flatten_selected_sections(sections: tuple[SelectedSection, ...]) -> list[_SelectedTableRow]:
    rows: list[_SelectedTableRow] = []
    for section in sections:
        rows.append(_SelectedTableRow(section.title, "", is_section=True))
        for field in section.rows:
            values = field.values if field.values else ("-",)
            for value in values:
                rows.append(_SelectedTableRow(field.name, value))
    return rows


class HdMapWindow(QMainWindow):
    """Main window hosting the 3D scene and the inspector dock."""

    def __init__(self, seg_cfg: SegmentationConfig, viz_cfg: VizConfig) -> None:
        super().__init__()
        self.setWindowTitle("shp2xodr — (no folder)")
        self.resize(1500, 950)

        self._seg_cfg = seg_cfg
        self._viz_cfg = viz_cfg
        self.viz: HdMapViz | None = None

        self.qt_plotter = QtInteractor(self)
        self.setCentralWidget(self.qt_plotter)
        self.qt_plotter.enable_parallel_projection()
        self.qt_plotter.view_xy()

        # Widget handles — populated in _build_dock(); read in load_folder()
        # to carry user preferences forward into a freshly-attached viz.
        self._layer_cbs: dict[str, QCheckBox] = {}
        self._segmentation_level_buttons: dict[int, QRadioButton] = {}
        self._segmentation_level = 0
        self._data_group: QGroupBox
        self._segmentation_group: QGroupBox

        self._build_menu()
        self._build_dock()
        self._set_dock_enabled(False)

    # ---- Menu construction ---------------------------------------------------

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        open_act = QAction("&Open SHP folder…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_folder)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    # ---- Dock construction ---------------------------------------------------

    def _build_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        # The dock is pinned to one side and not closable: float / X buttons
        # don't add anything useful here, so strip every dock feature.
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        self._data_group = self._build_data_group()
        self._segmentation_group = self._build_segmentation_group()
        layout.addWidget(self._data_group)
        layout.addWidget(self._segmentation_group)
        layout.addWidget(self._build_select_panel(), stretch=1)
        dock.setWidget(container)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_data_group(self) -> QGroupBox:
        gb = QGroupBox("Data layers")
        v = QVBoxLayout(gb)
        for name, label in _LAYERS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, n=name: self._on_layer_toggle(n, on))
            v.addWidget(cb)
            self._layer_cbs[name] = cb
        return gb

    def _build_segmentation_group(self) -> QGroupBox:
        gb = QGroupBox("Segmentation level")
        v = QVBoxLayout(gb)
        self._segmentation_button_group = QButtonGroup(gb)
        for level, label in enumerate(segmentation_level_labels()):
            rb = QRadioButton(f"{level}  {label}")
            rb.setChecked(level == self._segmentation_level)
            self._segmentation_button_group.addButton(rb, level)
            v.addWidget(rb)
            self._segmentation_level_buttons[level] = rb
        self._segmentation_button_group.idToggled.connect(self._on_segmentation_level_changed)
        return gb

    def _build_select_panel(self) -> QGroupBox:
        gb = QGroupBox("Selected")
        v = QVBoxLayout(gb)
        self._select_header = QLabel(_SELECT_PLACEHOLDER)
        self._select_header.setStyleSheet("font-weight: bold;")
        self._select_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._select_header)

        self._select_table = QTableWidget(0, 2)
        self._select_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._select_table.verticalHeader().setVisible(False)
        self._select_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._select_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._select_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._select_table.setShowGrid(True)
        self._select_table.setAlternatingRowColors(True)
        header = self._select_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self._select_table, stretch=1)
        return gb

    def _set_dock_enabled(self, on: bool) -> None:
        self._data_group.setEnabled(on)
        self._segmentation_group.setEnabled(on)

    # ---- Dock event handlers (no-op when no viz) -----------------------------

    def _on_layer_toggle(self, name: str, on: bool) -> None:
        if self.viz is not None:
            self.viz.set_layer_visible(name, on)

    def _on_segmentation_level_changed(self, level: int, checked: bool) -> None:
        if not checked:
            return
        self._segmentation_level = level
        if self.viz is not None:
            self.viz.set_segmentation_level(level)

    # ---- Open-folder flow ----------------------------------------------------

    def _open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open NGII SHP section folder")
        if not d:
            return
        self.load_folder(Path(d))

    def load_folder(self, shp_dir: Path) -> None:
        """Tear down any existing viz, build a fresh one against ``shp_dir``,
        and sync the dock-toggle states into the new viz.

        Wraps the constructor (which performs I/O) in a single try/except
        scoped to the two specific exceptions that mean "user selected the
        wrong folder" — a real I/O boundary with a meaningful recovery.
        """
        if self.viz is not None:
            self.viz.detach()
            self.viz = None

        try:
            viz = HdMapViz(
                shp_dir,
                seg_cfg=self._seg_cfg,
                viz_cfg=self._viz_cfg,
                plotter=self.qt_plotter,
                on_select=self._on_select,
            )
            viz.attach()
        except (FileNotFoundError, NotADirectoryError) as e:
            log.warning("cannot open %s: %s", shp_dir, e)
            QMessageBox.warning(
                self,
                "Cannot open folder",
                f"Not a valid NGII section folder:\n{shp_dir}\n\n{e}",
            )
            self.setWindowTitle("shp2xodr — (no folder)")
            self._set_dock_enabled(False)
            self._reset_select_panel(_SELECT_PLACEHOLDER)
            self.qt_plotter.render()
            return

        self.viz = viz
        # Carry layer visibility forward; a fresh viz defaults every layer on.
        for name, cb in self._layer_cbs.items():
            viz.set_layer_visible(name, cb.isChecked())
        viz.set_segmentation_level(self._segmentation_level)

        self.qt_plotter.view_xy()
        self.qt_plotter.reset_camera()
        self.setWindowTitle(f"shp2xodr — {shp_dir.name}")
        self._set_dock_enabled(True)
        self._reset_select_panel(_SELECT_PROMPT)
        log.info("loaded %s", shp_dir)

    def _reset_select_panel(self, header_text: str) -> None:
        self._select_header.setText(header_text)
        self._select_table.clearSpans()
        self._select_table.clearContents()
        self._select_table.setRowCount(0)

    # ---- Select → table dispatch -----------------------------------------------

    def _on_select(self, kind: str, idx: int) -> None:
        title = _LAYER_TITLE.get(kind)
        if title is None:
            log.warning("select from unknown layer kind: %s", kind)
            return
        sections = self._fields_for(kind, idx)
        # The first row is always ID, so use it for the header line.
        feature_id = sections[0].rows[0].values[0] if sections and sections[0].rows else ""
        self._select_header.setText(f"{title}    {feature_id}")
        table_rows = _flatten_selected_sections(sections)
        self._select_table.clearSpans()
        self._select_table.clearContents()
        self._select_table.setRowCount(len(table_rows))
        for r, row in enumerate(table_rows):
            if row.is_section:
                item = QTableWidgetItem(row.field)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QBrush(_SECTION_BG))
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self._select_table.setItem(r, 0, item)
                self._select_table.setSpan(r, 0, 1, 2)
            else:
                self._select_table.setItem(r, 0, QTableWidgetItem(row.field))
                self._select_table.setItem(r, 1, QTableWidgetItem(row.value))
        log.info("selected %s[%d]", kind, idx)

    def _fields_for(self, kind: str, idx: int) -> tuple[SelectedSection, ...]:
        """Single boundary that narrows ``self.viz``: dispatch field-builders
        with the resolved :class:`HdMapViz` so each builder takes it as a
        non-Optional argument.

        Returns ``[]`` when no section is loaded — the dock is disabled in
        that state, so this is a defensive no-op for any stray callback.
        """
        viz = self.viz
        if viz is None:
            return ()
        fn = {
            "A1": self._fields_a1,
            "A2": self._fields_a2,
            "A3": self._fields_a3,
            "A4": self._fields_a4,
            "B2": self._fields_b2,
            "C3": self._fields_c3,
        }.get(kind)
        return fn(viz, idx) if fn is not None else ()

    def _fields_a1(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.a1
        x, y, z = d.points[idx]
        return (
            _raw_section(
                (
                    _field("ID", str(d.ids[idx])),
                    _field("NodeType", _coded(d.node_types[idx], A1Data.NODE_TYPE_LABEL)),
                    _field("ITS NodeID", _opt(d.its_node_ids[idx])),
                    _field("X (m)", f"{float(x):.3f}"),
                    _field("Y (m)", f"{float(y):.3f}"),
                    _field("Z (m)", f"{float(z):.3f}"),
                )
            ),
        )

    def _fields_a2(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.a2
        raw = _raw_section(
            (
                _field("ID", str(d.ids[idx])),
                _field("RoadRank", _coded(d.road_ranks[idx], A2Data.ROAD_RANK_LABEL)),
                _field("RoadType", _coded(d.road_types[idx], A2Data.ROAD_TYPE_LABEL)),
                _field("RoadNo", _opt(d.road_nos[idx])),
                _field("LinkType", _coded(d.link_types[idx], A2Data.LINK_TYPE_LABEL)),
                _field("LaneNo", str(int(d.lane_nos[idx]))),
                _field("FromNode", str(d.from_node_ids[idx])),
                _field("ToNode", str(d.to_node_ids[idx])),
                _field("R_LinkID", _opt(d.r_link_ids[idx])),
                _field("L_LinkID", _opt(d.l_link_ids[idx])),
                _field("SectionID", _opt(d.section_ids[idx])),
                _field("Length (m)", f"{float(d.lengths_m[idx]):.2f}"),
                _field("ITS_LinkID", _opt(d.its_link_ids[idx])),
            )
        )
        segmentation_rows = viz.segmentation.selected_fields_for_link(idx)
        if not segmentation_rows:
            return (raw,)
        return (raw, SelectedSection("Segmentation", segmentation_rows))

    def _fields_a3(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.a3
        if d is None:
            msg = "A3 layer not loaded"
            raise RuntimeError(msg)
        return (
            _raw_section(
                (
                    _field("ID", str(d.ids[idx])),
                    _field("Kind", _coded(d.kinds[idx], A3Data.KIND_LABEL)),
                    _field("RoadType", _coded(d.road_types[idx], A3Data.ROAD_TYPE_LABEL)),
                    _field("Remark", _opt(d.remarks[idx])),
                )
            ),
        )

    def _fields_a4(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.a4
        if d is None:
            msg = "A4 layer not loaded"
            raise RuntimeError(msg)
        return (
            _raw_section(
                (
                    _field("ID", str(d.ids[idx])),
                    _field("SubType", _coded(d.subtypes[idx], A4Data.SUBTYPE_LABEL)),
                    _field("Name", _opt(d.names[idx])),
                    _field("Direction", _opt(d.directions[idx])),
                    _field("GasStation", _opt(d.gas_stations[idx])),
                    _field("LPGStation", _opt(d.lpg_stations[idx])),
                    _field("EVCharger", _opt(d.ev_chargers[idx])),
                    _field("Toilet", _opt(d.toilets[idx])),
                )
            ),
        )

    def _fields_b2(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.b2
        type_code = str(d.types[idx])
        color_label = B2Data.TYPE_COLOR_LABEL.get(type_code[:1], "")
        type_text = f"{type_code} ({color_label})" if color_label else type_code
        return (
            _raw_section(
                (
                    _field("ID", str(d.ids[idx])),
                    _field("Type", type_text),
                    _field("Kind", _coded(d.kinds[idx], B2Data.KIND_LABEL)),
                    _field("R_LinkID", _opt(d.r_link_ids[idx])),
                    _field("L_LinkID", _opt(d.l_link_ids[idx])),
                )
            ),
        )

    def _fields_c3(self, viz: HdMapViz, idx: int) -> tuple[SelectedSection, ...]:
        d = viz.c3
        return (
            _raw_section(
                (
                    _field("ID", str(d.ids[idx])),
                    _field("Type", _coded(d.types[idx], C3Data.TYPE_LABEL)),
                    _field("IsCentral", _coded(d.is_central[idx], C3Data.IS_CENTRAL_LABEL)),
                    _field("LowHigh", _coded(d.low_high[idx], C3Data.LOW_HIGH_LABEL)),
                    _field("Ref_ID", _opt(d.ref_ids[idx])),
                )
            ),
        )

    # ---- Qt lifecycle ---------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 (Qt API)
        if self.viz is not None:
            self.viz.detach()
            self.viz = None
        self.qt_plotter.close()
        super().closeEvent(event)  # type: ignore[arg-type]
