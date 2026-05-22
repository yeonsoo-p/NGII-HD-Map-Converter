"""Qt inspector window that hosts :class:`HdMapViz`.

The window starts empty: pick a section directory via **File → Open SHP
folder…** (Ctrl+O) and the scene is built against it. Opening another
folder tears down the previous viz and rebuilds. The window itself never
closes the underlying ``QtInteractor``, so the plotter persists across
reloads.

The right-side dock has three sections:

* **Data layers** — one checkbox per NGII layer (A1 / A2 / A3 / A4 / B2 /
  C3), toggling base-actor visibility.
* **Abstraction** — three radio buttons selecting the A2 / B2 coloring
  level: ``1`` None, ``2`` Group, ``3`` Road & Junction. C3 / A3 / A4 /
  A1 colors stay on their natural NGII codes at every level.
* **Picked** — a header line naming the picked layer + ID, and a key/value
  table populated from the data class and segmentation results. Only one
  feature can be picked at a time, so one panel is sufficient.

Dock widgets stay disabled until the first successful load.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
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
from shp2xodr.shp.segmentation import SegmentationConfig
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

# Layer code → header text shown above the picked-fields table.
_LAYER_TITLE: dict[str, str] = {
    "A1": "A1 Node",
    "A2": "A2 Link",
    "A3": "A3 Driveway Section",
    "A4": "A4 Subsidiary Section",
    "B2": "B2 Surface Line Mark",
    "C3": "C3 Vehicle Protection Safety",
}

_PICK_PLACEHOLDER = "— open a folder to begin —"
_PICK_PROMPT = "— shift-click a feature to inspect —"


def _coded(value: str, table: dict[str, str]) -> str:
    """Format a coded field as ``"<code> (<label>)"`` or ``"-"`` if empty."""
    if not value:
        return "-"
    label = table.get(value)
    return f"{value} ({label})" if label else value


def _opt(value: str) -> str:
    """Empty-string sentinel → ``"-"``."""
    return value if value else "-"


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
        self._level_buttons: dict[int, QRadioButton] = {}
        self._abstraction_level: int = viz_cfg.default_abstraction_level
        self._data_group: QGroupBox
        self._abstraction_group: QGroupBox

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
        self._abstraction_group = self._build_abstraction_group()
        layout.addWidget(self._data_group)
        layout.addWidget(self._abstraction_group)
        layout.addWidget(self._build_pick_panel(), stretch=1)
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

    def _build_abstraction_group(self) -> QGroupBox:
        gb = QGroupBox("Abstraction")
        v = QVBoxLayout(gb)
        # QButtonGroup is mutually-exclusive by default and owns the buttons'
        # ID assignment; we use level ints 1-3 directly as the button IDs.
        self._level_button_group = QButtonGroup(gb)
        for level, label in (
            (1, "1  None\t(raw layer colors)"),
            (2, "2  Group\t(A2 / B2 colored by SHP group)"),
            (3, "3  Road & Junction\t(A2 / B2 by OpenDRIVE entity)"),
        ):
            rb = QRadioButton(label)
            rb.setChecked(level == self._abstraction_level)
            self._level_button_group.addButton(rb, level)
            v.addWidget(rb)
            self._level_buttons[level] = rb
        self._level_button_group.idToggled.connect(self._on_abstraction_changed)
        return gb

    def _build_pick_panel(self) -> QGroupBox:
        gb = QGroupBox("Picked")
        v = QVBoxLayout(gb)
        self._pick_header = QLabel(_PICK_PLACEHOLDER)
        self._pick_header.setStyleSheet("font-weight: bold;")
        self._pick_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._pick_header)

        self._pick_table = QTableWidget(0, 2)
        self._pick_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._pick_table.verticalHeader().setVisible(False)
        self._pick_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._pick_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._pick_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._pick_table.setShowGrid(True)
        self._pick_table.setAlternatingRowColors(True)
        header = self._pick_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self._pick_table, stretch=1)
        return gb

    def _set_dock_enabled(self, on: bool) -> None:
        self._data_group.setEnabled(on)
        self._abstraction_group.setEnabled(on)

    # ---- Dock event handlers (no-op when no viz) -----------------------------

    def _on_layer_toggle(self, name: str, on: bool) -> None:
        if self.viz is not None:
            self.viz.set_layer_visible(name, on)

    def _on_abstraction_changed(self, level: int, checked: bool) -> None:
        # idToggled fires twice per click — once for the deselected button
        # (checked=False) and once for the newly selected one (checked=True).
        # We only care about the selection edge.
        if not checked:
            return
        self._abstraction_level = level
        if self.viz is not None:
            self.viz.set_abstraction_level(level)

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
        scoped to the two specific exceptions that mean "user picked the
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
                on_pick=self._on_pick,
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
            self._reset_pick_panel(_PICK_PLACEHOLDER)
            self.qt_plotter.render()
            return

        self.viz = viz
        # Carry dock state forward (the new viz defaults are all layers on
        # and level 3; the user may have changed either).
        for name, cb in self._layer_cbs.items():
            viz.set_layer_visible(name, cb.isChecked())
        viz.set_abstraction_level(self._abstraction_level)

        self.qt_plotter.view_xy()
        self.qt_plotter.reset_camera()
        self.setWindowTitle(f"shp2xodr — {shp_dir.name}")
        self._set_dock_enabled(True)
        self._reset_pick_panel(_PICK_PROMPT)
        log.info("loaded %s", shp_dir)

    def _reset_pick_panel(self, header_text: str) -> None:
        self._pick_header.setText(header_text)
        self._pick_table.setRowCount(0)

    # ---- Pick → table dispatch -----------------------------------------------

    def _on_pick(self, kind: str, idx: int) -> None:
        title = _LAYER_TITLE.get(kind)
        if title is None:
            log.warning("pick from unknown layer kind: %s", kind)
            return
        fields = self._fields_for(kind, idx)
        # The first row is always ID, so use it for the header line.
        feature_id = fields[0][1] if fields else ""
        self._pick_header.setText(f"{title}    {feature_id}")
        self._pick_table.setRowCount(len(fields))
        for r, (field, value) in enumerate(fields):
            self._pick_table.setItem(r, 0, QTableWidgetItem(field))
            self._pick_table.setItem(r, 1, QTableWidgetItem(value))
        log.info("picked %s[%d]", kind, idx)

    def _fields_for(self, kind: str, idx: int) -> list[tuple[str, str]]:
        """Single boundary that narrows ``self.viz``: dispatch field-builders
        with the resolved :class:`HdMapViz` so each builder takes it as a
        non-Optional argument.

        Returns ``[]`` when no section is loaded — the dock is disabled in
        that state, so this is a defensive no-op for any stray callback.
        """
        viz = self.viz
        if viz is None:
            return []
        fn = {
            "A1": self._fields_a1,
            "A2": self._fields_a2,
            "A3": self._fields_a3,
            "A4": self._fields_a4,
            "B2": self._fields_b2,
            "C3": self._fields_c3,
        }.get(kind)
        return fn(viz, idx) if fn is not None else []

    def _fields_a1(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.a1
        seg = viz.segmentation
        x, y, z = d.points[idx]
        jid = int(seg.node_junction_id[idx])
        return [
            ("ID", str(d.ids[idx])),
            ("NodeType", _coded(d.node_types[idx], A1Data.NODE_TYPE_LABEL)),
            ("ITS NodeID", _opt(d.its_node_ids[idx])),
            ("X (m)", f"{float(x):.3f}"),
            ("Y (m)", f"{float(y):.3f}"),
            ("Z (m)", f"{float(z):.3f}"),
            ("Junction", str(jid) if jid >= 0 else "-"),
        ]

    def _fields_a2(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.a2
        seg = viz.segmentation
        jid = int(seg.junction_id[idx])
        return [
            ("ID", str(d.ids[idx])),
            ("RoadRank", _coded(d.road_ranks[idx], A2Data.ROAD_RANK_LABEL)),
            ("RoadType", _coded(d.road_types[idx], A2Data.ROAD_TYPE_LABEL)),
            ("RoadNo", _opt(d.road_nos[idx])),
            ("LinkType", _coded(d.link_types[idx], A2Data.LINK_TYPE_LABEL)),
            ("LaneNo", str(int(d.lane_nos[idx]))),
            ("FromNode", str(d.from_node_ids[idx])),
            ("ToNode", str(d.to_node_ids[idx])),
            ("R_LinkID", _opt(d.r_link_ids[idx])),
            ("L_LinkID", _opt(d.l_link_ids[idx])),
            ("SectionID", _opt(d.section_ids[idx])),
            ("Length (m)", f"{float(d.lengths_m[idx]):.2f}"),
            ("ITS_LinkID", _opt(d.its_link_ids[idx])),
            ("Group", str(int(seg.group_id[idx]))),
            ("Junction", str(jid) if jid >= 0 else "-"),
        ]

    def _fields_a3(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.a3
        if d is None:
            msg = "A3 layer not loaded"
            raise RuntimeError(msg)
        return [
            ("ID", str(d.ids[idx])),
            ("Kind", _coded(d.kinds[idx], A3Data.KIND_LABEL)),
            ("RoadType", _coded(d.road_types[idx], A3Data.ROAD_TYPE_LABEL)),
            ("Remark", _opt(d.remarks[idx])),
        ]

    def _fields_a4(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.a4
        if d is None:
            msg = "A4 layer not loaded"
            raise RuntimeError(msg)
        return [
            ("ID", str(d.ids[idx])),
            ("SubType", _coded(d.subtypes[idx], A4Data.SUBTYPE_LABEL)),
            ("Name", _opt(d.names[idx])),
            ("Direction", _opt(d.directions[idx])),
            ("GasStation", _opt(d.gas_stations[idx])),
            ("LPGStation", _opt(d.lpg_stations[idx])),
            ("EVCharger", _opt(d.ev_chargers[idx])),
            ("Toilet", _opt(d.toilets[idx])),
        ]

    def _fields_b2(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.b2
        seg = viz.segmentation
        type_code = str(d.types[idx])
        color_label = B2Data.TYPE_COLOR_LABEL.get(type_code[:1], "")
        type_text = f"{type_code} ({color_label})" if color_label else type_code
        r_b = int(seg.b2_r_group[idx])
        l_b = int(seg.b2_l_group[idx])
        return [
            ("ID", str(d.ids[idx])),
            ("Type", type_text),
            ("Kind", _coded(d.kinds[idx], B2Data.KIND_LABEL)),
            ("R_LinkID", _opt(d.r_link_ids[idx])),
            ("L_LinkID", _opt(d.l_link_ids[idx])),
            ("R Group", str(r_b) if r_b >= 0 else "-"),
            ("L Group", str(l_b) if l_b >= 0 else "-"),
        ]

    def _fields_c3(self, viz: HdMapViz, idx: int) -> list[tuple[str, str]]:
        d = viz.c3
        if d is None:
            msg = "C3 layer not loaded"
            raise RuntimeError(msg)
        return [
            ("ID", str(d.ids[idx])),
            ("Type", _coded(d.types[idx], C3Data.TYPE_LABEL)),
            ("IsCentral", _coded(d.is_central[idx], C3Data.IS_CENTRAL_LABEL)),
            ("LowHigh", _coded(d.low_high[idx], C3Data.LOW_HIGH_LABEL)),
            ("Ref_ID", _opt(d.ref_ids[idx])),
        ]

    # ---- Qt lifecycle ---------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 (Qt API)
        if self.viz is not None:
            self.viz.detach()
            self.viz = None
        self.qt_plotter.close()
        super().closeEvent(event)  # type: ignore[arg-type]
