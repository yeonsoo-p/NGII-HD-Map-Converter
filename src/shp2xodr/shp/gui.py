"""Qt inspector window that hosts :class:`HdMapViz`.

The window provides a side dock with three sections:

* **Data layers** — one checkbox per NGII layer (A1 / A2 / A3 / A4 / B2 /
  C3), toggling base-actor visibility.
* **Abstractions** — toggles for segmentation-derived overlays: the A2
  bundle palette, and the junction-hulls overlay.
* **Picked** — a header line naming the picked layer + ID, and a key/value
  table populated from the data class and segmentation results. Only one
  feature can be picked at a time, so one panel is sufficient.

The viz layer is plotter-agnostic; this module supplies a
``pyvistaqt.QtInteractor`` as the plotter and an ``on_pick`` callback
that routes pick events into the panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from shp2xodr.shp.data import A1Data, A2Data, A3Data, A4Data, B2Data, C3Data
from shp2xodr.shp.viz import HdMapViz

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

    def __init__(self, shp_dir: Path, junction_merge_dist_m: float = 0.0) -> None:
        super().__init__()
        self.setWindowTitle(f"shp2xodr — {shp_dir.name}")
        self.resize(1500, 950)

        self.qt_plotter = QtInteractor(self)
        self.setCentralWidget(self.qt_plotter)

        # Build the scene against the embedded plotter, route picks back.
        self.viz = HdMapViz(
            shp_dir,
            junction_merge_dist_m=junction_merge_dist_m,
            plotter=self.qt_plotter,
            on_pick=self._on_pick,
        )
        self.viz.attach()
        self.qt_plotter.enable_parallel_projection()
        self.qt_plotter.view_xy()

        self._build_dock()

    # ---- Dock construction ----------------------------------------------------

    def _build_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        # The dock is pinned to one side and not closable: float / X buttons
        # don't add anything useful here, so strip every dock feature.
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_abstraction_group())
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
            cb.toggled.connect(lambda on, n=name: self.viz.set_layer_visible(n, on))
            v.addWidget(cb)
        return gb

    def _build_abstraction_group(self) -> QGroupBox:
        gb = QGroupBox("Abstractions")
        v = QVBoxLayout(gb)
        bp = QCheckBox("Bundle palette  (A2 colored by segmentation)")
        bp.setChecked(True)
        bp.toggled.connect(self.viz.set_bundle_palette_on)
        jh = QCheckBox("Junction hulls  (convex hull per junction)")
        jh.setChecked(False)
        jh.toggled.connect(self.viz.set_junction_hulls_visible)
        v.addWidget(bp)
        v.addWidget(jh)
        return gb

    def _build_pick_panel(self) -> QGroupBox:
        gb = QGroupBox("Picked")
        v = QVBoxLayout(gb)
        self._pick_header = QLabel("— shift-click a feature to inspect —")
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
        fn = {
            "A1": self._fields_a1,
            "A2": self._fields_a2,
            "A3": self._fields_a3,
            "A4": self._fields_a4,
            "B2": self._fields_b2,
            "C3": self._fields_c3,
        }.get(kind)
        return fn(idx) if fn is not None else []

    def _fields_a1(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.a1
        seg = self.viz.segmentation
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

    def _fields_a2(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.a2
        seg = self.viz.segmentation
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
            ("Bundle", str(int(seg.bundle_id[idx]))),
            ("Junction", str(jid) if jid >= 0 else "-"),
        ]

    def _fields_a3(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.a3
        return [
            ("ID", str(d.ids[idx])),
            ("Kind", _coded(d.kinds[idx], A3Data.KIND_LABEL)),
            ("RoadType", _coded(d.road_types[idx], A3Data.ROAD_TYPE_LABEL)),
            ("Remark", _opt(d.remarks[idx])),
        ]

    def _fields_a4(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.a4
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

    def _fields_b2(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.b2
        seg = self.viz.segmentation
        type_code = str(d.types[idx])
        color_label = B2Data.TYPE_COLOR_LABEL.get(type_code[:1], "")
        type_text = f"{type_code} ({color_label})" if color_label else type_code
        r_b = int(seg.b2_r_bundle[idx])
        l_b = int(seg.b2_l_bundle[idx])
        return [
            ("ID", str(d.ids[idx])),
            ("Type", type_text),
            ("Kind", _coded(d.kinds[idx], B2Data.KIND_LABEL)),
            ("R_LinkID", _opt(d.r_link_ids[idx])),
            ("L_LinkID", _opt(d.l_link_ids[idx])),
            ("R Bundle", str(r_b) if r_b >= 0 else "-"),
            ("L Bundle", str(l_b) if l_b >= 0 else "-"),
        ]

    def _fields_c3(self, idx: int) -> list[tuple[str, str]]:
        d = self.viz.c3
        return [
            ("ID", str(d.ids[idx])),
            ("Type", _coded(d.types[idx], C3Data.TYPE_LABEL)),
            ("IsCentral", _coded(d.is_central[idx], C3Data.IS_CENTRAL_LABEL)),
            ("LowHigh", _coded(d.low_high[idx], C3Data.LOW_HIGH_LABEL)),
            ("Ref_ID", _opt(d.ref_ids[idx])),
        ]

    # ---- Qt lifecycle ---------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 (Qt API)
        self.qt_plotter.close()
        super().closeEvent(event)  # type: ignore[arg-type]
