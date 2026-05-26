"""Cached dataset view used by segmentation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import LineFeature
from ngii2xodr.ngii.segmentation.model import SegmentationConfig


@dataclass(slots=True)
class SegmentationContext:
    dataset: NGIIDataset
    cfg: SegmentationConfig
    node_attr: str = field(init=False)
    link_attr: str = field(init=False)
    lane_line_attr: str | None = field(init=False)
    node_store: LayerStore[Any] = field(init=False)
    link_store: LayerStore[Any] = field(init=False)
    lane_line_store: LayerStore[Any] | None = field(init=False)
    link_refs: tuple[FeatureRef, ...] = field(init=False)
    lane_line_refs: tuple[FeatureRef, ...] = field(init=False)
    link_lines: tuple[shapely.LineString | None, ...] = field(init=False)
    lane_line_lines: tuple[shapely.LineString | None, ...] = field(init=False)
    link_line_rows: tuple[int, ...] = field(init=False)
    lane_line_rows: tuple[int, ...] = field(init=False)
    link_line_tree: shapely.STRtree | None = field(init=False)
    lane_line_tree: shapely.STRtree | None = field(init=False)
    node_points: tuple[shapely.Point, ...] = field(init=False)
    node_point_tree: shapely.STRtree | None = field(init=False)
    junction_node_indices: tuple[int, ...] = field(init=False)
    junction_node_points: tuple[shapely.Point, ...] = field(init=False)
    junction_node_tree: shapely.STRtree | None = field(init=False)
    _rows_by_filter: dict[str, tuple[int, ...]] = field(init=False)

    def __post_init__(self) -> None:
        self.node_attr = self.dataset.schema.attr_for_role("node")
        self.link_attr = self.dataset.schema.attr_for_role("link")
        lane_line_attrs = self.dataset.schema.attrs_for_role("lane_line")
        self.lane_line_attr = lane_line_attrs[0] if lane_line_attrs else None
        self.node_store = self.dataset.store_for_attr(self.node_attr)
        self.link_store = self.dataset.store_for_attr(self.link_attr)
        self.lane_line_store = (
            None
            if self.lane_line_attr is None
            else self.dataset.store_for_attr(self.lane_line_attr)
        )

        self.link_refs = tuple(
            FeatureRef(self.link_attr, feature.id) for feature in self.link_store.features
        )
        self.lane_line_refs = (
            ()
            if self.lane_line_store is None
            else tuple(
                FeatureRef(self.lane_line_attr or "", feature.id)
                for feature in self.lane_line_store.features
            )
        )
        self.link_lines = tuple(_line_or_none(feature) for feature in self.link_store.features)
        self.lane_line_lines = (
            ()
            if self.lane_line_store is None
            else tuple(_line_or_none(feature) for feature in self.lane_line_store.features)
        )
        self.link_line_rows = tuple(i for i, line in enumerate(self.link_lines) if line is not None)
        self.lane_line_rows = tuple(
            i for i, line in enumerate(self.lane_line_lines) if line is not None
        )
        self.link_line_tree = self._line_tree(self.link_lines, self.link_line_rows)
        self.lane_line_tree = self._line_tree(self.lane_line_lines, self.lane_line_rows)
        self.node_points = tuple(shapely.Point(node.point[:2]) for node in self.node_store.features)
        self.node_point_tree = shapely.STRtree(self.node_points) if self.node_points else None
        self._rows_by_filter = self._build_rows_by_filter()
        self.junction_node_indices = self._rows_by_filter.get("junction_node", ())
        self.junction_node_points = tuple(
            shapely.Point(self.node_store.features[i].point[:2]) for i in self.junction_node_indices
        )
        self.junction_node_tree = (
            shapely.STRtree(self.junction_node_points) if self.junction_node_points else None
        )

    def _build_rows_by_filter(self) -> dict[str, tuple[int, ...]]:
        rows_by_filter: dict[str, tuple[int, ...]] = {}
        for role_filter in self.dataset.schema.role_filters:
            try:
                store = self.dataset.store_for_role(role_filter.role)
            except KeyError:
                rows_by_filter[role_filter.name] = ()
                continue
            rows_by_filter[role_filter.name] = tuple(
                i
                for i, feature in enumerate(store.features)
                if str(getattr(feature, role_filter.attr_name, "")) in role_filter.values
            )
        return rows_by_filter

    def _line_tree(
        self,
        lines: tuple[shapely.LineString | None, ...],
        rows: tuple[int, ...],
    ) -> shapely.STRtree | None:
        geometries = [line for row in rows if (line := lines[row]) is not None]
        return shapely.STRtree(geometries) if geometries else None

    def nearest_type1_node_id(self, xy: NDArray[np.float64], tolerance_m: float) -> str | None:
        if self.junction_node_tree is None:
            return None
        point = shapely.Point(float(xy[0]), float(xy[1]))
        best_node_id: str | None = None
        best_dist = tolerance_m
        for pos_raw in self.junction_node_tree.query(point.buffer(tolerance_m)):
            pos = int(pos_raw)
            candidate = self.junction_node_points[pos]
            dist = point.distance(candidate)
            if dist <= best_dist:
                best_dist = dist
                best_node_id = self.node_store.features[self.junction_node_indices[pos]].id
        return best_node_id

    def ref_for_link_index(self, index: int) -> FeatureRef:
        return self.link_refs[index]

    def ref_for_lane_line_index(self, index: int) -> FeatureRef:
        return self.lane_line_refs[index]

    def rows_for_filter(self, name: str) -> tuple[int, ...]:
        return self._rows_by_filter.get(name, ())


def _line_or_none(feature: Any) -> shapely.LineString | None:
    if not isinstance(feature, LineFeature) or len(feature.polyline) < 2:
        return None
    return shapely.LineString(feature.polyline[:, :2])
