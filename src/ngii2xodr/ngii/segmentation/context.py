"""Cached dataset view used by segmentation stages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.data.features import LineFeature
from ngii2xodr.ngii.segmentation.model import SegmentationConfig


@dataclass(slots=True)
class SegmentationContext:
    dataset: NGIIDataset
    cfg: SegmentationConfig
    a2_refs: tuple[FeatureRef, ...] = field(init=False)
    b2_refs: tuple[FeatureRef, ...] = field(init=False)
    a2_lines: tuple[shapely.LineString | None, ...] = field(init=False)
    b2_lines: tuple[shapely.LineString | None, ...] = field(init=False)
    a2_rows_by_link_type: dict[str, tuple[int, ...]] = field(init=False)
    b2_rows_by_kind: dict[str, tuple[int, ...]] = field(init=False)
    a2_line_rows: tuple[int, ...] = field(init=False)
    b2_line_rows: tuple[int, ...] = field(init=False)
    a2_line_tree: shapely.STRtree | None = field(init=False)
    b2_line_tree: shapely.STRtree | None = field(init=False)
    a1_points: tuple[shapely.Point, ...] = field(init=False)
    a1_point_tree: shapely.STRtree | None = field(init=False)
    type1_node_indices: tuple[int, ...] = field(init=False)
    type1_node_points: tuple[shapely.Point, ...] = field(init=False)
    type1_node_tree: shapely.STRtree | None = field(init=False)

    def __post_init__(self) -> None:
        self.a2_refs = tuple(FeatureRef("a2_link", feature.id) for feature in self.dataset.a2_link)
        self.b2_refs = tuple(
            FeatureRef("b2_surfacelinemark", feature.id)
            for feature in self.dataset.b2_surfacelinemark
        )
        self.a2_lines = tuple(_line_or_none(feature) for feature in self.dataset.a2_link)
        self.b2_lines = tuple(_line_or_none(feature) for feature in self.dataset.b2_surfacelinemark)
        self.a2_rows_by_link_type = _rows_by_attr(self.dataset.a2_link.features, "link_type")
        self.b2_rows_by_kind = _rows_by_attr(self.dataset.b2_surfacelinemark.features, "kind")
        self.a2_line_rows = tuple(i for i, line in enumerate(self.a2_lines) if line is not None)
        self.b2_line_rows = tuple(i for i, line in enumerate(self.b2_lines) if line is not None)
        self.a2_line_tree = self._line_tree(self.a2_lines, self.a2_line_rows)
        self.b2_line_tree = self._line_tree(self.b2_lines, self.b2_line_rows)
        self.a1_points = tuple(shapely.Point(node.point[:2]) for node in self.dataset.a1_node)
        self.a1_point_tree = shapely.STRtree(self.a1_points) if self.a1_points else None
        self.type1_node_indices = tuple(
            i
            for i, node in enumerate(self.dataset.a1_node.features)
            if getattr(node, "node_type", "") == "1"
        )
        self.type1_node_points = tuple(
            shapely.Point(self.dataset.a1_node.features[i].point[:2])
            for i in self.type1_node_indices
        )
        self.type1_node_tree = (
            shapely.STRtree(self.type1_node_points) if self.type1_node_points else None
        )

    def _line_tree(
        self,
        lines: tuple[shapely.LineString | None, ...],
        rows: tuple[int, ...],
    ) -> shapely.STRtree | None:
        geometries = [line for row in rows if (line := lines[row]) is not None]
        return shapely.STRtree(geometries) if geometries else None

    def nearest_type1_node_id(self, xy: NDArray[np.float64], tolerance_m: float) -> str | None:
        if self.type1_node_tree is None:
            return None
        point = shapely.Point(float(xy[0]), float(xy[1]))
        best_node_id: str | None = None
        best_dist = tolerance_m
        for pos_raw in self.type1_node_tree.query(point.buffer(tolerance_m)):
            pos = int(pos_raw)
            candidate = self.type1_node_points[pos]
            dist = point.distance(candidate)
            if dist <= best_dist:
                best_dist = dist
                best_node_id = self.dataset.a1_node.features[self.type1_node_indices[pos]].id
        return best_node_id

    def ref_for_a2_index(self, index: int) -> FeatureRef:
        return self.a2_refs[index]

    def ref_for_b2_index(self, index: int) -> FeatureRef:
        return self.b2_refs[index]

    def a2_rows_for_link_type(self, link_type: str) -> tuple[int, ...]:
        return self.a2_rows_by_link_type.get(link_type, ())

    def b2_rows_for_kind(self, kind: str) -> tuple[int, ...]:
        return self.b2_rows_by_kind.get(kind, ())


def _rows_by_attr(features: list[Any], attr: str) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, feature in enumerate(features):
        value = getattr(feature, attr, "")
        if value:
            grouped[str(value)].append(index)
    return {key: tuple(rows) for key, rows in grouped.items()}


def _line_or_none(feature: Any) -> shapely.LineString | None:
    if not isinstance(feature, LineFeature) or len(feature.polyline) < 2:
        return None
    return shapely.LineString(feature.polyline[:, :2])
