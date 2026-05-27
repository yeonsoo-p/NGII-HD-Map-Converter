"""Cached dataset view used by segmentation stages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import LineFeature
from ngii2xodr.ngii.data.schema import RoleFilter, RoleKey
from ngii2xodr.ngii.segmentation.model import EndpointSide, SegmentationConfig


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
    junction_node_refs: frozenset[FeatureRef] = field(init=False)
    incoming_link_refs_by_node_id: dict[str, tuple[FeatureRef, ...]] = field(init=False)
    outgoing_link_refs_by_node_id: dict[str, tuple[FeatureRef, ...]] = field(init=False)
    _link_index_by_ref: dict[FeatureRef, int] = field(init=False)
    _node_index_by_ref: dict[FeatureRef, int] = field(init=False)
    _explicit_left_ref_by_link_id: dict[str, FeatureRef] = field(init=False)
    _explicit_right_ref_by_link_id: dict[str, FeatureRef] = field(init=False)
    _inverse_right_refs_by_link_id: dict[str, tuple[FeatureRef, ...]] = field(init=False)
    _inverse_left_refs_by_link_id: dict[str, tuple[FeatureRef, ...]] = field(init=False)
    _rows_by_filter: dict[str, tuple[int, ...]] = field(init=False)
    _semantic_keys_by_name: dict[str, dict[FeatureRef, tuple[str, ...]]] = field(init=False)

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
        self._link_index_by_ref = {ref: i for i, ref in enumerate(self.link_refs)}
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
        self._node_index_by_ref = {
            FeatureRef(self.node_attr, feature.id): i
            for i, feature in enumerate(self.node_store.features)
        }
        self._build_link_maps()
        self._rows_by_filter = self._build_rows_by_filter()
        junction_node_indices = self._rows_by_filter.get("junction_node", ())
        self.junction_node_refs = frozenset(
            FeatureRef(self.node_attr, self.node_store.features[i].id)
            for i in junction_node_indices
        )
        self._semantic_keys_by_name = self._build_semantic_keys_by_name()

    def _build_link_maps(self) -> None:
        incoming: dict[str, list[FeatureRef]] = {}
        outgoing: dict[str, list[FeatureRef]] = {}
        explicit_left: dict[str, FeatureRef] = {}
        explicit_right: dict[str, FeatureRef] = {}
        inverse_left: dict[str, list[FeatureRef]] = {}
        inverse_right: dict[str, list[FeatureRef]] = {}
        for i, link in enumerate(self.link_store.features):
            link_ref = self.ref_for_link_index(i)
            from_node_id = _optional_text(getattr(link, "from_node_id", ""))
            to_node_id = _optional_text(getattr(link, "to_node_id", ""))
            if from_node_id and self.node_store.get(from_node_id) is not None:
                outgoing.setdefault(from_node_id, []).append(link_ref)
            if to_node_id and self.node_store.get(to_node_id) is not None:
                incoming.setdefault(to_node_id, []).append(link_ref)

            right_id = _optional_text(getattr(link, "r_link_id", ""))
            left_id = _optional_text(getattr(link, "l_link_id", ""))
            if right_id and self.link_store.get(right_id) is not None:
                right_ref = FeatureRef(self.link_attr, right_id)
                explicit_right[link.id] = right_ref
                inverse_right.setdefault(right_id, []).append(link_ref)
            if left_id and self.link_store.get(left_id) is not None:
                left_ref = FeatureRef(self.link_attr, left_id)
                explicit_left[link.id] = left_ref
                inverse_left.setdefault(left_id, []).append(link_ref)

        self.incoming_link_refs_by_node_id = {
            node_id: tuple(refs) for node_id, refs in incoming.items()
        }
        self.outgoing_link_refs_by_node_id = {
            node_id: tuple(refs) for node_id, refs in outgoing.items()
        }
        self._explicit_left_ref_by_link_id = explicit_left
        self._explicit_right_ref_by_link_id = explicit_right
        self._inverse_left_refs_by_link_id = {
            feature_id: tuple(refs) for feature_id, refs in inverse_left.items()
        }
        self._inverse_right_refs_by_link_id = {
            feature_id: tuple(refs) for feature_id, refs in inverse_right.items()
        }

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
                if _matches_role_filter(feature, role_filter)
            )
        return rows_by_filter

    def _build_semantic_keys_by_name(self) -> dict[str, dict[FeatureRef, tuple[str, ...]]]:
        keys_by_name: dict[str, dict[FeatureRef, tuple[str, ...]]] = {}
        for role_key in self.dataset.schema.role_keys:
            keys_by_name[role_key.name] = self._semantic_keys_for_role_key(role_key)
        return keys_by_name

    def _semantic_keys_for_role_key(self, role_key: RoleKey) -> dict[FeatureRef, tuple[str, ...]]:
        try:
            store = self.dataset.store_for_role(role_key.role)
        except KeyError:
            return {}
        keys_by_ref: dict[FeatureRef, tuple[str, ...]] = {}
        for feature in store.features:
            values = tuple(
                f"{role_key.name}:{value}"
                for attr in role_key.attrs
                if (value := _optional_text(getattr(feature, attr, "")).strip())
            )
            if values:
                keys_by_ref[FeatureRef(store.spec.python_attr, feature.id)] = values
        return keys_by_ref

    def _line_tree(
        self,
        lines: tuple[shapely.LineString | None, ...],
        rows: tuple[int, ...],
    ) -> shapely.STRtree | None:
        geometries = [line for row in rows if (line := lines[row]) is not None]
        return shapely.STRtree(geometries) if geometries else None

    def ref_for_link_index(self, index: int) -> FeatureRef:
        return self.link_refs[index]

    def ref_for_lane_line_index(self, index: int) -> FeatureRef:
        return self.lane_line_refs[index]

    def rows_for_filter(self, name: str) -> tuple[int, ...]:
        return self._rows_by_filter.get(name, ())

    def link_index_for_ref(self, ref: FeatureRef) -> int | None:
        return self._link_index_by_ref.get(ref)

    def link_for_ref(self, ref: FeatureRef) -> Any | None:
        index = self.link_index_for_ref(ref)
        return None if index is None else self.link_store.features[index]

    def node_ref_for_id(self, node_id: str | None) -> FeatureRef | None:
        if not node_id or self.node_store.get(node_id) is None:
            return None
        return FeatureRef(self.node_attr, node_id)

    def node_point_for_ref(self, ref: FeatureRef) -> shapely.Point | None:
        index = self._node_index_by_ref.get(ref)
        return None if index is None else self.node_points[index]

    def is_junction_node_ref(self, ref: FeatureRef) -> bool:
        return ref in self.junction_node_refs

    def semantic_keys_for_ref(self, name: str, ref: FeatureRef) -> tuple[str, ...]:
        return self._semantic_keys_by_name.get(name, {}).get(ref, ())

    def semantic_keys_for_refs(self, name: str, refs: tuple[FeatureRef, ...]) -> tuple[str, ...]:
        keys: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            for key in self.semantic_keys_for_ref(name, ref):
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
        return tuple(keys)

    def endpoint_node_distance_m(
        self,
        left_refs: tuple[FeatureRef, ...],
        right_refs: tuple[FeatureRef, ...],
    ) -> float | None:
        best_distance_m: float | None = None
        for left_ref in left_refs:
            left_point = self.node_point_for_ref(left_ref)
            if left_point is None:
                continue
            for right_ref in right_refs:
                right_point = self.node_point_for_ref(right_ref)
                if right_point is None:
                    continue
                distance_m = float(left_point.distance(right_point))
                if best_distance_m is None or distance_m < best_distance_m:
                    best_distance_m = distance_m
        return best_distance_m

    def endpoint_node_id_for_link_index(self, link_index: int, side: EndpointSide) -> str | None:
        link = self.link_store.features[link_index]
        attr = "from_node_id" if side == "from" else "to_node_id"
        return _optional_text(getattr(link, attr, ""))

    def endpoint_node_ref_for_link_index(
        self, link_index: int, side: EndpointSide
    ) -> FeatureRef | None:
        return self.node_ref_for_id(self.endpoint_node_id_for_link_index(link_index, side))

    def endpoint_node_ref_for_link_ref(
        self, link_ref: FeatureRef, side: EndpointSide
    ) -> FeatureRef | None:
        index = self.link_index_for_ref(link_ref)
        return None if index is None else self.endpoint_node_ref_for_link_index(index, side)

    def endpoint_node_refs(
        self, link_refs: tuple[FeatureRef, ...], side: EndpointSide
    ) -> tuple[FeatureRef, ...]:
        node_refs: list[FeatureRef] = []
        seen: set[FeatureRef] = set()
        for link_ref in link_refs:
            node_ref = self.endpoint_node_ref_for_link_ref(link_ref, side)
            if node_ref is None or node_ref in seen:
                continue
            seen.add(node_ref)
            node_refs.append(node_ref)
        return tuple(node_refs)

    def endpoint_node_refs_for_link_ref(self, link_ref: FeatureRef) -> tuple[FeatureRef, ...]:
        node_refs: list[FeatureRef] = []
        for side in ("from", "to"):
            node_ref = self.endpoint_node_ref_for_link_ref(link_ref, side)
            if node_ref is not None and node_ref not in node_refs:
                node_refs.append(node_ref)
        return tuple(node_refs)

    def lateral_neighbor_refs_for_link_index(
        self, link_index: int, side: str
    ) -> tuple[FeatureRef, ...]:
        link = self.link_store.features[link_index]
        refs: list[FeatureRef] = []
        if side == "left":
            explicit = self._explicit_left_ref_by_link_id.get(link.id)
            inverse = self._inverse_right_refs_by_link_id.get(link.id, ())
        elif side == "right":
            explicit = self._explicit_right_ref_by_link_id.get(link.id)
            inverse = self._inverse_left_refs_by_link_id.get(link.id, ())
        else:
            msg = f"unsupported lateral side {side!r}"
            raise ValueError(msg)
        if explicit is not None:
            refs.append(explicit)
        for ref in inverse:
            if ref not in refs:
                refs.append(ref)
        return tuple(refs)

    def lateral_neighbor_rows_for_link_index(
        self, link_index: int, side: str | None = None
    ) -> tuple[int, ...]:
        sides = ("left", "right") if side is None else (side,)
        rows: list[int] = []
        seen: set[int] = set()
        for item in sides:
            for ref in self.lateral_neighbor_refs_for_link_index(link_index, item):
                row = self.link_index_for_ref(ref)
                if row is not None and row not in seen:
                    seen.add(row)
                    rows.append(row)
        return tuple(rows)

    def incoming_link_refs(self, node_ref: FeatureRef) -> tuple[FeatureRef, ...]:
        return self.incoming_link_refs_by_node_id.get(node_ref.feature_id, ())

    def outgoing_link_refs(self, node_ref: FeatureRef) -> tuple[FeatureRef, ...]:
        return self.outgoing_link_refs_by_node_id.get(node_ref.feature_id, ())

    def link_turn_for_ref(self, link_ref: FeatureRef) -> str:
        link = self.link_for_ref(link_ref)
        return "" if link is None else _optional_text(getattr(link, "turn", ""))

    def link_lane_no_for_ref(self, link_ref: FeatureRef) -> int | None:
        link = self.link_for_ref(link_ref)
        if link is None:
            return None
        value = getattr(link, "lane_no", None)
        return value if isinstance(value, int) else None

    def link_polyline_for_ref(self, link_ref: FeatureRef) -> NDArray[np.float64] | None:
        link = self.link_for_ref(link_ref)
        if not isinstance(link, LineFeature):
            return None
        return link.polyline

    def endpoint_geometry_for_link_ref(
        self, link_ref: FeatureRef, side: EndpointSide
    ) -> tuple[tuple[float, float, float], tuple[float, float]] | None:
        polyline = self.link_polyline_for_ref(link_ref)
        if polyline is None or len(polyline) < 2:
            return None
        if side == "from":
            anchor = polyline[0]
            tangent = polyline[1, :2] - polyline[0, :2]
        else:
            anchor = polyline[-1]
            tangent = polyline[-1, :2] - polyline[-2, :2]
        norm = float(np.hypot(tangent[0], tangent[1]))
        if norm <= 0.0:
            return None
        return (
            (float(anchor[0]), float(anchor[1]), float(anchor[2])),
            (float(tangent[0]) / norm, float(tangent[1]) / norm),
        )


def _line_or_none(feature: Any) -> shapely.LineString | None:
    if not isinstance(feature, LineFeature) or len(feature.polyline) < 2:
        return None
    return shapely.LineString(feature.polyline[:, :2])


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _matches_role_filter(feature: Any, role_filter: RoleFilter) -> bool:
    for attr in role_filter.attrs:
        value = getattr(feature, attr, None)
        if role_filter.numeric_min is not None:
            number = _number_or_none(value)
            if number is not None and number >= role_filter.numeric_min:
                return True
            continue
        if str(value) in role_filter.values:
            return True
    return False


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _number_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if value is None:
        return None
    text = str(value).strip()
    if not _NUMBER_RE.fullmatch(text):
        return None
    return float(text)
