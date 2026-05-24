"""Small staged segmentation pipeline for SHP-derived abstract entities.

Each stage is an explicit class with a tiny public contract: it reads typed
layer data, emits typed entities, and derives per-layer membership arrays for
fast visualization. The GUI builds its segmentation-level selector from
``SEGMENTATION_STAGES`` so adding or removing a stage is a code-local change.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, Self

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import nearest_points

from shp2xodr.shp.data import A1Data, A2Data, B2Data

log = logging.getLogger(__name__)

_UTURN_LINK_TYPE = "6"
_LATERAL_GROUP_LINK_TYPE = "6"
_JUNCTION_LINK_TYPE = "1"
_UTURN_MARKER_KIND = "502"
_A1_JUNCTION_NODE_TYPE = "1"
_NODE_ENDPOINT_MATCH_TOL_M = 0.5

StageTarget = Literal["A1", "A2"]


@dataclass(slots=True, frozen=True)
class SegmentationConfig:
    """Configuration shared by geometry-based segmentation stages."""

    z_intersection_tol_m: float
    junction_proximity_merge_dist_m: float
    junction_connection_node_merge_dist_m: float


@dataclass(slots=True, frozen=True)
class SelectedField:
    """One selected-table field, with one or more display values."""

    name: str
    values: tuple[str, ...]

    @classmethod
    def scalar(cls, name: str, value: str | int | float) -> Self:
        return cls(name=name, values=(str(value),))

    @classmethod
    def list(cls, name: str, values: tuple[str, ...]) -> Self:
        return cls(name=name, values=values if values else ("-",))


@dataclass(slots=True, frozen=True)
class SelectedSection:
    """A titled group of selected-table fields."""

    title: str
    rows: tuple[SelectedField, ...]


@dataclass(slots=True, frozen=True)
class SegmentationInput:
    """Layer data and config needed by segmentation stages."""

    a1: A1Data
    a2: A2Data
    b2: B2Data
    cfg: SegmentationConfig

    @classmethod
    def from_layers(cls, a1: A1Data, a2: A2Data, b2: B2Data, cfg: SegmentationConfig) -> Self:
        return cls(a1=a1, a2=a2, b2=b2, cfg=cfg)

    @property
    def n_links(self) -> int:
        return len(self.a2.ids)

    @property
    def n_nodes(self) -> int:
        return len(self.a1.ids)


class SegmentEntity(Protocol):
    """Class interface shared by all staged segmentation entities."""

    @property
    def id(self) -> int: ...

    def selected_fields(self) -> tuple[SelectedField, ...]: ...


@dataclass(slots=True, frozen=True)
class UTurn:
    """One U-turn entity: a single Type=6 A2 link plus B2 502 marker(s)."""

    id: int
    link_index: int
    link_id: str
    marker_indices: tuple[int, ...]
    marker_ids: tuple[str, ...]

    @property
    def link_indices(self) -> tuple[int, ...]:
        return (self.link_index,)

    @property
    def link_ids(self) -> tuple[str, ...]:
        return (self.link_id,)

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("U-turn", self.id),
            SelectedField.list("U-turn marker ID", self.marker_ids),
        )


@dataclass(slots=True, frozen=True)
class JunctionCache:
    """Derived geometry cache for one provisional or final junction."""

    representative_link_index: int
    lines: tuple[shapely.LineString, ...]
    geom: shapely.Geometry


@dataclass(slots=True, frozen=True)
class Junction:
    """One junction entity: a connected component of Type=1 A2 links.

    ``id`` is ``-1`` while the junction is provisional during staging and is
    assigned a non-negative final value before being exposed in ``StageResult``.
    """

    id: int
    link_indices: tuple[int, ...]
    link_ids: tuple[str, ...]
    cache: JunctionCache

    @classmethod
    def from_link_indices(
        cls,
        *,
        entity_id: int,
        link_indices: tuple[int, ...],
        a2: A2Data,
    ) -> Self:
        sorted_indices = tuple(sorted(link_indices))
        lines = tuple(a2.xy_lines[i] for i in sorted_indices if _can_make_line(a2.polylines[i]))
        geom = shapely.unary_union(lines) if lines else shapely.LineString()
        representative_link_index = min(sorted_indices) if sorted_indices else -1
        return cls(
            id=entity_id,
            link_indices=sorted_indices,
            link_ids=tuple(str(a2.ids[i]) for i in sorted_indices),
            cache=JunctionCache(
                representative_link_index=representative_link_index,
                lines=lines,
                geom=geom,
            ),
        )

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Junction", self.id),
            SelectedField.list("Junction link ID", self.link_ids),
        )


@dataclass(slots=True, frozen=True)
class LateralGroup:
    """One lateral bundle of ordinary Type=6 A2 links."""

    id: int
    link_indices: tuple[int, ...]
    link_ids: tuple[str, ...]

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Lateral group", self.id),
            SelectedField.list("Lateral group link ID", self.link_ids),
        )


@dataclass(slots=True, frozen=True)
class JunctionConnection:
    """One group of A1 Type=1 nodes connecting lateral traffic to a junction."""

    id: int
    junction_id: int
    node_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    lateral_group_ids: tuple[int, ...]

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Junction connection", self.id),
            SelectedField.scalar("Connection junction", self.junction_id),
            SelectedField.list("Connection node ID", self.node_ids),
            SelectedField.list(
                "Connection lateral group",
                tuple(str(group_id) for group_id in self.lateral_group_ids),
            ),
        )


@dataclass(slots=True, frozen=True)
class _ConnectionEndpoint:
    node_index: int
    node_id: str
    link_index: int
    lane_no: int


@dataclass(slots=True, frozen=True)
class _ConnectionNodeGroup:
    junction_id: int
    lateral_group_id: int
    node_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    endpoint_records: tuple[_ConnectionEndpoint, ...]
    representative_node_index: int


@dataclass(slots=True, frozen=True)
class StageResult:
    """Output of one segmentation stage."""

    stage_id: str
    label: str
    entity_label: str
    entities: tuple[SegmentEntity, ...]
    entity_id_per_link: NDArray[np.int32]
    target_layer: StageTarget = "A2"
    entity_id_per_node: NDArray[np.int32] | None = None
    entity_ids_per_node: tuple[tuple[int, ...], ...] | None = None

    def selected_fields_for_link(self, link_index: int) -> tuple[SelectedField, ...]:
        if self.target_layer != "A2":
            return ()
        entity_id = int(self.entity_id_per_link[link_index])
        if entity_id < 0:
            return ()
        return self.entities[entity_id].selected_fields()

    def selected_fields_for_node(self, node_index: int) -> tuple[SelectedField, ...]:
        if self.target_layer != "A1":
            return ()
        if self.entity_ids_per_node is not None:
            entity_ids = self.entity_ids_per_node[node_index]
            if not entity_ids:
                return ()
            fields: list[SelectedField] = []
            for entity_id in entity_ids:
                fields.extend(self.entities[entity_id].selected_fields())
            return tuple(fields)
        if self.entity_id_per_node is None:
            return ()
        entity_id = int(self.entity_id_per_node[node_index])
        if entity_id < 0:
            return ()
        return self.entities[entity_id].selected_fields()


class SegmentationStage:
    """Base class for code-registered segmentation stages."""

    id: ClassVar[str]
    label: ClassVar[str]
    entity_label: ClassVar[str]

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        raise NotImplementedError


class UTurnStage(SegmentationStage):
    """Detect U-turn links from Type=6 A2 links crossing B2 Kind=502 markers."""

    id = "uturn"
    label = "U-turns"
    entity_label = "U-turn"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        marker_rows = [
            i
            for i, kind in enumerate(data.b2.kinds)
            if kind == _UTURN_MARKER_KIND and _can_make_line(data.b2.polylines[i])
        ]
        if not marker_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        marker_lines = [data.b2.xy_lines[i] for i in marker_rows]
        tree = shapely.STRtree(marker_lines)
        entities: list[UTurn] = []

        for link_i, link_type in enumerate(data.a2.link_types):
            if link_type != _UTURN_LINK_TYPE or not _can_make_line(data.a2.polylines[link_i]):
                continue
            link_line = data.a2.xy_lines[link_i]
            matched_marker_rows: list[int] = []
            for marker_pos_raw in tree.query(link_line, predicate="intersects"):
                marker_pos = int(marker_pos_raw)
                marker_i = marker_rows[marker_pos]
                if _intersects_within_z_tol(
                    data.a2.polylines[link_i],
                    link_line,
                    data.b2.polylines[marker_i],
                    marker_lines[marker_pos],
                    data.cfg.z_intersection_tol_m,
                ):
                    matched_marker_rows.append(marker_i)
            if not matched_marker_rows:
                continue
            entity_id = len(entities)
            entity_id_per_link[link_i] = np.int32(entity_id)
            marker_indices = tuple(sorted(set(matched_marker_rows)))
            entities.append(
                UTurn(
                    id=entity_id,
                    link_index=link_i,
                    link_id=str(data.a2.ids[link_i]),
                    marker_indices=marker_indices,
                    marker_ids=tuple(str(data.b2.ids[i]) for i in marker_indices),
                )
            )

        if entities:
            log.info("UTurnStage: detected %d U-turn link(s)", len(entities))
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_per_link
        )


class JunctionStage(SegmentationStage):
    """Detect junctions as connected components of Type=1 A2 links."""

    id = "junction"
    label = "Junctions"
    entity_label = "Junction"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        candidate_rows = _junction_candidate_rows(data)
        if not candidate_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)

        _union_junction_link_refs(data, candidate_rows, row_to_candidate, parent)
        _union_junction_intersections(data, candidate_rows, row_to_candidate, parent)
        if data.cfg.junction_proximity_merge_dist_m > 0.0:
            provisional_junctions = _junctions_from_parent(
                data, candidate_rows, row_to_candidate, parent
            )
            _union_junctions_by_proximity(data, provisional_junctions, row_to_candidate, parent)

        entities = _junctions_from_parent(
            data,
            candidate_rows,
            row_to_candidate,
            parent,
            entity_id_per_link=entity_id_per_link,
        )
        if entities:
            log.info("JunctionStage: detected %d junction component(s)", len(entities))
        return StageResult(self.id, self.label, self.entity_label, entities, entity_id_per_link)


class LateralGroupStage(SegmentationStage):
    """Group ordinary Type=6 A2 links by lateral R/L link references."""

    id = "lateral_group"
    label = "Lateral groups"
    entity_label = "Lateral group"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        candidate_rows = _lateral_group_candidate_rows(data, previous_results)
        if not candidate_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)

        _union_lateral_group_link_refs(data, candidate_rows, row_to_candidate, parent)

        root_to_entity: dict[int, int] = {}
        rows_by_entity: dict[int, list[int]] = {}
        for row_i in candidate_rows:
            root = _uf_find(parent, row_to_candidate[row_i])
            entity_id = root_to_entity.get(root)
            if entity_id is None:
                entity_id = len(root_to_entity)
                root_to_entity[root] = entity_id
            entity_id_per_link[row_i] = np.int32(entity_id)
            rows_by_entity.setdefault(entity_id, []).append(row_i)

        entities = tuple(
            LateralGroup(
                id=entity_id,
                link_indices=tuple(rows_by_entity[entity_id]),
                link_ids=tuple(str(data.a2.ids[i]) for i in rows_by_entity[entity_id]),
            )
            for entity_id in range(len(rows_by_entity))
        )
        if entities:
            log.info("LateralGroupStage: detected %d lateral group(s)", len(entities))
        return StageResult(self.id, self.label, self.entity_label, entities, entity_id_per_link)


class JunctionConnectionStage(SegmentationStage):
    """Group A1 Type=1 nodes that form lateral connections to junctions."""

    id = "junction_connection"
    label = "Junction connections"
    entity_label = "Junction connection"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        entity_id_per_node = np.full(data.n_nodes, -1, dtype=np.int32)
        lateral_result = previous_results.get(LateralGroupStage.id)
        junction_result = previous_results.get(JunctionStage.id)
        if lateral_result is None or junction_result is None:
            return StageResult(
                self.id,
                self.label,
                self.entity_label,
                (),
                entity_id_per_link,
                target_layer="A1",
                entity_id_per_node=entity_id_per_node,
                entity_ids_per_node=tuple(() for _ in range(data.n_nodes)),
            )

        node_to_junction_ids = _junction_ids_by_type1_node(data, junction_result)
        node_groups = _connection_node_groups(data, lateral_result, node_to_junction_ids)
        if not node_groups:
            return StageResult(
                self.id,
                self.label,
                self.entity_label,
                (),
                entity_id_per_link,
                target_layer="A1",
                entity_id_per_node=entity_id_per_node,
                entity_ids_per_node=tuple(() for _ in range(data.n_nodes)),
            )

        parent = np.arange(len(node_groups), dtype=np.int32)
        _union_connection_node_groups(data, node_groups, parent)
        entities, entity_ids_per_node = _junction_connections_from_parent(
            data, node_groups, parent, entity_id_per_node
        )

        if entities:
            log.info(
                "JunctionConnectionStage: detected %d A1 node connection group(s)",
                len(entities),
            )
        return StageResult(
            self.id,
            self.label,
            self.entity_label,
            entities,
            entity_id_per_link,
            target_layer="A1",
            entity_id_per_node=entity_id_per_node,
            entity_ids_per_node=entity_ids_per_node,
        )


SEGMENTATION_STAGES: tuple[type[SegmentationStage], ...] = (
    UTurnStage,
    LateralGroupStage,
    JunctionStage,
    JunctionConnectionStage,
)


@dataclass(slots=True, frozen=True)
class Segmentation:
    """All stage outputs for one loaded section."""

    stage_results: tuple[StageResult, ...]

    @classmethod
    def from_layers(cls, a1: A1Data, a2: A2Data, b2: B2Data, cfg: SegmentationConfig) -> Self:
        data = SegmentationInput.from_layers(a1, a2, b2, cfg)
        results: list[StageResult] = []
        by_id: dict[str, StageResult] = {}
        for stage_cls in SEGMENTATION_STAGES:
            result = stage_cls().run(data, by_id)
            results.append(result)
            by_id[result.stage_id] = result
        return cls(stage_results=tuple(results))

    @property
    def level_labels(self) -> tuple[str, ...]:
        return segmentation_level_labels()

    def active_results(self, level: int) -> tuple[StageResult, ...]:
        if level < 0 or level > len(self.stage_results):
            raise ValueError(level)
        return self.stage_results[:level]

    def result(self, stage_id: str) -> StageResult:
        for result in self.stage_results:
            if result.stage_id == stage_id:
                return result
        raise KeyError(stage_id)

    def entity_id_per_link(self, stage_id: str) -> NDArray[np.int32]:
        return self.result(stage_id).entity_id_per_link

    def entity_id_per_node(self, stage_id: str) -> NDArray[np.int32]:
        result = self.result(stage_id)
        if result.entity_id_per_node is None:
            msg = f"{stage_id!r} does not produce A1 node membership"
            raise ValueError(msg)
        return result.entity_id_per_node

    def selected_fields_for_link(self, link_index: int) -> tuple[SelectedField, ...]:
        fields: list[SelectedField] = []
        for result in self.stage_results:
            fields.extend(result.selected_fields_for_link(link_index))
        return tuple(fields)

    def selected_fields_for_node(self, node_index: int) -> tuple[SelectedField, ...]:
        fields: list[SelectedField] = []
        for result in self.stage_results:
            fields.extend(result.selected_fields_for_node(node_index))
        return tuple(fields)


def segmentation_level_labels() -> tuple[str, ...]:
    """Labels for GUI selector levels: raw plus one entry per stage."""
    return ("Raw", *(stage.label for stage in SEGMENTATION_STAGES))


def _can_make_line(polyline: NDArray[np.float64]) -> bool:
    return len(polyline) >= 2


def _junction_candidate_rows(data: SegmentationInput) -> list[int]:
    return [i for i, link_type in enumerate(data.a2.link_types) if link_type == _JUNCTION_LINK_TYPE]


def _lateral_group_candidate_rows(
    data: SegmentationInput,
    previous_results: Mapping[str, StageResult],
) -> list[int]:
    uturn_result = previous_results.get(UTurnStage.id)
    uturn_entity_id_per_link = uturn_result.entity_id_per_link if uturn_result is not None else None
    return [
        i
        for i, link_type in enumerate(data.a2.link_types)
        if link_type == _LATERAL_GROUP_LINK_TYPE
        and (uturn_entity_id_per_link is None or int(uturn_entity_id_per_link[i]) < 0)
    ]


def _junction_ids_by_type1_node(
    data: SegmentationInput,
    junction_result: StageResult,
) -> dict[str, set[int]]:
    node_to_junction_ids: dict[str, set[int]] = {}
    for entity in junction_result.entities:
        if not isinstance(entity, Junction):
            continue
        for link_i in entity.link_indices:
            for node_id in _link_endpoint_type1_node_ids(data, link_i):
                node_to_junction_ids.setdefault(node_id, set()).add(entity.id)
    return node_to_junction_ids


def _link_endpoint_type1_node_ids(data: SegmentationInput, link_i: int) -> tuple[str, ...]:
    polyline = data.a2.polylines[link_i]
    if len(polyline) == 0:
        return ()
    endpoint_specs = (
        (polyline[0, :2], str(data.a2.from_node_ids[link_i])),
        (polyline[-1, :2], str(data.a2.to_node_ids[link_i])),
    )
    node_ids: list[str] = []
    for endpoint_xy, fallback_node_id in endpoint_specs:
        node_id = _type1_node_id_at_xy(data, float(endpoint_xy[0]), float(endpoint_xy[1]))
        if node_id is None:
            node_id = _type1_fallback_node_id(data, fallback_node_id)
        if node_id is not None and node_id not in node_ids:
            node_ids.append(node_id)
    return tuple(node_ids)


def _type1_node_id_at_xy(data: SegmentationInput, x: float, y: float) -> str | None:
    best_node_id: str | None = None
    best_dist = _NODE_ENDPOINT_MATCH_TOL_M
    for node_i, node_type in enumerate(data.a1.node_types):
        if node_type != _A1_JUNCTION_NODE_TYPE:
            continue
        dx = float(data.a1.points[node_i, 0] - x)
        dy = float(data.a1.points[node_i, 1] - y)
        dist = float(np.hypot(dx, dy))
        if dist <= best_dist:
            best_dist = dist
            best_node_id = str(data.a1.ids[node_i])
    return best_node_id


def _type1_fallback_node_id(data: SegmentationInput, node_id: str) -> str | None:
    node_i = data.a1.id_to_index.get(node_id)
    if node_i is None or data.a1.node_types[node_i] != _A1_JUNCTION_NODE_TYPE:
        return None
    return node_id


def _connection_node_groups(
    data: SegmentationInput,
    lateral_result: StageResult,
    node_to_junction_ids: Mapping[str, set[int]],
) -> tuple[_ConnectionNodeGroup, ...]:
    groups: list[_ConnectionNodeGroup] = []
    for entity in lateral_result.entities:
        if not isinstance(entity, LateralGroup):
            continue
        endpoint_records_by_junction: dict[int, list[_ConnectionEndpoint]] = {}
        for link_i in entity.link_indices:
            lane_no = int(data.a2.lane_nos[link_i])
            for node_id in _link_endpoint_type1_node_ids(data, link_i):
                node_i = data.a1.id_to_index[node_id]
                for junction_id in node_to_junction_ids.get(node_id, ()):
                    endpoint_records_by_junction.setdefault(junction_id, []).append(
                        _ConnectionEndpoint(
                            node_index=node_i,
                            node_id=node_id,
                            link_index=link_i,
                            lane_no=lane_no,
                        )
                    )
        for junction_id in sorted(endpoint_records_by_junction):
            endpoint_records = tuple(endpoint_records_by_junction[junction_id])
            node_indices = tuple(sorted({record.node_index for record in endpoint_records}))
            representative = _leftmost_connection_endpoint(endpoint_records)
            groups.append(
                _ConnectionNodeGroup(
                    junction_id=junction_id,
                    lateral_group_id=entity.id,
                    node_indices=node_indices,
                    node_ids=tuple(str(data.a1.ids[i]) for i in node_indices),
                    endpoint_records=endpoint_records,
                    representative_node_index=representative.node_index,
                )
            )
    return tuple(groups)


def _leftmost_connection_endpoint(
    endpoint_records: tuple[_ConnectionEndpoint, ...],
) -> _ConnectionEndpoint:
    return min(endpoint_records, key=_connection_endpoint_leftmost_key)


def _connection_endpoint_leftmost_key(record: _ConnectionEndpoint) -> tuple[int, int, int, int]:
    lane_no = record.lane_no
    if lane_no >= 90:
        return (0, -lane_no, record.link_index, record.node_index)
    if lane_no == 1:
        return (1, 0, record.link_index, record.node_index)
    if lane_no > 0:
        return (2, lane_no, record.link_index, record.node_index)
    return (3, lane_no, record.link_index, record.node_index)


def _union_connection_node_groups(
    data: SegmentationInput,
    node_groups: tuple[_ConnectionNodeGroup, ...],
    parent: NDArray[np.int32],
) -> None:
    max_distance = data.cfg.junction_connection_node_merge_dist_m
    if max_distance <= 0.0 or len(node_groups) < 2:
        return

    connected_node_pairs = _a2_endpoint_node_pairs(data)
    edges: list[tuple[float, int, int]] = []
    for a_i, group_a in enumerate(node_groups):
        for b_i in range(a_i + 1, len(node_groups)):
            group_b = node_groups[b_i]
            if group_a.junction_id != group_b.junction_id:
                continue
            dist = _connection_node_group_distance(data, group_a, group_b)
            if dist <= max_distance:
                edges.append((dist, a_i, b_i))

    n_unioned = 0
    for _dist, a_i, b_i in sorted(edges):
        root_a = _uf_find(parent, a_i)
        root_b = _uf_find(parent, b_i)
        if root_a == root_b:
            continue
        if _connection_components_directly_connected(
            node_groups,
            parent,
            root_a,
            root_b,
            connected_node_pairs,
        ):
            continue
        _uf_union(parent, root_a, root_b)
        n_unioned += 1
    if n_unioned:
        log.info(
            "JunctionConnectionStage: proximity-merged %d A1 node group pair(s) within %.2f m",
            n_unioned,
            max_distance,
        )


def _a2_endpoint_node_pairs(data: SegmentationInput) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for link_i in range(data.n_links):
        node_ids = _link_endpoint_type1_node_ids(data, link_i)
        if len(node_ids) == 2:
            pairs.add(_node_pair_key(node_ids[0], node_ids[1]))
    return pairs


def _connection_node_group_distance(
    data: SegmentationInput,
    group_a: _ConnectionNodeGroup,
    group_b: _ConnectionNodeGroup,
) -> float:
    node_a = group_a.representative_node_index
    node_b = group_b.representative_node_index
    xy_a = (float(data.a1.points[node_a, 0]), float(data.a1.points[node_a, 1]))
    xy_b = (float(data.a1.points[node_b, 0]), float(data.a1.points[node_b, 1]))
    return _xy_distance(xy_a, xy_b)


def _connection_components_directly_connected(
    node_groups: tuple[_ConnectionNodeGroup, ...],
    parent: NDArray[np.int32],
    root_a: int,
    root_b: int,
    connected_node_pairs: set[tuple[str, str]],
) -> bool:
    node_ids_a = _component_node_ids(node_groups, parent, root_a)
    node_ids_b = _component_node_ids(node_groups, parent, root_b)
    return any(
        _node_pair_key(node_id_a, node_id_b) in connected_node_pairs
        for node_id_a in node_ids_a
        for node_id_b in node_ids_b
    )


def _component_node_ids(
    node_groups: tuple[_ConnectionNodeGroup, ...],
    parent: NDArray[np.int32],
    root: int,
) -> tuple[str, ...]:
    node_ids: set[str] = set()
    for group_i, group in enumerate(node_groups):
        if _uf_find(parent, group_i) == root:
            node_ids.update(group.node_ids)
    return tuple(sorted(node_ids))


def _node_pair_key(node_id_a: str, node_id_b: str) -> tuple[str, str]:
    return (node_id_a, node_id_b) if node_id_a <= node_id_b else (node_id_b, node_id_a)


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _junction_connections_from_parent(
    data: SegmentationInput,
    node_groups: tuple[_ConnectionNodeGroup, ...],
    parent: NDArray[np.int32],
    entity_id_per_node: NDArray[np.int32],
) -> tuple[tuple[JunctionConnection, ...], tuple[tuple[int, ...], ...]]:
    root_to_entity: dict[int, int] = {}
    group_indices_by_entity: dict[int, list[int]] = {}
    for group_i in range(len(node_groups)):
        root = _uf_find(parent, group_i)
        entity_id = root_to_entity.get(root)
        if entity_id is None:
            entity_id = len(root_to_entity)
            root_to_entity[root] = entity_id
        group_indices_by_entity.setdefault(entity_id, []).append(group_i)

    node_entity_lists: list[list[int]] = [[] for _ in range(data.n_nodes)]
    entities: list[JunctionConnection] = []
    for entity_id in range(len(group_indices_by_entity)):
        group_indices = group_indices_by_entity[entity_id]
        groups = tuple(node_groups[i] for i in group_indices)
        junction_id = groups[0].junction_id
        node_indices = tuple(sorted({node_i for group in groups for node_i in group.node_indices}))
        lateral_group_ids = tuple(sorted({group.lateral_group_id for group in groups}))
        for node_i in node_indices:
            if int(entity_id_per_node[node_i]) < 0:
                entity_id_per_node[node_i] = np.int32(entity_id)
            node_entity_lists[node_i].append(entity_id)
        entities.append(
            JunctionConnection(
                id=entity_id,
                junction_id=junction_id,
                node_indices=node_indices,
                node_ids=tuple(str(data.a1.ids[i]) for i in node_indices),
                lateral_group_ids=lateral_group_ids,
            )
        )

    return tuple(entities), tuple(tuple(entity_ids) for entity_ids in node_entity_lists)


def _union_junction_link_refs(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    for row_i in candidate_rows:
        candidate_i = row_to_candidate[row_i]
        for neighbour_id in (data.a2.r_link_ids[row_i], data.a2.l_link_ids[row_i]):
            neighbour_row = data.a2.id_to_index.get(str(neighbour_id))
            if neighbour_row is None or neighbour_row not in row_to_candidate:
                continue
            _uf_union(parent, candidate_i, row_to_candidate[neighbour_row])


def _union_lateral_group_link_refs(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    for row_i in candidate_rows:
        candidate_i = row_to_candidate[row_i]
        for neighbour_id in (data.a2.r_link_ids[row_i], data.a2.l_link_ids[row_i]):
            neighbour_row = data.a2.id_to_index.get(str(neighbour_id))
            if neighbour_row is None or neighbour_row not in row_to_candidate:
                continue
            _uf_union(parent, candidate_i, row_to_candidate[neighbour_row])


def _union_junction_intersections(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    geometry_rows = [row_i for row_i in candidate_rows if _can_make_line(data.a2.polylines[row_i])]
    if len(geometry_rows) < 2:
        return

    lines = [data.a2.xy_lines[row_i] for row_i in geometry_rows]
    tree = shapely.STRtree(lines)
    for a_pos, line in enumerate(lines):
        a_row = geometry_rows[a_pos]
        for b_pos_raw in tree.query(line, predicate="intersects"):
            b_pos = int(b_pos_raw)
            if b_pos <= a_pos:
                continue
            b_row = geometry_rows[b_pos]
            if _intersects_within_z_tol(
                data.a2.polylines[a_row],
                line,
                data.a2.polylines[b_row],
                lines[b_pos],
                data.cfg.z_intersection_tol_m,
            ):
                _uf_union(parent, row_to_candidate[a_row], row_to_candidate[b_row])


def _intersects_within_z_tol(
    poly_a: NDArray[np.float64],
    line_a: shapely.LineString,
    poly_b: NDArray[np.float64],
    line_b: shapely.LineString,
    z_tol_m: float,
) -> bool:
    if not _can_make_line(poly_a) or not _can_make_line(poly_b):
        return False
    if not line_a.intersects(line_b):
        return False
    intersection = line_a.intersection(line_b)
    for point in _sample_intersection_points(intersection):
        z_a = _z_at_xy(poly_a, point.x, point.y)
        z_b = _z_at_xy(poly_b, point.x, point.y)
        if abs(z_a - z_b) <= z_tol_m:
            return True
    return False


def _junctions_from_parent(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
    *,
    entity_id_per_link: NDArray[np.int32] | None = None,
) -> tuple[Junction, ...]:
    root_to_entity: dict[int, int] = {}
    rows_by_entity: dict[int, list[int]] = {}
    for row_i in candidate_rows:
        root = _uf_find(parent, row_to_candidate[row_i])
        entity_id = root_to_entity.get(root)
        if entity_id is None:
            entity_id = len(root_to_entity)
            root_to_entity[root] = entity_id
        if entity_id_per_link is not None:
            entity_id_per_link[row_i] = np.int32(entity_id)
        rows_by_entity.setdefault(entity_id, []).append(row_i)

    final_ids = entity_id_per_link is not None
    return tuple(
        Junction.from_link_indices(
            entity_id=entity_id if final_ids else -1,
            link_indices=tuple(rows_by_entity[entity_id]),
            a2=data.a2,
        )
        for entity_id in range(len(rows_by_entity))
    )


def _union_junctions_by_proximity(
    data: SegmentationInput,
    junctions: tuple[Junction, ...],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    merge_candidates = tuple(junction for junction in junctions if not junction.cache.geom.is_empty)

    max_distance = data.cfg.junction_proximity_merge_dist_m
    if len(merge_candidates) < 2:
        return

    tree = shapely.STRtree([junction.cache.geom for junction in merge_candidates])
    n_unioned = 0
    for a_pos, junction_a in enumerate(merge_candidates):
        for b_pos_raw in tree.query(junction_a.cache.geom.buffer(max_distance)):
            b_pos = int(b_pos_raw)
            if b_pos <= a_pos:
                continue
            junction_b = merge_candidates[b_pos]
            candidate_a = row_to_candidate[junction_a.cache.representative_link_index]
            candidate_b = row_to_candidate[junction_b.cache.representative_link_index]
            if _uf_find(parent, candidate_a) == _uf_find(
                parent,
                candidate_b,
            ):
                continue
            if junction_a.cache.geom.distance(junction_b.cache.geom) > max_distance:
                continue
            if not _nearest_points_within_z_tol(
                data,
                junction_a,
                junction_b,
                data.cfg.z_intersection_tol_m,
            ):
                continue
            _uf_union(parent, candidate_a, candidate_b)
            n_unioned += 1
    if n_unioned:
        log.info(
            "JunctionStage: proximity-merged %d junction component pair(s) within %.2f m",
            n_unioned,
            max_distance,
        )


def _nearest_points_within_z_tol(
    data: SegmentationInput,
    junction_a: Junction,
    junction_b: Junction,
    z_tol_m: float,
) -> bool:
    point_a, point_b = nearest_points(junction_a.cache.geom, junction_b.cache.geom)
    z_a = _z_at_nearest_junction_link(data.a2, junction_a, point_a)
    z_b = _z_at_nearest_junction_link(data.a2, junction_b, point_b)
    return abs(z_a - z_b) <= z_tol_m


def _z_at_nearest_junction_link(
    a2: A2Data,
    junction: Junction,
    point: shapely.Point,
) -> float:
    valid_link_indices = tuple(
        link_i for link_i in junction.link_indices if _can_make_line(a2.polylines[link_i])
    )
    best_link_i = min(valid_link_indices, key=lambda link_i: a2.xy_lines[link_i].distance(point))
    return _z_at_xy(a2.polylines[best_link_i], point.x, point.y)


def _sample_intersection_points(geometry: object) -> list[shapely.Point]:
    points: list[shapely.Point] = []
    if not isinstance(geometry, shapely.Geometry):
        return points
    if geometry.is_empty:
        return points
    if isinstance(geometry, shapely.Point):
        points.append(geometry)
    elif isinstance(geometry, shapely.MultiPoint):
        points.extend(geometry.geoms)
    elif isinstance(geometry, shapely.LineString):
        points.append(geometry.interpolate(0.5, normalized=True))
    elif isinstance(geometry, shapely.MultiLineString):
        points.extend(line.interpolate(0.5, normalized=True) for line in geometry.geoms)
    elif isinstance(geometry, shapely.GeometryCollection):
        for part in geometry.geoms:
            points.extend(_sample_intersection_points(part))
    return points


def _z_at_xy(poly_xyz: NDArray[np.float64], x: float, y: float) -> float:
    if len(poly_xyz) == 0:
        return 0.0
    if len(poly_xyz) == 1:
        return float(poly_xyz[0, 2])
    xy = poly_xyz[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    line = shapely.LineString(xy)
    s = float(line.project(shapely.Point(x, y)))
    idx = int(np.searchsorted(cum, s) - 1)
    idx = max(0, min(idx, len(poly_xyz) - 2))
    seg_len = float(seg_lens[idx])
    if seg_len <= 0.0:
        return float(poly_xyz[idx, 2])
    t = (s - cum[idx]) / seg_len
    t = max(0.0, min(1.0, t))
    return float((1.0 - t) * poly_xyz[idx, 2] + t * poly_xyz[idx + 1, 2])


def _uf_find(parent: NDArray[np.int32], x: int) -> int:
    root = x
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[x]) != root:
        parent[x], x = np.int32(root), int(parent[x])
    return root


def _uf_union(parent: NDArray[np.int32], a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[rb] = np.int32(ra)
