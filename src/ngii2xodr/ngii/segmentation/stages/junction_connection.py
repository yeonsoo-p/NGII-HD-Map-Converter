"""Junction-connection segmentation stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import (
    EndpointSide,
    Junction,
    JunctionConnection,
    LateralLinkGroup,
    LateralNodeGroup,
    NodeLinkRelation,
    StageResult,
)
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage
from ngii2xodr.ngii.segmentation.stages.lateral_node_group import LateralNodeGroupStage
from ngii2xodr.ngii.segmentation.stages.node_link_relations import NodeLinkRelationsStage


class JunctionConnectionStage:
    id: ClassVar[str] = "junction_connection"
    label: ClassVar[str] = "Junction connections"
    entity_label: ClassVar[str] = "Junction connection"
    enabled_attr: ClassVar[str] = "enable_junction_connection"
    requires: ClassVar[tuple[str, ...]] = (
        NodeLinkRelationsStage.id,
        LateralLinkGroupStage.id,
        LateralNodeGroupStage.id,
        JunctionStage.id,
    )

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        node_relation_result = previous_results[NodeLinkRelationsStage.id]
        link_group_result = previous_results[LateralLinkGroupStage.id]
        node_group_result = previous_results[LateralNodeGroupStage.id]
        junction_result = previous_results[JunctionStage.id]

        node_relations = {
            relation.node_ref: relation
            for relation in node_relation_result.entities
            if isinstance(relation, NodeLinkRelation)
        }
        link_groups = {
            link_group.id: link_group
            for link_group in link_group_result.entities
            if isinstance(link_group, LateralLinkGroup)
        }
        node_groups = tuple(
            node_group
            for node_group in node_group_result.entities
            if isinstance(node_group, LateralNodeGroup)
        )
        junctions = tuple(
            junction for junction in junction_result.entities if isinstance(junction, Junction)
        )
        if not node_groups or not junctions:
            return empty_result(self)

        matches: list[_EndpointMatch] = []
        for node_group in node_groups:
            link_group = link_groups.get(node_group.lateral_link_group_id)
            if link_group is None:
                continue
            match = _best_junction_match(context, node_group, junctions, node_relations)
            if match is None:
                continue
            matches.append(
                _EndpointMatch(
                    node_group=node_group,
                    link_group=link_group,
                    junction_match=match,
                )
            )
        if not matches:
            return empty_result(self)

        components = _connection_components(context, matches)

        entities: list[JunctionConnection] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = {}
        for component in components:
            component_matches = tuple(matches[i] for i in component.match_indices)
            entity_id = len(entities)
            junction = component_matches[0].junction_match.junction
            node_refs = _unique_refs(
                ref for match in component_matches for ref in match.node_group.node_refs
            )
            junction_node_refs = _unique_refs(
                ref for match in component_matches for ref in match.junction_match.node_refs
            )
            link_refs = _unique_refs(
                ref for match in component_matches for ref in match.link_group.link_refs
            )
            lateral_link_group_ids = _unique_ints(
                match.link_group.id for match in component_matches
            )
            endpoint_sides = _unique_sides(match.node_group.side for match in component_matches)
            match_methods = _unique_strings(
                (
                    *(match.junction_match.method for match in component_matches),
                    *component.methods,
                )
            )
            entities.append(
                JunctionConnection(
                    id=entity_id,
                    junction_id=junction.id,
                    lateral_link_group_ids=lateral_link_group_ids,
                    endpoint_sides=endpoint_sides,
                    node_refs=node_refs,
                    junction_node_refs=junction_node_refs,
                    link_refs=link_refs,
                    match_methods=match_methods,
                )
            )
            for ref in (
                *link_refs,
                *node_refs,
                *junction_node_refs,
                *junction.link_refs,
            ):
                entity_ids_by_ref.setdefault(ref, []).append(entity_id)

        return StageResult(
            stage_id=self.id,
            label=self.label,
            entity_label=self.entity_label,
            entities=tuple(entities),
            entity_id_by_ref={},
            entity_ids_by_ref={ref: tuple(ids) for ref, ids in entity_ids_by_ref.items()},
        )


@dataclass(slots=True, frozen=True)
class _JunctionMatch:
    priority: int
    distance_m: float
    method: str
    junction: Junction
    node_refs: tuple[FeatureRef, ...]


@dataclass(slots=True, frozen=True)
class _EndpointMatch:
    node_group: LateralNodeGroup
    link_group: LateralLinkGroup
    junction_match: _JunctionMatch


@dataclass(slots=True, frozen=True)
class _ConnectionComponent:
    match_indices: tuple[int, ...]
    methods: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _PairCandidate:
    to_index: int
    from_index: int
    distance_m: float
    priority: int
    methods: tuple[str, ...]


def _best_junction_match(
    context: SegmentationContext,
    node_group: LateralNodeGroup,
    junctions: tuple[Junction, ...],
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> _JunctionMatch | None:
    matches: list[_JunctionMatch] = []
    for junction in junctions:
        shared = tuple(ref for ref in node_group.node_refs if ref in junction.endpoint_node_refs)
        if shared:
            matches.append(_JunctionMatch(0, 0.0, "shared_node", junction, shared))
            continue

        adjacent = _adjacent_node_refs(node_group.node_refs, junction, node_relations)
        if adjacent:
            matches.append(_JunctionMatch(1, 0.0, "graph_adjacency", junction, adjacent))
            continue

        nearby = _nearby_junction_node_refs(context, node_group.node_refs, junction)
        if nearby is not None:
            matches.append(nearby)

    if not matches:
        return None
    return min(matches, key=lambda match: (match.priority, match.distance_m, match.junction.id))


def _adjacent_node_refs(
    node_refs: tuple[FeatureRef, ...],
    junction: Junction,
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> tuple[FeatureRef, ...]:
    junction_link_refs = set(junction.link_refs)
    adjacent_refs: list[FeatureRef] = []
    for node_ref in node_refs:
        relation = node_relations.get(node_ref)
        if relation is None:
            continue
        if junction_link_refs.intersection(
            (*relation.incoming_link_refs, *relation.outgoing_link_refs)
        ):
            adjacent_refs.append(node_ref)
    return tuple(adjacent_refs)


def _nearby_junction_node_refs(
    context: SegmentationContext,
    node_refs: tuple[FeatureRef, ...],
    junction: Junction,
) -> _JunctionMatch | None:
    best_distance_m = context.cfg.junction_connection_node_merge_dist_m
    best_node_ref: FeatureRef | None = None
    for node_ref in node_refs:
        if not context.is_junction_node_ref(node_ref):
            continue
        node_point = context.node_point_for_ref(node_ref)
        if node_point is None:
            continue
        for junction_node_ref in junction.endpoint_node_refs:
            junction_point = context.node_point_for_ref(junction_node_ref)
            if junction_point is None:
                continue
            distance_m = node_point.distance(junction_point)
            if distance_m <= best_distance_m:
                best_distance_m = distance_m
                best_node_ref = junction_node_ref
    if best_node_ref is None:
        return None
    return _JunctionMatch(2, best_distance_m, "node_proximity", junction, (best_node_ref,))


def _connection_components(
    context: SegmentationContext,
    matches: list[_EndpointMatch],
) -> tuple[_ConnectionComponent, ...]:
    indices_by_junction: dict[int, list[int]] = {}
    for index, match in enumerate(matches):
        indices_by_junction.setdefault(match.junction_match.junction.id, []).append(index)

    components: list[_ConnectionComponent] = []
    for junction_id in sorted(indices_by_junction):
        indices = tuple(indices_by_junction[junction_id])
        pair_candidates = sorted(
            (
                candidate
                for left_pos, left in enumerate(indices)
                for right in indices[left_pos + 1 :]
                if (candidate := _pair_candidate(context, matches, left, right)) is not None
            ),
            key=lambda candidate: (
                candidate.priority,
                candidate.distance_m,
                candidate.to_index,
                candidate.from_index,
            ),
        )
        used: set[int] = set()
        for candidate in pair_candidates:
            if candidate.to_index in used or candidate.from_index in used:
                continue
            used.update((candidate.to_index, candidate.from_index))
            components.append(
                _ConnectionComponent(
                    match_indices=(candidate.to_index, candidate.from_index),
                    methods=candidate.methods,
                )
            )
        for index in indices:
            if index not in used:
                components.append(
                    _ConnectionComponent(match_indices=(index,), methods=("one_sided",))
                )
    return tuple(components)


def _pair_candidate(
    context: SegmentationContext,
    matches: list[_EndpointMatch],
    left_index: int,
    right_index: int,
) -> _PairCandidate | None:
    left = matches[left_index]
    right = matches[right_index]
    if left.node_group.side == right.node_group.side:
        return None
    to_index = left_index if left.node_group.side == "to" else right_index
    from_index = left_index if left.node_group.side == "from" else right_index
    to_match = matches[to_index]
    from_match = matches[from_index]

    distance_m = context.endpoint_node_distance_m(
        to_match.node_group.node_refs,
        from_match.node_group.node_refs,
    )
    if distance_m is None or distance_m > context.cfg.junction_connection_node_merge_dist_m:
        return None
    dot = _endpoint_tangent_dot(context, to_match, from_match)
    if dot is None or dot > -context.cfg.junction_connection_opposite_direction_dot_min:
        return None

    shared_keys = set(to_match.node_group.node_group_keys) & set(
        from_match.node_group.node_group_keys
    )
    methods = ("node_group_key", "opposite_direction") if shared_keys else ("opposite_direction",)
    priority = 0 if shared_keys else 1
    return _PairCandidate(
        to_index=to_index,
        from_index=from_index,
        distance_m=distance_m,
        priority=priority,
        methods=methods,
    )


def _endpoint_tangent_dot(
    context: SegmentationContext,
    to_match: _EndpointMatch,
    from_match: _EndpointMatch,
) -> float | None:
    to_tangent = _endpoint_tangent(context, to_match)
    from_tangent = _endpoint_tangent(context, from_match)
    if to_tangent is None or from_tangent is None:
        return None
    return to_tangent[0] * from_tangent[0] + to_tangent[1] * from_tangent[1]


def _endpoint_tangent(
    context: SegmentationContext,
    match: _EndpointMatch,
) -> tuple[float, float] | None:
    link_ref = match.link_group.reference_link_ref or _first_ref(match.link_group.link_refs)
    if link_ref is None:
        return None
    endpoint_geometry = context.endpoint_geometry_for_link_ref(link_ref, match.node_group.side)
    if endpoint_geometry is None:
        return None
    return endpoint_geometry[1]


def _first_ref(refs: tuple[FeatureRef, ...]) -> FeatureRef | None:
    return refs[0] if refs else None


def _unique_refs(refs: Iterable[FeatureRef]) -> tuple[FeatureRef, ...]:
    output: list[FeatureRef] = []
    seen: set[FeatureRef] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        output.append(ref)
    return tuple(output)


def _unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


def _unique_sides(values: Iterable[EndpointSide]) -> tuple[EndpointSide, ...]:
    output: list[EndpointSide] = []
    seen: set[EndpointSide] = set()
    for value in values:
        if value not in {"from", "to"} or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
