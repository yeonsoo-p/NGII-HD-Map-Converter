"""Junction-connection segmentation stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import connected_components_from_pairs
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

        merge_pairs = _merge_pairs(context, matches, node_relations)
        components = connected_components_from_pairs(
            tuple(range(len(matches))),
            ((pair.left, pair.right) for pair in merge_pairs),
        )
        methods_by_component = _merge_methods_by_component(components, merge_pairs)

        entities: list[JunctionConnection] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = {}
        for component_i, match_indices in enumerate(components):
            component_matches = tuple(matches[i] for i in match_indices)
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
                    *methods_by_component[component_i],
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
class _MergePair:
    left: int
    right: int
    method: str


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


def _merge_pairs(
    context: SegmentationContext,
    matches: list[_EndpointMatch],
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> tuple[_MergePair, ...]:
    pairs: list[_MergePair] = []
    for left in range(len(matches)):
        for right in range(left + 1, len(matches)):
            if (
                matches[left].junction_match.junction.id
                != matches[right].junction_match.junction.id
            ):
                continue
            method = _merge_method(context, matches[left], matches[right], node_relations)
            if method:
                pairs.append(_MergePair(left, right, method))
    return tuple(pairs)


def _merge_method(
    context: SegmentationContext,
    left: _EndpointMatch,
    right: _EndpointMatch,
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> str:
    left_nodes = {*left.node_group.node_refs, *left.junction_match.node_refs}
    right_nodes = {*right.node_group.node_refs, *right.junction_match.node_refs}
    if left_nodes & right_nodes:
        return "shared_node"
    if _endpoint_groups_are_graph_adjacent(left, right, node_relations):
        return "graph_adjacency"
    if _endpoint_groups_are_nearby(context, left.node_group.node_refs, right.node_group.node_refs):
        return "node_proximity"
    return ""


def _endpoint_groups_are_graph_adjacent(
    left: _EndpointMatch,
    right: _EndpointMatch,
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> bool:
    left_links = set(left.link_group.link_refs)
    right_links = set(right.link_group.link_refs)
    return _nodes_touch_links(left.node_group.node_refs, right_links, node_relations) or (
        _nodes_touch_links(right.node_group.node_refs, left_links, node_relations)
    )


def _nodes_touch_links(
    node_refs: tuple[FeatureRef, ...],
    link_refs: set[FeatureRef],
    node_relations: Mapping[FeatureRef, NodeLinkRelation],
) -> bool:
    for node_ref in node_refs:
        relation = node_relations.get(node_ref)
        if relation is None:
            continue
        if link_refs.intersection((*relation.incoming_link_refs, *relation.outgoing_link_refs)):
            return True
    return False


def _endpoint_groups_are_nearby(
    context: SegmentationContext,
    left_node_refs: tuple[FeatureRef, ...],
    right_node_refs: tuple[FeatureRef, ...],
) -> bool:
    max_distance_m = context.cfg.junction_connection_node_merge_dist_m
    if max_distance_m <= 0.0:
        return False
    for left_ref in left_node_refs:
        left_point = context.node_point_for_ref(left_ref)
        if left_point is None:
            continue
        for right_ref in right_node_refs:
            right_point = context.node_point_for_ref(right_ref)
            if right_point is not None and left_point.distance(right_point) <= max_distance_m:
                return True
    return False


def _merge_methods_by_component(
    components: tuple[tuple[int, ...], ...],
    merge_pairs: tuple[_MergePair, ...],
) -> tuple[tuple[str, ...], ...]:
    methods_by_component: list[tuple[str, ...]] = []
    for component in components:
        component_set = set(component)
        methods_by_component.append(
            _unique_strings(
                pair.method
                for pair in merge_pairs
                if pair.left in component_set and pair.right in component_set
            )
        )
    return tuple(methods_by_component)


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
