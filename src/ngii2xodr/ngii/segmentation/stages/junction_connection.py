"""Junction-connection segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import (
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

        entities: list[JunctionConnection] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = {}
        for node_group in node_groups:
            link_group = link_groups.get(node_group.lateral_link_group_id)
            if link_group is None:
                continue
            match = _best_junction_match(context, node_group, junctions, node_relations)
            if match is None:
                continue
            entity_id = len(entities)
            entities.append(
                JunctionConnection(
                    id=entity_id,
                    junction_id=match.junction.id,
                    lateral_link_group_id=link_group.id,
                    endpoint_side=node_group.side,
                    lateral_node_refs=node_group.node_refs,
                    junction_node_refs=match.node_refs,
                    link_refs=link_group.link_refs,
                )
            )
            for ref in (
                *link_group.link_refs,
                *node_group.node_refs,
                *match.node_refs,
                *match.junction.link_refs,
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
    junction: Junction
    node_refs: tuple[FeatureRef, ...]


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
            matches.append(_JunctionMatch(0, 0.0, junction, shared))
            continue

        adjacent = _adjacent_node_refs(node_group.node_refs, junction, node_relations)
        if adjacent:
            matches.append(_JunctionMatch(1, 0.0, junction, adjacent))
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
    return _JunctionMatch(2, best_distance_m, junction, (best_node_ref,))
