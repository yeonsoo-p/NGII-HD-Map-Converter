"""Connection-reference segmentation stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import (
    ConnectionReference,
    EndpointSide,
    JunctionConnection,
    LateralLinkGroup,
    StageResult,
)
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage


class ConnectionReferenceStage:
    id: ClassVar[str] = "connection_reference"
    label: ClassVar[str] = "Connection references"
    entity_label: ClassVar[str] = "Connection reference"
    enabled_attr: ClassVar[str] = "enable_connection_reference"
    requires: ClassVar[tuple[str, ...]] = (
        LateralLinkGroupStage.id,
        JunctionConnectionStage.id,
    )

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        link_group_result = previous_results[LateralLinkGroupStage.id]
        connection_result = previous_results[JunctionConnectionStage.id]
        link_groups = {
            link_group.id: link_group
            for link_group in link_group_result.entities
            if isinstance(link_group, LateralLinkGroup)
        }
        connections = tuple(
            connection
            for connection in connection_result.entities
            if isinstance(connection, JunctionConnection)
        )
        if not connections:
            return empty_result(self)

        entities: list[ConnectionReference] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = defaultdict(list)
        for connection in connections:
            candidate = _reference_candidate(context, connection, link_groups)
            if candidate is None:
                continue
            endpoint_geometry = context.endpoint_geometry_for_link_ref(
                candidate.link_ref, candidate.endpoint_side
            )
            if endpoint_geometry is None:
                continue
            anchor_xyz, native_tangent_xy = endpoint_geometry
            reversed_from_source = candidate.endpoint_side == "from"
            tangent_xy = (
                (-native_tangent_xy[0], -native_tangent_xy[1])
                if reversed_from_source
                else native_tangent_xy
            )
            entity_id = len(entities)
            reference = ConnectionReference(
                id=entity_id,
                connection_id=connection.id,
                junction_id=connection.junction_id,
                endpoint_side=candidate.endpoint_side,
                link_ref=candidate.link_ref,
                anchor_xyz=anchor_xyz,
                tangent_xy=tangent_xy,
                reversed_from_source=reversed_from_source,
                selection_source=candidate.selection_source,
            )
            entities.append(reference)
            for ref in (
                reference.link_ref,
                *connection.node_refs,
                *connection.junction_node_refs,
            ):
                entity_ids_by_ref[ref].append(entity_id)

        return StageResult(
            stage_id=self.id,
            label=self.label,
            entity_label=self.entity_label,
            entities=tuple(entities),
            entity_id_by_ref={},
            entity_ids_by_ref={ref: tuple(ids) for ref, ids in entity_ids_by_ref.items()},
        )


@dataclass(slots=True, frozen=True)
class _ReferenceCandidate:
    link_ref: FeatureRef
    endpoint_side: EndpointSide
    lateral_link_group_id: int
    lane_no: int | None
    selection_source: str


def _reference_candidate(
    context: SegmentationContext,
    connection: JunctionConnection,
    link_groups: Mapping[int, LateralLinkGroup],
) -> _ReferenceCandidate | None:
    candidates: list[_ReferenceCandidate] = []
    for group_id in connection.lateral_link_group_ids:
        link_group = link_groups.get(group_id)
        if link_group is None or link_group.reference_link_ref is None:
            continue
        endpoint_side = _endpoint_side_for_reference_link(
            context, connection, link_group.reference_link_ref
        )
        if endpoint_side is None:
            continue
        source = link_group.ordering_source or "lateral_topology"
        if endpoint_side == "to":
            source = f"{source}:toward_junction"
        else:
            source = f"{source}:reversed_toward_junction"
        candidates.append(
            _ReferenceCandidate(
                link_ref=link_group.reference_link_ref,
                endpoint_side=endpoint_side,
                lateral_link_group_id=group_id,
                lane_no=context.link_lane_no_for_ref(link_group.reference_link_ref),
                selection_source=source,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=_reference_candidate_key)


def _endpoint_side_for_reference_link(
    context: SegmentationContext,
    connection: JunctionConnection,
    link_ref: FeatureRef,
) -> EndpointSide | None:
    connection_nodes = {*connection.node_refs, *connection.junction_node_refs}
    for side in ("to", "from"):
        node_ref = context.endpoint_node_ref_for_link_ref(link_ref, side)
        if node_ref in connection_nodes:
            return side
    return None


def _reference_candidate_key(candidate: _ReferenceCandidate) -> tuple[int, int, int, str]:
    direction_priority = 0 if candidate.endpoint_side == "to" else 1
    lane_no_priority = candidate.lane_no if candidate.lane_no is not None else 1_000_000
    return (
        direction_priority,
        lane_no_priority,
        candidate.lateral_link_group_id,
        candidate.link_ref.feature_id,
    )
