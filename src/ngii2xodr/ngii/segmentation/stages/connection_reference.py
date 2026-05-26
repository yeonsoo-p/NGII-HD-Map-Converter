"""Connection-reference segmentation stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
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
            link_group = link_groups.get(connection.lateral_link_group_id)
            if link_group is None or link_group.reference_link_ref is None:
                continue
            endpoint_side = _endpoint_side_for_reference_link(context, connection, link_group)
            if endpoint_side is None:
                continue
            entity_id = len(entities)
            reference = ConnectionReference(
                id=entity_id,
                connection_id=connection.id,
                junction_id=connection.junction_id,
                endpoint_side=endpoint_side,
                link_ref=link_group.reference_link_ref,
                reversed_from_source=endpoint_side == "from",
            )
            entities.append(reference)
            for ref in (
                reference.link_ref,
                *connection.lateral_node_refs,
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


def _endpoint_side_for_reference_link(
    context: SegmentationContext,
    connection: JunctionConnection,
    link_group: LateralLinkGroup,
) -> EndpointSide | None:
    link_ref = link_group.reference_link_ref
    if link_ref is None:
        return None
    expected_node_ref = context.endpoint_node_ref_for_link_ref(link_ref, connection.endpoint_side)
    if expected_node_ref in connection.lateral_node_refs:
        return connection.endpoint_side
    for side in ("from", "to"):
        node_ref = context.endpoint_node_ref_for_link_ref(link_ref, side)
        if node_ref in connection.lateral_node_refs:
            return side
    return connection.endpoint_side
