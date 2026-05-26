"""Node-to-link graph relationship stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import NodeLinkRelation, StageResult


class NodeLinkRelationsStage:
    id: ClassVar[str] = "node_link_relations"
    label: ClassVar[str] = "Node link relations"
    entity_label: ClassVar[str] = "Node link relation"
    enabled_attr: ClassVar[str] = "enable_node_link_relations"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        entities: list[NodeLinkRelation] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}
        for node in context.node_store.features:
            node_ref = FeatureRef(context.node_attr, node.id)
            entity_id = len(entities)
            entity_id_by_ref[node_ref] = entity_id
            entities.append(
                NodeLinkRelation(
                    id=entity_id,
                    node_ref=node_ref,
                    incoming_link_refs=context.incoming_link_refs(node_ref),
                    outgoing_link_refs=context.outgoing_link_refs(node_ref),
                )
            )
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )
