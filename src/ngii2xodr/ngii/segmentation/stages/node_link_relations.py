"""Node-to-link graph relationship stage."""

from __future__ import annotations

from collections import defaultdict
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
        incoming: dict[str, list[FeatureRef]] = defaultdict(list)
        outgoing: dict[str, list[FeatureRef]] = defaultdict(list)
        for link_i, link in enumerate(context.link_store.features):
            link_ref = context.ref_for_link_index(link_i)
            from_node_id = getattr(link, "from_node_id", "")
            to_node_id = getattr(link, "to_node_id", "")
            if from_node_id and context.node_store.get(from_node_id) is not None:
                outgoing[from_node_id].append(link_ref)
            if to_node_id and context.node_store.get(to_node_id) is not None:
                incoming[to_node_id].append(link_ref)

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
                    incoming_link_refs=tuple(incoming.get(node.id, ())),
                    outgoing_link_refs=tuple(outgoing.get(node.id, ())),
                )
            )
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )
