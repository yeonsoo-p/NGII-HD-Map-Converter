"""Endpoint-node grouping for lateral link groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import LateralLinkGroup, LateralNodeGroup, StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage


class LateralNodeGroupStage:
    id: ClassVar[str] = "lateral_node_group"
    label: ClassVar[str] = "Lateral node groups"
    entity_label: ClassVar[str] = "Lateral node group"
    enabled_attr: ClassVar[str] = "enable_lateral_node_group"
    requires: ClassVar[tuple[str, ...]] = (LateralLinkGroupStage.id,)

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        link_group_result = previous_results.get(LateralLinkGroupStage.id)
        if link_group_result is None or not link_group_result.entities:
            return empty_result(self)

        entities: list[LateralNodeGroup] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = defaultdict(list)
        for link_group in link_group_result.entities:
            if not isinstance(link_group, LateralLinkGroup):
                continue
            for side in ("from", "to"):
                node_refs = context.endpoint_node_refs(link_group.link_refs, side)
                if not node_refs:
                    continue
                entity_id = len(entities)
                entities.append(
                    LateralNodeGroup(
                        id=entity_id,
                        lateral_link_group_id=link_group.id,
                        side=side,
                        link_refs=link_group.link_refs,
                        node_refs=node_refs,
                    )
                )
                for ref in (*link_group.link_refs, *node_refs):
                    entity_ids_by_ref[ref].append(entity_id)

        return StageResult(
            stage_id=self.id,
            label=self.label,
            entity_label=self.entity_label,
            entities=tuple(entities),
            entity_id_by_ref={},
            entity_ids_by_ref={ref: tuple(ids) for ref, ids in entity_ids_by_ref.items()},
        )
