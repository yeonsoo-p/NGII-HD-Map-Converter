"""Lateral lane-link grouping stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import uf_find, uf_union
from ngii2xodr.ngii.segmentation.model import LateralLinkGroup, StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.uturn import UTurnStage

_LATERAL_GROUP_LINK_TYPE = "6"


class LateralLinkGroupStage:
    id: ClassVar[str] = "lateral_link_group"
    label: ClassVar[str] = "Lateral link groups"
    entity_label: ClassVar[str] = "Lateral link group"
    enabled_attr: ClassVar[str] = "enable_lateral_link_group"
    requires: ClassVar[tuple[str, ...]] = (UTurnStage.id,)

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        uturn_refs = set(
            previous_results.get(UTurnStage.id, empty_result(UTurnStage)).entity_id_by_ref
        )
        candidate_rows = [
            i
            for i in context.a2_rows_for_link_type(_LATERAL_GROUP_LINK_TYPE)
            if context.ref_for_a2_index(i) not in uturn_refs
        ]
        if not candidate_rows:
            return empty_result(self)
        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)
        for row_i in candidate_rows:
            link = context.dataset.a2_link.features[row_i]
            for neighbour_id in (link.r_link_id, link.l_link_id):
                neighbour_row = context.dataset.a2_link.id_to_index.get(str(neighbour_id))
                if neighbour_row is not None and neighbour_row in row_to_candidate:
                    uf_union(parent, row_to_candidate[row_i], row_to_candidate[neighbour_row])

        rows_by_entity: dict[int, list[int]] = {}
        root_to_entity: dict[int, int] = {}
        for row_i in candidate_rows:
            root = uf_find(parent, row_to_candidate[row_i])
            entity_id = root_to_entity.setdefault(root, len(root_to_entity))
            rows_by_entity.setdefault(entity_id, []).append(row_i)

        entities: list[LateralLinkGroup] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}
        for entity_id in range(len(rows_by_entity)):
            refs = tuple(context.ref_for_a2_index(i) for i in rows_by_entity[entity_id])
            for ref in refs:
                entity_id_by_ref[ref] = entity_id
            entities.append(LateralLinkGroup(id=entity_id, link_refs=refs))
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )
