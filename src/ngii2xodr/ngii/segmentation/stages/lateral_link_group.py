"""Lateral lane-link grouping stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import connected_components_from_pairs
from ngii2xodr.ngii.segmentation.model import LateralLinkGroup, StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.uturn import UTurnStage


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
            for i in context.rows_for_filter("ordinary_link")
            if context.ref_for_link_index(i) not in uturn_refs
        ]
        if not candidate_rows:
            return empty_result(self, skipped_reason="schema has no lateral link candidates")
        candidate_set = set(candidate_rows)
        components = connected_components_from_pairs(
            candidate_rows,
            (
                (row_i, neighbour_row)
                for row_i in candidate_rows
                for neighbour_row in context.lateral_neighbor_rows_for_link_index(row_i)
                if neighbour_row in candidate_set
            ),
        )

        entities: list[LateralLinkGroup] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}
        for entity_id, rows in enumerate(components):
            refs = tuple(context.ref_for_link_index(i) for i in rows)
            pocket_refs = _pocket_link_refs(context, refs, uturn_refs)
            reference_ref, ordering_source, ordering_warning = _reference_link_ref(
                context, rows, pocket_refs
            )
            for ref in refs:
                entity_id_by_ref[ref] = entity_id
            entities.append(
                LateralLinkGroup(
                    id=entity_id,
                    link_refs=refs,
                    pocket_link_refs=pocket_refs,
                    reference_link_ref=reference_ref,
                    ordering_source=ordering_source,
                    ordering_warning=ordering_warning,
                )
            )
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )


def _pocket_link_refs(
    context: SegmentationContext,
    refs: tuple[FeatureRef, ...],
    uturn_refs: set[FeatureRef],
) -> tuple[FeatureRef, ...]:
    pocket_refs: list[FeatureRef] = []
    for ref in refs:
        if ref in uturn_refs or context.link_turn_for_ref(ref) in {"1", "3"}:
            pocket_refs.append(ref)
    return tuple(pocket_refs)


def _reference_link_ref(
    context: SegmentationContext,
    rows: tuple[int, ...],
    pocket_refs: tuple[FeatureRef, ...],
) -> tuple[FeatureRef | None, str, str]:
    row_refs = {context.ref_for_link_index(row_i) for row_i in rows}
    pocket_set = set(pocket_refs)
    non_pocket_refs = row_refs - pocket_set
    if not non_pocket_refs:
        return None, "lateral_topology", "no non-pocket links"

    candidates: list[FeatureRef] = []
    for row_i in rows:
        ref = context.ref_for_link_index(row_i)
        if ref in pocket_set:
            continue
        left_refs = context.lateral_neighbor_refs_for_link_index(row_i, "left")
        if not any(left_ref in non_pocket_refs for left_ref in left_refs):
            candidates.append(ref)
    if len(candidates) == 1:
        return candidates[0], "lateral_topology", ""
    if not candidates:
        return None, "lateral_topology", "no non-pocket left edge"
    return None, "lateral_topology", "multiple non-pocket left edges"
