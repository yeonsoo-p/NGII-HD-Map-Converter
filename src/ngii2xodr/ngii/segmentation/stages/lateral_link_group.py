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
        pocket_rows = set(context.rows_for_filter("pocket_link"))
        pocket_filter_refs = {context.ref_for_link_index(i) for i in pocket_rows}
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
            pocket_refs = _pocket_link_refs(refs, uturn_refs, pocket_filter_refs)
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
    refs: tuple[FeatureRef, ...],
    uturn_refs: set[FeatureRef],
    pocket_filter_refs: set[FeatureRef],
) -> tuple[FeatureRef, ...]:
    return tuple(ref for ref in refs if ref in uturn_refs or ref in pocket_filter_refs)


def _reference_link_ref(
    context: SegmentationContext,
    rows: tuple[int, ...],
    pocket_refs: tuple[FeatureRef, ...],
) -> tuple[FeatureRef | None, str, str]:
    row_refs = {context.ref_for_link_index(row_i) for row_i in rows}
    pocket_set = set(pocket_refs)
    non_pocket_refs = row_refs - pocket_set
    if not non_pocket_refs:
        return None, "lane_no", "no non-pocket links"

    lane_no_candidates = [
        ref
        for ref in sorted(non_pocket_refs, key=lambda item: item.feature_id)
        if context.link_lane_no_for_ref(ref) == 1
    ]
    if len(lane_no_candidates) == 1:
        return lane_no_candidates[0], "lane_no", ""
    lane_no_warning = ""
    if len(lane_no_candidates) > 1:
        lane_no_warning = "multiple lane_no_1 candidates; "

    candidates: list[FeatureRef] = []
    for row_i in rows:
        ref = context.ref_for_link_index(row_i)
        if ref in pocket_set:
            continue
        left_refs = context.lateral_neighbor_refs_for_link_index(row_i, "left")
        if not any(left_ref in non_pocket_refs for left_ref in left_refs):
            candidates.append(ref)
    if len(candidates) == 1:
        warning = f"{lane_no_warning}topology selected".strip() if lane_no_warning else ""
        return candidates[0], "lateral_topology", warning
    if not candidates:
        return None, "lateral_topology", f"{lane_no_warning}no non-pocket left edge".strip()
    return None, "lateral_topology", f"{lane_no_warning}multiple non-pocket left edges".strip()
