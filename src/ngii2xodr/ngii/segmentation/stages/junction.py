"""Junction segmentation stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import ClassVar

import shapely

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import (
    connected_components_from_pairs,
    intersects_within_z_tol,
)
from ngii2xodr.ngii.segmentation.model import (
    EndpointSide,
    Junction,
    LateralLinkGroup,
    LateralNodeGroup,
    StageResult,
)
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage
from ngii2xodr.ngii.segmentation.stages.lateral_node_group import LateralNodeGroupStage


class JunctionStage:
    id: ClassVar[str] = "junction"
    label: ClassVar[str] = "Junctions"
    entity_label: ClassVar[str] = "Junction"
    enabled_attr: ClassVar[str] = "enable_junction"
    requires: ClassVar[tuple[str, ...]] = (
        LateralLinkGroupStage.id,
        LateralNodeGroupStage.id,
    )

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        link_group_result = previous_results[LateralLinkGroupStage.id]
        node_group_result = previous_results[LateralNodeGroupStage.id]
        link_groups = tuple(
            link_group
            for link_group in link_group_result.entities
            if isinstance(link_group, LateralLinkGroup)
        )
        node_groups = tuple(
            node_group
            for node_group in node_group_result.entities
            if isinstance(node_group, LateralNodeGroup)
        )
        candidate_rows = list(context.rows_for_filter("junction_link"))
        if not candidate_rows:
            return empty_result(self, skipped_reason="schema has no junction link candidates")

        candidate_set = set(candidate_rows)
        link_group_by_ref = _lateral_link_group_by_ref(link_groups)
        promoted_rows, promotion_pairs = _promotion_rows_and_pairs(
            context,
            candidate_rows,
            candidate_set,
            link_group_by_ref,
        )
        component_rows = _unique_rows((*candidate_rows, *promoted_rows))
        components = connected_components_from_pairs(
            component_rows,
            (
                *_lateral_pairs(context, candidate_rows, candidate_set),
                *_intersection_pairs(context, candidate_rows),
                *_shared_endpoint_node_pairs(context, candidate_rows),
                *_endpoint_scoped_pairs(context, node_groups, candidate_set),
                *promotion_pairs,
            ),
        )

        entities: list[Junction] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}
        for entity_id, rows in enumerate(components):
            refs = tuple(context.ref_for_link_index(i) for i in rows)
            endpoint_node_refs = _junction_endpoint_node_refs(context, refs)
            for ref in refs:
                entity_id_by_ref[ref] = entity_id
            for node_ref in endpoint_node_refs:
                entity_id_by_ref[node_ref] = entity_id
            entities.append(
                Junction(id=entity_id, link_refs=refs, endpoint_node_refs=endpoint_node_refs)
            )
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )


def _lateral_pairs(
    context: SegmentationContext,
    candidate_rows: list[int],
    candidate_set: set[int],
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for row_i in candidate_rows:
        for neighbour_row in context.lateral_neighbor_rows_for_link_index(row_i):
            if neighbour_row in candidate_set:
                pairs.append((row_i, neighbour_row))
    return tuple(pairs)


def _lateral_link_group_by_ref(
    link_groups: tuple[LateralLinkGroup, ...],
) -> dict[FeatureRef, LateralLinkGroup]:
    return {ref: link_group for link_group in link_groups for ref in link_group.link_refs}


def _promotion_rows_and_pairs(
    context: SegmentationContext,
    seed_rows: list[int],
    seed_set: set[int],
    link_group_by_ref: Mapping[FeatureRef, LateralLinkGroup],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    rows: list[int] = []
    pairs: list[tuple[int, int]] = []
    seen_rows: set[int] = set()
    for seed_row in seed_rows:
        seed_ref = context.ref_for_link_index(seed_row)
        for side in ("from", "to"):
            for promoted_ref in _direction_compatible_link_refs(context, seed_ref, side):
                promoted_row = context.link_index_for_ref(promoted_ref)
                if promoted_row is None or promoted_row in seed_set:
                    continue
                promoted_refs = _expanded_lateral_group_refs(
                    promoted_ref,
                    link_group_by_ref,
                )
                for expanded_ref in promoted_refs:
                    expanded_row = context.link_index_for_ref(expanded_ref)
                    if expanded_row is None or expanded_row in seed_set:
                        continue
                    pairs.append((seed_row, expanded_row))
                    if expanded_row not in seen_rows:
                        seen_rows.add(expanded_row)
                        rows.append(expanded_row)
    return tuple(rows), tuple(pairs)


def _direction_compatible_link_refs(
    context: SegmentationContext,
    seed_ref: FeatureRef,
    side: EndpointSide,
) -> tuple[FeatureRef, ...]:
    node_ref = context.endpoint_node_ref_for_link_ref(seed_ref, side)
    if node_ref is None:
        return ()
    if side == "from":
        return context.outgoing_link_refs(node_ref)
    return context.incoming_link_refs(node_ref)


def _expanded_lateral_group_refs(
    promoted_ref: FeatureRef,
    link_group_by_ref: Mapping[FeatureRef, LateralLinkGroup],
) -> tuple[FeatureRef, ...]:
    link_group = link_group_by_ref.get(promoted_ref)
    return (promoted_ref,) if link_group is None else link_group.link_refs


def _unique_rows(rows: Iterable[int]) -> tuple[int, ...]:
    output: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        output.append(row)
    return tuple(output)


def _intersection_pairs(
    context: SegmentationContext,
    candidate_rows: list[int],
) -> tuple[tuple[int, int], ...]:
    geometry_rows: list[int] = []
    lines: list[shapely.LineString] = []
    for row_i in candidate_rows:
        line = context.link_lines[row_i]
        if line is not None:
            geometry_rows.append(row_i)
            lines.append(line)
    if len(geometry_rows) < 2:
        return ()
    pairs: list[tuple[int, int]] = []
    tree = shapely.STRtree(lines)
    for a_pos, line in enumerate(lines):
        a_row = geometry_rows[a_pos]
        for b_pos_raw in tree.query(line, predicate="intersects"):
            b_pos = int(b_pos_raw)
            if b_pos <= a_pos:
                continue
            b_row = geometry_rows[b_pos]
            other = lines[b_pos]
            if intersects_within_z_tol(
                context.link_store.features[a_row].polyline,
                line,
                context.link_store.features[b_row].polyline,
                other,
                context.cfg.z_intersection_tol_m,
            ):
                pairs.append((a_row, b_row))
    return tuple(pairs)


def _shared_endpoint_node_pairs(
    context: SegmentationContext,
    candidate_rows: list[int],
) -> tuple[tuple[int, int], ...]:
    rows_by_node_ref: dict[FeatureRef, list[int]] = defaultdict(list)
    for row_i in candidate_rows:
        link_ref = context.ref_for_link_index(row_i)
        for node_ref in context.endpoint_node_refs_for_link_ref(link_ref):
            rows_by_node_ref[node_ref].append(row_i)
    pairs: list[tuple[int, int]] = []
    for rows in rows_by_node_ref.values():
        pairs.extend(_pairs_from_rows(rows))
    return tuple(pairs)


def _endpoint_scoped_pairs(
    context: SegmentationContext,
    node_groups: tuple[LateralNodeGroup, ...],
    candidate_set: set[int],
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for node_group in node_groups:
        rows = _junction_rows_for_node_group(context, node_group, candidate_set)
        pairs.extend(_pairs_from_rows(rows))
    return tuple(pairs)


def _junction_rows_for_node_group(
    context: SegmentationContext,
    node_group: LateralNodeGroup,
    candidate_set: set[int],
) -> tuple[int, ...]:
    rows: list[int] = []
    seen: set[int] = set()
    for node_ref in node_group.node_refs:
        for link_ref in (
            *context.incoming_link_refs(node_ref),
            *context.outgoing_link_refs(node_ref),
        ):
            row_i = context.link_index_for_ref(link_ref)
            if row_i is None or row_i not in candidate_set or row_i in seen:
                continue
            seen.add(row_i)
            rows.append(row_i)
    return tuple(rows)


def _pairs_from_rows(rows: Iterable[int]) -> tuple[tuple[int, int], ...]:
    row_tuple = tuple(rows)
    return tuple(
        (left, right)
        for left_pos, left in enumerate(row_tuple)
        for right in row_tuple[left_pos + 1 :]
    )


def _junction_endpoint_node_refs(
    context: SegmentationContext, link_refs: tuple[FeatureRef, ...]
) -> tuple[FeatureRef, ...]:
    node_refs: list[FeatureRef] = []
    seen: set[FeatureRef] = set()
    for link_ref in link_refs:
        for node_ref in context.endpoint_node_refs_for_link_ref(link_ref):
            if node_ref in seen:
                continue
            seen.add(node_ref)
            node_refs.append(node_ref)
    return tuple(node_refs)
