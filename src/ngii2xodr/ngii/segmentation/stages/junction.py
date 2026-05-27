"""Junction segmentation stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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

        seed_set = set(candidate_rows)
        link_group_by_ref = _lateral_link_group_by_ref(link_groups)
        endpoint_promoted_rows, endpoint_promotion_pairs = _promotion_rows_and_pairs(
            context,
            candidate_rows,
            seed_set,
            link_group_by_ref,
        )
        lateral_promoted_rows, lateral_promotion_pairs = _lateral_promotion_rows_and_pairs(
            context,
            candidate_rows,
            seed_set,
            link_group_by_ref,
        )
        promoted_rows = _unique_rows((*endpoint_promoted_rows, *lateral_promoted_rows))
        bridge_result = _bounded_lateral_node_bridges(
            context,
            node_groups,
            candidate_rows,
            promoted_rows,
        )
        component_rows = _unique_rows((*candidate_rows, *promoted_rows, *bridge_result.owner_rows))
        components = connected_components_from_pairs(
            component_rows,
            (
                *_lateral_pairs(context, candidate_rows, seed_set),
                *_intersection_pairs(context, candidate_rows),
                *_shared_endpoint_node_pairs(context, candidate_rows),
                *endpoint_promotion_pairs,
                *lateral_promotion_pairs,
                *bridge_result.pairs,
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


@dataclass(slots=True, frozen=True)
class _BoundedBridgeResult:
    owner_rows: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


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


def _bounded_lateral_node_bridges(
    context: SegmentationContext,
    node_groups: tuple[LateralNodeGroup, ...],
    seed_rows: list[int],
    promoted_rows: tuple[int, ...],
) -> _BoundedBridgeResult:
    seed_set = set(seed_rows)
    promoted_set = set(promoted_rows)
    bridge_set = seed_set | promoted_set
    owner_rows: list[int] = []
    owner_seen: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for node_group in node_groups:
        rows_by_side = _junction_rows_by_endpoint_side_for_node_group(
            context, node_group, bridge_set
        )
        for side, rows in rows_by_side.items():
            if len(rows) < 2:
                continue
            pairs.extend(_pairs_from_rows(rows))
            if not _bridges_seed_and_promoted(rows, seed_set, promoted_set):
                continue
            for owner_row in _owner_rows_for_bridge_side(context, node_group, side):
                pairs.extend((owner_row, row) for row in rows)
                if owner_row not in bridge_set and owner_row not in owner_seen:
                    owner_seen.add(owner_row)
                    owner_rows.append(owner_row)
    return _BoundedBridgeResult(owner_rows=tuple(owner_rows), pairs=tuple(pairs))


def _bridges_seed_and_promoted(
    rows: tuple[int, ...], seed_set: set[int], promoted_set: set[int]
) -> bool:
    return any(row in seed_set for row in rows) and any(row in promoted_set for row in rows)


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


def _lateral_promotion_rows_and_pairs(
    context: SegmentationContext,
    seed_rows: list[int],
    seed_set: set[int],
    link_group_by_ref: Mapping[FeatureRef, LateralLinkGroup],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    rows: list[int] = []
    pairs: list[tuple[int, int]] = []
    seen_rows: set[int] = set()
    for seed_row in seed_rows:
        for neighbour_row in context.lateral_neighbor_rows_for_link_index(seed_row):
            if neighbour_row in seed_set:
                continue
            neighbour_ref = context.ref_for_link_index(neighbour_row)
            link_group = link_group_by_ref.get(neighbour_ref)
            if link_group is None:
                continue
            for promoted_ref in link_group.link_refs:
                promoted_row = context.link_index_for_ref(promoted_ref)
                if promoted_row is None or promoted_row in seed_set:
                    continue
                pairs.append((seed_row, promoted_row))
                if promoted_row not in seen_rows:
                    seen_rows.add(promoted_row)
                    rows.append(promoted_row)
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


def _junction_rows_by_endpoint_side_for_node_group(
    context: SegmentationContext,
    node_group: LateralNodeGroup,
    candidate_set: set[int],
) -> dict[EndpointSide, tuple[int, ...]]:
    rows_by_side: dict[EndpointSide, list[int]] = {"from": [], "to": []}
    seen_by_side: dict[EndpointSide, set[int]] = {"from": set(), "to": set()}
    for node_ref in node_group.node_refs:
        for link_ref in context.outgoing_link_refs(node_ref):
            row_i = context.link_index_for_ref(link_ref)
            if row_i is None or row_i not in candidate_set or row_i in seen_by_side["from"]:
                continue
            seen_by_side["from"].add(row_i)
            rows_by_side["from"].append(row_i)
        for link_ref in context.incoming_link_refs(node_ref):
            row_i = context.link_index_for_ref(link_ref)
            if row_i is None or row_i not in candidate_set or row_i in seen_by_side["to"]:
                continue
            seen_by_side["to"].add(row_i)
            rows_by_side["to"].append(row_i)
    return {side: tuple(rows) for side, rows in rows_by_side.items()}


def _owner_rows_for_bridge_side(
    context: SegmentationContext,
    node_group: LateralNodeGroup,
    side: EndpointSide,
) -> tuple[int, ...]:
    rows: list[int] = []
    seen: set[int] = set()
    for owner_ref in node_group.link_refs:
        if context.endpoint_node_ref_for_link_ref(owner_ref, side) not in node_group.node_refs:
            continue
        owner_row = context.link_index_for_ref(owner_ref)
        if owner_row is None or owner_row in seen:
            continue
        seen.add(owner_row)
        rows.append(owner_row)
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
