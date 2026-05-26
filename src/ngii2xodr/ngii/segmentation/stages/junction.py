"""Junction segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import shapely

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import (
    connected_components_from_pairs,
    intersects_within_z_tol,
)
from ngii2xodr.ngii.segmentation.model import Junction, StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result


class JunctionStage:
    id: ClassVar[str] = "junction"
    label: ClassVar[str] = "Junctions"
    entity_label: ClassVar[str] = "Junction"
    enabled_attr: ClassVar[str] = "enable_junction"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        candidate_rows = list(context.rows_for_filter("junction_link"))
        if not candidate_rows:
            return empty_result(self, skipped_reason="schema has no junction link candidates")

        candidate_set = set(candidate_rows)
        components = connected_components_from_pairs(
            candidate_rows,
            (
                *_lateral_pairs(context, candidate_rows, candidate_set),
                *_intersection_pairs(context, candidate_rows),
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
