"""Junction segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import shapely

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import intersects_within_z_tol, uf_find, uf_union
from ngii2xodr.ngii.segmentation.model import Junction, StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result

_JUNCTION_LINK_TYPE = "1"


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
        candidate_rows = list(context.a2_rows_for_link_type(_JUNCTION_LINK_TYPE))
        if not candidate_rows:
            return empty_result(self)

        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)
        _union_link_refs(context, candidate_rows, row_to_candidate, parent)
        _union_intersections(context, candidate_rows, row_to_candidate, parent)

        rows_by_entity: dict[int, list[int]] = {}
        root_to_entity: dict[int, int] = {}
        for row_i in candidate_rows:
            root = uf_find(parent, row_to_candidate[row_i])
            entity_id = root_to_entity.setdefault(root, len(root_to_entity))
            rows_by_entity.setdefault(entity_id, []).append(row_i)

        entities: list[Junction] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}
        for entity_id in range(len(rows_by_entity)):
            refs = tuple(context.ref_for_a2_index(i) for i in rows_by_entity[entity_id])
            for ref in refs:
                entity_id_by_ref[ref] = entity_id
            entities.append(Junction(id=entity_id, link_refs=refs))
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )


def _union_link_refs(
    context: SegmentationContext,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: np.ndarray,
) -> None:
    for row_i in candidate_rows:
        link = context.dataset.a2_link.features[row_i]
        for neighbour_id in (link.r_link_id, link.l_link_id):
            neighbour_row = context.dataset.a2_link.id_to_index.get(str(neighbour_id))
            if neighbour_row is not None and neighbour_row in row_to_candidate:
                uf_union(parent, row_to_candidate[row_i], row_to_candidate[neighbour_row])


def _union_intersections(
    context: SegmentationContext,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: np.ndarray,
) -> None:
    geometry_rows: list[int] = []
    lines: list[shapely.LineString] = []
    for row_i in candidate_rows:
        line = context.a2_lines[row_i]
        if line is not None:
            geometry_rows.append(row_i)
            lines.append(line)
    if len(geometry_rows) < 2:
        return
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
                context.dataset.a2_link.features[a_row].polyline,
                line,
                context.dataset.a2_link.features[b_row].polyline,
                other,
                context.cfg.z_intersection_tol_m,
            ):
                uf_union(parent, row_to_candidate[a_row], row_to_candidate[b_row])
