"""U-turn segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import shapely

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.helpers import intersects_within_z_tol
from ngii2xodr.ngii.segmentation.model import StageResult, UTurn
from ngii2xodr.ngii.segmentation.stage import empty_result

_UTURN_LINK_TYPE = "6"
_UTURN_MARKER_KIND = "502"


class UTurnStage:
    id: ClassVar[str] = "uturn"
    label: ClassVar[str] = "U-turns"
    entity_label: ClassVar[str] = "U-turn"
    enabled_attr: ClassVar[str] = "enable_uturn"
    requires: ClassVar[tuple[str, ...]] = ()

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        marker_rows: list[int] = []
        marker_lines: list[shapely.LineString] = []
        for i in context.b2_rows_for_kind(_UTURN_MARKER_KIND):
            marker_line = context.b2_lines[i]
            if marker_line is not None:
                marker_rows.append(i)
                marker_lines.append(marker_line)
        if not marker_rows:
            return empty_result(self)

        tree = shapely.STRtree(marker_lines)
        entities: list[UTurn] = []
        entity_id_by_ref: dict[FeatureRef, int] = {}

        for link_i in context.a2_rows_for_link_type(_UTURN_LINK_TYPE):
            link = context.dataset.a2_link.features[link_i]
            link_line = context.a2_lines[link_i]
            if link_line is None:
                continue
            matched_marker_refs: list[FeatureRef] = []
            for marker_pos_raw in tree.query(link_line, predicate="intersects"):
                marker_pos = int(marker_pos_raw)
                marker_i = marker_rows[marker_pos]
                marker_line = marker_lines[marker_pos]
                if intersects_within_z_tol(
                    link.polyline,
                    link_line,
                    context.dataset.b2_surfacelinemark.features[marker_i].polyline,
                    marker_line,
                    context.cfg.z_intersection_tol_m,
                ):
                    matched_marker_refs.append(context.ref_for_b2_index(marker_i))
            if not matched_marker_refs:
                continue
            entity_id = len(entities)
            link_ref = context.ref_for_a2_index(link_i)
            entity_id_by_ref[link_ref] = entity_id
            entities.append(
                UTurn(
                    id=entity_id,
                    link_ref=link_ref,
                    marker_refs=tuple(
                        sorted(set(matched_marker_refs), key=lambda ref: ref.feature_id)
                    ),
                )
            )
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_by_ref
        )
