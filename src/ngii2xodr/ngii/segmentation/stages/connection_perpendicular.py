"""Connection-perpendicular segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.connection_reference import ConnectionReferenceStage


class ConnectionPerpendicularStage:
    id: ClassVar[str] = "connection_perpendicular"
    label: ClassVar[str] = "Connection perpendiculars"
    entity_label: ClassVar[str] = "Connection perpendicular"
    enabled_attr: ClassVar[str] = "enable_connection_perpendicular"
    requires: ClassVar[tuple[str, ...]] = (ConnectionReferenceStage.id,)

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del context, previous_results
        return empty_result(self, skipped_reason="not implemented in dataset-native v1")
