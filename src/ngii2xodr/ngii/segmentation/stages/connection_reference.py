"""Connection-reference segmentation stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage


class ConnectionReferenceStage:
    id: ClassVar[str] = "connection_reference"
    label: ClassVar[str] = "Connection references"
    entity_label: ClassVar[str] = "Connection reference"
    enabled_attr: ClassVar[str] = "enable_connection_reference"
    requires: ClassVar[tuple[str, ...]] = (JunctionConnectionStage.id,)

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del context, previous_results
        return empty_result(self, skipped_reason="not implemented in dataset-native v1")
