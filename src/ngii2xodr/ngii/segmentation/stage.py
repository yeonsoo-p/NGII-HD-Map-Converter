"""Stage protocol for dataset-native segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol

from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import StageResult


class SegmentationStage(Protocol):
    id: ClassVar[str]
    label: ClassVar[str]
    entity_label: ClassVar[str]
    enabled_attr: ClassVar[str]
    requires: ClassVar[tuple[str, ...]]

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult: ...


def empty_result(
    stage: type[SegmentationStage] | SegmentationStage,
    *,
    skipped_reason: str = "",
) -> StageResult:
    return StageResult(
        stage_id=stage.id,
        label=stage.label,
        entity_label=stage.entity_label,
        entities=(),
        entity_id_by_ref={},
        skipped_reason=skipped_reason,
    )
