"""Junction-connection segmentation stage.

The stage is explicit and dependency-aware, but intentionally conservative:
post-junction semantics need more map-specific validation before they mutate
the user experience, so v1 emits no entities and never performs pairwise
global scans.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import StageResult
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage


class JunctionConnectionStage:
    id: ClassVar[str] = "junction_connection"
    label: ClassVar[str] = "Junction connections"
    entity_label: ClassVar[str] = "Junction connection"
    enabled_attr: ClassVar[str] = "enable_junction_connection"
    requires: ClassVar[tuple[str, ...]] = (JunctionStage.id, LateralLinkGroupStage.id)

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del context, previous_results
        return empty_result(self, skipped_reason="not implemented in dataset-native v1")
