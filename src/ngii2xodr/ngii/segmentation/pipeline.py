"""Hydra-enabled segmentation orchestration."""

from __future__ import annotations

import logging
from typing import Any

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import (
    SegmentationConfig,
    SegmentationResult,
    SelectedField,
    StageResult,
)
from ngii2xodr.ngii.segmentation.stage import SegmentationStage, empty_result
from ngii2xodr.ngii.segmentation.stages.connection_perpendicular import (
    ConnectionPerpendicularStage,
)
from ngii2xodr.ngii.segmentation.stages.connection_reference import ConnectionReferenceStage
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage
from ngii2xodr.ngii.segmentation.stages.lateral_node_group import LateralNodeGroupStage
from ngii2xodr.ngii.segmentation.stages.uturn import UTurnStage
from ngii2xodr.profile import PerformanceProfile

log = logging.getLogger(__name__)

SEGMENTATION_STAGES: tuple[type[SegmentationStage], ...] = (
    UTurnStage,
    LateralLinkGroupStage,
    LateralNodeGroupStage,
    JunctionStage,
    JunctionConnectionStage,
    ConnectionReferenceStage,
    ConnectionPerpendicularStage,
)


class Segmentation(SegmentationResult):
    """Compatibility alias for the dataset-native result type."""

    @classmethod
    def from_dataset(cls, dataset: NGIIDataset, cfg: SegmentationConfig) -> Segmentation:
        context = SegmentationContext(dataset=dataset, cfg=cfg)
        profile = PerformanceProfile()
        results: list[StageResult] = []
        by_id: dict[str, StageResult] = {}
        for stage_cls in SEGMENTATION_STAGES:
            if not getattr(cfg, stage_cls.enabled_attr):
                reason = "disabled by Hydra"
                log.info("%s skipped: %s", stage_cls.__name__, reason)
                result = empty_result(stage_cls, skipped_reason=reason)
                by_id[result.stage_id] = result
                continue
            missing = [stage_id for stage_id in stage_cls.requires if stage_id not in by_id]
            skipped = [
                stage_id
                for stage_id in stage_cls.requires
                if stage_id in by_id and by_id[stage_id].skipped_reason
            ]
            if missing or skipped:
                reason = f"missing dependencies={missing}; skipped dependencies={skipped}"
                log.info("%s skipped: %s", stage_cls.__name__, reason)
                result = empty_result(stage_cls, skipped_reason=reason)
                by_id[result.stage_id] = result
                continue
            stage = stage_cls()
            with profile.timed(stage_cls.id) as timer:
                result = stage.run(context, by_id)
                timer.detail = f"{len(result.entities)} entities"
            log.info(
                "%s: %.3fs, %d %s entity(s)",
                stage_cls.__name__,
                timer.duration_s,
                len(result.entities),
                stage_cls.entity_label,
            )
            results.append(result)
            by_id[result.stage_id] = result
        return cls(stage_results=tuple(results), profile=profile)

    @classmethod
    def from_layers(cls, *_args: Any, **_kwargs: Any) -> Segmentation:
        msg = "Segmentation.from_layers was removed; use Segmentation.from_dataset(dataset, cfg)"
        raise RuntimeError(msg)

    def selected_fields_for_ref(self, ref: FeatureRef) -> tuple[SelectedField, ...]:
        return super().selected_fields_for_ref(ref)


def segmentation_level_labels() -> tuple[str, ...]:
    return ("Raw", *(stage.label for stage in SEGMENTATION_STAGES))
