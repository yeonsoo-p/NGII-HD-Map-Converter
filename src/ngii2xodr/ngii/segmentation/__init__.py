"""Dataset-native segmentation package."""

from ngii2xodr.ngii.segmentation.model import (
    Junction,
    JunctionConnection,
    JunctionEdge,
    JunctionReference,
    LateralLinkGroup,
    LateralNodeGroup,
    SegmentationConfig,
    SegmentationResult,
    SelectedField,
    StageResult,
    UTurn,
)
from ngii2xodr.ngii.segmentation.pipeline import (
    SEGMENTATION_STAGES,
    Segmentation,
    segmentation_level_labels,
)
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage
from ngii2xodr.ngii.segmentation.stages.junction_edge import (
    JunctionEdgeStage,
)
from ngii2xodr.ngii.segmentation.stages.junction_reference import JunctionReferenceStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage
from ngii2xodr.ngii.segmentation.stages.lateral_node_group import LateralNodeGroupStage
from ngii2xodr.ngii.segmentation.stages.uturn import UTurnStage

__all__ = [
    "SEGMENTATION_STAGES",
    "Junction",
    "JunctionConnection",
    "JunctionConnectionStage",
    "JunctionEdge",
    "JunctionEdgeStage",
    "JunctionReference",
    "JunctionReferenceStage",
    "JunctionStage",
    "LateralLinkGroup",
    "LateralLinkGroupStage",
    "LateralNodeGroup",
    "LateralNodeGroupStage",
    "Segmentation",
    "SegmentationConfig",
    "SegmentationResult",
    "SelectedField",
    "StageResult",
    "UTurn",
    "UTurnStage",
    "segmentation_level_labels",
]
