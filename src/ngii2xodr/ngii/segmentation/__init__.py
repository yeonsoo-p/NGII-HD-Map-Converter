"""Dataset-native segmentation package."""

from ngii2xodr.ngii.segmentation.model import (
    ConnectionPerpendicular,
    ConnectionReference,
    Junction,
    JunctionConnection,
    LateralLinkGroup,
    LateralNodeGroup,
    NodeLinkRelation,
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
from ngii2xodr.ngii.segmentation.stages.connection_perpendicular import (
    ConnectionPerpendicularStage,
)
from ngii2xodr.ngii.segmentation.stages.connection_reference import ConnectionReferenceStage
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage
from ngii2xodr.ngii.segmentation.stages.lateral_link_group import LateralLinkGroupStage
from ngii2xodr.ngii.segmentation.stages.lateral_node_group import LateralNodeGroupStage
from ngii2xodr.ngii.segmentation.stages.node_link_relations import NodeLinkRelationsStage
from ngii2xodr.ngii.segmentation.stages.uturn import UTurnStage

__all__ = [
    "SEGMENTATION_STAGES",
    "ConnectionPerpendicular",
    "ConnectionPerpendicularStage",
    "ConnectionReference",
    "ConnectionReferenceStage",
    "Junction",
    "JunctionConnection",
    "JunctionConnectionStage",
    "JunctionStage",
    "LateralLinkGroup",
    "LateralLinkGroupStage",
    "LateralNodeGroup",
    "LateralNodeGroupStage",
    "NodeLinkRelation",
    "NodeLinkRelationsStage",
    "Segmentation",
    "SegmentationConfig",
    "SegmentationResult",
    "SelectedField",
    "StageResult",
    "UTurn",
    "UTurnStage",
    "segmentation_level_labels",
]
