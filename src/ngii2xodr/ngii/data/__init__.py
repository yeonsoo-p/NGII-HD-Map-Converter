"""NGII 2023 data loading API."""

from ngii2xodr.ngii.data.config import (
    NGIIConfig,
    NGIISanityConfig,
    NGIISanityRepairConfig,
    NGIISanityWarningConfig,
    NGIITextCorrection,
    NGIITextCorrectionConfig,
)
from ngii2xodr.ngii.data.dataset import AmbiguousFeatureIDError, LayerStore, NGIIDataset
from ngii2xodr.ngii.data.layers import (
    A1_NODE,
    A2_LINK,
    A3_DRIVEWAYSECTION,
    A4_SUBSIDIARYSECTION,
    A5_PARKINGLOT,
    B1_SAFETYSIGN,
    B2_SURFACELINEMARK,
    B3_SURFACEMARK,
    C1_TRAFFICLIGHT,
    C2_KILOPOST,
    C3_VEHICLEPROTECTIONSAFETY,
    C4_SPEEDBUMP,
    C5_HEIGHTBARRIER,
    C6_POSTPOINT,
)
from ngii2xodr.ngii.data.loader import load_ngii
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning

__all__ = [
    "A1_NODE",
    "A2_LINK",
    "A3_DRIVEWAYSECTION",
    "A4_SUBSIDIARYSECTION",
    "A5_PARKINGLOT",
    "B1_SAFETYSIGN",
    "B2_SURFACELINEMARK",
    "B3_SURFACEMARK",
    "C1_TRAFFICLIGHT",
    "C2_KILOPOST",
    "C3_VEHICLEPROTECTIONSAFETY",
    "C4_SPEEDBUMP",
    "C5_HEIGHTBARRIER",
    "C6_POSTPOINT",
    "AmbiguousFeatureIDError",
    "LayerStore",
    "NGIIConfig",
    "NGIIDataset",
    "NGIISanityConfig",
    "NGIISanityRepairConfig",
    "NGIISanityWarningConfig",
    "NGIITextCorrection",
    "NGIITextCorrectionConfig",
    "SanityAction",
    "SanityReport",
    "SanityWarning",
    "load_ngii",
]
