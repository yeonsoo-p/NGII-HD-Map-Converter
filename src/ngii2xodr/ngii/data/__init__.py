"""NGII data loading API and common canonical contracts."""

from ngii2xodr.ngii.data.config import (
    NGIIConfig,
    NGIISanityConfig,
    NGIISanityRepairConfig,
    NGIISanityWarningConfig,
    NGIITextCorrection,
    NGIITextCorrectionConfig,
)
from ngii2xodr.ngii.data.dataset import AmbiguousFeatureIDError, LayerStore, NGIIDataset
from ngii2xodr.ngii.data.loader import NGIIVersion, detect_ngii_version, load_ngii
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning

__all__ = [
    "AmbiguousFeatureIDError",
    "LayerStore",
    "NGIIConfig",
    "NGIIDataset",
    "NGIISanityConfig",
    "NGIISanityRepairConfig",
    "NGIISanityWarningConfig",
    "NGIITextCorrection",
    "NGIITextCorrectionConfig",
    "NGIIVersion",
    "SanityAction",
    "SanityReport",
    "SanityWarning",
    "detect_ngii_version",
    "load_ngii",
]
