"""NGII data loading API and common canonical contracts."""

from ngii2xodr.ngii.data.config import (
    NGIIConfig,
    NGIIEncodingConfig,
    NGIIGeometryConfig,
    NGIISanityConfig,
    NGIISanityRepairConfig,
    NGIISanityWarningConfig,
    NGIITextCorrectionConfig,
)
from ngii2xodr.ngii.data.dataset import AmbiguousFeatureIDError, LayerStore, NGIIDataset
from ngii2xodr.ngii.data.loader import NGIIVersion, detect_ngii_version, load_ngii
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning
from ngii2xodr.ngii.data.schema import (
    FieldRule,
    LayerSpec,
    RelationshipRule,
    RoleFilter,
    SchemaDefinition,
)

__all__ = [
    "AmbiguousFeatureIDError",
    "FieldRule",
    "LayerSpec",
    "LayerStore",
    "NGIIConfig",
    "NGIIDataset",
    "NGIIEncodingConfig",
    "NGIIGeometryConfig",
    "NGIISanityConfig",
    "NGIISanityRepairConfig",
    "NGIISanityWarningConfig",
    "NGIITextCorrectionConfig",
    "NGIIVersion",
    "RelationshipRule",
    "RoleFilter",
    "SanityAction",
    "SanityReport",
    "SanityWarning",
    "SchemaDefinition",
    "detect_ngii_version",
    "load_ngii",
]
