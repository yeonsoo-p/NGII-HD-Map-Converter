"""NGII data loading API and common canonical contracts."""

from __future__ import annotations

from pathlib import Path

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
from ngii2xodr.ngii.data.loader import NGIILoadResult, detect_schema
from ngii2xodr.ngii.data.loader import load_ngii as _load_ngii
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning
from ngii2xodr.ngii.data.schema import (
    FieldRule,
    LayerSpec,
    RelationshipRule,
    RoleFilter,
    RoleKey,
    SchemaDefinition,
)
from ngii2xodr.ngii.data.v2023 import SCHEMA as V2023_SCHEMA
from ngii2xodr.ngii.data.v2025 import SCHEMA as V2025_SCHEMA

SUPPORTED_SCHEMAS: tuple[SchemaDefinition, ...] = (V2023_SCHEMA, V2025_SCHEMA)


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIILoadResult:
    return _load_ngii(root, coordinate, cfg, SUPPORTED_SCHEMAS)


def detect_ngii_version(root: Path, coordinate: str) -> str:
    return detect_schema(root, coordinate, SUPPORTED_SCHEMAS).version


__all__ = [
    "SUPPORTED_SCHEMAS",
    "AmbiguousFeatureIDError",
    "FieldRule",
    "LayerSpec",
    "LayerStore",
    "NGIIConfig",
    "NGIIDataset",
    "NGIIEncodingConfig",
    "NGIIGeometryConfig",
    "NGIILoadResult",
    "NGIISanityConfig",
    "NGIISanityRepairConfig",
    "NGIISanityWarningConfig",
    "NGIITextCorrectionConfig",
    "RelationshipRule",
    "RoleFilter",
    "RoleKey",
    "SanityAction",
    "SanityReport",
    "SanityWarning",
    "SchemaDefinition",
    "detect_ngii_version",
    "detect_schema",
    "load_ngii",
]
