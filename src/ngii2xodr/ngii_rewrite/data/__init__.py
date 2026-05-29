"""NGII rewrite data loading API and canonical contracts."""

from __future__ import annotations

from pathlib import Path

from ngii2xodr.ngii.data.config import (
    NGIIConfig,
    NGIIEncodingConfig,
    NGIIGeometryConfig,
    NGIISanityChecksConfig,
    NGIISanityConfig,
    SanityMode,
)
from ngii2xodr.ngii_rewrite.data.dataset import (
    AmbiguousFeatureIDError,
    LayerStore,
    NGIIDataset,
)
from ngii2xodr.ngii_rewrite.data.features import FeatureRef, ResolvedReference
from ngii2xodr.ngii_rewrite.data.loader import NGIILoadResult, detect_schema
from ngii2xodr.ngii_rewrite.data.loader import load_ngii as _load_ngii
from ngii2xodr.ngii_rewrite.data.metadata import (
    FieldDef,
    LayerDef,
    ReciprocalReferenceRule,
    ReferenceDef,
    RoleFilter,
    RoleKey,
    Schema,
)
from ngii2xodr.ngii_rewrite.data.sanity import SanityAction, SanityReport, SanityWarning
from ngii2xodr.ngii_rewrite.data.v2023 import SCHEMA as V2023_SCHEMA
from ngii2xodr.ngii_rewrite.data.v2025 import SCHEMA as V2025_SCHEMA

SUPPORTED_SCHEMAS: tuple[Schema, ...] = (V2023_SCHEMA, V2025_SCHEMA)


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIILoadResult:
    return _load_ngii(root, coordinate, cfg, SUPPORTED_SCHEMAS)


def detect_ngii_version(root: Path, coordinate: str) -> str:
    return detect_schema(root, coordinate, SUPPORTED_SCHEMAS).version


__all__ = [
    "SUPPORTED_SCHEMAS",
    "AmbiguousFeatureIDError",
    "FeatureRef",
    "FieldDef",
    "LayerDef",
    "LayerStore",
    "NGIIConfig",
    "NGIIDataset",
    "NGIIEncodingConfig",
    "NGIIGeometryConfig",
    "NGIILoadResult",
    "NGIISanityChecksConfig",
    "NGIISanityConfig",
    "ReciprocalReferenceRule",
    "ReferenceDef",
    "ResolvedReference",
    "RoleFilter",
    "RoleKey",
    "SanityAction",
    "SanityMode",
    "SanityReport",
    "SanityWarning",
    "Schema",
    "detect_ngii_version",
    "detect_schema",
    "load_ngii",
]
