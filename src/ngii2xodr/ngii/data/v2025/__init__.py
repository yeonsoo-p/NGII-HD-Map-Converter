"""NGII 2025.12 version package."""

from __future__ import annotations

from ngii2xodr.ngii.data.schema import FieldRule, LayerSpec, RelationshipRule
from ngii2xodr.ngii.data.v2025.definitions import (
    LAYER_SPECS,
    SCHEMA,
    SPECS_BY_FILENAME,
    SPECS_BY_LAYER_NAME,
)

__all__ = [
    "LAYER_SPECS",
    "SCHEMA",
    "SPECS_BY_FILENAME",
    "SPECS_BY_LAYER_NAME",
    "FieldRule",
    "LayerSpec",
    "RelationshipRule",
]
