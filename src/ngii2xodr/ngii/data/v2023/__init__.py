"""NGII 2023.07 version package."""

from __future__ import annotations

from ngii2xodr.ngii.data.schema import FieldRule, LayerSpec, RelationshipRule
from ngii2xodr.ngii.data.v2023.definitions import (
    LAYER_SPECS,
    SCHEMA,
    SPECS_BY_FILENAME,
    SPECS_BY_LAYER_NAME,
)
from ngii2xodr.ngii.data.v2023.loader import load_ngii

__all__ = [
    "LAYER_SPECS",
    "SCHEMA",
    "SPECS_BY_FILENAME",
    "SPECS_BY_LAYER_NAME",
    "FieldRule",
    "LayerSpec",
    "RelationshipRule",
    "load_ngii",
]
