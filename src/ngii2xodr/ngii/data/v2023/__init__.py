"""NGII 2023.07 version package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LAYER_SPECS",
    "SPECS_BY_FILENAME",
    "SPECS_BY_LAYER_NAME",
    "FieldRule",
    "LayerSpec",
    "RelationshipRule",
    "load_ngii",
]


def __getattr__(name: str) -> Any:
    if name == "load_ngii":
        from ngii2xodr.ngii.data.v2023.loader import load_ngii  # noqa: PLC0415

        return load_ngii
    if name in {
        "FieldRule",
        "LAYER_SPECS",
        "LayerSpec",
        "RelationshipRule",
        "SPECS_BY_FILENAME",
        "SPECS_BY_LAYER_NAME",
    }:
        from ngii2xodr.ngii.data.v2023 import definitions  # noqa: PLC0415

        return getattr(definitions, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
