"""NGII 2025.12 loader."""

from __future__ import annotations

from pathlib import Path

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.data.engine import load_schema
from ngii2xodr.ngii.data.v2025.definitions import SCHEMA


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIIDataset:
    """Load one 2025 NGII coordinate product."""
    return load_schema(root, coordinate, cfg, SCHEMA)
