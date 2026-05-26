"""NGII 2025 loader placeholder."""

from __future__ import annotations

from pathlib import Path

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import NGIIDataset


def load_ngii(_root: Path, _coordinate: str, _cfg: NGIIConfig) -> NGIIDataset:
    """Raise until NGII 2025 definitions and rules are implemented."""
    msg = "NGII v2025 loading is not implemented yet"
    raise NotImplementedError(msg)
