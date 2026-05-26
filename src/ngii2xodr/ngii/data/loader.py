"""Version-dispatching public NGII loader facade."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.data.engine import coordinate_dirs_for
from ngii2xodr.ngii.data.v2023.definitions import SPECS_BY_FILENAME as V2023_FILENAMES
from ngii2xodr.ngii.data.v2023.loader import load_ngii as load_ngii_v2023
from ngii2xodr.ngii.data.v2025.definitions import SPECS_BY_FILENAME as V2025_FILENAMES
from ngii2xodr.ngii.data.v2025.loader import load_ngii as load_ngii_v2025

NGIIVersion = Literal["v2023", "v2025"]


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIIDataset:
    """Load one NGII coordinate product using the detected manual version."""
    version = detect_ngii_version(root, coordinate)
    if version == "v2023":
        return load_ngii_v2023(root, coordinate, cfg)
    return load_ngii_v2025(root, coordinate, cfg)


def detect_ngii_version(root: Path, coordinate: str) -> NGIIVersion:
    """Infer the NGII manual version for the requested coordinate product."""
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    coordinate_dirs = coordinate_dirs_for(root, coordinate)
    if not coordinate_dirs:
        msg = f"coordinate folder {coordinate!r} was not found under {root}"
        raise FileNotFoundError(msg)

    found_names = {
        shp_path.name.upper()
        for coordinate_dir in coordinate_dirs
        for shp_path in coordinate_dir.rglob("*.shp")
    }
    has_v2023 = bool(found_names & set(V2023_FILENAMES))
    has_v2025 = bool(found_names & set(V2025_FILENAMES))
    if has_v2023 and has_v2025:
        msg = (
            f"ambiguous NGII manual version for {root} [{coordinate}]: "
            "both 2023 and 2025 layer signatures were found"
        )
        raise ValueError(msg)
    if has_v2023:
        return "v2023"
    if has_v2025:
        return "v2025"

    msg = f"could not detect an implemented NGII manual version for {root} [{coordinate}]"
    raise ValueError(msg)
