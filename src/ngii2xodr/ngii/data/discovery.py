"""Recursive discovery for one requested NGII coordinate product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ngii2xodr.ngii.data.manual_2023 import SPECS_BY_FILENAME, LayerSpec
from ngii2xodr.ngii.data.sanity import SanityReport


@dataclass(slots=True, frozen=True)
class DiscoveredLayerFile:
    spec: LayerSpec
    path: Path


def discover_layer_files(
    root: Path,
    coordinate: str,
    sanity: SanityReport,
    *,
    warn_unknown_layers: bool,
) -> list[DiscoveredLayerFile]:
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    coordinate_dirs = _coordinate_dirs(root, coordinate)
    if not coordinate_dirs:
        msg = f"coordinate folder {coordinate!r} was not found under {root}"
        raise FileNotFoundError(msg)

    discovered: list[DiscoveredLayerFile] = []
    for coordinate_dir in coordinate_dirs:
        for shp_path in sorted(coordinate_dir.rglob("*.shp")):
            spec = SPECS_BY_FILENAME.get(shp_path.name.upper())
            if spec is None:
                if warn_unknown_layers:
                    sanity.warn(
                        "unknown-layer",
                        f"unknown SHP layer {shp_path.name!r} in requested coordinate product",
                        source_path=shp_path,
                    )
                continue
            discovered.append(DiscoveredLayerFile(spec=spec, path=shp_path))
    return discovered


def _coordinate_dirs(root: Path, coordinate: str) -> list[Path]:
    if root.name == coordinate:
        return [root]
    return sorted(path for path in root.rglob(coordinate) if path.is_dir())
