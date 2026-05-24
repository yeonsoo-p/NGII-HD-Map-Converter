"""Hydra entry point: ``uv run python -m shp2xodr``.

The window opens empty by default; select a section directory via
**File → Open SHP folder…** (Ctrl+O). Pass ``shp_dir=/path/to/section`` on
the CLI (or set it in ``conf/config.yaml``) to auto-load on startup.

This module is the only place that knows about both Hydra/OmegaConf
``DictConfig`` and typed config dataclasses.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PySide6.QtWidgets import QApplication

from shp2xodr.shp.gui import HdMapWindow
from shp2xodr.shp.segmentation import SegmentationConfig
from shp2xodr.shp.viz import VizConfig

log = logging.getLogger(__name__)


def _rgb_int(v: list[int]) -> tuple[int, int, int]:
    return (int(v[0]), int(v[1]), int(v[2]))


def _rgb_float(v: list[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _rgb_int_dict(v: dict[str, list[int]]) -> dict[str, tuple[int, int, int]]:
    return {k: _rgb_int(c) for k, c in v.items()}


def _build_seg_cfg(cfg: DictConfig) -> SegmentationConfig:
    return SegmentationConfig(
        z_intersection_tol_m=float(cfg.segmentation.z_intersection_tol_m),
        junction_proximity_merge_dist_m=float(cfg.segmentation.junction_proximity_merge_dist_m),
        junction_connection_node_merge_dist_m=float(
            cfg.segmentation.junction_connection_node_merge_dist_m
        ),
    )


def _build_viz_cfg(cfg: DictConfig) -> VizConfig:
    raw = OmegaConf.to_container(cfg.viz, resolve=True)
    if not isinstance(raw, dict):
        msg = f"viz config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return VizConfig(
        node_point_size=float(raw["node_point_size"]),
        poly_opacity=float(raw["poly_opacity"]),
        poly_depth_offset_factor=float(raw["poly_depth_offset_factor"]),
        poly_depth_offset_units=float(raw["poly_depth_offset_units"]),
        line_width_a2=float(raw["line_width_a2"]),
        line_width_b2=float(raw["line_width_b2"]),
        line_width_c3=float(raw["line_width_c3"]),
        selector_tol_a1=float(raw["selector_tol_a1"]),
        selector_tol_a2=float(raw["selector_tol_a2"]),
        selector_tol_thin=float(raw["selector_tol_thin"]),
        selector_tol_poly=float(raw["selector_tol_poly"]),
        background_color=_rgb_float(raw["background_color"]),
        highlight_rgb=_rgb_int(raw["highlight_rgb"]),
        a2_uniform_rgb=_rgb_int(raw["a2_uniform_rgb"]),
        a3_road_type_rgb=_rgb_int_dict(raw["a3_road_type_rgb"]),
        a3_protected_rgb=_rgb_int(raw["a3_protected_rgb"]),
        a3_fallback_rgb=_rgb_int(raw["a3_fallback_rgb"]),
        a4_subtype_rgb=_rgb_int_dict(raw["a4_subtype_rgb"]),
        a4_fallback_rgb=_rgb_int(raw["a4_fallback_rgb"]),
        b2_paint_rgb=_rgb_int_dict(raw["b2_paint_rgb"]),
        b2_paint_fallback_rgb=_rgb_int(raw["b2_paint_fallback_rgb"]),
        c3_type_rgb=_rgb_int_dict(raw["c3_type_rgb"]),
        c3_type_fallback_rgb=_rgb_int(raw["c3_type_fallback_rgb"]),
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(seg_cfg=_build_seg_cfg(cfg), viz_cfg=_build_viz_cfg(cfg))
    window.show()
    if cfg.shp_dir is not None:
        # to_absolute_path resolves against the invocation cwd, not Hydra's
        # per-run output dir, which is what the user means by a relative path.
        window.load_folder(Path(to_absolute_path(str(cfg.shp_dir))).expanduser())
    app.exec()


if __name__ == "__main__":
    # Windows stdio defaults to cp1252; UTF-8 makes logging non-ASCII-safe.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
