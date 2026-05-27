"""Hydra entry point: ``uv run python -m ngii2xodr``.

The window opens empty by default; select a section directory via
**File → Open NGII folder…** (Ctrl+O). Pass ``ngii_dir=/path/to/section`` on
the CLI (or set it in ``conf/config.yaml``) to auto-load on startup.

This module is the only place that knows about both Hydra/OmegaConf
``DictConfig`` and typed config dataclasses.
"""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PySide6.QtWidgets import QApplication

from ngii2xodr.gui import HdMapWindow
from ngii2xodr.ngii.data import (
    NGIIConfig,
    NGIIEncodingConfig,
    NGIIGeometryConfig,
    NGIISanityChecksConfig,
    NGIISanityConfig,
    SanityMode,
)
from ngii2xodr.ngii.segmentation import SegmentationConfig
from ngii2xodr.ngii.viz import VizCameraFocusConfig, VizConfig, VizLayerConfig
from ngii2xodr.profile import ViewportProfilingConfig

log = logging.getLogger(__name__)

_NGII_CONFIG_KEYS = frozenset({"coordinate", "geometry", "encoding", "sanity"})
_SANITY_CONFIG_KEYS = frozenset(
    {"node_match_tolerance_m", "direction_parallel_dot_min", "link_min_length_m", "checks"}
)
_WARNING_ONLY_CHECKS = frozenset(
    {
        "layer_required_missing",
        "layer_unknown",
        "shp_sidecar_missing",
        "dbf_column_case_duplicate",
        "manual_column_missing",
        "manual_column_unknown",
        "manual_field_required_missing",
        "manual_field_length_exceeded",
        "manual_field_type_invalid",
        "manual_field_code_invalid",
        "manual_hist_type_invalid",
        "geometry_missing",
        "geometry_invalid",
        "text_utf8_dbf_row_mismatch",
        "text_utf8_decode_replacement",
        "text_replacement_char",
        "feature_id_missing",
        "feature_id_duplicate_identical",
        "global_id_collision",
        "reciprocal_reference_conflict",
        "link_endpoint_order_ambiguous",
        "link_endpoint_misaligned",
    }
)


def _rgb_int(v: list[int]) -> tuple[int, int, int]:
    return (int(v[0]), int(v[1]), int(v[2]))


def _rgb_float(v: list[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _build_seg_cfg(cfg: DictConfig) -> SegmentationConfig:
    return SegmentationConfig(
        z_intersection_tol_m=float(cfg.segmentation.z_intersection_tol_m),
        junction_connection_node_merge_dist_m=float(
            cfg.segmentation.junction_connection_node_merge_dist_m
        ),
        junction_connection_opposite_direction_dot_min=_ratio_float(
            cfg.segmentation.junction_connection_opposite_direction_dot_min,
            "segmentation.junction_connection_opposite_direction_dot_min",
        ),
        endpoint_tangent_lookback_m=_positive_float(
            cfg.segmentation.endpoint_tangent_lookback_m,
            "segmentation.endpoint_tangent_lookback_m",
        ),
        enable_uturn=bool(cfg.segmentation.enable_uturn),
        enable_lateral_link_group=bool(cfg.segmentation.enable_lateral_link_group),
        enable_lateral_node_group=bool(cfg.segmentation.enable_lateral_node_group),
        enable_junction=bool(cfg.segmentation.enable_junction),
        enable_junction_connection=bool(cfg.segmentation.enable_junction_connection),
        enable_junction_reference=bool(cfg.segmentation.enable_junction_reference),
        enable_junction_edge=bool(cfg.segmentation.enable_junction_edge),
    )


def _build_viz_cfg(cfg: DictConfig) -> VizConfig:
    raw = OmegaConf.to_container(cfg.viz, resolve=True)
    if not isinstance(raw, dict):
        msg = f"viz config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    layers_raw = raw["layers"]
    if not isinstance(layers_raw, dict):
        msg = f"viz.layers config: expected dict, got {type(layers_raw).__name__}"
        raise TypeError(msg)
    return VizConfig(
        background_color=_rgb_float(raw["background_color"]),
        highlight_rgb=_rgb_int(raw["highlight_rgb"]),
        junction_connection_node_point_size=float(raw["junction_connection_node_point_size"]),
        junction_reference_arrow_length_m=float(raw["junction_reference_arrow_length_m"]),
        junction_reference_arrow_rgb=_rgb_int(raw["junction_reference_arrow_rgb"]),
        junction_edge_rgb=_rgb_int(raw["junction_edge_rgb"]),
        junction_edge_line_width=float(raw["junction_edge_line_width"]),
        selector_tol_point=float(raw["selector_tol_point"]),
        selector_tol_line=float(raw["selector_tol_line"]),
        selector_tol_poly=float(raw["selector_tol_poly"]),
        point_hit_radius_px=float(raw["point_hit_radius_px"]),
        poly_depth_offset_factor=float(raw["poly_depth_offset_factor"]),
        poly_depth_offset_units=float(raw["poly_depth_offset_units"]),
        segmentation_seed=int(raw["segmentation_seed"]),
        camera_focus=_build_camera_focus_cfg(raw["camera_focus"]),
        profiling=_build_viewport_profiling_cfg(raw["profiling"]),
        layers={
            str(attr): _build_viz_layer_cfg(value)
            for attr, value in layers_raw.items()
            if isinstance(value, dict)
        },
    )


def _build_camera_focus_cfg(raw: object) -> VizCameraFocusConfig:
    if not isinstance(raw, dict):
        msg = f"viz.camera_focus config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return VizCameraFocusConfig(
        padding_m=float(cast(Any, raw["padding_m"])),
        min_scale_m=float(cast(Any, raw["min_scale_m"])),
        max_scale_m=float(cast(Any, raw["max_scale_m"])),
    )


def _build_viewport_profiling_cfg(raw: object) -> ViewportProfilingConfig:
    if not isinstance(raw, dict):
        msg = f"viz.profiling config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return ViewportProfilingConfig(
        enabled=bool(cast(Any, raw["enabled"])),
        slow_frame_ms=float(cast(Any, raw["slow_frame_ms"])),
        log_every_n_interactions=int(cast(Any, raw["log_every_n_interactions"])),
    )


def _build_ngii_cfg(cfg: DictConfig) -> NGIIConfig:
    ngii_raw = _raw_mapping(OmegaConf.to_container(cfg.ngii, resolve=True), "ngii")
    _reject_unknown_keys(ngii_raw, _NGII_CONFIG_KEYS, "ngii")
    sanity_raw = _raw_mapping(OmegaConf.to_container(cfg.ngii.sanity, resolve=True), "ngii.sanity")
    _reject_unknown_keys(sanity_raw, _SANITY_CONFIG_KEYS, "ngii.sanity")
    return NGIIConfig(
        sanity=NGIISanityConfig(
            node_match_tolerance_m=float(cfg.ngii.sanity.node_match_tolerance_m),
            direction_parallel_dot_min=float(cfg.ngii.sanity.direction_parallel_dot_min),
            link_min_length_m=_non_negative_float(
                cfg.ngii.sanity.link_min_length_m,
                "ngii.sanity.link_min_length_m",
            ),
            checks=_build_sanity_checks_cfg(cfg.ngii.sanity.checks),
        ),
        geometry=NGIIGeometryConfig(
            multipart_snap_tolerance_m=_non_negative_float(
                cfg.ngii.geometry.multipart_snap_tolerance_m,
                "ngii.geometry.multipart_snap_tolerance_m",
            ),
        ),
        encoding=NGIIEncodingConfig(
            utf8_dbf_invalid_non_ascii_ratio_max=_ratio_float(
                cfg.ngii.encoding.utf8_dbf_invalid_non_ascii_ratio_max,
                "ngii.encoding.utf8_dbf_invalid_non_ascii_ratio_max",
            ),
        ),
    )


def _build_sanity_checks_cfg(raw: object) -> NGIISanityChecksConfig:
    checks_raw = _raw_mapping(OmegaConf.to_container(raw, resolve=True), "ngii.sanity.checks")
    expected_names = {field.name for field in dataclass_fields(NGIISanityChecksConfig)}
    _reject_unknown_keys(checks_raw, expected_names, "ngii.sanity.checks")
    missing = sorted(expected_names - checks_raw.keys())
    if missing:
        msg = f"ngii.sanity.checks is missing required checks: {', '.join(missing)}"
        raise ValueError(msg)
    modes = {
        name: _sanity_mode(checks_raw[name], f"ngii.sanity.checks.{name}")
        for name in expected_names
    }
    invalid_repair = sorted(
        name for name in _WARNING_ONLY_CHECKS if modes[name] is SanityMode.REPAIR
    )
    if invalid_repair:
        msg = f"warning-only sanity checks cannot use repair mode: {', '.join(invalid_repair)}"
        raise ValueError(msg)
    return NGIISanityChecksConfig(**cast(Any, modes))


def _sanity_mode(raw: object, name: str) -> SanityMode | None:
    if raw is None:
        return None
    if raw == SanityMode.WARN.value:
        return SanityMode.WARN
    if raw == SanityMode.REPAIR.value:
        return SanityMode.REPAIR
    msg = f"{name} must be null, 'warn', or 'repair', got {raw!r}"
    raise ValueError(msg)


def _raw_mapping(raw: object, name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        msg = f"{name} config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return {str(key): value for key, value in raw.items()}


def _reject_unknown_keys(
    raw: dict[str, object], expected: frozenset[str] | set[str], name: str
) -> None:
    unknown = sorted(raw.keys() - expected)
    if unknown:
        msg = f"{name} contains unknown keys: {', '.join(unknown)}"
        raise ValueError(msg)


def _non_negative_float(raw: object, name: str) -> float:
    value = float(cast(Any, raw))
    if value < 0.0:
        msg = f"{name} must be >= 0.0, got {value}"
        raise ValueError(msg)
    return value


def _positive_float(raw: object, name: str) -> float:
    value = float(cast(Any, raw))
    if value <= 0.0:
        msg = f"{name} must be > 0.0, got {value}"
        raise ValueError(msg)
    return value


def _ratio_float(raw: object, name: str) -> float:
    value = float(cast(Any, raw))
    if value < 0.0 or value > 1.0:
        msg = f"{name} must be between 0.0 and 1.0, got {value}"
        raise ValueError(msg)
    return value


def _build_viz_layer_cfg(raw: dict[str, object]) -> VizLayerConfig:
    rgb = raw["rgb"]
    if not isinstance(rgb, list):
        msg = f"viz layer rgb: expected list, got {type(rgb).__name__}"
        raise TypeError(msg)
    return VizLayerConfig(
        visible=bool(raw["visible"]),
        rgb=_rgb_int(cast(list[int], rgb)),
        point_size=float(cast(Any, raw["point_size"])),
        line_width=float(cast(Any, raw["line_width"])),
        opacity=float(cast(Any, raw["opacity"])),
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(
        seg_cfg=_build_seg_cfg(cfg),
        viz_cfg=_build_viz_cfg(cfg),
        ngii_cfg=_build_ngii_cfg(cfg),
        coordinate=str(cfg.ngii.coordinate),
    )
    window.show()
    if cfg.ngii_dir is not None:
        # to_absolute_path resolves against the invocation cwd, not Hydra's
        # per-run output dir, which is what the user means by a relative path.
        window.load_folder(Path(to_absolute_path(str(cfg.ngii_dir))).expanduser())
    app.exec()


if __name__ == "__main__":
    # Windows stdio defaults to cp1252; UTF-8 makes logging non-ASCII-safe.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
