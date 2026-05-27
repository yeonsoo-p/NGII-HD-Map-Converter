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
    NGIISanityConfig,
    NGIISanityRepairConfig,
    NGIISanityWarningConfig,
    NGIITextCorrectionConfig,
)
from ngii2xodr.ngii.segmentation import SegmentationConfig
from ngii2xodr.ngii.viz import VizCameraFocusConfig, VizConfig, VizLayerConfig, VizPointConfig
from ngii2xodr.profile import ViewportProfilingConfig

log = logging.getLogger(__name__)


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
        connection_perpendicular_half_length_m=float(
            cfg.segmentation.connection_perpendicular_half_length_m
        ),
        enable_node_link_relations=bool(cfg.segmentation.enable_node_link_relations),
        enable_uturn=bool(cfg.segmentation.enable_uturn),
        enable_lateral_link_group=bool(cfg.segmentation.enable_lateral_link_group),
        enable_lateral_node_group=bool(cfg.segmentation.enable_lateral_node_group),
        enable_junction=bool(cfg.segmentation.enable_junction),
        enable_junction_connection=bool(cfg.segmentation.enable_junction_connection),
        enable_connection_reference=bool(cfg.segmentation.enable_connection_reference),
        enable_connection_perpendicular=bool(cfg.segmentation.enable_connection_perpendicular),
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
        connection_reference_arrow_length_m=float(raw["connection_reference_arrow_length_m"]),
        connection_reference_arrow_rgb=_rgb_int(raw["connection_reference_arrow_rgb"]),
        selector_tol_point=float(raw["selector_tol_point"]),
        selector_tol_line=float(raw["selector_tol_line"]),
        selector_tol_poly=float(raw["selector_tol_poly"]),
        point_hit_radius_px=float(raw["point_hit_radius_px"]),
        poly_depth_offset_factor=float(raw["poly_depth_offset_factor"]),
        poly_depth_offset_units=float(raw["poly_depth_offset_units"]),
        segmentation_seed=int(raw["segmentation_seed"]),
        camera_focus=_build_camera_focus_cfg(raw["camera_focus"]),
        points=_build_point_cfg(raw["points"]),
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


def _build_point_cfg(raw: object) -> VizPointConfig:
    if not isinstance(raw, dict):
        msg = f"viz.points config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return VizPointConfig(render_as_spheres=bool(cast(Any, raw["render_as_spheres"])))


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
    warning_cfg = cfg.ngii.sanity.warnings
    repair_cfg = cfg.ngii.sanity.repairs
    return NGIIConfig(
        sanity=NGIISanityConfig(
            node_match_tolerance_m=float(cfg.ngii.sanity.node_match_tolerance_m),
            direction_parallel_dot_min=float(cfg.ngii.sanity.direction_parallel_dot_min),
            warnings=NGIISanityWarningConfig(
                missing_required_layers=bool(warning_cfg.missing_required_layers),
                missing_sidecars=bool(warning_cfg.missing_sidecars),
                duplicate_column_capitalization=bool(warning_cfg.duplicate_column_capitalization),
                manual_field_rules=bool(warning_cfg.manual_field_rules),
                unknown_manual_columns=bool(warning_cfg.unknown_manual_columns),
                invalid_code_values=bool(warning_cfg.invalid_code_values),
                unsupported_geometry=bool(warning_cfg.unsupported_geometry),
                unknown_layers=bool(warning_cfg.unknown_layers),
                unresolved_relationships=bool(warning_cfg.unresolved_relationships),
                global_id_collision=bool(warning_cfg.global_id_collision),
                duplicate_identical_ids=bool(warning_cfg.duplicate_identical_ids),
                duplicate_conflicting_ids=bool(warning_cfg.duplicate_conflicting_ids),
                link_endpoint_alignment=_cfg_bool(
                    warning_cfg, "link_endpoint_alignment", legacy_name="a2_endpoint_alignment"
                ),
                link_direction_ambiguous=_cfg_bool(
                    warning_cfg, "link_direction_ambiguous", legacy_name="a2_direction_ambiguous"
                ),
                link_topology_direction=_cfg_bool(
                    warning_cfg, "link_topology_direction", legacy_name="a2_topology_direction"
                ),
            ),
            repairs=NGIISanityRepairConfig(
                duplicate_conflicting_id_drop=bool(repair_cfg.duplicate_conflicting_id_drop),
                link_endpoint_direction_swap=_cfg_bool(
                    repair_cfg,
                    "link_endpoint_direction_swap",
                    legacy_name="a2_endpoint_direction_swap",
                ),
                link_missing_node_ref_nearest=_cfg_bool(
                    repair_cfg,
                    "link_missing_node_ref_nearest",
                    legacy_name="a2_missing_node_ref_nearest",
                ),
                link_missing_node_ref_remove=_cfg_bool(
                    repair_cfg,
                    "link_missing_node_ref_remove",
                    legacy_name="a2_missing_node_ref_remove",
                ),
                link_topology_direction_swap=_cfg_bool(
                    repair_cfg,
                    "link_topology_direction_swap",
                    legacy_name="a2_topology_direction_swap",
                ),
            ),
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
        text_repair=_build_text_correction_cfg(cfg),
    )


def _cfg_bool(raw: DictConfig, name: str, *, legacy_name: str | None = None) -> bool:
    if name in raw:
        return bool(raw[name])
    if legacy_name is not None and legacy_name in raw:
        return bool(raw[legacy_name])
    raise KeyError(name)


def _non_negative_float(raw: object, name: str) -> float:
    value = float(cast(Any, raw))
    if value < 0.0:
        msg = f"{name} must be >= 0.0, got {value}"
        raise ValueError(msg)
    return value


def _ratio_float(raw: object, name: str) -> float:
    value = float(cast(Any, raw))
    if value < 0.0 or value > 1.0:
        msg = f"{name} must be between 0.0 and 1.0, got {value}"
        raise ValueError(msg)
    return value


def _build_text_correction_cfg(cfg: DictConfig) -> NGIITextCorrectionConfig:
    raw = OmegaConf.to_container(cfg.ngii.text_repair, resolve=True)
    if not isinstance(raw, dict):
        msg = f"ngii.text_repair config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return NGIITextCorrectionConfig(
        enabled=bool(raw["enabled"]),
        repair_mojibake=bool(raw["repair_mojibake"]),
        warn_unrepaired_replacement_chars=bool(raw["warn_unrepaired_replacement_chars"]),
    )


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
