"""Shared runtime configuration builders and validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ngii2xodr.ngii.data.config import (
    NGIIConfig,
    NGIIEncodingConfig,
    NGIIGeometryConfig,
    NGIISanityChecksConfig,
    NGIISanityConfig,
    SanityCheckMode,
    SanityMode,
)
from ngii2xodr.ngii.segmentation import SegmentationConfig
from ngii2xodr.ngii.viz import VizCameraFocusConfig, VizConfig, VizLayerConfig
from ngii2xodr.profile import ViewportProfilingConfig

NGII_CONFIG_KEYS = frozenset({"coordinate", "geometry", "encoding", "sanity"})
SANITY_CONFIG_KEYS = frozenset(
    {"node_match_tolerance_m", "direction_parallel_dot_min", "link_min_length_m", "checks"}
)
SANITY_CHECK_NAMES = tuple(field.name for field in dataclass_fields(NGIISanityChecksConfig))
WARNING_ONLY_SANITY_CHECKS = frozenset(
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
        "link_endpoint_misaligned",
    }
)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """Typed in-memory configuration for one GUI/CLI runtime."""

    coordinate: str
    ngii: NGIIConfig
    segmentation: SegmentationConfig
    viz: VizConfig


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    runtime_cfg = RuntimeConfig(
        coordinate=str(cfg.ngii.coordinate),
        ngii=build_ngii_config(cfg),
        segmentation=build_segmentation_config(cfg),
        viz=build_viz_config(cfg),
    )
    validate_runtime_config(runtime_cfg)
    return runtime_cfg


def build_segmentation_config(cfg: DictConfig) -> SegmentationConfig:
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


def build_viz_config(cfg: DictConfig) -> VizConfig:
    raw = _raw_mapping(OmegaConf.to_container(cfg.viz, resolve=True), "viz")
    layers_raw = _raw_mapping(raw["layers"], "viz.layers")
    return VizConfig(
        background_color=_rgb_float(cast(Sequence[object], raw["background_color"])),
        highlight_rgb=_rgb_int(cast(Sequence[object], raw["highlight_rgb"])),
        junction_connection_node_point_size=_float_value(
            raw["junction_connection_node_point_size"]
        ),
        junction_reference_arrow_length_m=_float_value(raw["junction_reference_arrow_length_m"]),
        junction_reference_arrow_rgb=_rgb_int(
            cast(Sequence[object], raw["junction_reference_arrow_rgb"])
        ),
        junction_edge_rgb=_rgb_int(cast(Sequence[object], raw["junction_edge_rgb"])),
        junction_edge_line_width=_float_value(raw["junction_edge_line_width"]),
        selector_tol_point=_float_value(raw["selector_tol_point"]),
        selector_tol_line=_float_value(raw["selector_tol_line"]),
        selector_tol_poly=_float_value(raw["selector_tol_poly"]),
        point_hit_radius_px=_float_value(raw["point_hit_radius_px"]),
        poly_depth_offset_factor=_float_value(raw["poly_depth_offset_factor"]),
        poly_depth_offset_units=_float_value(raw["poly_depth_offset_units"]),
        segmentation_seed=_int_value(raw["segmentation_seed"]),
        camera_focus=build_camera_focus_config(raw["camera_focus"]),
        profiling=build_viewport_profiling_config(raw["profiling"]),
        layers={
            str(attr): build_viz_layer_config(_raw_mapping(value, f"viz.layers.{attr}"))
            for attr, value in layers_raw.items()
        },
    )


def build_camera_focus_config(raw: object) -> VizCameraFocusConfig:
    mapping = _raw_mapping(raw, "viz.camera_focus")
    return VizCameraFocusConfig(
        padding_m=_float_value(mapping["padding_m"]),
        min_scale_m=_float_value(mapping["min_scale_m"]),
        max_scale_m=_float_value(mapping["max_scale_m"]),
    )


def build_viewport_profiling_config(raw: object) -> ViewportProfilingConfig:
    mapping = _raw_mapping(raw, "viz.profiling")
    return ViewportProfilingConfig(
        enabled=bool(mapping["enabled"]),
        slow_frame_ms=_float_value(mapping["slow_frame_ms"]),
        log_every_n_interactions=_int_value(mapping["log_every_n_interactions"]),
    )


def build_ngii_config(cfg: DictConfig) -> NGIIConfig:
    ngii_raw = _raw_mapping(OmegaConf.to_container(cfg.ngii, resolve=True), "ngii")
    _reject_unknown_keys(ngii_raw, NGII_CONFIG_KEYS, "ngii")
    sanity_raw = _raw_mapping(OmegaConf.to_container(cfg.ngii.sanity, resolve=True), "ngii.sanity")
    _reject_unknown_keys(sanity_raw, SANITY_CONFIG_KEYS, "ngii.sanity")
    return NGIIConfig(
        sanity=NGIISanityConfig(
            node_match_tolerance_m=float(cfg.ngii.sanity.node_match_tolerance_m),
            direction_parallel_dot_min=float(cfg.ngii.sanity.direction_parallel_dot_min),
            link_min_length_m=_non_negative_float(
                cfg.ngii.sanity.link_min_length_m,
                "ngii.sanity.link_min_length_m",
            ),
            checks=build_sanity_checks_config(cfg.ngii.sanity.checks),
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


def build_sanity_checks_config(raw: object) -> NGIISanityChecksConfig:
    checks_raw = _raw_mapping(OmegaConf.to_container(raw, resolve=True), "ngii.sanity.checks")
    _reject_unknown_keys(checks_raw, set(SANITY_CHECK_NAMES), "ngii.sanity.checks")
    missing = sorted(set(SANITY_CHECK_NAMES) - checks_raw.keys())
    if missing:
        msg = f"ngii.sanity.checks is missing required checks: {', '.join(missing)}"
        raise ValueError(msg)
    return make_sanity_checks_config(
        {
            name: sanity_mode(checks_raw[name], f"ngii.sanity.checks.{name}")
            for name in SANITY_CHECK_NAMES
        }
    )


def make_sanity_checks_config(
    modes: Mapping[str, SanityCheckMode],
) -> NGIISanityChecksConfig:
    _reject_unknown_keys(dict(modes), set(SANITY_CHECK_NAMES), "ngii.sanity.checks")
    missing = sorted(set(SANITY_CHECK_NAMES) - modes.keys())
    if missing:
        msg = f"ngii.sanity.checks is missing required checks: {', '.join(missing)}"
        raise ValueError(msg)
    invalid_repair = sorted(
        name for name in WARNING_ONLY_SANITY_CHECKS if modes[name] is SanityMode.REPAIR
    )
    if invalid_repair:
        msg = f"warning-only sanity checks cannot use repair mode: {', '.join(invalid_repair)}"
        raise ValueError(msg)
    return NGIISanityChecksConfig(**cast(Any, {name: modes[name] for name in SANITY_CHECK_NAMES}))


def sanity_mode(raw: object, name: str) -> SanityCheckMode:
    if raw is None:
        return None
    if raw == SanityMode.WARN.value or raw is SanityMode.WARN:
        return SanityMode.WARN
    if raw == SanityMode.REPAIR.value or raw is SanityMode.REPAIR:
        return SanityMode.REPAIR
    msg = f"{name} must be null, 'warn', or 'repair', got {raw!r}"
    raise ValueError(msg)


def build_viz_layer_config(raw: Mapping[str, object]) -> VizLayerConfig:
    return VizLayerConfig(
        visible=bool(raw["visible"]),
        rgb=_rgb_int(cast(Sequence[object], raw["rgb"])),
        point_size=_float_value(raw["point_size"]),
        line_width=_float_value(raw["line_width"]),
        opacity=_float_value(raw["opacity"]),
    )


def validate_runtime_config(cfg: RuntimeConfig) -> None:
    if not cfg.coordinate.strip():
        msg = "ngii.coordinate must not be empty"
        raise ValueError(msg)
    validate_ngii_config(cfg.ngii)
    validate_segmentation_config(cfg.segmentation)
    validate_viz_config(cfg.viz)


def validate_ngii_config(cfg: NGIIConfig) -> None:
    _non_negative_float(
        cfg.geometry.multipart_snap_tolerance_m,
        "ngii.geometry.multipart_snap_tolerance_m",
    )
    _ratio_float(
        cfg.encoding.utf8_dbf_invalid_non_ascii_ratio_max,
        "ngii.encoding.utf8_dbf_invalid_non_ascii_ratio_max",
    )
    _non_negative_float(cfg.sanity.link_min_length_m, "ngii.sanity.link_min_length_m")
    make_sanity_checks_config(
        {name: getattr(cfg.sanity.checks, name) for name in SANITY_CHECK_NAMES}
    )


def validate_segmentation_config(cfg: SegmentationConfig) -> None:
    _ratio_float(
        cfg.junction_connection_opposite_direction_dot_min,
        "segmentation.junction_connection_opposite_direction_dot_min",
    )
    _positive_float(
        cfg.endpoint_tangent_lookback_m,
        "segmentation.endpoint_tangent_lookback_m",
    )


def validate_viz_config(cfg: VizConfig) -> None:
    _rgb_float(cfg.background_color)
    _rgb_int(cfg.highlight_rgb)
    _positive_float(
        cfg.junction_connection_node_point_size,
        "viz.junction_connection_node_point_size",
    )
    _positive_float(
        cfg.junction_reference_arrow_length_m,
        "viz.junction_reference_arrow_length_m",
    )
    _rgb_int(cfg.junction_reference_arrow_rgb)
    _rgb_int(cfg.junction_edge_rgb)
    _positive_float(cfg.junction_edge_line_width, "viz.junction_edge_line_width")
    _non_negative_float(cfg.selector_tol_point, "viz.selector_tol_point")
    _non_negative_float(cfg.selector_tol_line, "viz.selector_tol_line")
    _non_negative_float(cfg.selector_tol_poly, "viz.selector_tol_poly")
    _positive_float(cfg.point_hit_radius_px, "viz.point_hit_radius_px")
    _non_negative_float(cfg.camera_focus.padding_m, "viz.camera_focus.padding_m")
    _positive_float(cfg.camera_focus.min_scale_m, "viz.camera_focus.min_scale_m")
    _positive_float(cfg.camera_focus.max_scale_m, "viz.camera_focus.max_scale_m")
    if cfg.camera_focus.max_scale_m < cfg.camera_focus.min_scale_m:
        msg = "viz.camera_focus.max_scale_m must be >= viz.camera_focus.min_scale_m"
        raise ValueError(msg)
    _positive_float(cfg.profiling.slow_frame_ms, "viz.profiling.slow_frame_ms")
    if cfg.profiling.log_every_n_interactions < 1:
        msg = "viz.profiling.log_every_n_interactions must be >= 1"
        raise ValueError(msg)
    for attr, layer in cfg.layers.items():
        _rgb_int(layer.rgb)
        _positive_float(layer.point_size, f"viz.layers.{attr}.point_size")
        _positive_float(layer.line_width, f"viz.layers.{attr}.line_width")
        _ratio_float(layer.opacity, f"viz.layers.{attr}.opacity")


def with_viz_layer_visibility(cfg: RuntimeConfig, layer_attr: str, visible: bool) -> RuntimeConfig:
    layer_cfg = cfg.viz.layers.get(layer_attr)
    if layer_cfg is None:
        return cfg
    layers = dict(cfg.viz.layers)
    layers[layer_attr] = replace(layer_cfg, visible=visible)
    return replace(cfg, viz=replace(cfg.viz, layers=layers))


def _raw_mapping(raw: object, name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        msg = f"{name} config: expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return {str(key): value for key, value in raw.items()}


def _reject_unknown_keys(
    raw: Mapping[str, object], expected: frozenset[str] | set[str], name: str
) -> None:
    unknown = sorted(raw.keys() - expected)
    if unknown:
        msg = f"{name} contains unknown keys: {', '.join(unknown)}"
        raise ValueError(msg)


def _non_negative_float(raw: object, name: str) -> float:
    value = _float_value(raw)
    if value < 0.0:
        msg = f"{name} must be >= 0.0, got {value}"
        raise ValueError(msg)
    return value


def _positive_float(raw: object, name: str) -> float:
    value = _float_value(raw)
    if value <= 0.0:
        msg = f"{name} must be > 0.0, got {value}"
        raise ValueError(msg)
    return value


def _ratio_float(raw: object, name: str) -> float:
    value = _float_value(raw)
    if value < 0.0 or value > 1.0:
        msg = f"{name} must be between 0.0 and 1.0, got {value}"
        raise ValueError(msg)
    return value


def _rgb_int(v: Sequence[object]) -> tuple[int, int, int]:
    if len(v) != 3:
        msg = f"RGB values must have three channels, got {len(v)}"
        raise ValueError(msg)
    rgb = (_int_value(v[0]), _int_value(v[1]), _int_value(v[2]))
    if any(channel < 0 or channel > 255 for channel in rgb):
        msg = f"RGB channels must be between 0 and 255, got {rgb!r}"
        raise ValueError(msg)
    return rgb


def _rgb_float(v: Sequence[object]) -> tuple[float, float, float]:
    if len(v) != 3:
        msg = f"RGB values must have three channels, got {len(v)}"
        raise ValueError(msg)
    rgb = (_float_value(v[0]), _float_value(v[1]), _float_value(v[2]))
    if any(channel < 0.0 or channel > 1.0 for channel in rgb):
        msg = f"RGB float channels must be between 0.0 and 1.0, got {rgb!r}"
        raise ValueError(msg)
    return rgb


def _float_value(raw: object) -> float:
    return float(cast(Any, raw))


def _int_value(raw: object) -> int:
    return int(cast(Any, raw))
