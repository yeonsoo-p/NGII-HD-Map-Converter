"""Hydra-built configuration objects for NGII data loading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class NGIISanityWarningConfig:
    missing_required_layers: bool
    missing_sidecars: bool
    duplicate_column_capitalization: bool
    manual_field_rules: bool
    unknown_manual_columns: bool
    invalid_code_values: bool
    unsupported_geometry: bool
    unknown_layers: bool
    unresolved_relationships: bool
    global_id_collision: bool
    duplicate_identical_ids: bool
    duplicate_conflicting_ids: bool
    link_endpoint_alignment: bool
    link_direction_ambiguous: bool
    link_topology_direction: bool
    link_lateral_longitudinal_conflict: bool
    link_lateral_reciprocal_conflict: bool
    too_short_links: bool
    singular_links: bool
    dangling_relationships: bool
    dangling_nodes: bool
    reciprocal_relationships: bool


@dataclass(slots=True, frozen=True)
class NGIISanityRepairConfig:
    duplicate_conflicting_id_drop: bool
    link_too_short_remove: bool
    link_singular_remove: bool
    link_endpoint_direction_swap: bool
    link_missing_node_ref_nearest: bool
    link_missing_node_ref_remove: bool
    link_topology_direction_swap: bool
    link_lateral_longitudinal_conflict_clear: bool
    link_lateral_reciprocal_conflict_repair: bool
    dangling_relationship_remove: bool
    dangling_node_remove: bool
    reciprocal_relationship_fill: bool


@dataclass(slots=True, frozen=True)
class NGIISanityConfig:
    """Thresholds used by sanity-action checks."""

    node_match_tolerance_m: float
    direction_parallel_dot_min: float
    link_min_length_m: float
    warnings: NGIISanityWarningConfig
    repairs: NGIISanityRepairConfig


@dataclass(slots=True, frozen=True)
class NGIIGeometryConfig:
    """Geometry conversion thresholds."""

    multipart_snap_tolerance_m: float


@dataclass(slots=True, frozen=True)
class NGIIEncodingConfig:
    """Text decoding heuristics."""

    utf8_dbf_invalid_non_ascii_ratio_max: float


@dataclass(slots=True, frozen=True)
class NGIITextCorrectionConfig:
    enabled: bool
    repair_mojibake: bool
    warn_unrepaired_replacement_chars: bool


@dataclass(slots=True, frozen=True)
class NGIIConfig:
    """Configuration consumed by :func:`ngii2xodr.ngii.data.load_ngii`."""

    sanity: NGIISanityConfig
    geometry: NGIIGeometryConfig
    encoding: NGIIEncodingConfig
    text_repair: NGIITextCorrectionConfig
