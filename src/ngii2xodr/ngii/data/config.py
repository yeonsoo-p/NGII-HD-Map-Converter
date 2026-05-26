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
    a2_endpoint_alignment: bool
    a2_direction_ambiguous: bool
    a2_topology_direction: bool


@dataclass(slots=True, frozen=True)
class NGIISanityRepairConfig:
    duplicate_conflicting_id_drop: bool
    a2_endpoint_direction_swap: bool
    a2_missing_node_ref_nearest: bool
    a2_missing_node_ref_remove: bool
    a2_topology_direction_swap: bool


@dataclass(slots=True, frozen=True)
class NGIISanityConfig:
    """Thresholds used by sanity-action checks."""

    node_match_tolerance_m: float
    direction_parallel_dot_min: float
    warnings: NGIISanityWarningConfig
    repairs: NGIISanityRepairConfig


@dataclass(slots=True, frozen=True)
class NGIITextCorrection:
    layer_name: str
    feature_id: str
    field: str
    old: str
    new: str


@dataclass(slots=True, frozen=True)
class NGIITextCorrectionConfig:
    enabled: bool
    repair_mojibake: bool
    apply_exact_corrections: bool
    warn_unrepaired_replacement_chars: bool
    corrections: tuple[NGIITextCorrection, ...]


@dataclass(slots=True, frozen=True)
class NGIIConfig:
    """Configuration consumed by :func:`ngii2xodr.ngii.data.load_ngii`."""

    sanity: NGIISanityConfig
    text_repair: NGIITextCorrectionConfig
