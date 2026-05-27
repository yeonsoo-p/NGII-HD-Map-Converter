"""Hydra-built configuration objects for NGII data loading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SanityMode(StrEnum):
    WARN = "warn"
    REPAIR = "repair"


SanityCheckMode = SanityMode | None


def check_reports(mode: SanityCheckMode) -> bool:
    return mode is not None


def check_repairs(mode: SanityCheckMode) -> bool:
    return mode is SanityMode.REPAIR


@dataclass(slots=True, frozen=True)
class NGIISanityChecksConfig:
    layer_required_missing: SanityCheckMode
    layer_unknown: SanityCheckMode
    shp_sidecar_missing: SanityCheckMode
    dbf_column_case_duplicate: SanityCheckMode
    manual_column_missing: SanityCheckMode
    manual_column_unknown: SanityCheckMode
    manual_field_required_missing: SanityCheckMode
    manual_field_length_exceeded: SanityCheckMode
    manual_field_type_invalid: SanityCheckMode
    manual_field_code_invalid: SanityCheckMode
    manual_hist_type_invalid: SanityCheckMode
    geometry_missing: SanityCheckMode
    geometry_invalid: SanityCheckMode
    text_utf8_dbf_row_mismatch: SanityCheckMode
    text_utf8_decode_replacement: SanityCheckMode
    text_mojibake: SanityCheckMode
    text_replacement_char: SanityCheckMode
    feature_id_missing: SanityCheckMode
    feature_id_duplicate_identical: SanityCheckMode
    feature_id_duplicate_conflicting: SanityCheckMode
    global_id_collision: SanityCheckMode
    reference_unresolved: SanityCheckMode
    reciprocal_reference_missing: SanityCheckMode
    reciprocal_reference_conflict: SanityCheckMode
    link_too_short: SanityCheckMode
    link_endpoint_isolated: SanityCheckMode
    link_endpoint_unresolved: SanityCheckMode
    link_endpoint_reversed: SanityCheckMode
    link_endpoint_order_ambiguous: SanityCheckMode
    link_endpoint_misaligned: SanityCheckMode
    link_flow_reversed: SanityCheckMode
    node_unreferenced: SanityCheckMode
    link_side_reference_longitudinal: SanityCheckMode
    link_side_reference_nonreciprocal: SanityCheckMode


@dataclass(slots=True, frozen=True)
class NGIISanityConfig:
    """Thresholds and modes used by sanity checks."""

    node_match_tolerance_m: float
    direction_parallel_dot_min: float
    link_min_length_m: float
    checks: NGIISanityChecksConfig


@dataclass(slots=True, frozen=True)
class NGIIGeometryConfig:
    """Geometry conversion thresholds."""

    multipart_snap_tolerance_m: float


@dataclass(slots=True, frozen=True)
class NGIIEncodingConfig:
    """Text decoding heuristics."""

    utf8_dbf_invalid_non_ascii_ratio_max: float


@dataclass(slots=True, frozen=True)
class NGIIConfig:
    """Configuration consumed by :func:`ngii2xodr.ngii.data.load_ngii`."""

    sanity: NGIISanityConfig
    geometry: NGIIGeometryConfig
    encoding: NGIIEncodingConfig
