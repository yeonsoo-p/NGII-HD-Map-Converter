"""Public NGII rewrite loader facade and SHP row IO helpers."""

from __future__ import annotations

import logging
import struct
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ngii2xodr.ngii.data.config import NGIIConfig, check_reports
from ngii2xodr.ngii_rewrite.data.dataset import NGIIDataset
from ngii2xodr.ngii_rewrite.data.features import (
    FeatureRecord,
    NGIIFeature,
    is_float_value,
    is_integer_value,
    optional_text,
    same_feature,
)
from ngii2xodr.ngii_rewrite.data.geometry import convert_geometry
from ngii2xodr.ngii_rewrite.data.metadata import (
    FieldDef,
    LayerDef,
    LayerType,
    Schema,
    columns_by_alias,
    field_defs,
    field_defs_by_column,
    manual_columns,
)
from ngii2xodr.ngii_rewrite.data.sanity import (
    SanityReport,
    check_manual_field_values,
    check_unresolved_references,
    log_sanity_report,
    warn_missing_columns,
    warn_missing_required_layers,
    warn_missing_sidecars,
    warn_unknown_columns,
)
from ngii2xodr.profile import log_profile_events

log = logging.getLogger(__name__)

_REPLACEMENT_CHAR = "\ufffd"


@dataclass(slots=True, frozen=True)
class DiscoveredLayerFile:
    layer: LayerDef
    path: Path
    is_filename_alias: bool


@dataclass(slots=True, frozen=True)
class NGIILoadResult:
    dataset: NGIIDataset
    sanity: SanityReport


def load_schema(root: Path, coordinate: str, cfg: NGIIConfig, schema: Schema) -> NGIILoadResult:
    sanity = SanityReport()
    dataset = NGIIDataset(root=root, coordinate=coordinate, schema=schema)

    with dataset.load_profile.timed("discover", coordinate):
        discovered = discover_layer_files(
            root,
            coordinate,
            schema,
            sanity,
            report_unknown_layers=check_reports(cfg.sanity.checks.layer_unknown),
        )

    if check_reports(cfg.sanity.checks.layer_required_missing):
        warn_missing_required_layers(discovered, schema, sanity)

    layer_order = {layer.layer_name: index for index, layer in enumerate(schema.layers)}
    for layer_file in sorted(discovered, key=lambda item: layer_file_sort_key(item, layer_order)):
        if check_reports(cfg.sanity.checks.shp_sidecar_missing):
            warn_missing_sidecars(layer_file.path, sanity)
        with dataset.load_profile.timed("read_layer", layer_file.path.name):
            features = read_layer_file(layer_file, sanity, cfg)
        with dataset.load_profile.timed("merge_layer", layer_file.path.name):
            merge_features(
                dataset.store_for_attr(layer_file.layer.layer_attr),
                features,
                sanity,
                cfg,
            )

    with dataset.load_profile.timed("bind_initial"):
        dataset.bind(
            sanity,
            warn_global_id_collision=check_reports(cfg.sanity.checks.global_id_collision),
        )
    if _reports_manual_field_values(cfg):
        with dataset.load_profile.timed("manual_validation"):
            check_manual_field_values(dataset, sanity, cfg)
    with dataset.load_profile.timed("reference_edges"):
        dataset.rebuild_reference_edges()
    if check_reports(cfg.sanity.checks.reference_unresolved):
        with dataset.load_profile.timed("reference_validation"):
            check_unresolved_references(dataset, sanity)
    log_sanity_report(dataset, sanity, log)
    log_profile_events(
        dataset.load_profile,
        log,
        title="NGII rewrite load profile",
        context=f"{dataset.root} [{dataset.coordinate}]",
    )
    return NGIILoadResult(dataset=dataset, sanity=sanity)


def load_ngii(
    root: Path,
    coordinate: str,
    cfg: NGIIConfig,
    schemas: Sequence[Schema],
) -> NGIILoadResult:
    schema = detect_schema(root, coordinate, schemas)
    return load_schema(root, coordinate, cfg, schema)


def detect_schema(root: Path, coordinate: str, schemas: Sequence[Schema]) -> Schema:
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
        for shp_path in coordinate_dir.glob("*.shp")
    }
    matches = [schema for schema in schemas if found_names & set(schema.layers_by_filename)]
    if len(matches) > 1:
        msg = (
            f"ambiguous NGII schema for {root} [{coordinate}]: "
            f"{tuple(schema.version for schema in matches)!r}"
        )
        raise ValueError(msg)
    if matches:
        return matches[0]

    msg = f"could not detect an implemented NGII schema for {root} [{coordinate}]"
    raise ValueError(msg)


def coordinate_dirs_for(root: Path, coordinate: str) -> list[Path]:
    if root.name == coordinate:
        return [root]
    return sorted(path for path in root.rglob(coordinate) if path.is_dir())


def discover_layer_files(
    root: Path,
    coordinate: str,
    schema: Schema,
    sanity: SanityReport,
    *,
    report_unknown_layers: bool,
) -> list[DiscoveredLayerFile]:
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    coordinate_dirs = coordinate_dirs_for(root, coordinate)
    if not coordinate_dirs:
        msg = f"coordinate folder {coordinate!r} was not found under {root}"
        raise FileNotFoundError(msg)

    layers_by_filename = schema.layers_by_filename
    discovered: list[DiscoveredLayerFile] = []
    for coordinate_dir in coordinate_dirs:
        candidates: dict[str, list[DiscoveredLayerFile]] = defaultdict(list)
        for shp_path in sorted(coordinate_dir.glob("*.shp")):
            layer = layers_by_filename.get(shp_path.name.upper())
            if layer is None:
                if report_unknown_layers:
                    sanity.warn(
                        "unknown-layer",
                        f"unknown SHP layer {shp_path.name!r} in requested coordinate product",
                        source_path=shp_path,
                    )
                continue
            candidates[layer.layer_name].append(
                DiscoveredLayerFile(
                    layer=layer,
                    path=shp_path,
                    is_filename_alias=shp_path.name.upper() != layer.filename.upper(),
                )
            )
        for layer_candidates in candidates.values():
            canonical = [item for item in layer_candidates if not item.is_filename_alias]
            discovered.extend(canonical or layer_candidates)
    return discovered


def layer_file_sort_key(
    layer_file: DiscoveredLayerFile, layer_order: dict[str, int]
) -> tuple[int, int, str]:
    alias_order = 1 if layer_file.is_filename_alias else 0
    return (layer_order[layer_file.layer.layer_name], alias_order, str(layer_file.path))


def read_layer_file(
    layer_file: DiscoveredLayerFile, sanity: SanityReport, cfg: NGIIConfig
) -> list[NGIIFeature]:
    gdf = _load_gdf(layer_file.path, layer_file.layer, sanity, cfg)
    gdf = _normalize_columns(
        gdf,
        layer_file.path,
        sanity,
        layer_file.layer,
        report_duplicate_columns=check_reports(cfg.sanity.checks.dbf_column_case_duplicate),
    )
    if check_reports(cfg.sanity.checks.manual_column_missing):
        warn_missing_columns(gdf, layer_file.layer, layer_file.path, sanity)
    if check_reports(cfg.sanity.checks.manual_column_unknown):
        warn_unknown_columns(gdf, layer_file.layer, layer_file.path, sanity)
    features: list[NGIIFeature] = []
    documented_columns = manual_columns(layer_file.layer.layer_type)
    for row_idx, row in enumerate(gdf.itertuples(index=False), start=0):
        row_dict = dict(zip(gdf.columns, row, strict=True))
        geometry = row_dict.pop("geometry")
        if not isinstance(geometry, shapely.geometry.base.BaseGeometry):
            if check_reports(cfg.sanity.checks.geometry_missing):
                sanity.warn(
                    "missing-geometry",
                    f"{layer_file.layer.layer_name} row {row_idx} has no geometry",
                    layer_name=layer_file.layer.layer_name,
                    source_path=layer_file.path,
                )
            continue
        attrs = {
            key: _clean_value(value) for key, value in row_dict.items() if key in documented_columns
        }
        undocumented_attrs = {
            key: _clean_value(value)
            for key, value in row_dict.items()
            if key not in documented_columns
        }
        row_id = str(attrs.get("ID", ""))
        try:
            geometry_kind, converted_geometry = convert_geometry(
                layer_file.layer.layer_type,
                geometry,
                row_id=row_id,
                multipart_snap_tolerance_m=cfg.geometry.multipart_snap_tolerance_m,
            )
        except (TypeError, ValueError) as e:
            if check_reports(cfg.sanity.checks.geometry_invalid):
                sanity.warn(
                    "geometry-invalid",
                    str(e),
                    layer_name=layer_file.layer.layer_name,
                    feature_id=row_id,
                    source_path=layer_file.path,
                )
            continue
        _warn_row_numeric_parse_issues(layer_file, attrs, row_idx, row_id, sanity, cfg)
        feature = feature_from_record(
            layer_file.layer.layer_type,
            FeatureRecord(
                layer_name=layer_file.layer.layer_name,
                source_path=layer_file.path,
                source_row=row_idx,
                attributes=attrs,
                geometry_kind=geometry_kind,
                geometry=converted_geometry,
            ),
            undocumented_attrs,
        )
        features.append(feature)
    return features


def feature_from_record(
    layer_type: LayerType,
    record: FeatureRecord,
    undocumented_attributes: dict[str, Any],
) -> NGIIFeature:
    kwargs: dict[str, Any] = {
        "id": record.text("ID"),
        "source_path": record.source_path,
        "source_row": record.source_row,
    }
    _add_geometry_kwargs(kwargs, layer_type, record)
    for definition in field_defs(layer_type):
        if definition.attr == "id":
            continue
        kwargs[definition.attr] = _value_from_record(record, definition)
    feature = cast(NGIIFeature, cast(Any, layer_type)(**kwargs))
    feature.undocumented_attributes = undocumented_attributes
    return feature


def merge_features(
    store: Any,
    features: list[NGIIFeature],
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    for feature in features:
        if not feature.id:
            if check_reports(cfg.sanity.checks.feature_id_missing):
                sanity.warn(
                    "feature-id-missing",
                    f"{feature.layer_name} row has an empty ID and cannot be globally indexed",
                    layer_name=feature.layer_name,
                    source_path=feature.source_path,
                )
            continue
        existing = store.by_id.get(feature.id)
        if existing is None:
            store.id_to_index[feature.id] = len(store.features)
            store.by_id[feature.id] = feature
            store.features.append(feature)
            continue
        if same_feature(existing, feature):
            if check_reports(cfg.sanity.checks.feature_id_duplicate_identical):
                sanity.warn(
                    "feature-id-duplicate-identical",
                    f"{feature.layer_name} ID {feature.id!r} appears more than once "
                    "with identical data",
                    layer_name=feature.layer_name,
                    feature_id=feature.id,
                    source_path=feature.source_path,
                )
            continue
        if check_reports(cfg.sanity.checks.feature_id_duplicate_conflicting):
            sanity.warn(
                "feature-id-duplicate-conflicting",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row dropped",
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )


def _reports_manual_field_values(cfg: NGIIConfig) -> bool:
    checks = cfg.sanity.checks
    return any(
        check_reports(mode)
        for mode in (
            checks.manual_field_required_missing,
            checks.manual_field_length_exceeded,
            checks.manual_field_type_invalid,
            checks.manual_field_code_invalid,
            checks.manual_hist_type_invalid,
        )
    )


def _add_geometry_kwargs(
    kwargs: dict[str, Any], layer_type: LayerType, record: FeatureRecord
) -> None:
    constructor_fields = {item.name for item in fields(layer_type) if item.init}
    if record.geometry_kind == "point":
        if "point" in constructor_fields:
            kwargs["point"] = record.geometry
        if "ring" in constructor_fields:
            kwargs["ring"] = None
        return
    if record.geometry_kind == "line":
        if "polyline" in constructor_fields:
            kwargs["polyline"] = record.geometry
        return
    if "ring" in constructor_fields:
        kwargs["ring"] = record.geometry
    if "point" in constructor_fields:
        kwargs["point"] = None


def _value_from_record(record: FeatureRecord, definition: FieldDef) -> str | int | float | None:
    if definition.is_reference:
        return record.optional_ref(definition.name)
    if definition.field_type == "integer":
        return record.integer(definition.name)
    if definition.field_type == "float":
        return record.floating(definition.name)
    return record.text(definition.name)


def _warn_row_numeric_parse_issues(
    layer_file: DiscoveredLayerFile,
    attrs: dict[str, Any],
    row_idx: int,
    row_id: str,
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    if not check_reports(cfg.sanity.checks.manual_field_type_invalid):
        return
    feature_label = row_id or f"row {row_idx}"
    feature_id = optional_text(row_id) or None
    for definition in field_defs(layer_file.layer.layer_type):
        value = attrs.get(definition.name, "")
        invalid_integer = definition.field_type == "integer" and not is_integer_value(value)
        invalid_float = definition.field_type == "float" and not is_float_value(value)
        if not invalid_integer and not invalid_float:
            continue
        sanity.warn(
            "manual-field-type-invalid",
            f"{layer_file.layer.layer_name} {feature_label}: {definition.name}="
            f"{optional_text(value)!r} is not a valid {definition.field_type}",
            layer_name=layer_file.layer.layer_name,
            feature_id=feature_id,
            source_path=layer_file.path,
        )


def _load_gdf(
    shp_path: Path, layer: LayerDef, sanity: SanityReport, cfg: NGIIConfig
) -> gpd.GeoDataFrame:
    overlay_utf8_dbf_text = False
    try:
        gdf = gpd.read_file(shp_path)
        retry_reason = "mojibake heuristic" if _looks_like_cp949_mojibake(gdf) else None
    except UnicodeDecodeError:
        retry_reason = "UTF-8 decode failure"
        overlay_utf8_dbf_text = _declares_utf8(shp_path)
    if retry_reason is not None:
        log.debug("retrying %s with cp949 after %s", shp_path.name, retry_reason)
        gdf = gpd.read_file(shp_path, encoding="cp949")
        if overlay_utf8_dbf_text:
            _overlay_utf8_dbf_text(gdf, shp_path, layer, sanity, cfg)
    return gdf


def _declares_utf8(shp_path: Path) -> bool:
    cpg_path = shp_path.with_suffix(".cpg")
    if not cpg_path.is_file():
        return False
    declared = cpg_path.read_text(encoding="ascii", errors="ignore").strip().casefold()
    normalized = declared.replace("-", "").replace("_", "")
    return normalized in {"utf8", "65001"}


def _overlay_utf8_dbf_text(
    gdf: gpd.GeoDataFrame,
    shp_path: Path,
    layer: LayerDef,
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    records, replacements, non_ascii_cells = _read_utf8_dbf_text_records(
        shp_path.with_suffix(".dbf"), layer
    )
    if not records:
        return
    if (
        non_ascii_cells > 0
        and len(replacements) / non_ascii_cells > cfg.encoding.utf8_dbf_invalid_non_ascii_ratio_max
    ):
        log.debug(
            "skipping UTF-8 DBF text overlay for %s: %d/%d non-ASCII text cells "
            "decode with replacement characters",
            shp_path.name,
            len(replacements),
            non_ascii_cells,
        )
        return
    if len(records) != len(gdf):
        if check_reports(cfg.sanity.checks.text_utf8_dbf_row_mismatch):
            sanity.warn(
                "text-utf8-dbf-row-mismatch",
                f"{layer.layer_name}: raw DBF text row count {len(records)} does not match "
                f"geometry row count {len(gdf)}; UTF-8 text overlay skipped",
                layer_name=layer.layer_name,
                source_path=shp_path,
            )
        return
    columns = sorted({column for record in records for column in record})
    for column in columns:
        if column in gdf.columns:
            gdf[column] = [record.get(column, "") for record in records]
    for row_idx, feature_id, column in replacements:
        if check_reports(cfg.sanity.checks.text_utf8_decode_replacement):
            sanity.warn(
                "text-utf8-decode-replacement",
                f"{layer.layer_name} {feature_id or f'row {row_idx}'}: {column} contained "
                "malformed UTF-8 bytes and was decoded with replacement characters",
                layer_name=layer.layer_name,
                feature_id=feature_id or None,
                source_path=shp_path,
            )


def _read_utf8_dbf_text_records(
    dbf_path: Path, layer: LayerDef
) -> tuple[list[dict[str, str]], list[tuple[int, str, str]], int]:
    if not dbf_path.is_file():
        return [], [], 0
    data = dbf_path.read_bytes()
    if len(data) < 32:
        return [], [], 0
    num_records = struct.unpack_from("<I", data, 4)[0]
    header_len = struct.unpack_from("<H", data, 8)[0]
    record_len = struct.unpack_from("<H", data, 10)[0]
    text_field_by_lower = _text_field_by_lower(layer)
    fields_: list[tuple[str, str, int, int]] = []
    row_offset = 1
    pos = 32
    while pos + 32 <= len(data) and data[pos] != 0x0D:
        desc = data[pos : pos + 32]
        name = desc[:11].split(b"\0", maxsplit=1)[0].decode("ascii", errors="replace")
        field_type = chr(desc[11])
        length = int(desc[16])
        canonical_name = text_field_by_lower.get(name.lower())
        if field_type == "C" and canonical_name is not None:
            fields_.append((name, canonical_name, row_offset, length))
        row_offset += length
        pos += 32
    records: list[dict[str, str]] = []
    replacements: list[tuple[int, str, str]] = []
    non_ascii_cells = 0
    for row_idx in range(num_records):
        start = header_len + row_idx * record_len
        end = start + record_len
        if end > len(data):
            break
        row = data[start:end]
        if row[:1] == b"*":
            continue
        record: dict[str, str] = {}
        replaced_columns: list[str] = []
        for source_name, canonical_name, offset, length in fields_:
            raw = row[offset : offset + length].strip(b" \0")
            if any(byte >= 0x80 for byte in raw):
                non_ascii_cells += 1
            value = raw.decode("utf-8", errors="replace") if raw else ""
            record[source_name] = value
            if _REPLACEMENT_CHAR in value:
                replaced_columns.append(canonical_name)
        feature_id = record.get("ID", "")
        for column in replaced_columns:
            replacements.append((row_idx, feature_id, column))
        records.append(record)
    return records, replacements, non_ascii_cells


def _text_field_by_lower(layer: LayerDef) -> dict[str, str]:
    return {
        column.lower(): definition.name
        for column, definition in field_defs_by_column(layer.layer_type).items()
        if definition.field_type == "text"
    }


def _looks_like_cp949_mojibake(gdf: gpd.GeoDataFrame) -> bool:
    obj_cols = gdf.select_dtypes(include=["object", "str"]).columns.drop(
        "geometry", errors="ignore"
    )
    if obj_cols.empty:
        return False
    return bool(gdf[obj_cols].stack().astype(str).str.contains(r"[｡-ﾟ]", regex=True).any())


def _normalize_columns(
    gdf: gpd.GeoDataFrame,
    shp_path: Path,
    sanity: SanityReport,
    layer: LayerDef,
    *,
    report_duplicate_columns: bool,
) -> gpd.GeoDataFrame:
    canonical_columns = {
        column.lower(): canonical
        for column, canonical in columns_by_alias(layer.layer_type).items()
    }
    canonical_columns["geometry"] = "geometry"
    by_lower: dict[str, list[str]] = {}
    for col in gdf.columns:
        by_lower.setdefault(str(col).lower(), []).append(str(col))
    rename: dict[str, str] = {}
    for lower, cols in by_lower.items():
        canon = canonical_columns.get(lower)
        if canon is None:
            continue
        if len(cols) > 1:
            if report_duplicate_columns:
                sanity.warn(
                    "dbf-column-case-duplicate",
                    f"{shp_path.name}: column {canon} has multiple capitalizations {cols}",
                    source_path=shp_path,
                )
            continue
        if cols[0] != canon:
            rename[cols[0]] = canon
    if rename:
        log.debug("normalized columns in %s: %s", shp_path.name, rename)
        return gdf.rename(columns=rename)
    return gdf


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
