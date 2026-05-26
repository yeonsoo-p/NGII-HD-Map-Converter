"""Schema-driven NGII loader engine."""

from __future__ import annotations

import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import (
    FeatureGeometryKind,
    FeatureRecord,
    NGIIFeature,
    same_feature,
)
from ngii2xodr.ngii.data.geometry import (
    point_xyz,
    polygon_outer_ring_xyz,
    polyline_xyz,
)
from ngii2xodr.ngii.data.repairs import DEFAULT_REPAIR_HOOKS, RepairHook
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning
from ngii2xodr.ngii.data.schema import FieldRule, LayerSpec, SchemaDefinition

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DiscoveredLayerFile:
    spec: LayerSpec
    path: Path
    is_filename_alias: bool


_REPLACEMENT_CHAR = "\ufffd"


def load_schema(
    root: Path,
    coordinate: str,
    cfg: NGIIConfig,
    schema: SchemaDefinition,
    *,
    repair_hooks: tuple[tuple[str, RepairHook], ...] = DEFAULT_REPAIR_HOOKS,
) -> NGIIDataset:
    """Load one NGII coordinate product into a canonical object dataset."""
    sanity = SanityReport()
    dataset = NGIIDataset(root=root, coordinate=coordinate, schema=schema, sanity=sanity)
    dataset.warn_global_id_collision = cfg.sanity.warnings.global_id_collision

    with dataset.load_profile.timed("discover", coordinate):
        discovered = discover_layer_files(
            root,
            coordinate,
            schema,
            sanity,
            warn_unknown_layers=cfg.sanity.warnings.unknown_layers,
        )

    if cfg.sanity.warnings.missing_required_layers:
        _warn_missing_required_layers(discovered, schema, sanity)
    layer_order = {spec.layer_name: index for index, spec in enumerate(schema.layer_specs)}
    for layer_file in sorted(discovered, key=lambda item: _layer_file_sort_key(item, layer_order)):
        if cfg.sanity.warnings.missing_sidecars:
            _warn_missing_sidecars(layer_file.path, sanity)
        with dataset.load_profile.timed("read_layer", layer_file.path.name):
            features = _read_layer_file(layer_file, sanity, cfg)
        with dataset.load_profile.timed("merge_layer", layer_file.path.name):
            _merge_features(
                dataset.store_for_attr(layer_file.spec.python_attr), features, sanity, cfg
            )

    with dataset.load_profile.timed("bind_initial"):
        dataset.bind()
    with dataset.load_profile.timed("text_repair"):
        _apply_text_corrections(dataset, cfg)
    if cfg.sanity.warnings.manual_field_rules or cfg.sanity.warnings.invalid_code_values:
        with dataset.load_profile.timed("manual_validation"):
            _warn_manual_field_values(dataset, cfg)
    for hook_name, hook in repair_hooks:
        with dataset.load_profile.timed(hook_name):
            hook(dataset, cfg)
    with dataset.load_profile.timed("bind_final"):
        dataset.bind()
    if cfg.sanity.warnings.unresolved_relationships:
        with dataset.load_profile.timed("relationship_validation"):
            _warn_unresolved_relationships(dataset)
    _log_sanity_report(dataset)
    _log_load_profile(dataset)
    return dataset


def _warn_missing_required_layers(
    discovered: list[DiscoveredLayerFile], schema: SchemaDefinition, sanity: SanityReport
) -> None:
    seen = {item.spec.layer_name for item in discovered}
    for spec in schema.layer_specs:
        if spec.required_layer and spec.layer_name not in seen:
            sanity.warn(
                "missing-required-layer",
                f"required NGII layer {spec.layer_name} was not found",
                layer_name=spec.layer_name,
            )


def _layer_file_sort_key(
    layer_file: DiscoveredLayerFile, layer_order: dict[str, int]
) -> tuple[int, int, str]:
    alias_order = 1 if layer_file.is_filename_alias else 0
    return (layer_order[layer_file.spec.layer_name], alias_order, str(layer_file.path))


def discover_layer_files(
    root: Path,
    coordinate: str,
    schema: SchemaDefinition,
    sanity: SanityReport,
    *,
    warn_unknown_layers: bool,
) -> list[DiscoveredLayerFile]:
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    coordinate_dirs = coordinate_dirs_for(root, coordinate)
    if not coordinate_dirs:
        msg = f"coordinate folder {coordinate!r} was not found under {root}"
        raise FileNotFoundError(msg)

    specs_by_filename = schema.specs_by_filename
    discovered: list[DiscoveredLayerFile] = []
    for coordinate_dir in coordinate_dirs:
        candidates: dict[str, list[DiscoveredLayerFile]] = defaultdict(list)
        for shp_path in sorted(coordinate_dir.rglob("*.shp")):
            spec = specs_by_filename.get(shp_path.name.upper())
            if spec is None:
                if warn_unknown_layers:
                    sanity.warn(
                        "unknown-layer",
                        f"unknown SHP layer {shp_path.name!r} in requested coordinate product",
                        source_path=shp_path,
                    )
                continue
            candidates[spec.layer_name].append(
                DiscoveredLayerFile(
                    spec=spec,
                    path=shp_path,
                    is_filename_alias=shp_path.name.upper() != spec.filename.upper(),
                )
            )
        for layer_candidates in candidates.values():
            canonical = [item for item in layer_candidates if not item.is_filename_alias]
            discovered.extend(canonical or layer_candidates)
    return discovered


def coordinate_dirs_for(root: Path, coordinate: str) -> list[Path]:
    if root.name == coordinate or any(root.glob("*.shp")):
        return [root]
    return sorted(path for path in root.rglob(coordinate) if path.is_dir())


def _warn_missing_sidecars(shp_path: Path, sanity: SanityReport) -> None:
    for suffix in (".dbf", ".shx", ".prj"):
        sidecar = shp_path.with_suffix(suffix)
        if not sidecar.is_file():
            sanity.warn(
                "missing-shp-sidecar",
                f"{shp_path.name} is missing required sidecar {sidecar.name}",
                source_path=shp_path,
            )


def _read_layer_file(
    layer_file: DiscoveredLayerFile, sanity: SanityReport, cfg: NGIIConfig
) -> list[NGIIFeature]:
    gdf = _load_gdf(layer_file.path, layer_file.spec, sanity, cfg)
    gdf = _normalize_columns(
        gdf,
        layer_file.path,
        sanity,
        layer_file.spec,
        warn_duplicate_columns=cfg.sanity.warnings.duplicate_column_capitalization,
    )
    if cfg.sanity.warnings.manual_field_rules:
        _warn_missing_columns(gdf, layer_file.spec, layer_file.path, sanity)
    if cfg.sanity.warnings.unknown_manual_columns:
        _warn_unknown_columns(gdf, layer_file.spec, layer_file.path, sanity)
    features: list[NGIIFeature] = []
    manual_columns = _manual_columns(layer_file.spec)
    for row_idx, row in enumerate(gdf.itertuples(index=False), start=0):
        row_dict = dict(zip(gdf.columns, row, strict=True))
        geometry = row_dict.pop("geometry")
        if not isinstance(geometry, shapely.geometry.base.BaseGeometry):
            if cfg.sanity.warnings.unsupported_geometry:
                sanity.warn(
                    "missing-geometry",
                    f"{layer_file.spec.layer_name} row {row_idx} has no geometry",
                    layer_name=layer_file.spec.layer_name,
                    source_path=layer_file.path,
                )
            continue
        attrs = {
            key: _clean_value(value) for key, value in row_dict.items() if key in manual_columns
        }
        row_id = str(attrs.get("ID", ""))
        try:
            geometry_kind, converted_geometry = _convert_geometry(
                layer_file.spec, geometry, row_id, cfg
            )
            feature = layer_file.spec.factory(
                FeatureRecord(
                    layer_name=layer_file.spec.layer_name,
                    source_path=layer_file.path,
                    source_row=row_idx,
                    attributes=attrs,
                    geometry_kind=geometry_kind,
                    geometry=converted_geometry,
                )
            )
        except (TypeError, ValueError) as e:
            if cfg.sanity.warnings.unsupported_geometry:
                sanity.warn(
                    "parse-row-failed",
                    str(e),
                    layer_name=layer_file.spec.layer_name,
                    feature_id=row_id,
                    source_path=layer_file.path,
                )
            continue
        features.append(feature)
    return features


def _load_gdf(
    shp_path: Path, spec: LayerSpec, sanity: SanityReport, cfg: NGIIConfig
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
            _overlay_utf8_dbf_text(
                gdf,
                shp_path,
                spec,
                sanity,
                cfg.encoding.utf8_dbf_invalid_non_ascii_ratio_max,
            )
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
    spec: LayerSpec,
    sanity: SanityReport,
    invalid_non_ascii_ratio_max: float,
) -> None:
    records, replacements, non_ascii_cells = _read_utf8_dbf_text_records(
        shp_path.with_suffix(".dbf"), spec
    )
    if not records:
        return
    if non_ascii_cells > 0 and len(replacements) / non_ascii_cells > invalid_non_ascii_ratio_max:
        log.debug(
            "skipping UTF-8 DBF text overlay for %s: %d/%d non-ASCII text cells "
            "decode with replacement characters",
            shp_path.name,
            len(replacements),
            non_ascii_cells,
        )
        return
    if len(records) != len(gdf):
        sanity.warn(
            "dbf-text-row-count-mismatch",
            f"{spec.layer_name}: raw DBF text row count {len(records)} does not match "
            f"geometry row count {len(gdf)}; UTF-8 text overlay skipped",
            layer_name=spec.layer_name,
            source_path=shp_path,
        )
        return
    columns = sorted({column for record in records for column in record})
    for column in columns:
        if column in gdf.columns:
            gdf[column] = [record.get(column, "") for record in records]
    for row_idx, feature_id, column in replacements:
        sanity.warn(
            "corrupt-text-decode-replaced",
            f"{spec.layer_name} {feature_id or f'row {row_idx}'}: {column} contained "
            "malformed UTF-8 bytes and was decoded with replacement characters",
            layer_name=spec.layer_name,
            feature_id=feature_id or None,
            source_path=shp_path,
        )


def _read_utf8_dbf_text_records(
    dbf_path: Path, spec: LayerSpec
) -> tuple[list[dict[str, str]], list[tuple[int, str, str]], int]:
    if not dbf_path.is_file():
        return [], [], 0
    data = dbf_path.read_bytes()
    if len(data) < 32:
        return [], [], 0
    num_records = struct.unpack_from("<I", data, 4)[0]
    header_len = struct.unpack_from("<H", data, 8)[0]
    record_len = struct.unpack_from("<H", data, 10)[0]
    text_field_by_lower = _text_field_by_lower(spec)
    fields: list[tuple[str, str, int, int]] = []
    row_offset = 1
    pos = 32
    while pos + 32 <= len(data) and data[pos] != 0x0D:
        desc = data[pos : pos + 32]
        name = desc[:11].split(b"\0", maxsplit=1)[0].decode("ascii", errors="replace")
        field_type = chr(desc[11])
        length = int(desc[16])
        canonical_name = text_field_by_lower.get(name.lower())
        if field_type == "C" and canonical_name is not None:
            fields.append((name, canonical_name, row_offset, length))
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
        for source_name, canonical_name, offset, length in fields:
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


def _text_field_by_lower(spec: LayerSpec) -> dict[str, str]:
    result: dict[str, str] = {}
    for rule in spec.field_rules:
        if rule.field_type != "text":
            continue
        for column in rule.columns:
            result[column.lower()] = rule.name
    return result


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
    spec: LayerSpec,
    *,
    warn_duplicate_columns: bool,
) -> gpd.GeoDataFrame:
    canonical_columns = {}
    for rule in spec.field_rules:
        for column in rule.columns:
            canonical_columns[column.lower()] = rule.name
    for rel in spec.relationships:
        canonical_columns[rel.column_name.lower()] = rel.column_name
    canonical_columns["geometry"] = "geometry"
    by_lower: dict[str, list[str]] = {}
    for col in gdf.columns:
        by_lower.setdefault(col.lower(), []).append(col)
    rename: dict[str, str] = {}
    for lower, cols in by_lower.items():
        canon = canonical_columns.get(lower)
        if canon is None:
            continue
        if len(cols) > 1:
            if warn_duplicate_columns:
                sanity.warn(
                    "duplicate-column-capitalization",
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


def _warn_missing_columns(
    gdf: gpd.GeoDataFrame, spec: LayerSpec, shp_path: Path, sanity: SanityReport
) -> None:
    missing = sorted(rule.name for rule in spec.field_rules if rule.name not in gdf.columns)
    if missing:
        sanity.warn(
            "missing-manual-columns",
            f"{spec.layer_name} is missing manual fields: {', '.join(missing)}",
            layer_name=spec.layer_name,
            source_path=shp_path,
        )


def _warn_unknown_columns(
    gdf: gpd.GeoDataFrame, spec: LayerSpec, shp_path: Path, sanity: SanityReport
) -> None:
    allowed = _manual_columns(spec) | {"geometry"}
    unknown = sorted(str(column) for column in gdf.columns if str(column) not in allowed)
    if not unknown:
        return
    sanity.warn(
        "unknown-manual-column",
        f"{spec.layer_name} has undocumented DBF columns that were omitted: {', '.join(unknown)}",
        layer_name=spec.layer_name,
        source_path=shp_path,
    )


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _value_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _is_integer(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer()


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_valid_hist_type(spec: LayerSpec, value: str) -> bool:
    rule = next((item for item in spec.field_rules if item.name == "HistType"), None)
    if rule is not None and rule.max_length == 3:
        allowed = {f"{code:03d}" for code in range(1, 9)}
        return value in allowed
    layer_prefix = spec.layer_name.split("_", maxsplit=1)[0]
    allowed = {f"{layer_prefix}{code:03d}" for code in range(1, 9)}
    return value in allowed


def _convert_geometry(
    spec: LayerSpec,
    geometry: shapely.geometry.base.BaseGeometry,
    row_id: str,
    cfg: NGIIConfig,
) -> tuple[FeatureGeometryKind, np.ndarray]:
    geometry_kind = _geometry_kind_for(geometry, row_id)
    if geometry_kind not in spec.geometry_kinds:
        expected = ", ".join(spec.geometry_kinds)
        msg = (
            f"row ID={row_id!r}: {spec.layer_name} does not accept "
            f"{type(geometry).__name__}; expected {expected}"
        )
        raise TypeError(msg)
    if geometry_kind == "point":
        return geometry_kind, point_xyz(geometry, row_id=row_id)
    if geometry_kind == "line":
        return geometry_kind, polyline_xyz(
            geometry,
            row_id=row_id,
            multipart_snap_tolerance_m=cfg.geometry.multipart_snap_tolerance_m,
        )
    return geometry_kind, polygon_outer_ring_xyz(geometry, row_id=row_id)


def _geometry_kind_for(
    geometry: shapely.geometry.base.BaseGeometry, row_id: str
) -> FeatureGeometryKind:
    if isinstance(geometry, shapely.Point):
        return "point"
    if isinstance(geometry, (shapely.LineString, shapely.MultiLineString)):
        return "line"
    if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):
        return "polygon"
    msg = f"row ID={row_id!r}: unsupported geometry {type(geometry).__name__}"
    raise TypeError(msg)


def _manual_columns(spec: LayerSpec) -> set[str]:
    manual_columns = {rule.name for rule in spec.field_rules}
    relationship_columns = {relationship.column_name for relationship in spec.relationships}
    return manual_columns | relationship_columns


def _merge_features(
    store: LayerStore[Any],
    features: list[NGIIFeature],
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    for feature in features:
        if not feature.id:
            if cfg.sanity.warnings.manual_field_rules:
                sanity.warn(
                    "missing-feature-id",
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
            if cfg.sanity.warnings.duplicate_identical_ids:
                sanity.warn(
                    "duplicate-identical-id",
                    f"{feature.layer_name} ID {feature.id!r} appears more than once "
                    "with identical data",
                    layer_name=feature.layer_name,
                    feature_id=feature.id,
                    source_path=feature.source_path,
                )
            continue
        before = {
            "kept_source": str(existing.source_path),
            "dropped_source": str(feature.source_path),
        }
        after = {"canonical_source": str(existing.source_path)}
        if cfg.sanity.repairs.duplicate_conflicting_id_drop:
            sanity.action(
                "duplicate-conflicting-id-dropped",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row dropped",
                before=before,
                after=after,
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )
        elif cfg.sanity.warnings.duplicate_conflicting_ids:
            sanity.warn(
                "duplicate-conflicting-id-drop-disabled",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row could not become canonical because duplicate repair is disabled",
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )


def _apply_text_corrections(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    if not cfg.text_repair.enabled:
        return
    if cfg.text_repair.repair_mojibake:
        _repair_mojibake_text(dataset)
    if cfg.text_repair.warn_unrepaired_replacement_chars:
        _warn_unrepaired_replacement_chars(dataset)


def _repair_mojibake_text(dataset: NGIIDataset) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in _iter_feature_text_fields(feature):
                repaired = _repair_cp949_latin1_mojibake(value)
                if repaired is None:
                    continue
                _set_feature_text_field(feature, field_name, repaired)
                dataset.sanity.action(
                    "text-mojibake-repaired",
                    f"{feature.layer_name} {feature.id} {field_name} repaired by "
                    "latin1-to-cp949 mojibake rule",
                    before={field_name: value},
                    after={field_name: repaired},
                    layer_name=feature.layer_name,
                    feature_id=feature.id,
                    source_path=feature.source_path,
                )


def _repair_cp949_latin1_mojibake(value: str) -> str | None:
    if not value or _REPLACEMENT_CHAR in value:
        return None
    try:
        candidate = value.encode("latin1").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if candidate == value or _REPLACEMENT_CHAR in candidate:
        return None
    original_hangul = _hangul_count(value)
    candidate_hangul = _hangul_count(candidate)
    if candidate_hangul <= original_hangul:
        return None
    if _mojibake_marker_count(value) == 0 and candidate_hangul < 2:
        return None
    return candidate


def _hangul_count(value: str) -> int:
    return sum(1 for char in value if "\uac00" <= char <= "\ud7a3")


def _mojibake_marker_count(value: str) -> int:
    return sum(1 for char in value if "\u00a1" <= char <= "\u00ff" or "\uff61" <= char <= "\uff9f")


def _warn_unrepaired_replacement_chars(dataset: NGIIDataset) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in _iter_feature_text_fields(feature):
                if _REPLACEMENT_CHAR in value:
                    dataset.sanity.warn(
                        "corrupt-text-unrepaired",
                        f"{feature.layer_name} {feature.id}: {field_name} contains "
                        "Unicode replacement characters",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def _iter_feature_text_fields(feature: NGIIFeature) -> list[tuple[str, str]]:
    dataset = feature._require_dataset()
    spec = dataset.schema.spec_for_layer_name(feature.layer_name)
    if spec is None:
        return []
    values: list[tuple[str, str]] = []
    for rule in spec.field_rules:
        if rule.name == "ID" or rule.field_type != "text":
            continue
        attr_name = rule.attr
        if not hasattr(feature, attr_name):
            continue
        value = getattr(feature, attr_name)
        if isinstance(value, str):
            values.append((rule.name, value))
    return values


def _set_feature_text_field(feature: NGIIFeature, field_name: str, value: str) -> None:
    _set_feature_column(feature, field_name, value)


def _feature_value_for_column(feature: NGIIFeature, column_name: str) -> Any:
    attr_name = _attr_for_feature_column(feature, column_name)
    if attr_name and hasattr(feature, attr_name):
        return getattr(feature, attr_name)
    return ""


def _set_feature_column(feature: NGIIFeature, column_name: str, value: Any) -> None:
    attr_name = _attr_for_feature_column(feature, column_name)
    if attr_name and hasattr(feature, attr_name):
        setattr(feature, attr_name, value)


def _attr_for_feature_column(feature: NGIIFeature, column_name: str) -> str:
    dataset = feature._require_dataset()
    spec = dataset.schema.spec_for_layer_name(feature.layer_name)
    if spec is None:
        return ""
    for rule in spec.field_rules:
        if rule.name == column_name:
            return rule.attr
    for relationship in spec.relationships:
        if relationship.column_name == column_name:
            return relationship.source_attr
    return ""


def _warn_manual_field_values(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    for spec in dataset.schema.layer_specs:
        store = dataset.store_for_attr(spec.python_attr)
        for feature in store.features:
            for rule in spec.field_rules:
                _warn_manual_field_value(dataset, spec, feature, rule, cfg)


def _warn_manual_field_value(
    dataset: NGIIDataset,
    spec: LayerSpec,
    feature: NGIIFeature,
    rule: FieldRule,
    cfg: NGIIConfig,
) -> None:
    value = _feature_value_for_column(feature, rule.name)
    value_text = _value_text(value)
    if not value_text:
        if rule.required and cfg.sanity.warnings.manual_field_rules:
            dataset.sanity.warn(
                "empty-required-field",
                f"{spec.layer_name} {feature.id}: {rule.name} is required by the "
                f"{dataset.schema.version} manual",
                layer_name=spec.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )
        return
    if (
        rule.max_length is not None
        and len(value_text) > rule.max_length
        and cfg.sanity.warnings.manual_field_rules
    ):
        dataset.sanity.warn(
            "field-too-long",
            f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} exceeds "
            f"VARCHAR2({rule.max_length})",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    invalid_integer = rule.field_type == "integer" and not _is_integer(value)
    invalid_float = rule.field_type == "float" and not _is_float(value)
    if (invalid_integer or invalid_float) and cfg.sanity.warnings.manual_field_rules:
        _warn_invalid_type(dataset, spec, feature, rule, value_text)
    if (
        rule.code_list is not None
        and value_text not in rule.code_list
        and cfg.sanity.warnings.invalid_code_values
    ):
        dataset.sanity.warn(
            "invalid-code",
            f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} is not "
            f"in the {dataset.schema.version} code list",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    if (
        rule.name == "HistType"
        and not _is_valid_hist_type(spec, value_text)
        and cfg.sanity.warnings.manual_field_rules
    ):
        expected = _expected_hist_type_label(spec, rule)
        dataset.sanity.warn(
            "invalid-hist-type",
            f"{spec.layer_name} {feature.id}: HistType={value_text!r} must be {expected}",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )


def _expected_hist_type_label(spec: LayerSpec, rule: FieldRule) -> str:
    if rule.max_length == 3:
        return "001-008"
    layer_prefix = spec.layer_name.split("_", maxsplit=1)[0]
    return f"{layer_prefix}001-{layer_prefix}008"


def _warn_invalid_type(
    dataset: NGIIDataset,
    spec: LayerSpec,
    feature: NGIIFeature,
    rule: FieldRule,
    value_text: str,
) -> None:
    dataset.sanity.warn(
        "invalid-field-type",
        f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} is not "
        f"a valid {rule.field_type}",
        layer_name=spec.layer_name,
        feature_id=feature.id,
        source_path=feature.source_path,
    )


def _warn_unresolved_relationships(dataset: NGIIDataset) -> None:
    for spec in dataset.schema.layer_specs:
        store = dataset.store_for_attr(spec.python_attr)
        for feature in store.features:
            for relationship in spec.relationships:
                value = getattr(feature, relationship.source_attr)
                value_text = _value_text("" if value is None else value)
                target_layer = _relationship_target_label(dataset, relationship.target_attrs)
                if not value_text:
                    if relationship.required:
                        _warn_missing_relation(
                            dataset,
                            feature,
                            relationship.column_name,
                            target_layer,
                        )
                    continue
                if not _relationship_resolves(dataset, relationship.target_attrs, value_text):
                    _warn_missing_relation(
                        dataset,
                        feature,
                        relationship.column_name,
                        target_layer,
                    )


def _relationship_resolves(
    dataset: NGIIDataset, target_attrs: tuple[str, ...], feature_id: str
) -> bool:
    return any(
        dataset.store_for_attr(target_attr).get(feature_id) is not None
        for target_attr in target_attrs
    )


def _relationship_target_label(dataset: NGIIDataset, target_attrs: tuple[str, ...]) -> str:
    return "/".join(dataset.store_for_attr(target_attr).layer_name for target_attr in target_attrs)


def _warn_missing_relation(
    dataset: NGIIDataset, feature: NGIIFeature, column_name: str, target_layer: str
) -> None:
    dataset.sanity.warn(
        "missing-relation",
        f"{feature.layer_name} {feature.id} {column_name} does not resolve to {target_layer}",
        layer_name=feature.layer_name,
        feature_id=feature.id,
        source_path=feature.source_path,
    )


def _log_sanity_report(dataset: NGIIDataset) -> None:
    warning_count = len(dataset.sanity.warnings)
    action_count = len(dataset.sanity.actions)
    if warning_count == 0 and action_count == 0:
        log.info(
            "NGII sanity: no warnings or repairs for %s [%s]",
            dataset.root,
            dataset.coordinate,
        )
        return

    log.warning(
        "NGII sanity: %d warning(s), %d repair action(s) for %s [%s]",
        warning_count,
        action_count,
        dataset.root,
        dataset.coordinate,
    )
    _log_warning_summaries(dataset)
    _log_action_summaries(dataset)
    for warning in dataset.sanity.warnings:
        log.debug(
            "NGII sanity warning detail [%s] %s: %s",
            warning.code,
            _sanity_location(warning.layer_name, warning.feature_id, warning.source_path),
            warning.message,
        )
    for action in dataset.sanity.actions:
        log.debug(
            "NGII sanity repair detail [%s] %s: %s before=%s after=%s",
            action.code,
            _sanity_location(action.layer_name, action.feature_id, action.source_path),
            action.message,
            action.before,
            action.after,
        )


def _log_load_profile(dataset: NGIIDataset) -> None:
    if not dataset.load_profile.events:
        return
    log.info(
        "NGII load profile: total %.3fs for %s [%s]",
        dataset.load_profile.total_s,
        dataset.root,
        dataset.coordinate,
    )
    grouped: dict[tuple[str, str], float] = defaultdict(float)
    for event in dataset.load_profile.events:
        grouped[(event.name, event.detail)] += event.duration_s
    for (name, detail), duration_s in sorted(
        grouped.items(), key=lambda item: item[1], reverse=True
    ):
        suffix = f" {detail}" if detail else ""
        log.info("NGII load profile: %.3fs %s%s", duration_s, name, suffix)


def _log_warning_summaries(dataset: NGIIDataset) -> None:
    grouped: dict[tuple[str, str, str, str], list[SanityWarning]] = defaultdict(list)
    for warning in dataset.sanity.warnings:
        key = (
            warning.code,
            warning.layer_name or "-",
            _short_path(dataset.root, warning.source_path),
            _warning_summary_reason(warning),
        )
        grouped[key].append(warning)

    for (code, layer_name, source_path, reason), events in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        log.warning(
            "NGII sanity warning summary [%s] %s %s count=%d examples=%s reason=%s",
            code,
            layer_name,
            source_path,
            len(events),
            _example_ids(events),
            reason,
        )


def _log_action_summaries(dataset: NGIIDataset) -> None:
    grouped: dict[tuple[str, str, str], list[SanityAction]] = defaultdict(list)
    for action in dataset.sanity.actions:
        key = (action.code, action.layer_name or "-", _action_summary_reason(dataset, action))
        grouped[key].append(action)

    for (code, layer_name, reason), events in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        log.warning(
            "NGII sanity repair summary [%s] %s count=%d examples=%s reason=%s",
            code,
            layer_name,
            len(events),
            _example_ids(events),
            reason,
        )


def _warning_summary_reason(warning: SanityWarning) -> str:
    if warning.code == "parse-row-failed":
        _row_label, sep, reason = warning.message.partition(": ")
        if sep:
            return reason
    return _strip_feature_prefix(warning.message, warning.layer_name, warning.feature_id)


def _action_summary_reason(dataset: NGIIDataset, action: SanityAction) -> str:
    if action.code == "duplicate-conflicting-id-dropped":
        kept_source = _mapping_path(dataset.root, action.before, "kept_source")
        dropped_source = _mapping_path(dataset.root, action.before, "dropped_source")
        return f"duplicate ID conflict; kept={kept_source}; dropped={dropped_source}"
    return _strip_feature_prefix(action.message, action.layer_name, action.feature_id)


def _strip_feature_prefix(message: str, layer_name: str | None, feature_id: str | None) -> str:
    if layer_name is None or feature_id is None:
        return message
    colon_prefix = f"{layer_name} {feature_id}: "
    if message.startswith(colon_prefix):
        return message.removeprefix(colon_prefix)
    prefix = f"{layer_name} {feature_id} "
    return message.removeprefix(prefix)


def _short_path(root: Path, source_path: Path | None) -> str:
    if source_path is None:
        return "-"
    try:
        return str(source_path.relative_to(root))
    except ValueError:
        return str(source_path)


def _mapping_path(root: Path, values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        return "-"
    return _short_path(root, Path(value))


def _example_ids(events: list[SanityWarning] | list[SanityAction]) -> str:
    ids = [event.feature_id for event in events if event.feature_id]
    return ", ".join(ids[:8]) if ids else "-"


def _sanity_location(
    layer_name: str | None, feature_id: str | None, source_path: Path | None
) -> str:
    parts: list[str] = []
    if layer_name:
        parts.append(layer_name)
    if feature_id:
        parts.append(feature_id)
    if source_path is not None:
        parts.append(source_path.name)
    return " ".join(parts) if parts else "-"
