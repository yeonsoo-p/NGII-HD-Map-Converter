"""Public NGII loader and 2023 sanity-action pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ngii2xodr.ngii.data.config import NGIIConfig, NGIITextCorrection
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
    xy_distance,
)
from ngii2xodr.ngii.data.sanity import SanityAction, SanityReport, SanityWarning
from ngii2xodr.ngii.data.v2023.definitions import LAYER_SPECS, FieldRule, LayerSpec
from ngii2xodr.ngii.data.v2023.discovery import DiscoveredLayerFile, discover_layer_files
from ngii2xodr.ngii.data.v2023.layers.a1_node import A1_NODE
from ngii2xodr.ngii.data.v2023.layers.a2_link import A2_LINK

log = logging.getLogger(__name__)

_CANONICAL_COLUMNS: dict[str, str] = {
    "id": "ID",
    "admincode": "AdminCode",
    "nodetype": "NodeType",
    "itsnodeid": "ITSNodeID",
    "roadrank": "RoadRank",
    "roadtype": "RoadType",
    "roadno": "RoadNo",
    "linktype": "LinkType",
    "laneno": "LaneNo",
    "r_linkid": "R_LinkID",
    "l_linkid": "L_LinkID",
    "fromnodeid": "FromNodeID",
    "tonodeid": "ToNodeID",
    "sectionid": "SectionID",
    "length": "Length",
    "itslinkid": "ITSLinkID",
    "kind": "Kind",
    "subtype": "SubType",
    "name": "Name",
    "direction": "Direction",
    "gasstation": "GasStation",
    "lpgstation": "LpgStation",
    "evcharger": "EvCharger",
    "toilet": "Toilet",
    "type": "Type",
    "linkid": "LinkID",
    "ref_lane": "Ref_Lane",
    "postid": "PostID",
    "distance": "Distance",
    "origin": "Origin",
    "iscentral": "IsCentral",
    "lowhigh": "LowHigh",
    "ref_id": "Ref_ID",
    "maker": "Maker",
    "updatedate": "UpdateDate",
    "version": "Version",
    "remark": "Remark",
    "histtype": "HistType",
    "histremark": "HistRemark",
}

_FIELD_ATTRS: dict[str, str] = {
    "ID": "id",
    "AdminCode": "admin_code",
    "Maker": "maker",
    "UpdateDate": "update_date",
    "Version": "version",
    "Remark": "remark",
    "HistType": "hist_type",
    "HistRemark": "hist_remark",
    "NodeType": "node_type",
    "ITSNodeID": "its_node_id",
    "RoadRank": "road_rank",
    "RoadType": "road_type",
    "RoadNo": "road_no",
    "LinkType": "link_type",
    "LaneNo": "lane_no",
    "R_LinkID": "r_link_id",
    "L_LinkID": "l_link_id",
    "FromNodeID": "from_node_id",
    "ToNodeID": "to_node_id",
    "SectionID": "section_id",
    "Length": "length_m",
    "ITSLinkID": "its_link_id",
    "Kind": "kind",
    "SubType": "subtype",
    "Name": "name",
    "Direction": "direction",
    "GasStation": "gas_station",
    "LpgStation": "lpg_station",
    "EvCharger": "ev_charger",
    "Toilet": "toilet",
    "Type": "type",
    "LinkID": "link_id",
    "Ref_Lane": "ref_lane",
    "PostID": "post_id",
    "Distance": "distance",
    "Origin": "origin",
    "IsCentral": "is_central",
    "LowHigh": "low_high",
    "Ref_ID": "ref_id",
}
_REPLACEMENT_CHAR = "\ufffd"
_LAYER_ORDER = {spec.layer_name: index for index, spec in enumerate(LAYER_SPECS)}


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIIDataset:
    """Load one NGII coordinate product into a canonical object dataset."""
    sanity = SanityReport()
    dataset = NGIIDataset(root=root, coordinate=coordinate, sanity=sanity)
    dataset.warn_global_id_collision = cfg.sanity.warnings.global_id_collision

    with dataset.load_profile.timed("discover", coordinate):
        discovered = discover_layer_files(
            root,
            coordinate,
            sanity,
            warn_unknown_layers=cfg.sanity.warnings.unknown_layers,
        )

    if cfg.sanity.warnings.missing_required_layers:
        _warn_missing_required_layers(discovered, sanity)
    for layer_file in sorted(discovered, key=_layer_file_sort_key):
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
    with dataset.load_profile.timed("a2_endpoint_direction"):
        _repair_reversed_a2_links(dataset, cfg)
    with dataset.load_profile.timed("a2_missing_node_refs"):
        _repair_missing_a2_node_refs(dataset, cfg)
    with dataset.load_profile.timed("a2_topology_direction"):
        _repair_topology_direction(dataset, cfg)
    with dataset.load_profile.timed("bind_final"):
        dataset.bind()
    if cfg.sanity.warnings.unresolved_relationships:
        with dataset.load_profile.timed("relationship_validation"):
            _warn_unresolved_relationships(dataset)
    _log_sanity_report(dataset)
    _log_load_profile(dataset)
    return dataset


def _warn_missing_required_layers(
    discovered: list[DiscoveredLayerFile], sanity: SanityReport
) -> None:
    seen = {item.spec.layer_name for item in discovered}
    for spec in LAYER_SPECS:
        if spec.required_layer and spec.layer_name not in seen:
            sanity.warn(
                "missing-required-layer",
                f"required 2023 NGII layer {spec.layer_name} was not found",
                layer_name=spec.layer_name,
            )


def _layer_file_sort_key(layer_file: DiscoveredLayerFile) -> tuple[int, str]:
    return (_LAYER_ORDER[layer_file.spec.layer_name], str(layer_file.path))


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
    gdf = _load_gdf(layer_file.path)
    gdf = _normalize_columns(
        gdf,
        layer_file.path,
        sanity,
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
            geometry_kind, converted_geometry = _convert_geometry(layer_file.spec, geometry, row_id)
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


def _load_gdf(shp_path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(shp_path)
        retry_reason = "mojibake heuristic" if _looks_like_cp949_mojibake(gdf) else None
    except UnicodeDecodeError:
        retry_reason = "UTF-8 decode failure"
    if retry_reason is not None:
        log.debug("retrying %s with cp949 after %s", shp_path.name, retry_reason)
        gdf = gpd.read_file(shp_path, encoding="cp949")
    return gdf


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
    *,
    warn_duplicate_columns: bool,
) -> gpd.GeoDataFrame:
    by_lower: dict[str, list[str]] = {}
    for col in gdf.columns:
        by_lower.setdefault(col.lower(), []).append(col)
    rename: dict[str, str] = {}
    for lower, cols in by_lower.items():
        canon = _CANONICAL_COLUMNS.get(lower)
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


def _is_valid_hist_type(layer_name: str, value: str) -> bool:
    layer_prefix = layer_name.split("_", maxsplit=1)[0]
    allowed = {f"{layer_prefix}{code:03d}" for code in range(1, 9)}
    return value in allowed


def _convert_geometry(
    spec: LayerSpec, geometry: shapely.geometry.base.BaseGeometry, row_id: str
) -> tuple[FeatureGeometryKind, np.ndarray]:
    if spec.layer_name == "B1_SAFETYSIGN" and isinstance(geometry, shapely.Point):
        return "point", point_xyz(geometry, row_id=row_id)
    if spec.geometry_kind == "point":
        return "point", point_xyz(geometry, row_id=row_id)
    if spec.geometry_kind == "line":
        return "line", polyline_xyz(geometry, row_id=row_id)
    return "polygon", polygon_outer_ring_xyz(geometry, row_id=row_id)


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
    if cfg.text_repair.apply_exact_corrections:
        for correction in cfg.text_repair.corrections:
            _apply_text_correction(dataset, correction)
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


def _apply_text_correction(dataset: NGIIDataset, correction: NGIITextCorrection) -> None:
    feature = _feature_for_layer_name(dataset, correction.layer_name, correction.feature_id)
    if feature is None:
        return
    current = _value_text(_feature_value_for_column(feature, correction.field))
    if current != correction.old:
        return
    _set_feature_column(feature, correction.field, correction.new)
    dataset.sanity.action(
        "known-text-correction",
        f"{feature.layer_name} {feature.id} {correction.field} repaired from exact text map",
        before={correction.field: correction.old},
        after={correction.field: correction.new},
        layer_name=feature.layer_name,
        feature_id=feature.id,
        source_path=feature.source_path,
    )


def _feature_for_layer_name(
    dataset: NGIIDataset, layer_name: str, feature_id: str
) -> NGIIFeature | None:
    for spec in LAYER_SPECS:
        if spec.layer_name == layer_name:
            return dataset.store_for_attr(spec.python_attr).get(feature_id)
    return None


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
    values: list[tuple[str, str]] = []
    for column_name, attr_name in _FIELD_ATTRS.items():
        if column_name == "ID":
            continue
        if not hasattr(feature, attr_name):
            continue
        value = getattr(feature, attr_name)
        if isinstance(value, str):
            values.append((column_name, value))
    return values


def _set_feature_text_field(feature: NGIIFeature, field_name: str, value: str) -> None:
    _set_feature_column(feature, field_name, value)


def _feature_value_for_column(feature: NGIIFeature, column_name: str) -> Any:
    attr_name = _FIELD_ATTRS.get(column_name)
    if attr_name is not None and hasattr(feature, attr_name):
        return getattr(feature, attr_name)
    return ""


def _set_feature_column(feature: NGIIFeature, column_name: str, value: Any) -> None:
    attr_name = _FIELD_ATTRS.get(column_name)
    if attr_name is not None and hasattr(feature, attr_name):
        setattr(feature, attr_name, value)


def _warn_manual_field_values(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    for spec in LAYER_SPECS:
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
                f"{spec.layer_name} {feature.id}: {rule.name} is required by the 2023 manual",
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
            "in the 2023 code list",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    if (
        rule.name == "HistType"
        and not _is_valid_hist_type(spec.layer_name, value_text)
        and cfg.sanity.warnings.manual_field_rules
    ):
        dataset.sanity.warn(
            "invalid-hist-type",
            f"{spec.layer_name} {feature.id}: HistType={value_text!r} must be "
            f"{spec.layer_name.split('_', maxsplit=1)[0]}001-"
            f"{spec.layer_name.split('_', maxsplit=1)[0]}008",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )


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


def _repair_reversed_a2_links(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in dataset.a2_link.features:
        from_node = dataset.a1_node.get(link.from_node_id)
        to_node = dataset.a1_node.get(link.to_node_id)
        if from_node is None or to_node is None or len(link.polyline) < 2:
            continue
        start = link.polyline[0]
        end = link.polyline[-1]
        normal = (
            xy_distance(start, from_node.point) <= tolerance_m
            and xy_distance(end, to_node.point) <= tolerance_m
        )
        reversed_alignment = (
            xy_distance(start, to_node.point) <= tolerance_m
            and xy_distance(end, from_node.point) <= tolerance_m
        )
        if reversed_alignment and not normal:
            _swap_a2_endpoint_ids(
                dataset,
                link,
                "a2-direction-swapped",
                "had reversed FromNodeID/ToNodeID relative to geometry order",
                enabled=cfg.sanity.repairs.a2_endpoint_direction_swap,
                warn_disabled=cfg.sanity.warnings.a2_endpoint_alignment,
            )
        elif reversed_alignment and normal and cfg.sanity.warnings.a2_direction_ambiguous:
            dataset.sanity.warn(
                "a2-direction-ambiguous",
                f"A2_LINK {link.id} endpoints match both normal and reversed node ordering",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif not normal and cfg.sanity.warnings.a2_endpoint_alignment:
            dataset.sanity.warn(
                "a2-endpoint-alignment-mismatch",
                f"A2_LINK {link.id} endpoint nodes do not match polyline endpoints within "
                f"{tolerance_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )


def _repair_missing_a2_node_refs(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in dataset.a2_link.features:
        if len(link.polyline) < 2:
            continue
        _repair_link_endpoint(
            dataset, link, "from_node_id", "FromNodeID", link.polyline[0], cfg, tolerance_m
        )
        _repair_link_endpoint(
            dataset, link, "to_node_id", "ToNodeID", link.polyline[-1], cfg, tolerance_m
        )


def _repair_link_endpoint(
    dataset: NGIIDataset,
    link: A2_LINK,
    attr_name: str,
    column_name: str,
    endpoint_xyz: np.ndarray,
    cfg: NGIIConfig,
    tolerance_m: float,
) -> None:
    current_id = getattr(link, attr_name)
    if current_id is not None and dataset.a1_node.get(current_id) is not None:
        return
    nearby = _nearby_nodes(dataset.a1_node.features, endpoint_xyz, tolerance_m)
    before = {attr_name: current_id}
    if len(nearby) == 1:
        repaired_id = nearby[0].id
        if cfg.sanity.repairs.a2_missing_node_ref_nearest:
            setattr(link, attr_name, repaired_id)
            dataset.sanity.action(
                "a2-node-ref-nearest",
                f"A2_LINK {link.id} {column_name} repaired to nearby A1_NODE {repaired_id}",
                before=before,
                after={attr_name: repaired_id},
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif cfg.sanity.warnings.a2_endpoint_alignment:
            dataset.sanity.warn(
                "a2-node-ref-nearest-disabled",
                f"A2_LINK {link.id} {column_name} could be repaired to nearby "
                f"A1_NODE {repaired_id}, but nearest-node repair is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if len(nearby) > 1 and cfg.sanity.warnings.a2_endpoint_alignment:
        dataset.sanity.warn(
            "a2-node-ref-ambiguous",
            f"A2_LINK {link.id} {column_name} has {len(nearby)} nearby A1_NODE candidates",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    if cfg.sanity.repairs.a2_missing_node_ref_remove:
        setattr(link, attr_name, None)
        dataset.sanity.action(
            "a2-node-ref-removed",
            f"A2_LINK {link.id} {column_name} could not be resolved and was removed",
            before=before,
            after={attr_name: None},
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.a2_endpoint_alignment:
        dataset.sanity.warn(
            "a2-node-ref-remove-disabled",
            f"A2_LINK {link.id} {column_name} could not be resolved, but relation "
            "removal is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _nearby_nodes(
    nodes: list[A1_NODE], endpoint_xyz: np.ndarray, tolerance_m: float
) -> list[A1_NODE]:
    return [node for node in nodes if xy_distance(endpoint_xyz, node.point) <= tolerance_m]


def _repair_topology_direction(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    desired_flip = _desired_a2_flips_from_leaf_flow(dataset)
    candidates: list[A2_LINK] = []
    for link in dataset.a2_link.features:
        if not desired_flip.get(link.id, False):
            continue
        if _has_opposing_same_direction_neighbour(
            dataset, link, cfg.sanity.direction_parallel_dot_min
        ):
            candidates.append(link)

    if not candidates:
        return
    if len(candidates) > 1:
        if cfg.sanity.warnings.a2_topology_direction:
            dataset.sanity.warn(
                "a2-topology-direction-ambiguous",
                "topology flow and R/L same-direction evidence found multiple possible "
                f"backward A2 links: {', '.join(link.id for link in candidates[:8])}",
                layer_name="A2_LINK",
            )
        return
    if cfg.sanity.repairs.a2_topology_direction_swap:
        link = candidates[0]
        _swap_a2_endpoint_ids(
            dataset,
            link,
            "a2-topology-direction-swapped",
            "was reversed by topology flow and R/L same-direction evidence",
            enabled=True,
            warn_disabled=False,
        )
        return
    if cfg.sanity.warnings.a2_topology_direction:
        link = candidates[0]
        dataset.sanity.warn(
            "a2-topology-direction-swap-disabled",
            f"A2_LINK {link.id} appears reversed by topology flow and R/L "
            "same-direction evidence, but topology repair is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _desired_a2_flips_from_leaf_flow(dataset: NGIIDataset) -> dict[str, bool]:
    by_node = _a2_links_by_node(dataset)
    votes: dict[str, set[bool]] = defaultdict(set)
    queue = _leaf_flow_queue(by_node)

    seen: set[tuple[str, bool]] = set()
    while queue:
        link, flip = queue.pop(0)
        state = (link.id, flip)
        if state in seen:
            continue
        seen.add(state)
        votes[link.id].add(flip)
        _queue_downstream_links(queue, by_node, link, flip)
    return {
        link_id: next(iter(link_votes))
        for link_id, link_votes in votes.items()
        if len(link_votes) == 1
    }


def _a2_links_by_node(dataset: NGIIDataset) -> dict[str, list[A2_LINK]]:
    by_node: dict[str, list[A2_LINK]] = defaultdict(list)
    for link in dataset.a2_link.features:
        if link.from_node_id:
            by_node[link.from_node_id].append(link)
        if link.to_node_id:
            by_node[link.to_node_id].append(link)
    return by_node


def _leaf_flow_queue(by_node: dict[str, list[A2_LINK]]) -> list[tuple[A2_LINK, bool]]:
    leaves = {node_id for node_id, links in by_node.items() if len(links) == 1}
    queue: list[tuple[A2_LINK, bool]] = []
    for node_id in sorted(leaves):
        link = by_node[node_id][0]
        if link.from_node_id == node_id:
            queue.append((link, False))
        elif link.to_node_id == node_id:
            queue.append((link, True))
    return queue


def _queue_downstream_links(
    queue: list[tuple[A2_LINK, bool]],
    by_node: dict[str, list[A2_LINK]],
    link: A2_LINK,
    flip: bool,
) -> None:
    upstream = link.to_node_id if flip else link.from_node_id
    downstream = link.from_node_id if flip else link.to_node_id
    if upstream is None or downstream is None:
        return
    for next_link in by_node.get(downstream, ()):
        if next_link.id == link.id:
            continue
        if next_link.from_node_id == downstream:
            queue.append((next_link, False))
        elif next_link.to_node_id == downstream:
            queue.append((next_link, True))


def _has_opposing_same_direction_neighbour(
    dataset: NGIIDataset, link: A2_LINK, parallel_dot_min: float
) -> bool:
    link_vec = _a2_unit_vector(link)
    if link_vec is None:
        return False
    for neighbour_id in (link.r_link_id, link.l_link_id):
        neighbour = dataset.a2_link.get(neighbour_id)
        if neighbour is None:
            continue
        neighbour_vec = _a2_unit_vector(neighbour)
        if neighbour_vec is None:
            continue
        if float(np.dot(link_vec, neighbour_vec)) <= -parallel_dot_min:
            return True
    return False


def _a2_unit_vector(link: A2_LINK) -> np.ndarray | None:
    if len(link.polyline) < 2:
        return None
    vec = link.polyline[-1, :2] - link.polyline[0, :2]
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def _swap_a2_endpoint_ids(
    dataset: NGIIDataset,
    link: A2_LINK,
    code: str,
    reason: str,
    *,
    enabled: bool,
    warn_disabled: bool,
) -> None:
    before = {"from_node_id": link.from_node_id, "to_node_id": link.to_node_id}
    if not enabled:
        if warn_disabled:
            dataset.sanity.warn(
                f"{code}-disabled",
                f"A2_LINK {link.id} {reason}, but direction swap is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    link.from_node_id, link.to_node_id = link.to_node_id, link.from_node_id
    dataset.sanity.action(
        code,
        f"A2_LINK {link.id} {reason}",
        before=before,
        after={"from_node_id": link.from_node_id, "to_node_id": link.to_node_id},
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
    )


def _warn_unresolved_relationships(dataset: NGIIDataset) -> None:
    for spec in LAYER_SPECS:
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
