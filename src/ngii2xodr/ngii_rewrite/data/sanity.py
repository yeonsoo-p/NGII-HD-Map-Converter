"""Structured warning/action logs for the NGII rewrite loader."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ngii2xodr.ngii.data.config import NGIIConfig, check_reports
from ngii2xodr.ngii_rewrite.data.features import (
    NGIIFeature,
    is_float_value,
    is_integer_value,
    optional_text,
)
from ngii2xodr.ngii_rewrite.data.metadata import (
    FieldDef,
    LayerDef,
    ReferenceDef,
    Schema,
    field_defs,
    manual_columns,
    reference_defs,
)

if TYPE_CHECKING:
    from ngii2xodr.ngii_rewrite.data.dataset import LayerStore, NGIIDataset


@dataclass(slots=True, frozen=True)
class SanityWarning:
    code: str
    message: str
    layer_name: str | None = None
    feature_id: str | None = None
    source_path: Path | None = None


@dataclass(slots=True, frozen=True)
class SanityAction:
    code: str
    message: str
    before: dict[str, Any]
    after: dict[str, Any]
    layer_name: str | None = None
    feature_id: str | None = None
    source_path: Path | None = None


@dataclass(slots=True)
class SanityReport:
    warnings: list[SanityWarning] = field(default_factory=list)
    actions: list[SanityAction] = field(default_factory=list)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def warn(
        self,
        code: str,
        message: str,
        *,
        layer_name: str | None = None,
        feature_id: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.warnings.append(
            SanityWarning(
                code=code,
                message=message,
                layer_name=layer_name,
                feature_id=feature_id,
                source_path=source_path,
            )
        )

    def action(
        self,
        code: str,
        message: str,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        layer_name: str | None = None,
        feature_id: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.actions.append(
            SanityAction(
                code=code,
                message=message,
                before=before,
                after=after,
                layer_name=layer_name,
                feature_id=feature_id,
                source_path=source_path,
            )
        )


def warn_missing_required_layers(
    discovered: Sequence[Any], schema: Schema, sanity: SanityReport
) -> None:
    seen = {item.layer.layer_name for item in discovered}
    for layer in schema.layers:
        if layer.required_layer and layer.layer_name not in seen:
            sanity.warn(
                "missing-required-layer",
                f"required NGII layer {layer.layer_name} was not found",
                layer_name=layer.layer_name,
            )


def warn_missing_sidecars(shp_path: Path, sanity: SanityReport) -> None:
    for suffix in (".dbf", ".shx", ".prj"):
        sidecar = shp_path.with_suffix(suffix)
        if not sidecar.is_file():
            sanity.warn(
                "missing-shp-sidecar",
                f"{shp_path.name} is missing required sidecar {sidecar.name}",
                source_path=shp_path,
            )


def warn_missing_columns(gdf: Any, layer: LayerDef, shp_path: Path, sanity: SanityReport) -> None:
    missing = sorted(
        definition.name
        for definition in field_defs(layer.layer_type)
        if definition.name not in gdf.columns
    )
    if missing:
        sanity.warn(
            "missing-manual-columns",
            f"{layer.layer_name} is missing manual fields: {', '.join(missing)}",
            layer_name=layer.layer_name,
            source_path=shp_path,
        )


def warn_unknown_columns(gdf: Any, layer: LayerDef, shp_path: Path, sanity: SanityReport) -> None:
    allowed = manual_columns(layer.layer_type) | {"geometry"}
    unknown = sorted(str(column) for column in gdf.columns if str(column) not in allowed)
    if not unknown:
        return
    sanity.warn(
        "unknown-manual-column",
        f"{layer.layer_name} has undocumented DBF columns that were preserved: "
        f"{', '.join(unknown)}",
        layer_name=layer.layer_name,
        source_path=shp_path,
    )


def check_manual_field_values(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for definition in field_defs(store.layer_type):
                _warn_manual_field_value(dataset, sanity, store, feature, definition, cfg)


def _warn_manual_field_value(
    dataset: NGIIDataset,
    sanity: SanityReport,
    store: LayerStore[Any],
    feature: NGIIFeature,
    definition: FieldDef,
    cfg: NGIIConfig,
) -> None:
    value = store.value_for_column(feature, definition.name)
    value_text = _value_text(value)
    if not value_text:
        if definition.required and check_reports(cfg.sanity.checks.manual_field_required_missing):
            sanity.warn(
                "manual-field-required-missing",
                f"{store.layer_name} {feature.id}: {definition.name} is required by the "
                f"{dataset.schema.version} manual",
                layer_name=store.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )
        return
    if (
        definition.max_length is not None
        and len(value_text) > definition.max_length
        and check_reports(cfg.sanity.checks.manual_field_length_exceeded)
    ):
        sanity.warn(
            "manual-field-length-exceeded",
            f"{store.layer_name} {feature.id}: {definition.name}={value_text!r} exceeds "
            f"VARCHAR2({definition.max_length})",
            layer_name=store.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    invalid_integer = definition.field_type == "integer" and not is_integer_value(value)
    invalid_float = definition.field_type == "float" and not is_float_value(value)
    if (invalid_integer or invalid_float) and check_reports(
        cfg.sanity.checks.manual_field_type_invalid
    ):
        _warn_invalid_type(sanity, store.layer_name, feature, definition, value_text)
    if (
        definition.code_list is not None
        and value_text not in definition.code_list
        and check_reports(cfg.sanity.checks.manual_field_code_invalid)
    ):
        sanity.warn(
            "manual-field-code-invalid",
            f"{store.layer_name} {feature.id}: {definition.name}={value_text!r} is not "
            f"in the {dataset.schema.version} code list",
            layer_name=store.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )


def check_unresolved_references(dataset: NGIIDataset, sanity: SanityReport) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for reference in reference_defs(store.layer_type):
                value = getattr(feature, reference.source_attr)
                value_text = _value_text("" if value is None else value)
                target_layer = _reference_target_label(dataset, reference.target_attrs)
                if not value_text:
                    if reference.required:
                        _warn_unresolved_reference(sanity, feature, reference, target_layer)
                    continue
                if not _reference_resolves(dataset, reference.target_attrs, value_text):
                    _warn_unresolved_reference(sanity, feature, reference, target_layer)


def log_sanity_report(dataset: NGIIDataset, report: SanityReport, logger: logging.Logger) -> None:
    warning_count = report.warning_count
    action_count = report.action_count
    if warning_count == 0 and action_count == 0:
        logger.info(
            "NGII rewrite sanity: no warnings or repairs for %s [%s]",
            dataset.root,
            dataset.coordinate,
        )
        return

    logger.warning(
        "NGII rewrite sanity: %d warning(s), %d repair action(s) for %s [%s]",
        warning_count,
        action_count,
        dataset.root,
        dataset.coordinate,
    )
    _log_warning_summaries(dataset, report, logger)
    for warning in report.warnings:
        logger.debug(
            "NGII rewrite sanity warning detail [%s] %s: %s",
            warning.code,
            _sanity_location(warning.layer_name, warning.feature_id, warning.source_path),
            warning.message,
        )


def _value_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return optional_text(value)


def _warn_invalid_type(
    sanity: SanityReport,
    layer_name: str,
    feature: NGIIFeature,
    definition: FieldDef,
    value_text: str,
) -> None:
    sanity.warn(
        "manual-field-type-invalid",
        f"{layer_name} {feature.id}: {definition.name}={value_text!r} is not "
        f"a valid {definition.field_type}",
        layer_name=layer_name,
        feature_id=feature.id,
        source_path=feature.source_path,
    )


def _reference_resolves(
    dataset: NGIIDataset, target_attrs: tuple[str, ...], feature_id: str
) -> bool:
    return any(
        dataset.store_for_attr(target_attr).get(feature_id) is not None
        for target_attr in target_attrs
    )


def _reference_target_label(dataset: NGIIDataset, target_attrs: tuple[str, ...]) -> str:
    return "/".join(dataset.store_for_attr(target_attr).layer_name for target_attr in target_attrs)


def _warn_unresolved_reference(
    sanity: SanityReport, feature: NGIIFeature, reference: ReferenceDef, target_layer: str
) -> None:
    sanity.warn(
        "reference-unresolved",
        f"{feature.layer_name} {feature.id} {reference.column_name} does not resolve to "
        f"{target_layer}",
        layer_name=feature.layer_name,
        feature_id=feature.id,
        source_path=feature.source_path,
    )


def _log_warning_summaries(
    dataset: NGIIDataset, report: SanityReport, logger: logging.Logger
) -> None:
    grouped: dict[tuple[str, str, str, str], list[SanityWarning]] = defaultdict(list)
    for warning in report.warnings:
        key = (
            warning.code,
            warning.layer_name or "-",
            _short_path(dataset.root, warning.source_path),
            _strip_feature_prefix(warning.message, warning.layer_name, warning.feature_id),
        )
        grouped[key].append(warning)

    for (code, layer_name, source_path, reason), events in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        logger.warning(
            "NGII rewrite sanity warning summary [%s] %s %s count=%d examples=%s reason=%s",
            code,
            layer_name,
            source_path,
            len(events),
            _example_ids(events),
            reason,
        )


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


def _example_ids(events: list[SanityWarning]) -> str:
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
