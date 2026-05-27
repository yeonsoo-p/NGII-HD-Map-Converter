"""Structured warning/action logs produced while loading NGII data."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ngii2xodr.ngii.data.config import NGIIConfig, check_reports
from ngii2xodr.ngii.data.features import NGIIFeature, optional_text
from ngii2xodr.ngii.data.schema import FieldRule, LayerSpec, SchemaDefinition

if TYPE_CHECKING:
    from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset


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
    discovered: Sequence[Any], schema: SchemaDefinition, sanity: SanityReport
) -> None:
    seen = {item.spec.layer_name for item in discovered}
    for spec in schema.layer_specs:
        if spec.required_layer and spec.layer_name not in seen:
            sanity.warn(
                "missing-required-layer",
                f"required NGII layer {spec.layer_name} was not found",
                layer_name=spec.layer_name,
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


def warn_missing_columns(gdf: Any, spec: LayerSpec, shp_path: Path, sanity: SanityReport) -> None:
    missing = sorted(rule.name for rule in spec.field_rules if rule.name not in gdf.columns)
    if missing:
        sanity.warn(
            "missing-manual-columns",
            f"{spec.layer_name} is missing manual fields: {', '.join(missing)}",
            layer_name=spec.layer_name,
            source_path=shp_path,
        )


def warn_unknown_columns(gdf: Any, spec: LayerSpec, shp_path: Path, sanity: SanityReport) -> None:
    allowed = spec.manual_columns | {"geometry"}
    unknown = sorted(str(column) for column in gdf.columns if str(column) not in allowed)
    if not unknown:
        return
    sanity.warn(
        "unknown-manual-column",
        f"{spec.layer_name} has undocumented DBF columns that were omitted: {', '.join(unknown)}",
        layer_name=spec.layer_name,
        source_path=shp_path,
    )


def check_manual_field_values(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    for store in dataset.layer_stores:
        spec = store.spec
        for feature in store.features:
            for rule in spec.field_rules:
                _warn_manual_field_value(dataset, sanity, store, feature, rule, cfg)


def _warn_manual_field_value(
    dataset: NGIIDataset,
    sanity: SanityReport,
    store: LayerStore[Any],
    feature: NGIIFeature,
    rule: FieldRule,
    cfg: NGIIConfig,
) -> None:
    spec = store.spec
    value = store.value_for_column(feature, rule.name)
    value_text = _value_text(value)
    if not value_text:
        if rule.required and check_reports(cfg.sanity.checks.manual_field_required_missing):
            sanity.warn(
                "manual-field-required-missing",
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
        and check_reports(cfg.sanity.checks.manual_field_length_exceeded)
    ):
        sanity.warn(
            "manual-field-length-exceeded",
            f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} exceeds "
            f"VARCHAR2({rule.max_length})",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    invalid_integer = rule.field_type == "integer" and not _is_integer(value)
    invalid_float = rule.field_type == "float" and not _is_float(value)
    if (invalid_integer or invalid_float) and check_reports(
        cfg.sanity.checks.manual_field_type_invalid
    ):
        _warn_invalid_type(sanity, spec, feature, rule, value_text)
    if (
        rule.code_list is not None
        and value_text not in rule.code_list
        and check_reports(cfg.sanity.checks.manual_field_code_invalid)
    ):
        sanity.warn(
            "manual-field-code-invalid",
            f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} is not "
            f"in the {dataset.schema.version} code list",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )
    if (
        rule.name == "HistType"
        and not _is_valid_hist_type(spec, value_text)
        and check_reports(cfg.sanity.checks.manual_hist_type_invalid)
    ):
        expected = _expected_hist_type_label(spec, rule)
        sanity.warn(
            "manual-hist-type-invalid",
            f"{spec.layer_name} {feature.id}: HistType={value_text!r} must be {expected}",
            layer_name=spec.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )


def check_unresolved_references(dataset: NGIIDataset, sanity: SanityReport) -> None:
    for store in dataset.layer_stores:
        spec = store.spec
        for feature in store.features:
            for reference in spec.references:
                value = getattr(feature, reference.source_attr)
                value_text = _value_text("" if value is None else value)
                target_layer = _reference_target_label(dataset, reference.target_attrs)
                if not value_text:
                    if reference.required:
                        _warn_unresolved_reference(
                            sanity,
                            feature,
                            reference.column_name,
                            target_layer,
                        )
                    continue
                if not _reference_resolves(dataset, reference.target_attrs, value_text):
                    _warn_unresolved_reference(
                        sanity,
                        feature,
                        reference.column_name,
                        target_layer,
                    )


def log_sanity_report(dataset: NGIIDataset, report: SanityReport, logger: logging.Logger) -> None:
    warning_count = report.warning_count
    action_count = report.action_count
    if warning_count == 0 and action_count == 0:
        logger.info(
            "NGII sanity: no warnings or repairs for %s [%s]",
            dataset.root,
            dataset.coordinate,
        )
        return

    logger.warning(
        "NGII sanity: %d warning(s), %d repair action(s) for %s [%s]",
        warning_count,
        action_count,
        dataset.root,
        dataset.coordinate,
    )
    _log_warning_summaries(dataset, report, logger)
    _log_action_summaries(dataset, report, logger)
    for warning in report.warnings:
        logger.debug(
            "NGII sanity warning detail [%s] %s: %s",
            warning.code,
            _sanity_location(warning.layer_name, warning.feature_id, warning.source_path),
            warning.message,
        )
    for action in report.actions:
        logger.debug(
            "NGII sanity repair detail [%s] %s: %s before=%s after=%s",
            action.code,
            _sanity_location(action.layer_name, action.feature_id, action.source_path),
            action.message,
            action.before,
            action.after,
        )


def _value_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return optional_text(value)


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


def _expected_hist_type_label(spec: LayerSpec, rule: FieldRule) -> str:
    if rule.max_length == 3:
        return "001-008"
    layer_prefix = spec.layer_name.split("_", maxsplit=1)[0]
    return f"{layer_prefix}001-{layer_prefix}008"


def _warn_invalid_type(
    sanity: SanityReport,
    spec: LayerSpec,
    feature: NGIIFeature,
    rule: FieldRule,
    value_text: str,
) -> None:
    sanity.warn(
        "manual-field-type-invalid",
        f"{spec.layer_name} {feature.id}: {rule.name}={value_text!r} is not "
        f"a valid {rule.field_type}",
        layer_name=spec.layer_name,
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
    sanity: SanityReport, feature: NGIIFeature, column_name: str, target_layer: str
) -> None:
    sanity.warn(
        "reference-unresolved",
        f"{feature.layer_name} {feature.id} {column_name} does not resolve to {target_layer}",
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
            _warning_summary_reason(warning),
        )
        grouped[key].append(warning)

    for (code, layer_name, source_path, reason), events in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        logger.warning(
            "NGII sanity warning summary [%s] %s %s count=%d examples=%s reason=%s",
            code,
            layer_name,
            source_path,
            len(events),
            _example_ids(events),
            reason,
        )


def _log_action_summaries(
    dataset: NGIIDataset, report: SanityReport, logger: logging.Logger
) -> None:
    grouped: dict[tuple[str, str, str], list[SanityAction]] = defaultdict(list)
    for action in report.actions:
        key = (action.code, action.layer_name or "-", _action_summary_reason(dataset, action))
        grouped[key].append(action)

    for (code, layer_name, reason), events in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        logger.warning(
            "NGII sanity repair summary [%s] %s count=%d examples=%s reason=%s",
            code,
            layer_name,
            len(events),
            _example_ids(events),
            reason,
        )


def _warning_summary_reason(warning: SanityWarning) -> str:
    if warning.code == "geometry-invalid":
        _row_label, sep, reason = warning.message.partition(": ")
        if sep:
            return reason
    return _strip_feature_prefix(warning.message, warning.layer_name, warning.feature_id)


def _action_summary_reason(dataset: NGIIDataset, action: SanityAction) -> str:
    if action.code == "feature-id-duplicate-conflicting-dropped":
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
