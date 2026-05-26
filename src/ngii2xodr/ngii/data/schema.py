"""Version-agnostic NGII schema contracts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Literal

from ngii2xodr.ngii.data.features import (
    FeatureGeometryKind,
    FeatureRecord,
    LineFeature,
    NGIIFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
)

FieldType = Literal["text", "integer", "float"]
LayerRole = str


@dataclass(slots=True, frozen=True)
class FieldRule:
    name: str
    required: bool
    field_type: FieldType
    max_length: int | None = None
    code_list: dict[str, str] | None = None
    attr_name: str = ""
    column_aliases: tuple[str, ...] = ()
    array_aliases: tuple[str, ...] = ()

    @property
    def attr(self) -> str:
        return self.attr_name or column_to_attr(self.name)

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name, *self.column_aliases)


@dataclass(slots=True, frozen=True)
class RelationshipRule:
    column_name: str
    source_attr: str
    target_attrs: tuple[str, ...]
    required: bool


@dataclass(slots=True, frozen=True)
class RoleFilter:
    name: str
    role: LayerRole
    attr_name: str
    values: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class LayerSpec:
    layer_name: str
    python_attr: str
    filename: str
    geometry_kinds: tuple[FeatureGeometryKind, ...]
    feature_type: type[NGIIFeature]
    factory: Callable[[FeatureRecord], NGIIFeature]
    required_layer: bool
    field_rules: tuple[FieldRule, ...]
    relationships: tuple[RelationshipRule, ...] = ()
    roles: tuple[LayerRole, ...] = ()
    filename_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_feature_type(self)

    @property
    def filenames(self) -> tuple[str, ...]:
        return (self.filename, *self.filename_aliases)

    @property
    def field_attrs_by_column(self) -> dict[str, str]:
        return {column: rule.attr for rule in self.field_rules for column in rule.columns}

    @property
    def field_rules_by_column(self) -> dict[str, FieldRule]:
        return {column: rule for rule in self.field_rules for column in rule.columns}

    @property
    def manual_columns(self) -> set[str]:
        field_columns = {rule.name for rule in self.field_rules}
        relationship_columns = {relationship.column_name for relationship in self.relationships}
        return field_columns | relationship_columns


@dataclass(slots=True, frozen=True)
class SchemaDefinition:
    version: str
    layer_specs: tuple[LayerSpec, ...]
    role_filters: tuple[RoleFilter, ...] = ()

    @property
    def specs_by_filename(self) -> dict[str, LayerSpec]:
        return {filename.upper(): spec for spec in self.layer_specs for filename in spec.filenames}

    @property
    def specs_by_layer_name(self) -> dict[str, LayerSpec]:
        return {spec.layer_name: spec for spec in self.layer_specs}

    @property
    def specs_by_attr(self) -> dict[str, LayerSpec]:
        return {spec.python_attr: spec for spec in self.layer_specs}

    def spec_for_attr(self, attr: str) -> LayerSpec:
        return self.specs_by_attr[attr]

    def spec_for_layer_name(self, layer_name: str) -> LayerSpec | None:
        return self.specs_by_layer_name.get(layer_name)

    def attr_for_layer_name(self, layer_name: str) -> str | None:
        spec = self.spec_for_layer_name(layer_name)
        return None if spec is None else spec.python_attr

    def attrs_for_role(self, role: LayerRole) -> tuple[str, ...]:
        return tuple(spec.python_attr for spec in self.layer_specs if role in spec.roles)

    def attr_for_role(self, role: LayerRole) -> str:
        attrs = self.attrs_for_role(role)
        if not attrs:
            msg = f"schema {self.version} has no layer for role {role!r}"
            raise KeyError(msg)
        if len(attrs) > 1:
            msg = f"schema {self.version} has multiple layers for role {role!r}: {attrs}"
            raise KeyError(msg)
        return attrs[0]


def text_rule(
    name: str,
    max_length: int,
    *,
    required: bool,
    code_list: dict[str, str] | None = None,
    attr_name: str = "",
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> FieldRule:
    return FieldRule(
        name,
        required,
        "text",
        max_length,
        code_list,
        attr_name,
        column_aliases,
        array_aliases,
    )


def integer_rule(
    name: str,
    *,
    required: bool,
    attr_name: str = "",
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> FieldRule:
    return FieldRule(
        name,
        required,
        "integer",
        attr_name=attr_name,
        column_aliases=column_aliases,
        array_aliases=array_aliases,
    )


def float_rule(
    name: str,
    *,
    required: bool,
    attr_name: str = "",
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> FieldRule:
    return FieldRule(
        name,
        required,
        "float",
        attr_name=attr_name,
        column_aliases=column_aliases,
        array_aliases=array_aliases,
    )


def column_to_attr(column_name: str) -> str:
    if column_name == "ID":
        return "id"
    parts = column_name.split("_")
    return "_".join(_camel_to_snake(part) for part in parts if part).lower()


def _validate_feature_type(spec: LayerSpec) -> None:
    feature_type_object: object = spec.feature_type
    if not isinstance(feature_type_object, type) or not issubclass(
        feature_type_object, NGIIFeature
    ):
        msg = f"{spec.layer_name}: feature_type must be an NGIIFeature subclass"
        raise ValueError(msg)  # noqa: TRY004
    actual_layer_name = getattr(spec.feature_type, "layer_name", None)
    if actual_layer_name != spec.layer_name:
        msg = (
            f"{spec.layer_name}: feature_type {spec.feature_type.__name__} declares "
            f"layer_name={actual_layer_name!r}"
        )
        raise ValueError(msg)

    expected_base = _geometry_base_for(spec.geometry_kinds)
    if not issubclass(spec.feature_type, expected_base):
        msg = (
            f"{spec.layer_name}: feature_type {spec.feature_type.__name__} must inherit "
            f"{expected_base.__name__} for geometry_kinds={spec.geometry_kinds!r}"
        )
        raise ValueError(msg)  # noqa: TRY004

    feature_fields = {field.name for field in dataclass_fields(spec.feature_type)}
    missing_field_attrs = sorted({rule.attr for rule in spec.field_rules} - feature_fields)
    missing_relationship_attrs = sorted(
        {relationship.source_attr for relationship in spec.relationships} - feature_fields
    )
    if missing_field_attrs or missing_relationship_attrs:
        parts: list[str] = []
        if missing_field_attrs:
            parts.append(f"field attrs missing from feature_type: {missing_field_attrs}")
        if missing_relationship_attrs:
            parts.append(
                f"relationship attrs missing from feature_type: {missing_relationship_attrs}"
            )
        msg = f"{spec.layer_name}: {'; '.join(parts)}"
        raise ValueError(msg)


def _geometry_base_for(
    geometry_kinds: tuple[FeatureGeometryKind, ...],
) -> type[NGIIFeature]:
    kind_set = frozenset(geometry_kinds)
    if kind_set == frozenset({"point"}):
        return PointFeature
    if kind_set == frozenset({"line"}):
        return LineFeature
    if kind_set == frozenset({"polygon"}):
        return PolygonFeature
    if kind_set == frozenset({"point", "polygon"}):
        return PointOrPolygonFeature
    msg = f"unsupported geometry_kinds={geometry_kinds!r}"
    raise ValueError(msg)


def _camel_to_snake(value: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value
