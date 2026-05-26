"""Version-agnostic NGII schema contracts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ngii2xodr.ngii.data.features import FeatureGeometryKind, FeatureRecord, NGIIFeature

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

    @property
    def attr(self) -> str:
        return self.attr_name or column_to_attr(self.name)


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
    geometry_kind: FeatureGeometryKind
    feature_type: type[NGIIFeature]
    factory: Callable[[FeatureRecord], NGIIFeature]
    required_layer: bool
    field_rules: tuple[FieldRule, ...]
    relationships: tuple[RelationshipRule, ...] = ()
    roles: tuple[LayerRole, ...] = ()
    filename_aliases: tuple[str, ...] = ()
    accepted_geometry_kinds: tuple[FeatureGeometryKind, ...] = ()

    @property
    def filenames(self) -> tuple[str, ...]:
        return (self.filename, *self.filename_aliases)

    @property
    def geometry_kinds(self) -> tuple[FeatureGeometryKind, ...]:
        return self.accepted_geometry_kinds or (self.geometry_kind,)

    @property
    def field_attrs_by_column(self) -> dict[str, str]:
        return {rule.name: rule.attr for rule in self.field_rules}


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

    def role_filter(self, name: str) -> RoleFilter | None:
        for role_filter in self.role_filters:
            if role_filter.name == name:
                return role_filter
        return None


def text_rule(
    name: str,
    max_length: int,
    *,
    required: bool,
    code_list: dict[str, str] | None = None,
    attr_name: str = "",
) -> FieldRule:
    return FieldRule(name, required, "text", max_length, code_list, attr_name)


def integer_rule(name: str, *, required: bool, attr_name: str = "") -> FieldRule:
    return FieldRule(name, required, "integer", attr_name=attr_name)


def float_rule(name: str, *, required: bool, attr_name: str = "") -> FieldRule:
    return FieldRule(name, required, "float", attr_name=attr_name)


def column_to_attr(column_name: str) -> str:
    if column_name == "ID":
        return "id"
    parts = column_name.split("_")
    return "_".join(_camel_to_snake(part) for part in parts if part).lower()


def _camel_to_snake(value: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value
