"""Dataclass-backed NGII layer metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import Field, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ngii2xodr.ngii_rewrite.data.features import FeatureGeometryKind, NGIIFeature

FieldType = Literal["text", "integer", "float"]
LayerRole = str
LayerType = type["NGIIFeature"]

_FIELD_METADATA_KEY = "ngii_field"


@dataclass(slots=True, frozen=True)
class FieldDef:
    name: str
    attr: str
    required: bool
    field_type: FieldType
    max_length: int | None = None
    code_list: dict[str, str] | None = None
    column_aliases: tuple[str, ...] = ()
    array_aliases: tuple[str, ...] = ()
    target_layer_attrs: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name, *self.column_aliases)

    @property
    def is_reference(self) -> bool:
        return bool(self.target_layer_attrs)


@dataclass(slots=True, frozen=True)
class ReferenceDef:
    column_name: str
    source_attr: str
    target_attrs: tuple[str, ...]
    required: bool


@dataclass(slots=True, frozen=True)
class ReciprocalReferenceRule:
    layer_attr: str
    source_column: str
    source_attr: str
    reciprocal_column: str
    reciprocal_attr: str


@dataclass(slots=True, frozen=True)
class RoleFilter:
    name: str
    role: LayerRole
    attr_name: str = ""
    values: tuple[str, ...] = ()
    attr_names: tuple[str, ...] = ()
    numeric_min: float | None = None

    @property
    def attrs(self) -> tuple[str, ...]:
        if self.attr_names:
            return self.attr_names
        if self.attr_name:
            return (self.attr_name,)
        msg = f"role filter {self.name!r} must define attr_name or attr_names"
        raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class RoleKey:
    name: str
    role: LayerRole
    attr_name: str = ""
    attr_names: tuple[str, ...] = ()

    @property
    def attrs(self) -> tuple[str, ...]:
        if self.attr_names:
            return self.attr_names
        if self.attr_name:
            return (self.attr_name,)
        msg = f"role key {self.name!r} must define attr_name or attr_names"
        raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class LayerDef:
    layer_type: LayerType
    layer_name: str
    layer_attr: str
    filename: str
    geometry_kinds: tuple[FeatureGeometryKind, ...]
    required_layer: bool
    roles: tuple[LayerRole, ...]
    filename_aliases: tuple[str, ...] = ()

    @property
    def filenames(self) -> tuple[str, ...]:
        return (self.filename, *self.filename_aliases)


@dataclass(slots=True, frozen=True)
class Schema:
    version: str
    layer_types: tuple[LayerType, ...]
    role_filters: tuple[RoleFilter, ...] = ()
    role_keys: tuple[RoleKey, ...] = ()
    reciprocal_references: tuple[ReciprocalReferenceRule, ...] = ()

    @property
    def layers(self) -> tuple[LayerDef, ...]:
        return tuple(layer_def(layer_type) for layer_type in self.layer_types)

    @property
    def layers_by_filename(self) -> dict[str, LayerDef]:
        return {filename.upper(): layer for layer in self.layers for filename in layer.filenames}

    @property
    def layers_by_name(self) -> dict[str, LayerDef]:
        return {layer.layer_name: layer for layer in self.layers}

    @property
    def layers_by_attr(self) -> dict[str, LayerDef]:
        return {layer.layer_attr: layer for layer in self.layers}

    def layer_for_attr(self, attr: str) -> LayerDef:
        return self.layers_by_attr[attr]

    def layer_for_name(self, layer_name: str) -> LayerDef | None:
        return self.layers_by_name.get(layer_name)

    def attr_for_layer_name(self, layer_name: str) -> str | None:
        layer = self.layer_for_name(layer_name)
        return None if layer is None else layer.layer_attr

    def attrs_for_role(self, role: LayerRole) -> tuple[str, ...]:
        return tuple(layer.layer_attr for layer in self.layers if role in layer.roles)

    def attr_for_role(self, role: LayerRole) -> str:
        attrs = self.attrs_for_role(role)
        if not attrs:
            msg = f"schema {self.version} has no layer for role {role!r}"
            raise KeyError(msg)
        if len(attrs) > 1:
            msg = f"schema {self.version} has multiple layers for role {role!r}: {attrs}"
            raise KeyError(msg)
        return attrs[0]


def text_field(
    name: str,
    max_length: int,
    *,
    required: bool,
    code_list: dict[str, str] | None = None,
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> Any:
    return _field(
        FieldDef(
            name=name,
            attr="",
            required=required,
            field_type="text",
            max_length=max_length,
            code_list=code_list,
            column_aliases=column_aliases,
            array_aliases=array_aliases,
        )
    )


def integer_field(
    name: str,
    *,
    required: bool,
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> Any:
    return _field(
        FieldDef(
            name=name,
            attr="",
            required=required,
            field_type="integer",
            column_aliases=column_aliases,
            array_aliases=array_aliases,
        )
    )


def float_field(
    name: str,
    *,
    required: bool,
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> Any:
    return _field(
        FieldDef(
            name=name,
            attr="",
            required=required,
            field_type="float",
            column_aliases=column_aliases,
            array_aliases=array_aliases,
        )
    )


def ref_field(
    name: str,
    max_length: int,
    *,
    target_layer_attrs: tuple[str, ...],
    required: bool,
    column_aliases: tuple[str, ...] = (),
    array_aliases: tuple[str, ...] = (),
) -> Any:
    return _field(
        FieldDef(
            name=name,
            attr="",
            required=required,
            field_type="text",
            max_length=max_length,
            column_aliases=column_aliases,
            array_aliases=array_aliases,
            target_layer_attrs=target_layer_attrs,
        )
    )


def _field(definition: FieldDef) -> Any:
    return field(metadata={_FIELD_METADATA_KEY: definition})


def layer_def(layer_type: LayerType) -> LayerDef:
    return LayerDef(
        layer_type=layer_type,
        layer_name=str(layer_type.layer_name),
        layer_attr=str(layer_type.layer_attr),
        filename=str(layer_type.filename),
        filename_aliases=tuple(getattr(layer_type, "filename_aliases", ())),
        geometry_kinds=geometry_kinds_for_layer(layer_type),
        required_layer=bool(layer_type.required_layer),
        roles=tuple(getattr(layer_type, "roles", ())),
    )


def field_defs(layer_type: LayerType) -> tuple[FieldDef, ...]:
    id_field = FieldDef(
        name="ID",
        attr="id",
        required=True,
        field_type="text",
        max_length=int(layer_type.id_max_length),
    )
    return (id_field, *tuple(_field_def_for(item) for item in fields(layer_type) if _has_def(item)))


def reference_defs(layer_type: LayerType) -> tuple[ReferenceDef, ...]:
    return tuple(
        ReferenceDef(
            column_name=definition.name,
            source_attr=definition.attr,
            target_attrs=definition.target_layer_attrs,
            required=definition.required,
        )
        for definition in field_defs(layer_type)
        if definition.is_reference
    )


def manual_columns(layer_type: LayerType) -> set[str]:
    return {column for definition in field_defs(layer_type) for column in definition.columns}


def field_defs_by_column(layer_type: LayerType) -> dict[str, FieldDef]:
    return {
        column: definition for definition in field_defs(layer_type) for column in definition.columns
    }


def columns_by_alias(layer_type: LayerType) -> dict[str, str]:
    return {
        column: definition.name
        for definition in field_defs(layer_type)
        for column in definition.columns
    }


def geometry_kinds_for_layer(layer_type: LayerType) -> tuple[FeatureGeometryKind, ...]:
    from ngii2xodr.ngii_rewrite.data.features import (  # noqa: PLC0415
        LineFeature,
        PointFeature,
        PolygonFeature,
    )

    candidate = cast(type[Any], layer_type)
    kinds: list[FeatureGeometryKind] = []
    if issubclass(candidate, PointFeature):
        kinds.append("point")
    if issubclass(candidate, LineFeature):
        kinds.append("line")
    if issubclass(candidate, PolygonFeature):
        kinds.append("polygon")
    if not kinds:
        msg = f"{layer_type.__name__} must inherit at least one geometry mixin"
        raise ValueError(msg)
    return tuple(kinds)


def validate_layer_types(layer_types: Iterable[type[Any]]) -> None:
    from ngii2xodr.ngii_rewrite.data.features import NGIIFeature  # noqa: PLC0415

    for candidate in layer_types:
        if not issubclass(candidate, NGIIFeature):
            msg = f"{candidate.__name__} must inherit NGIIFeature"
            raise TypeError(msg)
        layer_type = candidate
        layer = layer_def(layer_type)
        if getattr(layer_type, "layer_name", "") != layer.layer_name:
            msg = f"{layer_type.__name__} has inconsistent layer_name"
            raise ValueError(msg)
        dataclass_field_names = {item.name for item in fields(layer_type)}
        missing_reference_attrs = sorted(
            reference.source_attr
            for reference in reference_defs(layer_type)
            if reference.source_attr not in dataclass_field_names
        )
        if missing_reference_attrs:
            msg = f"{layer.layer_name}: reference attrs missing: {missing_reference_attrs}"
            raise ValueError(msg)


def _has_def(item: Field[Any]) -> bool:
    return _FIELD_METADATA_KEY in item.metadata


def _field_def_for(item: Field[Any]) -> FieldDef:
    raw = item.metadata[_FIELD_METADATA_KEY]
    if not isinstance(raw, FieldDef):
        msg = f"{item.name} has invalid NGII field metadata"
        raise TypeError(msg)
    return FieldDef(
        name=raw.name,
        attr=item.name,
        required=raw.required,
        field_type=raw.field_type,
        max_length=raw.max_length,
        code_list=raw.code_list,
        column_aliases=raw.column_aliases,
        array_aliases=raw.array_aliases,
        target_layer_attrs=raw.target_layer_attrs,
    )


def column_to_attr(column_name: str) -> str:
    if column_name == "ID":
        return "id"
    parts = column_name.split("_")
    return "_".join(_camel_to_snake(part) for part in parts if part).lower()


def _camel_to_snake(value: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value


__all__ = [
    "FieldDef",
    "FieldType",
    "LayerDef",
    "LayerRole",
    "LayerType",
    "ReciprocalReferenceRule",
    "ReferenceDef",
    "RoleFilter",
    "RoleKey",
    "Schema",
    "columns_by_alias",
    "field_defs",
    "field_defs_by_column",
    "float_field",
    "integer_field",
    "layer_def",
    "manual_columns",
    "ref_field",
    "reference_defs",
    "text_field",
    "validate_layer_types",
]
