"""Canonical in-memory NGII dataset tree."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.data.features import (
    LineFeature,
    NGIIFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
)
from ngii2xodr.ngii.data.geometry import xy_line
from ngii2xodr.ngii.data.sanity import SanityReport
from ngii2xodr.ngii.data.schema import (
    FieldRule,
    LayerRole,
    LayerSpec,
    SchemaDefinition,
    column_to_attr,
)
from ngii2xodr.profile import PerformanceProfile

_ARRAY_UNIT_SUFFIXES = ("_per_m", "_m", "_mm", "_rad", "_deg")


class AmbiguousFeatureIDError(KeyError):
    """Raised when global ``dataset[id]`` would cross layer boundaries."""


@dataclass(slots=True)
class LayerStore[T: NGIIFeature](Mapping[str, T]):
    spec: LayerSpec
    features: list[T] = field(default_factory=list)
    by_id: dict[str, T] = field(default_factory=dict)
    id_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rebuild_index()

    @property
    def layer_name(self) -> str:
        return self.spec.layer_name

    def __getitem__(self, feature_id: str) -> T:
        return self.by_id[feature_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.by_id)

    def __len__(self) -> int:
        return len(self.features)

    def rebuild_index(self) -> None:
        self.by_id = {feature.id: feature for feature in self.features}
        self.id_to_index = {feature.id: i for i, feature in enumerate(self.features)}

    @property
    def ids(self) -> NDArray[np.str_]:
        return np.asarray([feature.id for feature in self.features], dtype=np.str_)

    @property
    def points(self) -> NDArray[np.float64]:
        points: list[NDArray[np.float64]] = []
        for feature in self.features:
            point: NDArray[np.float64] | None = None
            if isinstance(feature, (PointFeature, PointOrPolygonFeature)):
                point = feature.point
            if point is not None:
                points.append(point)
        return np.asarray(points, dtype=np.float64)

    @property
    def polylines(self) -> list[NDArray[np.float64]]:
        return [feature.polyline for feature in self.features if isinstance(feature, LineFeature)]

    @property
    def xy_lines(self) -> tuple[shapely.LineString, ...]:
        return tuple(xy_line(polyline) for polyline in self.polylines)

    @property
    def rings(self) -> list[NDArray[np.float64]]:
        rings: list[NDArray[np.float64]] = []
        for feature in self.features:
            ring: NDArray[np.float64] | None = None
            if isinstance(feature, (PolygonFeature, PointOrPolygonFeature)):
                ring = feature.ring
            if ring is not None:
                rings.append(ring)
        return rings

    def __getattr__(self, name: str) -> NDArray[Any]:
        rule = _array_rule_for(self.spec, name)
        if rule is None:
            msg = f"{type(self).__name__!s} has no attribute {name!r}"
            raise AttributeError(msg)
        attr = rule.attr
        values = [getattr(feature, attr) for feature in self.features]
        if rule.field_type == "integer":
            return np.asarray(values, dtype=np.int32)
        if rule.field_type == "float":
            return np.asarray(values, dtype=np.float64)
        return np.asarray(["" if value is None else value for value in values], dtype=np.str_)


@dataclass(slots=True)
class NGIIDataset(Mapping[str, NGIIFeature]):
    root: Path
    coordinate: str
    schema: SchemaDefinition
    sanity: SanityReport
    warn_global_id_collision: bool = True
    load_profile: PerformanceProfile = field(default_factory=PerformanceProfile)
    _stores: dict[str, LayerStore[Any]] = field(default_factory=dict, init=False)
    _global_index: dict[str, NGIIFeature] = field(default_factory=dict, init=False)
    _ambiguous_global_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._stores = {spec.python_attr: LayerStore(spec) for spec in self.schema.layer_specs}

    @property
    def layer_stores(self) -> tuple[LayerStore[Any], ...]:
        return tuple(self._stores[spec.python_attr] for spec in self.schema.layer_specs)

    def __getitem__(self, feature_id: str) -> NGIIFeature:
        if feature_id in self._ambiguous_global_ids:
            msg = f"feature id {feature_id!r} exists in multiple NGII layers"
            raise AmbiguousFeatureIDError(msg)
        return self._global_index[feature_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._global_index)

    def __len__(self) -> int:
        return len(self._global_index)

    def bind(self) -> None:
        self._global_index.clear()
        self._ambiguous_global_ids.clear()
        first_by_id: dict[str, NGIIFeature] = {}
        for store in self.layer_stores:
            store.rebuild_index()
            for feature in store.features:
                feature.bind_dataset(self)
                existing = first_by_id.get(feature.id)
                if existing is None:
                    first_by_id[feature.id] = feature
                    self._global_index[feature.id] = feature
                elif existing.layer_name != feature.layer_name:
                    self._global_index.pop(feature.id, None)
                    already_ambiguous = feature.id in self._ambiguous_global_ids
                    self._ambiguous_global_ids.add(feature.id)
                    if self.warn_global_id_collision and not already_ambiguous:
                        self.sanity.warn(
                            "global-id-collision",
                            f"ID {feature.id!r} exists in both {existing.layer_name} and "
                            f"{feature.layer_name}; dataset[id] is ambiguous",
                            feature_id=feature.id,
                            layer_name=feature.layer_name,
                            source_path=feature.source_path,
                        )

    def store_for_attr(self, attr: str) -> LayerStore[Any]:
        try:
            return self._stores[attr]
        except KeyError:
            msg = f"{attr!r} is not an NGII layer store in schema {self.schema.version}"
            raise KeyError(msg) from None

    def store_for_role(self, role: LayerRole) -> LayerStore[Any]:
        return self.store_for_attr(self.schema.attr_for_role(role))

    def store_for_layer_name(self, layer_name: str) -> LayerStore[Any] | None:
        attr = self.schema.attr_for_layer_name(layer_name)
        return None if attr is None else self.store_for_attr(attr)

    def feature_ref_attr(self, feature: NGIIFeature) -> str | None:
        return self.schema.attr_for_layer_name(feature.layer_name)

    def __getattr__(self, name: str) -> LayerStore[Any]:
        try:
            stores = cast(dict[str, LayerStore[Any]], object.__getattribute__(self, "_stores"))
        except AttributeError:
            stores = {}
        if name in stores:
            return stores[name]
        msg = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(msg)


def _array_rule_for(spec: LayerSpec, name: str) -> FieldRule | None:
    for rule in spec.field_rules:
        if rule.name == "ID":
            continue
        if name in _array_names_for_rule(rule):
            return rule
    return None


def _array_names_for_rule(rule: FieldRule) -> tuple[str, ...]:
    names = list(rule.array_aliases or (_array_name(rule.attr),))
    for column_alias in rule.column_aliases:
        alias_attr = column_to_attr(column_alias)
        alias_name = _array_name(alias_attr)
        if alias_attr != rule.attr and alias_name not in names:
            names.append(alias_name)
    return tuple(names)


def _array_name(attr: str) -> str:
    for suffix in _ARRAY_UNIT_SUFFIXES:
        if attr.endswith(suffix):
            return f"{attr[: -len(suffix)]}s{suffix}"
    return f"{attr}s"
