"""Canonical in-memory NGII dataset tree."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.data.features import (
    FeatureGeometryKind,
    FeatureRef,
    LineFeature,
    NGIIFeature,
    ResolvedRelationship,
)
from ngii2xodr.ngii.data.geometry import xy_line
from ngii2xodr.ngii.data.schema import (
    LayerRole,
    LayerSpec,
    SchemaDefinition,
)
from ngii2xodr.profile import PerformanceProfile

if TYPE_CHECKING:
    from ngii2xodr.ngii.data.sanity import SanityReport


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
            point = feature.point_xyz
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
            ring = feature.polygon_ring
            if ring is not None:
                rings.append(ring)
        return rings

    @property
    def observed_geometry_kinds(self) -> tuple[FeatureGeometryKind, ...]:
        if not self.features:
            return ()
        present = {feature.geometry_kind for feature in self.features}
        return tuple(kind for kind in self.spec.geometry_kinds if kind in present)

    def feature_indices_for_geometry_kind(self, kind: FeatureGeometryKind) -> tuple[int, ...]:
        return tuple(
            index for index, feature in enumerate(self.features) if feature.geometry_kind == kind
        )

    def related_feature(
        self, feature: T, column_or_attr: str, dataset: NGIIDataset
    ) -> NGIIFeature | None:
        relationship = next(
            (
                item
                for item in self.spec.relationships
                if column_or_attr in {item.column_name, item.source_attr}
            ),
            None,
        )
        if relationship is None:
            return None
        value = getattr(feature, relationship.source_attr, "")
        if value is None:
            return None
        feature_id = str(value)
        if not feature_id:
            return None
        for target_attr in relationship.target_attrs:
            related = dataset.store_for_attr(target_attr).get(feature_id)
            if isinstance(related, NGIIFeature):
                return related
        return None

    def iter_text_fields(self, feature: T) -> Iterator[tuple[str, str]]:
        for rule in self.spec.field_rules:
            if rule.name == "ID" or rule.field_type != "text":
                continue
            attr_name = rule.attr
            if not hasattr(feature, attr_name):
                continue
            value = getattr(feature, attr_name)
            if isinstance(value, str):
                yield rule.name, value

    def value_for_column(self, feature: T, column_name: str) -> Any:
        attr_name = self._attr_for_column(column_name)
        if attr_name and hasattr(feature, attr_name):
            return getattr(feature, attr_name)
        return ""

    def set_column(self, feature: T, column_name: str, value: Any) -> None:
        attr_name = self._attr_for_column(column_name)
        if attr_name and hasattr(feature, attr_name):
            setattr(feature, attr_name, value)

    def _attr_for_column(self, column_name: str) -> str:
        for rule in self.spec.field_rules:
            if rule.name == column_name:
                return rule.attr
        for relationship in self.spec.relationships:
            if relationship.column_name == column_name:
                return relationship.source_attr
        return ""


@dataclass(slots=True)
class NGIIDataset(Mapping[str, NGIIFeature]):
    root: Path
    coordinate: str
    schema: SchemaDefinition
    load_profile: PerformanceProfile = field(default_factory=PerformanceProfile)
    _stores: dict[str, LayerStore[Any]] = field(default_factory=dict, init=False)
    _global_index: dict[str, NGIIFeature] = field(default_factory=dict, init=False)
    _ambiguous_global_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._stores = {spec.python_attr: LayerStore(spec) for spec in self.schema.layer_specs}

    @property
    def layer_stores(self) -> tuple[LayerStore[Any], ...]:
        return tuple(self._stores[spec.python_attr] for spec in self.schema.layer_specs)

    @property
    def layer_items(self) -> tuple[tuple[str, LayerStore[Any]], ...]:
        return tuple(
            (spec.python_attr, self._stores[spec.python_attr]) for spec in self.schema.layer_specs
        )

    def __getitem__(self, feature_id: str) -> NGIIFeature:
        if feature_id in self._ambiguous_global_ids:
            msg = f"feature id {feature_id!r} exists in multiple NGII layers"
            raise AmbiguousFeatureIDError(msg)
        return self._global_index[feature_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._global_index)

    def __len__(self) -> int:
        return len(self._global_index)

    def bind(self, sanity: SanityReport, *, warn_global_id_collision: bool = True) -> None:
        self._global_index.clear()
        self._ambiguous_global_ids.clear()
        first_by_id: dict[str, NGIIFeature] = {}
        for store in self.layer_stores:
            store.rebuild_index()
            for feature in store.features:
                existing = first_by_id.get(feature.id)
                if existing is None:
                    first_by_id[feature.id] = feature
                    self._global_index[feature.id] = feature
                elif existing.layer_name != feature.layer_name:
                    self._global_index.pop(feature.id, None)
                    already_ambiguous = feature.id in self._ambiguous_global_ids
                    self._ambiguous_global_ids.add(feature.id)
                    if warn_global_id_collision and not already_ambiguous:
                        sanity.warn(
                            "global-id-collision",
                            f"ID {feature.id!r} exists in both {existing.layer_name} and "
                            f"{feature.layer_name}; dataset[id] is ambiguous",
                            feature_id=feature.id,
                            layer_name=feature.layer_name,
                            source_path=feature.source_path,
                        )

    def rebuild_relationship_edges(self) -> None:
        outgoing: dict[FeatureRef, list[ResolvedRelationship]] = {}
        incoming: dict[FeatureRef, list[ResolvedRelationship]] = {}
        for attr, store in self.layer_items:
            for feature in store.features:
                source_ref = FeatureRef(attr, feature.id)
                for relationship in store.spec.relationships:
                    target_ref = self._resolve_relationship_ref(
                        feature, relationship.source_attr, relationship.target_attrs
                    )
                    if target_ref is None:
                        continue
                    edge = ResolvedRelationship(
                        source_ref=source_ref,
                        target_ref=target_ref,
                        source_column=relationship.column_name,
                        source_attr=relationship.source_attr,
                        required=relationship.required,
                    )
                    outgoing.setdefault(source_ref, []).append(edge)
                    incoming.setdefault(target_ref, []).append(edge)

        for attr, store in self.layer_items:
            for feature in store.features:
                feature_ref = FeatureRef(attr, feature.id)
                feature.references = tuple(outgoing.get(feature_ref, ()))
                feature.referenced_by = tuple(incoming.get(feature_ref, ()))

    def _resolve_relationship_ref(
        self, feature: NGIIFeature, source_attr: str, target_attrs: tuple[str, ...]
    ) -> FeatureRef | None:
        value = getattr(feature, source_attr, None)
        if value is None:
            return None
        feature_id = str(value).strip()
        if not feature_id:
            return None
        matches = tuple(
            FeatureRef(target_attr, feature_id)
            for target_attr in target_attrs
            if self.store_for_attr(target_attr).get(feature_id) is not None
        )
        return matches[0] if len(matches) == 1 else None

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
