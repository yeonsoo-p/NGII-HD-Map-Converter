"""Canonical in-memory NGII dataset tree."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from ngii2xodr.profile import PerformanceProfile

_ARRAY_ATTRS = {
    "node_types": "node_type",
    "its_node_ids": "its_node_id",
    "road_ranks": "road_rank",
    "road_types": "road_type",
    "road_nos": "road_no",
    "remarks": "remark",
    "link_types": "link_type",
    "lane_nos": "lane_no",
    "r_link_ids": "r_link_id",
    "l_link_ids": "l_link_id",
    "from_node_ids": "from_node_id",
    "to_node_ids": "to_node_id",
    "section_ids": "section_id",
    "lengths_m": "length_m",
    "its_link_ids": "its_link_id",
    "kinds": "kind",
    "subtypes": "subtype",
    "names": "name",
    "directions": "direction",
    "gas_stations": "gas_station",
    "lpg_stations": "lpg_station",
    "ev_chargers": "ev_charger",
    "toilets": "toilet",
    "types": "type",
    "is_central": "is_central",
    "low_high": "low_high",
    "ref_ids": "ref_id",
}


class AmbiguousFeatureIDError(KeyError):
    """Raised when global ``dataset[id]`` would cross layer boundaries."""


@dataclass(slots=True)
class LayerStore[T: NGIIFeature]:
    layer_name: str
    features: list[T] = field(default_factory=list)
    by_id: dict[str, T] = field(default_factory=dict)
    id_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rebuild_index()

    def __getitem__(self, feature_id: str) -> T:
        return self.by_id[feature_id]

    def __iter__(self) -> Iterator[T]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def get(self, feature_id: str | None) -> T | None:
        if feature_id is None:
            return None
        return self.by_id.get(feature_id)

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
        attr = _ARRAY_ATTRS.get(name)
        if attr is None:
            msg = f"{type(self).__name__!s} has no attribute {name!r}"
            raise AttributeError(msg)
        values = [getattr(feature, attr) for feature in self.features]
        if attr in {"lane_no"}:
            return np.asarray(values, dtype=np.int32)
        if attr in {"length_m"}:
            return np.asarray(values, dtype=np.float64)
        return np.asarray(["" if value is None else value for value in values], dtype=np.str_)


@dataclass(slots=True)
class NGIIDataset:
    root: Path
    coordinate: str
    sanity: SanityReport
    warn_global_id_collision: bool = True
    load_profile: PerformanceProfile = field(default_factory=PerformanceProfile)
    a1_node: LayerStore[Any] = field(default_factory=lambda: LayerStore("A1_NODE"))
    a2_link: LayerStore[Any] = field(default_factory=lambda: LayerStore("A2_LINK"))
    a3_drivewaysection: LayerStore[Any] = field(
        default_factory=lambda: LayerStore("A3_DRIVEWAYSECTION")
    )
    a4_subsidiarysection: LayerStore[Any] = field(
        default_factory=lambda: LayerStore("A4_SUBSIDIARYSECTION")
    )
    a5_parkinglot: LayerStore[Any] = field(default_factory=lambda: LayerStore("A5_PARKINGLOT"))
    b1_safetysign: LayerStore[Any] = field(default_factory=lambda: LayerStore("B1_SAFETYSIGN"))
    b2_surfacelinemark: LayerStore[Any] = field(
        default_factory=lambda: LayerStore("B2_SURFACELINEMARK")
    )
    b3_surfacemark: LayerStore[Any] = field(default_factory=lambda: LayerStore("B3_SURFACEMARK"))
    c1_trafficlight: LayerStore[Any] = field(default_factory=lambda: LayerStore("C1_TRAFFICLIGHT"))
    c2_kilopost: LayerStore[Any] = field(default_factory=lambda: LayerStore("C2_KILOPOST"))
    c3_vehicleprotectionsafety: LayerStore[Any] = field(
        default_factory=lambda: LayerStore("C3_VEHICLEPROTECTIONSAFETY")
    )
    c4_speedbump: LayerStore[Any] = field(default_factory=lambda: LayerStore("C4_SPEEDBUMP"))
    c5_heightbarrier: LayerStore[Any] = field(
        default_factory=lambda: LayerStore("C5_HEIGHTBARRIER")
    )
    c6_postpoint: LayerStore[Any] = field(default_factory=lambda: LayerStore("C6_POSTPOINT"))
    _global_index: dict[str, NGIIFeature] = field(default_factory=dict, init=False)
    _ambiguous_global_ids: set[str] = field(default_factory=set, init=False)

    @property
    def layer_stores(self) -> tuple[LayerStore[Any], ...]:
        return (
            self.a1_node,
            self.a2_link,
            self.a3_drivewaysection,
            self.a4_subsidiarysection,
            self.a5_parkinglot,
            self.b1_safetysign,
            self.b2_surfacelinemark,
            self.b3_surfacemark,
            self.c1_trafficlight,
            self.c2_kilopost,
            self.c3_vehicleprotectionsafety,
            self.c4_speedbump,
            self.c5_heightbarrier,
            self.c6_postpoint,
        )

    def __getitem__(self, feature_id: str) -> NGIIFeature:
        if feature_id in self._ambiguous_global_ids:
            msg = f"feature id {feature_id!r} exists in multiple NGII layers"
            raise AmbiguousFeatureIDError(msg)
        return self._global_index[feature_id]

    def bind(self) -> None:
        self._global_index.clear()
        self._ambiguous_global_ids.clear()
        for store in self.layer_stores:
            store.rebuild_index()
            for feature in store.features:
                feature.bind_dataset(self)
                existing = self._global_index.get(feature.id)
                if existing is None:
                    self._global_index[feature.id] = feature
                elif existing.layer_name != feature.layer_name:
                    self._ambiguous_global_ids.add(feature.id)
                    if self.warn_global_id_collision:
                        self.sanity.warn(
                            "global-id-collision",
                            f"ID {feature.id!r} exists in both {existing.layer_name} and "
                            f"{feature.layer_name}; dataset[id] is ambiguous",
                            feature_id=feature.id,
                            layer_name=feature.layer_name,
                            source_path=feature.source_path,
                        )

    def store_for_attr(self, attr: str) -> LayerStore[Any]:
        store = getattr(self, attr)
        if not isinstance(store, LayerStore):
            msg = f"{attr!r} is not an NGII layer store"
            raise TypeError(msg)
        return store
