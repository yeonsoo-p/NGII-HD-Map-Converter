"""Base feature objects and parser records for NGII layer modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.data.geometry import xy_line

FeatureGeometryKind = Literal["point", "line", "polygon"]


@dataclass(slots=True, frozen=True)
class FeatureRecord:
    layer_name: str
    source_path: Path = field(compare=False)
    source_row: int = field(compare=False)
    attributes: dict[str, Any]
    geometry_kind: FeatureGeometryKind
    geometry: NDArray[np.float64]

    def text(self, column: str, default: str = "") -> str:
        value = self.attributes.get(column, default)
        return default if value == "" else str(value)

    def integer(self, column: str, default: int = -1) -> int:
        value = self.attributes.get(column, "")
        if value == "":
            return default
        return int(value)

    def floating(self, column: str, default: float = float("nan")) -> float:
        value = self.attributes.get(column, "")
        if value == "":
            return default
        return float(value)


@dataclass(slots=True)
class NGIIFeature:
    id: str
    source_path: Path = field(compare=False)
    source_row: int = field(compare=False)
    layer_name: ClassVar[str]

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        msg = f"{self.layer_name} {self.id!r} has no supported geometry signature"
        raise TypeError(msg)


@dataclass(slots=True)
class PointFeature(NGIIFeature):
    point: NDArray[np.float64] = field(compare=False)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        return (tuple(float(v) for v in self.point),)


@dataclass(slots=True)
class LineFeature(NGIIFeature):
    polyline: NDArray[np.float64] = field(compare=False)

    @property
    def xy_line(self) -> shapely.LineString:
        return xy_line(self.polyline)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in row) for row in self.polyline)


@dataclass(slots=True)
class PolygonFeature(NGIIFeature):
    ring: NDArray[np.float64] = field(compare=False)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in row) for row in self.ring)


@dataclass(slots=True)
class PointOrPolygonFeature(NGIIFeature):
    point: NDArray[np.float64] | None = field(compare=False)
    ring: NDArray[np.float64] | None = field(compare=False)

    @property
    def geometry_kind(self) -> Literal["point", "polygon"]:
        if self.point is not None:
            return "point"
        if self.ring is not None:
            return "polygon"
        msg = f"{self.layer_name} {self.id!r} has neither point nor polygon geometry"
        raise ValueError(msg)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        if self.point is not None:
            return (tuple(float(v) for v in self.point),)
        if self.ring is not None:
            return tuple(tuple(float(v) for v in row) for row in self.ring)
        msg = f"{self.layer_name} {self.id!r} has neither point nor polygon geometry"
        raise ValueError(msg)


def base_kwargs(record: FeatureRecord) -> dict[str, Any]:
    return {
        "id": record.text("ID"),
        "source_path": record.source_path,
        "source_row": record.source_row,
    }


def feature_geometry_kind(feature: NGIIFeature) -> FeatureGeometryKind:
    if isinstance(feature, PointFeature):
        return "point"
    if isinstance(feature, LineFeature):
        return "line"
    if isinstance(feature, PolygonFeature):
        return "polygon"
    if isinstance(feature, PointOrPolygonFeature):
        return feature.geometry_kind
    msg = f"{feature.layer_name} {feature.id!r} has no supported geometry"
    raise TypeError(msg)


def feature_point_xyz(feature: NGIIFeature) -> NDArray[np.float64] | None:
    if isinstance(feature, PointFeature):
        return feature.point
    if isinstance(feature, PointOrPolygonFeature):
        return feature.point
    return None


def feature_polygon_ring(feature: NGIIFeature) -> NDArray[np.float64] | None:
    if isinstance(feature, PolygonFeature):
        return feature.ring
    if isinstance(feature, PointOrPolygonFeature):
        return feature.ring
    return None


def feature_points(feature: NGIIFeature) -> NDArray[np.float64] | None:
    point = feature_point_xyz(feature)
    if point is not None:
        return point.reshape(1, 3)
    if isinstance(feature, LineFeature):
        return feature.polyline
    ring = feature_polygon_ring(feature)
    if ring is not None:
        return ring
    return None


def same_feature(a: NGIIFeature, b: NGIIFeature) -> bool:
    return type(a) is type(b) and a == b and a.geometry_signature == b.geometry_signature
