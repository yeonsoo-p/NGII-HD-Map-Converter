"""Core NGII rewrite feature objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii_rewrite.data.geometry import xy_line

FeatureGeometryKind = Literal["point", "line", "polygon"]
_INTEGER_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")


def optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_integer_value(value: object) -> bool:
    if value == "":
        return True
    if isinstance(value, Real):
        return float(value).is_integer()
    value_text = optional_text(value)
    if not value_text:
        return True
    if _INTEGER_RE.fullmatch(value_text):
        return True
    if not _FLOAT_RE.fullmatch(value_text):
        return False
    return float(value_text).is_integer()


def is_float_value(value: object) -> bool:
    if value == "":
        return True
    if isinstance(value, Real):
        return True
    value_text = optional_text(value)
    return not value_text or _FLOAT_RE.fullmatch(value_text) is not None


def _integer_or_default(value: object, default: int) -> int:
    if value == "":
        return default
    if isinstance(value, Real):
        value_float = float(value)
        return int(value_float) if value_float.is_integer() else default
    value_text = optional_text(value)
    if not value_text:
        return default
    if _INTEGER_RE.fullmatch(value_text):
        return int(value_text)
    if _FLOAT_RE.fullmatch(value_text):
        value_float = float(value_text)
        return int(value_float) if value_float.is_integer() else default
    return default


def _float_or_default(value: object, default: float) -> float:
    if value == "":
        return default
    if isinstance(value, Real):
        return float(value)
    value_text = optional_text(value)
    if not value_text or not _FLOAT_RE.fullmatch(value_text):
        return default
    return float(value_text)


@dataclass(slots=True, frozen=True)
class FeatureRef:
    layer_attr: str
    feature_id: str


@dataclass(slots=True, frozen=True)
class ResolvedReference:
    source_ref: FeatureRef
    target_ref: FeatureRef
    source_column: str
    source_attr: str
    required: bool


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
        return default if value == "" else optional_text(value)

    def optional_ref(self, column: str) -> str | None:
        return self.text(column) or None

    def integer(self, column: str, default: int = -1) -> int:
        return _integer_or_default(self.attributes.get(column, ""), default)

    def floating(self, column: str, default: float = float("nan")) -> float:
        return _float_or_default(self.attributes.get(column, ""), default)


@dataclass(slots=True)
class NGIIFeature:
    id: str
    source_path: Path = field(compare=False)
    source_row: int = field(compare=False)
    references: tuple[ResolvedReference, ...] = field(
        default_factory=tuple, init=False, compare=False
    )
    referenced_by: tuple[ResolvedReference, ...] = field(
        default_factory=tuple, init=False, compare=False
    )
    undocumented_attributes: dict[str, Any] = field(default_factory=dict, init=False, compare=False)
    layer_name: ClassVar[str]
    layer_attr: ClassVar[str]
    filename: ClassVar[str]
    filename_aliases: ClassVar[tuple[str, ...]] = ()
    id_max_length: ClassVar[int]
    required_layer: ClassVar[bool]
    roles: ClassVar[tuple[str, ...]] = ()

    @property
    def geometry_kind(self) -> FeatureGeometryKind:
        msg = f"{self.layer_name} {self.id!r} has no supported geometry"
        raise TypeError(msg)

    @property
    def point_xyz(self) -> NDArray[np.float64] | None:
        return None

    @property
    def polygon_ring(self) -> NDArray[np.float64] | None:
        return None

    @property
    def points(self) -> NDArray[np.float64] | None:
        return None

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        msg = f"{self.layer_name} {self.id!r} has no supported geometry signature"
        raise TypeError(msg)


class PointFeature:
    __slots__ = ()

    point: NDArray[np.float64] | None

    @property
    def geometry_kind(self) -> FeatureGeometryKind:
        return "point"

    @property
    def point_xyz(self) -> NDArray[np.float64] | None:
        return self.point

    @property
    def points(self) -> NDArray[np.float64] | None:
        if self.point is None:
            return None
        return self.point.reshape(1, 3)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        if self.point is None:
            return ()
        return (tuple(float(v) for v in self.point),)


class LineFeature:
    __slots__ = ()

    polyline: NDArray[np.float64]

    @property
    def geometry_kind(self) -> FeatureGeometryKind:
        return "line"

    @property
    def points(self) -> NDArray[np.float64]:
        return self.polyline

    @property
    def xy_line(self) -> shapely.LineString:
        return xy_line(self.polyline)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in row) for row in self.polyline)


class PolygonFeature:
    __slots__ = ()

    ring: NDArray[np.float64] | None

    @property
    def geometry_kind(self) -> FeatureGeometryKind:
        return "polygon"

    @property
    def polygon_ring(self) -> NDArray[np.float64] | None:
        return self.ring

    @property
    def points(self) -> NDArray[np.float64] | None:
        return self.ring

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        if self.ring is None:
            return ()
        return tuple(tuple(float(v) for v in row) for row in self.ring)


def same_feature(a: NGIIFeature, b: NGIIFeature) -> bool:
    return type(a) is type(b) and a == b and a.geometry_signature == b.geometry_signature


__all__ = [
    "FeatureGeometryKind",
    "FeatureRecord",
    "FeatureRef",
    "LineFeature",
    "NGIIFeature",
    "PointFeature",
    "PolygonFeature",
    "ResolvedReference",
    "is_float_value",
    "is_integer_value",
    "optional_text",
    "same_feature",
]
