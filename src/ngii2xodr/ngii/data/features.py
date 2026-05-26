"""Base feature objects and parser records for NGII layer modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import shapely
from numpy.typing import NDArray

from ngii2xodr.ngii.data.geometry import xy_line

if TYPE_CHECKING:
    from ngii2xodr.ngii.data.dataset import NGIIDataset

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
    admin_code: str
    maker: str
    update_date: str
    survey_date: str
    version: str
    remark: str
    hist_type: str
    hist_remark: str
    source_path: Path = field(compare=False)
    source_row: int = field(compare=False)
    layer_name: ClassVar[str]
    _dataset: NGIIDataset | None = field(default=None, init=False, repr=False, compare=False)

    def bind_dataset(self, dataset: NGIIDataset) -> None:
        self._dataset = dataset

    def _require_dataset(self) -> NGIIDataset:
        if self._dataset is None:
            msg = f"{self.layer_name} {self.id!r} is not bound to an NGII dataset"
            raise RuntimeError(msg)
        return self._dataset

    def resolve_relation(self, column_or_attr: str) -> NGIIFeature | None:
        """Resolve one documented relationship through the bound dataset schema."""
        dataset = self._require_dataset()
        spec = dataset.schema.spec_for_layer_name(self.layer_name)
        if spec is None:
            return None
        relationship = next(
            (
                item
                for item in spec.relationships
                if column_or_attr in {item.column_name, item.source_attr}
            ),
            None,
        )
        if relationship is None:
            return None
        value = getattr(self, relationship.source_attr, "")
        if value is None:
            return None
        feature_id = str(value)
        if not feature_id:
            return None
        for target_attr in relationship.target_attrs:
            feature = dataset.store_for_attr(target_attr).get(feature_id)
            if feature is not None:
                return cast(NGIIFeature, feature)
        return None

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


def common_kwargs(record: FeatureRecord) -> dict[str, Any]:
    return {
        "id": record.text("ID"),
        "admin_code": record.text("AdminCode"),
        "maker": record.text("Maker"),
        "update_date": record.text("UpdateDate"),
        "survey_date": record.text("SurveyDate"),
        "version": record.text("Version"),
        "remark": record.text("Remark"),
        "hist_type": record.text("HistType"),
        "hist_remark": record.text("HistRemark"),
        "source_path": record.source_path,
        "source_row": record.source_row,
    }


def same_feature(a: NGIIFeature, b: NGIIFeature) -> bool:
    return type(a) is type(b) and a == b and a.geometry_signature == b.geometry_signature


def relation_property(column_or_attr: str) -> property:
    def _resolve(self: NGIIFeature) -> NGIIFeature | None:
        return self.resolve_relation(column_or_attr)

    return property(_resolve)
