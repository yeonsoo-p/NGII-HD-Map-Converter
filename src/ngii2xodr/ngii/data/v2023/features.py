"""2023.07 NGII feature base classes and common field parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ngii2xodr.ngii.data.features import (
    FeatureRecord,
    LineFeature,
    PointFeature,
    PointOrPolygonFeature,
    PolygonFeature,
    base_kwargs,
)


@dataclass(slots=True)
class V2023PointFeature(PointFeature):
    admin_code: str
    maker: str
    update_date: str
    version: str
    remark: str
    hist_type: str
    hist_remark: str


@dataclass(slots=True)
class V2023LineFeature(LineFeature):
    admin_code: str
    maker: str
    update_date: str
    version: str
    remark: str
    hist_type: str
    hist_remark: str


@dataclass(slots=True)
class V2023PolygonFeature(PolygonFeature):
    admin_code: str
    maker: str
    update_date: str
    version: str
    remark: str
    hist_type: str
    hist_remark: str


@dataclass(slots=True)
class V2023PointOrPolygonFeature(PointOrPolygonFeature):
    admin_code: str
    maker: str
    update_date: str
    version: str
    remark: str
    hist_type: str
    hist_remark: str


def common_kwargs(record: FeatureRecord) -> dict[str, Any]:
    return {
        **base_kwargs(record),
        "admin_code": record.text("AdminCode"),
        "maker": record.text("Maker"),
        "update_date": record.text("UpdateDate"),
        "version": record.text("Version"),
        "remark": record.text("Remark"),
        "hist_type": record.text("HistType"),
        "hist_remark": record.text("HistRemark"),
    }
