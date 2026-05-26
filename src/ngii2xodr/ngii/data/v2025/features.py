"""2025.12 NGII feature base classes and common field parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ngii2xodr.ngii.data.features import (
    FeatureRecord,
    LineFeature,
    PointFeature,
    PolygonFeature,
    base_kwargs,
)


@dataclass(slots=True)
class V2025PointFeature(PointFeature):
    survey_date: str
    version: str
    remark: str
    hist_type: str


@dataclass(slots=True)
class V2025LineFeature(LineFeature):
    survey_date: str
    version: str
    remark: str
    hist_type: str


@dataclass(slots=True)
class V2025PolygonFeature(PolygonFeature):
    survey_date: str
    version: str
    remark: str
    hist_type: str


def common_kwargs(record: FeatureRecord) -> dict[str, Any]:
    return {
        **base_kwargs(record),
        "survey_date": record.text("SurveyDate"),
        "version": record.text("Version"),
        "remark": record.text("Remark"),
        "hist_type": record.text("HistType"),
    }
