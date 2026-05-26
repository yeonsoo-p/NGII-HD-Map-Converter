from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PolygonFeature, common_kwargs


@dataclass(slots=True)
class A3_DRIVEWAYSECTION(PolygonFeature):
    layer_name: ClassVar[str] = "A3_DRIVEWAYSECTION"
    KIND_LABEL: ClassVar[dict[str, str]] = {
        "1": "주행구간",
        "7": "자율주행금지구간",
    }
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반도로",
        "2": "터널",
        "3": "교량",
        "4": "지하도로",
        "5": "고가도로",
    }

    kind: str
    road_type: str


def make_feature(record: FeatureRecord) -> A3_DRIVEWAYSECTION:
    return A3_DRIVEWAYSECTION(
        **common_kwargs(record),
        ring=record.geometry,
        kind=record.text("Kind"),
        road_type=record.text("RoadType"),
    )
