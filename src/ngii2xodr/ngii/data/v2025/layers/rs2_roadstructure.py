from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PolygonFeature, common_kwargs


@dataclass(slots=True)
class RS2_ROADSTRUCTURE(V2025PolygonFeature):
    layer_name: ClassVar[str] = "RS2_ROADSTRUCTURE"
    RS_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "교량",
        "101": "고가차도",
        "102": "터널",
        "103": "지하차도",
        "200": "보호구역",
        "300": "톨게이트",
    }

    rs_type: str


def make_feature(record: FeatureRecord) -> RS2_ROADSTRUCTURE:
    return RS2_ROADSTRUCTURE(
        **common_kwargs(record),
        ring=record.geometry,
        rs_type=record.text("RSType"),
    )
