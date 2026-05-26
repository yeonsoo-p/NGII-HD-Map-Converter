from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PolygonFeature, common_kwargs


@dataclass(slots=True)
class RM3_PARKINGLOT(V2025PolygonFeature):
    layer_name: ClassVar[str] = "RM3_PARKINGLOT"
    PL_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "일반차량",
        "101": "우선주차",
        "102": "대형차량",
        "200": "장애인",
        "201": "전기차(충전)",
        "202": "환경친화(미충전)",
        "999": "기타",
    }

    pl_type: str


def make_feature(record: FeatureRecord) -> RM3_PARKINGLOT:
    return RM3_PARKINGLOT(
        **common_kwargs(record),
        ring=record.geometry,
        pl_type=record.text("PLType"),
    )
