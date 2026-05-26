from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PointFeature, common_kwargs


@dataclass(slots=True)
class SF3_TRAFFICLIGHT(PointFeature):
    layer_name: ClassVar[str] = "SF3_TRAFFICLIGHT"
    LIGHT_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "차량가로형-삼색등",
        "101": "차량가로형-사색등A",
        "102": "차량가로형-사색등B",
        "103": "차량가로형-화살표삼색등",
        "110": "차량세로형-삼색등",
        "111": "차량세로형-우회전삼색등",
        "112": "차량세로형-사색등",
        "120": "버스삼색등",
        "121": "노면전차육구등",
        "130": "가로형이색등",
        "140": "차량보조등-세로형삼색등",
        "141": "차량보조등-세로형사색등",
        "200": "가변등",
        "201": "경보형경보등",
        "300": "보행등",
        "400": "자전거세로형-삼색등",
        "401": "자전거세로형-이색등",
        "999": "기타 신호등 유형",
    }

    light_type: str
    post_id: str | None


def make_feature(record: FeatureRecord) -> SF3_TRAFFICLIGHT:
    post_id = record.text("PostID")
    return SF3_TRAFFICLIGHT(
        **common_kwargs(record),
        point=record.geometry,
        light_type=record.text("LightType"),
        post_id=post_id or None,
    )
