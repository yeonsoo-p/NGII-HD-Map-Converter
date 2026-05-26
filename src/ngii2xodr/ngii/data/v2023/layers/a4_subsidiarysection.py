from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2023.features import V2023PolygonFeature, common_kwargs


@dataclass(slots=True)
class A4_SUBSIDIARYSECTION(V2023PolygonFeature):
    layer_name: ClassVar[str] = "A4_SUBSIDIARYSECTION"
    SUBTYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "휴게소",
        "2": "졸음쉼터",
        "3": "보도",
        "4": "자전거도로",
        "9": "기타부속구간",
    }
    DIRECTION_LABEL: ClassVar[dict[str, str]] = {
        "1": "상행",
        "2": "하행",
        "3": "양방향",
    }
    PRESENCE_LABEL: ClassVar[dict[str, str]] = {
        "0": "미존재",
        "1": "존재",
    }

    subtype: str
    name: str
    direction: str
    gas_station: str
    lpg_station: str
    ev_charger: str
    toilet: str


def make_feature(record: FeatureRecord) -> A4_SUBSIDIARYSECTION:
    return A4_SUBSIDIARYSECTION(
        **common_kwargs(record),
        ring=record.geometry,
        subtype=record.text("SubType"),
        name=record.text("Name"),
        direction=record.text("Direction"),
        gas_station=record.text("GasStation"),
        lpg_station=record.text("LpgStation"),
        ev_charger=record.text("EvCharger"),
        toilet=record.text("Toilet"),
    )
