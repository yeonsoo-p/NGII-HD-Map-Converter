from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PolygonFeature, common_kwargs


@dataclass(slots=True)
class RS3_SUBSIDIARYSECTION(V2025PolygonFeature):
    layer_name: ClassVar[str] = "RS3_SUBSIDIARYSECTION"
    SUBS_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "휴게소",
        "101": "졸음쉼터",
        "999": "기타",
    }
    PRESENCE_LABEL: ClassVar[dict[str, str]] = {"0": "미존재", "1": "존재"}

    subs_type: str
    gas_station: str
    lpg_station: str
    ev_charger: str


def make_feature(record: FeatureRecord) -> RS3_SUBSIDIARYSECTION:
    return RS3_SUBSIDIARYSECTION(
        **common_kwargs(record),
        ring=record.geometry,
        subs_type=record.text("SubsType"),
        gas_station=record.text("GasStation"),
        lpg_station=record.text("LPGStation"),
        ev_charger=record.text("EVCharger"),
    )
