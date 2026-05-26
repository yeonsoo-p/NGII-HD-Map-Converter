from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, relation_property
from ngii2xodr.ngii.data.v2023.features import V2023LineFeature, common_kwargs


@dataclass(slots=True)
class C3_VEHICLEPROTECTIONSAFETY(V2023LineFeature):
    layer_name: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "2": "가드레일",
        "3": "콘크리트방호벽",
        "4": "콘크리트연석",
        "5": "무단횡단방지시설",
        "6": "중앙분리대개구부",
        "7": "임시구조물",
        "8": "벽",
        "99": "기타",
    }
    IS_CENTRAL_LABEL: ClassVar[dict[str, str]] = {
        "0": "도로 가장자리",
        "1": "중앙분리대",
    }
    LOW_HIGH_LABEL: ClassVar[dict[str, str]] = {
        "1": "상단",
        "2": "하단",
    }

    type: str
    is_central: str
    low_high: str
    ref_id: str | None

    reference = relation_property("Ref_ID")


def make_feature(record: FeatureRecord) -> C3_VEHICLEPROTECTIONSAFETY:
    ref_id = record.text("Ref_ID")
    return C3_VEHICLEPROTECTIONSAFETY(
        **common_kwargs(record),
        polyline=record.geometry,
        type=record.text("Type"),
        is_central=record.text("IsCentral"),
        low_high=record.text("LowHigh"),
        ref_id=ref_id or None,
    )
