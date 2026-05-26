from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025LineFeature, common_kwargs


@dataclass(slots=True)
class SF1_BARRIER(V2025LineFeature):
    layer_name: ClassVar[str] = "SF1_BARRIER"
    BARR_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "콘크리트방호벽",
        "101": "가드레일",
        "102": "펜스",
        "103": "임시구조물",
        "200": "벽",
        "300": "충격흡수시설",
        "301": "중앙분리대 개구부",
        "999": "기타",
    }

    barr_type: str
    r_link_id: str | None
    l_link_id: str | None


def _optional_id(value: str) -> str | None:
    return value or None


def make_feature(record: FeatureRecord) -> SF1_BARRIER:
    return SF1_BARRIER(
        **common_kwargs(record),
        polyline=record.geometry,
        barr_type=record.text("BarrType"),
        r_link_id=_optional_id(record.text("R_LinkID")),
        l_link_id=_optional_id(record.text("L_LinkID")),
    )
