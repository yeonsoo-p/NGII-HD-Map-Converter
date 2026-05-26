from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PolygonFeature, common_kwargs


@dataclass(slots=True)
class RM2_ROADMARKING(PolygonFeature):
    layer_name: ClassVar[str] = "RM2_ROADMARKING"
    MARK_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "화살표",
        "200": "횡단보도",
        "300": "도형",
        "500": "구역",
        "999": "기타",
    }
    MARK_KIND_LABEL: ClassVar[dict[str, str]] = {
        "522": "양보",
        "5232": "버스정차구획",
        "5233": "택시정차구획",
        "524": "정차금지대",
        "529": "횡단보도 예고",
        "5321": "횡단보도",
        "533": "고원식횡단보도",
        "534": "자전거횡단보도",
        "5371": "직진",
        "5372": "좌회전",
        "5373": "우회전",
        "5374": "좌우회전",
        "5379": "전방향",
        "5381": "직진 및 좌회전",
        "5382": "직진 및 우회전",
        "5391": "유턴",
        "5392": "좌회전 및 유턴",
        "5431": "차로변경(좌로합류)",
        "5432": "차로변경(우로합류)",
        "999": "기타 노면표시",
    }

    mark_type: str
    mark_kind: str


def make_feature(record: FeatureRecord) -> RM2_ROADMARKING:
    return RM2_ROADMARKING(
        **common_kwargs(record),
        ring=record.geometry,
        mark_type=record.text("MarkType"),
        mark_kind=record.text("MarkKind"),
    )
