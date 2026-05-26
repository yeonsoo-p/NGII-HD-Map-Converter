from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2023.features import V2023PointFeature, common_kwargs


@dataclass(slots=True)
class C1_TRAFFICLIGHT(V2023PointFeature):
    layer_name: ClassVar[str] = "C1_TRAFFICLIGHT"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "차량횡형-삼색등",
        "2": "차량횡형-사색등A",
        "3": "차량횡형-사색등B",
        "4": "차량횡형-화살표삼색등",
        "5": "차량종형-삼색등",
        "6": "차량종형-화살표삼색등",
        "7": "차량종형-사색등",
        "8": "버스삼색등",
        "9": "가변형 가변등",
        "10": "경보형 가변등",
        "11": "보행등",
        "12": "자전거종형-삼색등",
        "13": "자전거종형-이색등",
        "14": "차량보조등-종형삼색등",
        "15": "차량보조등-종형사색등",
        "99": "기타 신호등 유형",
    }

    type: str
    link_id: str | None
    ref_lane: int
    post_id: str | None


def make_feature(record: FeatureRecord) -> C1_TRAFFICLIGHT:
    link_id = record.text("LinkID")
    post_id = record.text("PostID")
    return C1_TRAFFICLIGHT(
        **common_kwargs(record),
        point=record.geometry,
        type=record.text("Type"),
        link_id=link_id or None,
        ref_lane=record.integer("Ref_Lane"),
        post_id=post_id or None,
    )
