from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PointFeature, common_kwargs


@dataclass(slots=True)
class NT1_NODE(V2025PointFeature):
    layer_name: ClassVar[str] = "NT1_NODE"
    NODE_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "평면교차로 시점/종점",
        "101": "입체교차로 시점/종점",
        "102": "회전교차로",
        "200": "차로 수 변화 지점",
        "201": "도로 및 차로 분기/합류 지점",
        "202": "가변차로 시점/종점",
        "203": "유턴 시점/종점",
        "300": "최고 제한속도 변화 시점",
        "400": "교량 시점/종점",
        "401": "고가차도 시점/종점",
        "402": "터널 시점/종점",
        "403": "지하차도 시점/종점",
        "404": "보호구역 시점/종점",
        "405": "부속구역 시점/종점",
        "406": "톨게이트 시점/종점",
        "500": "요금소(하이패스)",
        "501": "요금소(비하이패스)",
        "502": "과적검문소",
        "600": "행정경계",
        "999": "기타",
    }
    START_END_LABEL: ClassVar[dict[str, str]] = {"1": "시점", "2": "종점"}
    PSEUDO_LABEL: ClassVar[dict[str, str]] = {"0": "의사노드 아님", "1": "의사노드"}

    node_type1: str
    node_type2: str
    node_type3: str
    start_end1: str
    start_end2: str
    start_end3: str
    pseudo: str
    group_id: str


def make_feature(record: FeatureRecord) -> NT1_NODE:
    return NT1_NODE(
        **common_kwargs(record),
        point=record.geometry,
        node_type1=record.text("NodeType1"),
        node_type2=record.text("NodeType2"),
        node_type3=record.text("NodeType3"),
        start_end1=record.text("StartEnd1"),
        start_end2=record.text("StartEnd2"),
        start_end3=record.text("StartEnd3"),
        pseudo=record.text("Pseudo"),
        group_id=record.text("GroupID"),
    )
