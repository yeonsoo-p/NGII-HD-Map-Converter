from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PointFeature, common_kwargs


@dataclass(slots=True)
class A1_NODE(PointFeature):
    layer_name: ClassVar[str] = "A1_NODE"
    NODE_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "평면교차로",
        "2": "입체교차로",
        "3": "터널시종점",
        "4": "교량시종점",
        "5": "지하차도시종점",
        "6": "고가차도시종점",
        "7": "도로차로수변화",
        "8": "톨게이트시종점",
        "9": "요금소",
        "10": "회전교차로",
        "99": "기타유형",
    }

    node_type: str
    its_node_id: str


def make_feature(record: FeatureRecord) -> A1_NODE:
    return A1_NODE(
        **common_kwargs(record),
        point=record.geometry,
        node_type=record.text("NodeType"),
        its_node_id=record.text("ITSNodeID"),
    )
