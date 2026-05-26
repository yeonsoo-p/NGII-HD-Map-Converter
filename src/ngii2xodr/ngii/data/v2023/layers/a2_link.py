from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, relation_property
from ngii2xodr.ngii.data.v2023.features import V2023LineFeature, common_kwargs


@dataclass(slots=True)
class A2_LINK(V2023LineFeature):
    layer_name: ClassVar[str] = "A2_LINK"
    ROAD_RANK_LABEL: ClassVar[dict[str, str]] = {
        "1": "고속도로",
        "2": "국도",
        "3": "특별광역시도",
        "4": "국가지원지방도",
        "5": "지방도",
        "6": "시도",
        "7": "군도",
        "8": "구도",
        "9": "기타도로",
    }
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반도로",
        "2": "터널",
        "3": "교량",
        "4": "지하도로",
        "5": "고가도로",
    }
    LINK_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "교차로내주행경로",
        "2": "톨게이트차로(하이패스)",
        "3": "톨게이트차로(비하이패스)",
        "4": "버스전용차로",
        "5": "가변차선차로",
        "6": "일반주행차로",
        "7": "휴게소진입로",
        "8": "휴게소내주행경로",
        "9": "휴게소진출로",
        "10": "졸음쉼터진입로",
        "11": "졸음쉼터내주행경로",
        "12": "졸음쉼터진출로",
        "13": "교차로진입로",
        "14": "교차로진출로",
        "99": "기타차로",
    }

    road_rank: str
    road_type: str
    road_no: str
    link_type: str
    lane_no: int
    r_link_id: str | None
    l_link_id: str | None
    from_node_id: str | None
    to_node_id: str | None
    section_id: str
    length_m: float
    its_link_id: str

    from_node = relation_property("FromNodeID")
    to_node = relation_property("ToNodeID")
    right_link = relation_property("R_LinkID")
    left_link = relation_property("L_LinkID")


def _optional_id(value: str) -> str | None:
    return value or None


def make_feature(record: FeatureRecord) -> A2_LINK:
    return A2_LINK(
        **common_kwargs(record),
        polyline=record.geometry,
        road_rank=record.text("RoadRank"),
        road_type=record.text("RoadType"),
        road_no=record.text("RoadNo"),
        link_type=record.text("LinkType"),
        lane_no=record.integer("LaneNo"),
        r_link_id=_optional_id(record.text("R_LinkID")),
        l_link_id=_optional_id(record.text("L_LinkID")),
        from_node_id=_optional_id(record.text("FromNodeID")),
        to_node_id=_optional_id(record.text("ToNodeID")),
        section_id=record.text("SectionID"),
        length_m=record.floating("Length"),
        its_link_id=record.text("ITSLinkID"),
    )
