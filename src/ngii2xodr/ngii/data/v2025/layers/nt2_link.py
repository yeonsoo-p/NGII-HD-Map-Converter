from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025LineFeature, common_kwargs


@dataclass(slots=True)
class NT2_LINK(V2025LineFeature):
    layer_name: ClassVar[str] = "NT2_LINK"
    ROAD_RANK_LABEL: ClassVar[dict[str, str]] = {
        "100": "고속도로",
        "200": "일반국도",
        "300": "특별시도/광역시도",
        "400": "지방도",
        "500": "시도",
        "600": "군도",
        "700": "구도",
        "999": "기타도로",
    }
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "일반도로",
        "200": "생활(이면)도로",
        "300": "구간",
        "400": "부속구역",
    }
    DIRECTION_LABEL: ClassVar[dict[str, str]] = {
        "1": "종점방향",
        "2": "기점방향",
        "3": "양방향",
    }
    LINK_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "평면교차로 링크",
        "101": "입체교차로 링크",
        "102": "회전교차로 링크",
        "103": "회전차로 링크",
        "104": "변속차로 링크",
        "200": "버스전용차로 링크",
        "201": "가변차로 링크",
        "300": "일반차로 링크",
    }
    TURN_LABEL: ClassVar[dict[str, str]] = {"1": "좌회전", "2": "우회전", "3": "유턴"}

    admin_code: str
    road_rank: str
    road_no: str
    road_name: str
    m_road_rank: str
    m_road_no: str
    m_road_name: str
    road_type: str
    max_speed: str
    direction: str
    link_type: str
    turn: str
    r_link_id: str | None
    l_link_id: str | None
    from_node_id: str | None
    to_node_id: str | None
    road_type_id: str | None


def make_feature(record: FeatureRecord) -> NT2_LINK:
    return NT2_LINK(
        **common_kwargs(record),
        polyline=record.geometry,
        admin_code=record.text("AdminCode"),
        road_rank=record.text("RoadRank"),
        road_no=record.text("RoadNo"),
        road_name=record.text("RoadName"),
        m_road_rank=record.text("M_RoadRank"),
        m_road_no=record.text("M_RoadNo"),
        m_road_name=record.text("M_RoadName"),
        road_type=record.text("RoadType"),
        max_speed=record.text("MaxSpeed"),
        direction=record.text("Direction"),
        link_type=record.text("LinkType"),
        turn=record.text("Turn"),
        r_link_id=record.optional_ref("R_LinkID"),
        l_link_id=record.optional_ref("L_LinkID"),
        from_node_id=record.optional_ref("FromNodeID"),
        to_node_id=record.optional_ref("ToNodeID"),
        road_type_id=record.optional_ref("RoadTypeID"),
    )
