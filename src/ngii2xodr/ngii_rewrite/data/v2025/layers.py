# ruff: noqa: N801
"""Dataclass-native 2025.12 NGII layer feature classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii_rewrite.data.features import (
    LineFeature,
    NGIIFeature,
    PointFeature,
    PolygonFeature,
)
from ngii2xodr.ngii_rewrite.data.metadata import ref_field, text_field

HIST_TYPE_LABEL = {
    "001": "객체 생성",
    "002": "공간도형과 속성을 함께 수정",
    "003": "공간도형만 수정",
    "004": "공간도형 분할",
    "005": "공간도형 합병",
    "006": "위치이동",
    "007": "속성만 수정",
    "008": "객체 삭제",
}
PRESENCE_LABEL = {"0": "미존재", "1": "존재"}
BINARY_LABEL = {"0": "아님", "1": "해당"}


@dataclass(slots=True)
class NT1_NODE(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "NT1_NODE"
    layer_attr: ClassVar[str] = "nt1_node"
    filename: ClassVar[str] = "NT1_NODE.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("node",)
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

    point: NDArray[np.float64]
    node_type1: str = text_field("NodeType1", 3, required=True, code_list=NODE_TYPE_LABEL)
    node_type2: str = text_field("NodeType2", 3, required=False, code_list=NODE_TYPE_LABEL)
    node_type3: str = text_field("NodeType3", 3, required=False, code_list=NODE_TYPE_LABEL)
    start_end1: str = text_field("StartEnd1", 1, required=False, code_list=START_END_LABEL)
    start_end2: str = text_field("StartEnd2", 1, required=False, code_list=START_END_LABEL)
    start_end3: str = text_field("StartEnd3", 1, required=False, code_list=START_END_LABEL)
    pseudo: str = text_field("Pseudo", 1, required=True, code_list=PSEUDO_LABEL)
    group_id: str = text_field("GroupID", 13, required=True)
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)


@dataclass(slots=True)
class NT2_LINK(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "NT2_LINK"
    layer_attr: ClassVar[str] = "nt2_link"
    filename: ClassVar[str] = "NT2_LINK.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("link",)
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
    DIRECTION_LABEL: ClassVar[dict[str, str]] = {"1": "종점방향", "2": "기점방향", "3": "양방향"}
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

    polyline: NDArray[np.float64]
    road_rank: str = text_field("RoadRank", 3, required=True, code_list=ROAD_RANK_LABEL)
    road_no: str = text_field("RoadNo", 5, required=False)
    admin_code: str = text_field("AdminCode", 5, required=True)
    road_name: str = text_field("RoadName", 7, required=True)
    m_road_rank: str = text_field("M_RoadRank", 3, required=False, code_list=ROAD_RANK_LABEL)
    m_road_no: str = text_field("M_RoadNo", 5, required=False)
    m_road_name: str = text_field("M_RoadName", 7, required=False)
    road_type: str = text_field("RoadType", 3, required=True, code_list=ROAD_TYPE_LABEL)
    max_speed: str = text_field("MaxSpeed", 3, required=True)
    direction: str = text_field("Direction", 1, required=False, code_list=DIRECTION_LABEL)
    link_type: str = text_field("LinkType", 3, required=True, code_list=LINK_TYPE_LABEL)
    turn: str = text_field("Turn", 1, required=False, code_list=TURN_LABEL)
    r_link_id: str | None = ref_field(
        "R_LinkID", 13, target_layer_attrs=("nt2_link",), required=False
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        13,
        target_layer_attrs=("nt2_link",),
        required=False,
        column_aliases=("L_LinKID",),
    )
    from_node_id: str | None = ref_field(
        "FromNodeID", 13, target_layer_attrs=("nt1_node",), required=True
    )
    to_node_id: str | None = ref_field(
        "ToNodeID", 13, target_layer_attrs=("nt1_node",), required=True
    )
    road_type_id: str | None = ref_field(
        "RoadTypeID",
        13,
        target_layer_attrs=("rs2_roadstructure", "rs3_subsidiarysection"),
        required=False,
    )
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)


@dataclass(slots=True)
class RS1_ROADBORDER(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RS1_ROADBORDER"
    layer_attr: ClassVar[str] = "rs1_roadborder"
    filename: ClassVar[str] = "RS1_ROADBORDER.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("road_border",)

    polyline: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    kerb: str = text_field("Kerb", 1, required=True, code_list=BINARY_LABEL)
    tfc_island: str = text_field(
        "TFCIsland", 1, required=True, code_list=BINARY_LABEL, column_aliases=("TFCIsLand",)
    )
    r_link_id: str | None = ref_field(
        "R_LinkID", 13, target_layer_attrs=("nt2_link",), required=False
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        13,
        target_layer_attrs=("nt2_link",),
        required=False,
        column_aliases=("L_LinKID",),
    )
    pathway_id: str | None = ref_field(
        "PathwayID", 13, target_layer_attrs=("pw1_pathway",), required=False
    )


@dataclass(slots=True)
class RS2_ROADSTRUCTURE(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RS2_ROADSTRUCTURE"
    layer_attr: ClassVar[str] = "rs2_roadstructure"
    filename: ClassVar[str] = "RS2_ROADSTRUCTURE.shp"
    filename_aliases: ClassVar[tuple[str, ...]] = ("RS3_ROADSTRUCTURE.shp",)
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("road_structure",)
    RS_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "교량",
        "101": "고가차도",
        "102": "터널",
        "103": "지하차도",
        "200": "보호구역",
        "300": "톨게이트",
    }

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    rs_type: str = text_field("RSType", 3, required=True, code_list=RS_TYPE_LABEL)


@dataclass(slots=True)
class RS3_SUBSIDIARYSECTION(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RS3_SUBSIDIARYSECTION"
    layer_attr: ClassVar[str] = "rs3_subsidiarysection"
    filename: ClassVar[str] = "RS3_SUBSIDIARYSECTION.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("subsidiary_section",)
    SUBS_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "휴게소",
        "101": "졸음쉼터",
        "999": "기타",
    }

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    subs_type: str = text_field("SubsType", 3, required=True, code_list=SUBS_TYPE_LABEL)
    gas_station: str = text_field("GasStation", 1, required=False, code_list=PRESENCE_LABEL)
    lpg_station: str = text_field("LPGStation", 1, required=False, code_list=PRESENCE_LABEL)
    ev_charger: str = text_field("EVCharger", 1, required=False, code_list=PRESENCE_LABEL)


@dataclass(slots=True)
class PW1_PATHWAY(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "PW1_PATHWAY"
    layer_attr: ClassVar[str] = "pw1_pathway"
    filename: ClassVar[str] = "PW1_PATHWAY.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("pathway",)
    PATH_TYPE_LABEL: ClassVar[dict[str, str]] = {"100": "보도", "200": "자전거도로"}

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    path_type: str = text_field("PathType", 3, required=True, code_list=PATH_TYPE_LABEL)
    tfc_island: str = text_field(
        "TFCIsland", 1, required=True, code_list=BINARY_LABEL, column_aliases=("TFCIsLand",)
    )


@dataclass(slots=True)
class RM1_LANELINE(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RM1_LANELINE"
    layer_attr: ClassVar[str] = "rm1_laneline"
    filename: ClassVar[str] = "RM1_LANELINE.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("lane_line",)
    LINE_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "111": "황색-단선-실선",
        "112": "황색-단선-점선",
        "113": "황색-단선-좌점혼선",
        "114": "황색-단선-우점혼선",
        "121": "황색-겹선-실선",
        "122": "황색-겹선-점선",
        "123": "황색-겹선-좌점혼선",
        "124": "황색-겹선-우점혼선",
        "211": "백색-단선-실선",
        "212": "백색-단선-점선",
        "213": "백색-단선-좌점혼선",
        "214": "백색-단선-우점혼선",
        "221": "백색-겹선-실선",
        "222": "백색-겹선-점선",
        "223": "백색-겹선-좌점혼선",
        "224": "백색-겹선-우점혼선",
        "311": "청색-단선-실선",
        "312": "청색-단선-점선",
        "313": "청색-단선-좌점혼선",
        "314": "청색-단선-우점혼선",
        "321": "청색-겹선-실선",
        "322": "청색-겹선-점선",
        "323": "청색-겹선-좌점혼선",
        "324": "청색-겹선-우점혼선",
        "999": "기타",
    }
    LINE_KIND_LABEL: ClassVar[dict[str, str]] = {
        "501": "중앙선",
        "5011": "가변차선",
        "502": "유턴구역선",
        "503": "차선",
        "504": "버스전용차선",
        "505": "길가장자리구역선",
        "506": "진로변경제한선",
        "525": "유도선",
        "530": "정지선",
        "531": "안전지대",
        "535": "자전거도로",
        "599": "기타선",
    }

    polyline: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    line_type: str = text_field("LineType", 3, required=True, code_list=LINE_TYPE_LABEL)
    line_kind: str = text_field("LineKind", 4, required=True, code_list=LINE_KIND_LABEL)
    r_link_id: str | None = ref_field(
        "R_LinkID",
        13,
        target_layer_attrs=("nt2_link",),
        required=False,
        column_aliases=("R_linkID",),
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        13,
        target_layer_attrs=("nt2_link",),
        required=False,
        column_aliases=("L_linkID",),
    )


@dataclass(slots=True)
class RM2_ROADMARKING(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RM2_ROADMARKING"
    layer_attr: ClassVar[str] = "rm2_roadmarking"
    filename: ClassVar[str] = "RM2_ROADMARKING.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("road_marking",)
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
        "5383": "직진 및 유턴",
        "5391": "유턴",
        "5392": "좌회전 및 유턴",
        "5431": "차로변경(좌로합류)",
        "5432": "차로변경(우로합류)",
        "999": "기타 노면표시",
    }

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    mark_type: str = text_field("MarkType", 3, required=True, code_list=MARK_TYPE_LABEL)
    mark_kind: str = text_field("MarkKind", 4, required=True, code_list=MARK_KIND_LABEL)


@dataclass(slots=True)
class RM3_PARKINGLOT(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "RM3_PARKINGLOT"
    layer_attr: ClassVar[str] = "rm3_parkinglot"
    filename: ClassVar[str] = "RM3_PARKINGLOT.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("parking_lot",)
    PL_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "일반차량",
        "101": "우선주차",
        "102": "대형차량",
        "200": "장애인",
        "201": "전기차(충전)",
        "202": "환경친화(미충전)",
        "999": "기타",
    }

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    pl_type: str = text_field("PLType", 3, required=True, code_list=PL_TYPE_LABEL)


@dataclass(slots=True)
class SF1_BARRIER(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "SF1_BARRIER"
    layer_attr: ClassVar[str] = "sf1_barrier"
    filename: ClassVar[str] = "SF1_BARRIER.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("barrier",)
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

    polyline: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    barr_type: str = text_field("BarrType", 3, required=True, code_list=BARR_TYPE_LABEL)
    r_link_id: str | None = ref_field(
        "R_LinkID", 13, target_layer_attrs=("nt2_link",), required=False
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        13,
        target_layer_attrs=("nt2_link",),
        required=False,
        column_aliases=("L_LinKID", "L_linkID"),
    )


@dataclass(slots=True)
class SF2_TRAFFICSIGN(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "SF2_TRAFFICSIGN"
    layer_attr: ClassVar[str] = "sf2_trafficsign"
    filename: ClassVar[str] = "SF2_TRAFFICSIGN.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("traffic_sign",)
    SIGN_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "주의표지",
        "200": "규제표지",
        "300": "지시표지",
        "400": "보조표지",
    }

    point: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    sign_type: str = text_field("SignType", 3, required=True, code_list=SIGN_TYPE_LABEL)
    post_id: str | None = ref_field(
        "PostID", 13, target_layer_attrs=("sf4_supportpost",), required=False
    )


@dataclass(slots=True)
class SF3_TRAFFICLIGHT(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "SF3_TRAFFICLIGHT"
    layer_attr: ClassVar[str] = "sf3_trafficlight"
    filename: ClassVar[str] = "SF3_TRAFFICLIGHT.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("traffic_light",)
    LIGHT_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "차량가로형-삼색등",
        "101": "차량가로형-사색등A",
        "102": "차량가로형-사색등B",
        "103": "차량가로형-화살표삼색등",
        "110": "차량세로형-삼색등",
        "111": "차량세로형-우회전삼색등",
        "112": "차량세로형-사색등",
        "120": "버스삼색등",
        "121": "노면전차육구등",
        "130": "가로형이색등",
        "140": "차량보조등-세로형삼색등",
        "141": "차량보조등-세로형사색등",
        "200": "가변등",
        "201": "경보형경보등",
        "300": "보행등",
        "400": "자전거세로형-삼색등",
        "401": "자전거세로형-이색등",
        "999": "기타 신호등 유형",
    }

    point: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    light_type: str = text_field("LightType", 3, required=True, code_list=LIGHT_TYPE_LABEL)
    post_id: str | None = ref_field(
        "PostID", 13, target_layer_attrs=("sf4_supportpost",), required=False
    )


@dataclass(slots=True)
class SF4_SUPPORTPOST(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "SF4_SUPPORTPOST"
    layer_attr: ClassVar[str] = "sf4_supportpost"
    filename: ClassVar[str] = "SF4_SUPPORTPOST.shp"
    filename_aliases: ClassVar[tuple[str, ...]] = ("SF4_SUPPROTPOST.shp",)
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("support_post",)
    POST_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "교통 및 보행 신호기 지주",
        "200": "교통안전표지 지주",
    }

    point: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)
    post_type: str = text_field("PostType", 3, required=True, code_list=POST_TYPE_LABEL)


@dataclass(slots=True)
class SF5_SPEEDBUMP(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "SF5_SPEEDBUMP"
    layer_attr: ClassVar[str] = "sf5_speedbump"
    filename: ClassVar[str] = "SF5_SPEEDBUMP.shp"
    id_max_length: ClassVar[int] = 13
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("speed_bump",)

    ring: NDArray[np.float64]
    survey_date: str = text_field("SurveyDate", 8, required=True)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 3, required=False, code_list=HIST_TYPE_LABEL)


LAYER_TYPES = (
    NT1_NODE,
    NT2_LINK,
    RS1_ROADBORDER,
    RS2_ROADSTRUCTURE,
    RS3_SUBSIDIARYSECTION,
    PW1_PATHWAY,
    RM1_LANELINE,
    RM2_ROADMARKING,
    RM3_PARKINGLOT,
    SF1_BARRIER,
    SF2_TRAFFICSIGN,
    SF3_TRAFFICLIGHT,
    SF4_SUPPORTPOST,
    SF5_SPEEDBUMP,
)
