# ruff: noqa: N801
"""Dataclass-native 2023.07 NGII layer feature classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii_rewrite.data.features import (
    FeatureGeometryKind,
    LineFeature,
    NGIIFeature,
    PointFeature,
    PolygonFeature,
)
from ngii2xodr.ngii_rewrite.data.metadata import (
    float_field,
    integer_field,
    ref_field,
    text_field,
)

PRESENCE_LABEL = {"0": "미존재", "1": "존재"}


@dataclass(slots=True)
class A1_NODE(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "A1_NODE"
    layer_attr: ClassVar[str] = "a1_node"
    filename: ClassVar[str] = "A1_NODE.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("node",)
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

    point: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    node_type: str = text_field("NodeType", 2, required=True, code_list=NODE_TYPE_LABEL)
    its_node_id: str = text_field("ITSNodeID", 10, required=False)


@dataclass(slots=True)
class A2_LINK(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "A2_LINK"
    layer_attr: ClassVar[str] = "a2_link"
    filename: ClassVar[str] = "A2_LINK.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("link",)
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

    polyline: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    road_rank: str = text_field("RoadRank", 1, required=True, code_list=ROAD_RANK_LABEL)
    road_type: str = text_field("RoadType", 1, required=True, code_list=ROAD_TYPE_LABEL)
    road_no: str = text_field("RoadNo", 5, required=False)
    link_type: str = text_field("LinkType", 2, required=True, code_list=LINK_TYPE_LABEL)
    lane_no: int = integer_field("LaneNo", required=False)
    r_link_id: str | None = ref_field(
        "R_LinkID", 12, target_layer_attrs=("a2_link",), required=False
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        12,
        target_layer_attrs=("a2_link",),
        required=False,
        column_aliases=("L_LinKID",),
    )
    from_node_id: str | None = ref_field(
        "FromNodeID", 12, target_layer_attrs=("a1_node",), required=True
    )
    to_node_id: str | None = ref_field(
        "ToNodeID",
        12,
        target_layer_attrs=("a1_node",),
        required=True,
        column_aliases=("ToNodeId",),
    )
    section_id: str | None = ref_field(
        "SectionID",
        12,
        target_layer_attrs=("a3_drivewaysection", "a4_subsidiarysection"),
        required=False,
    )
    length_m: float = float_field("Length", required=False, array_aliases=("lengths_m",))
    its_link_id: str = text_field("ITSLinkID", 30, required=False)


@dataclass(slots=True)
class A3_DRIVEWAYSECTION(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "A3_DRIVEWAYSECTION"
    layer_attr: ClassVar[str] = "a3_drivewaysection"
    filename: ClassVar[str] = "A3_DRIVEWAYSECTION.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("driveway_section",)
    KIND_LABEL: ClassVar[dict[str, str]] = {"1": "주행구간", "7": "자율주행금지구간"}
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = A2_LINK.ROAD_TYPE_LABEL

    ring: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    kind: str = text_field("Kind", 1, required=True, code_list=KIND_LABEL)
    road_type: str = text_field("RoadType", 1, required=True, code_list=ROAD_TYPE_LABEL)


@dataclass(slots=True)
class A4_SUBSIDIARYSECTION(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "A4_SUBSIDIARYSECTION"
    layer_attr: ClassVar[str] = "a4_subsidiarysection"
    filename: ClassVar[str] = "A4_SUBSIDIARYSECTION.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("subsidiary_section",)
    SUBTYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "휴게소",
        "2": "졸음쉼터",
        "3": "보도",
        "4": "자전거도로",
        "9": "기타부속구간",
    }
    DIRECTION_LABEL: ClassVar[dict[str, str]] = {"1": "상행", "2": "하행", "3": "양방향"}

    ring: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    subtype: str = text_field("SubType", 1, required=True, code_list=SUBTYPE_LABEL)
    name: str = text_field("Name", 30, required=True)
    direction: str = text_field("Direction", 1, required=True, code_list=DIRECTION_LABEL)
    gas_station: str = text_field("GasStation", 1, required=True, code_list=PRESENCE_LABEL)
    lpg_station: str = text_field("LpgStation", 1, required=True, code_list=PRESENCE_LABEL)
    ev_charger: str = text_field("EvCharger", 1, required=True, code_list=PRESENCE_LABEL)
    toilet: str = text_field("Toilet", 1, required=True, code_list=PRESENCE_LABEL)


@dataclass(slots=True)
class A5_PARKINGLOT(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "A5_PARKINGLOT"
    layer_attr: ClassVar[str] = "a5_parkinglot"
    filename: ClassVar[str] = "A5_PARKINGLOT.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("parking_lot",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반주차장",
        "2": "화물차전용주차장",
        "3": "장애인전용주차장",
        "4": "노인전용주차장",
        "5": "여성전용주차장/가족배려주차장",
        "6": "버스전용주차장",
        "7": "전기차전용주차장",
        "9": "기타주차장",
    }

    ring: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)
    section_id: str | None = ref_field(
        "SectionID", 12, target_layer_attrs=("a4_subsidiarysection",), required=True
    )


@dataclass(slots=True)
class B1_SAFETYSIGN(PointFeature, PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "B1_SAFETYSIGN"
    layer_attr: ClassVar[str] = "b1_safetysign"
    filename: ClassVar[str] = "B1_SAFETYSIGN.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("traffic_sign",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "주의표지",
        "2": "지시표지",
        "3": "규제표지",
        "4": "보조표지",
    }

    point: NDArray[np.float64] | None
    ring: NDArray[np.float64] | None
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=True)
    ref_lane: int = integer_field("Ref_Lane", required=True)
    post_id: str | None = ref_field(
        "PostID", 12, target_layer_attrs=("c6_postpoint",), required=False
    )

    @property
    def geometry_kind(self) -> FeatureGeometryKind:
        if self.point is not None:
            return "point"
        if self.ring is not None:
            return "polygon"
        msg = f"{self.layer_name} {self.id!r} has neither point nor polygon geometry"
        raise ValueError(msg)

    @property
    def points(self) -> NDArray[np.float64]:
        if self.point is not None:
            return self.point.reshape(1, 3)
        if self.ring is not None:
            return self.ring
        msg = f"{self.layer_name} {self.id!r} has neither point nor polygon geometry"
        raise ValueError(msg)

    @property
    def geometry_signature(self) -> tuple[tuple[float, ...], ...]:
        if self.point is not None:
            return (tuple(float(v) for v in self.point),)
        if self.ring is not None:
            return tuple(tuple(float(v) for v in row) for row in self.ring)
        msg = f"{self.layer_name} {self.id!r} has neither point nor polygon geometry"
        raise ValueError(msg)


@dataclass(slots=True)
class B2_SURFACELINEMARK(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "B2_SURFACELINEMARK"
    layer_attr: ClassVar[str] = "b2_surfacelinemark"
    filename: ClassVar[str] = "B2_SURFACELINEMARK.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("lane_line",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {
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
    KIND_LABEL: ClassVar[dict[str, str]] = {
        "501": "중앙선",
        "5011": "가변차선",
        "502": "유턴구역선",
        "503": "차선",
        "504": "버스전용차선",
        "505": "길가장자리구역선",
        "506": "진로변경제한선",
        "515": "주정차금지선",
        "525": "유도선",
        "530": "정지선",
        "531": "안전지대",
        "535": "자전거도로",
        "599": "기타선",
    }

    polyline: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 3, required=True, code_list=TYPE_LABEL)
    kind: str = text_field("Kind", 5, required=True, code_list=KIND_LABEL)
    r_link_id: str | None = ref_field(
        "R_LinkID",
        12,
        target_layer_attrs=("a2_link",),
        required=False,
        column_aliases=("R_linkID",),
    )
    l_link_id: str | None = ref_field(
        "L_LinkID",
        12,
        target_layer_attrs=("a2_link",),
        required=False,
        column_aliases=("L_linkID",),
    )


@dataclass(slots=True)
class B3_SURFACEMARK(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "B3_SURFACEMARK"
    layer_attr: ClassVar[str] = "b3_surfacemark"
    filename: ClassVar[str] = "B3_SURFACEMARK.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("road_marking",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {"1": "화살표", "5": "횡단보도"}
    KIND_LABEL: ClassVar[dict[str, str]] = {
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
        "544": "오르막경사면",
        "599": "기타 지시표시",
    }

    ring: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)
    kind: str = text_field("Kind", 4, required=True, code_list=KIND_LABEL)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=False)


@dataclass(slots=True)
class C1_TRAFFICLIGHT(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C1_TRAFFICLIGHT"
    layer_attr: ClassVar[str] = "c1_trafficlight"
    filename: ClassVar[str] = "C1_TRAFFICLIGHT.shp"
    filename_aliases: ClassVar[tuple[str, ...]] = ("C1_TRAFFICLIGH.shp",)
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("traffic_light",)
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

    point: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 2, required=True, code_list=TYPE_LABEL)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=True)
    ref_lane: int = integer_field("Ref_Lane", required=True)
    post_id: str | None = ref_field(
        "PostID",
        12,
        target_layer_attrs=("c6_postpoint",),
        required=False,
        column_aliases=("postID",),
    )


@dataclass(slots=True)
class C2_KILOPOST(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C2_KILOPOST"
    layer_attr: ClassVar[str] = "c2_kilopost"
    filename: ClassVar[str] = "C2_KILOPOST.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("kilopost",)

    point: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    distance: float = float_field("Distance", required=True)
    origin: str = text_field("Origin", 30, required=False)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=True)
    ref_lane: int = integer_field("Ref_Lane", required=True)


@dataclass(slots=True)
class C3_VEHICLEPROTECTIONSAFETY(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY"
    layer_attr: ClassVar[str] = "c3_vehicleprotectionsafety"
    filename: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = True
    roles: ClassVar[tuple[str, ...]] = ("barrier",)
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
    IS_CENTRAL_LABEL: ClassVar[dict[str, str]] = {"0": "도로 가장자리", "1": "중앙분리대"}
    LOW_HIGH_LABEL: ClassVar[dict[str, str]] = {"1": "상단", "2": "하단"}

    polyline: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 2, required=True, code_list=TYPE_LABEL)
    is_central: str = text_field(
        "IsCentral",
        1,
        required=True,
        code_list=IS_CENTRAL_LABEL,
        column_aliases=("isCentral",),
        array_aliases=("is_central",),
    )
    low_high: str = text_field(
        "LowHigh",
        1,
        required=False,
        code_list=LOW_HIGH_LABEL,
        array_aliases=("low_high",),
    )
    ref_id: str | None = ref_field(
        "Ref_ID", 12, target_layer_attrs=("c3_vehicleprotectionsafety",), required=False
    )


@dataclass(slots=True)
class C4_SPEEDBUMP(PolygonFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C4_SPEEDBUMP"
    layer_attr: ClassVar[str] = "c4_speedbump"
    filename: ClassVar[str] = "C4_SPEEDBUMP.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("speed_bump",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "높이있는방지턱",
        "2": "높이없는방지턱표시",
        "3": "기타방지턱",
    }

    ring: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=True)
    ref_lane: int = integer_field("Ref_Lane", required=True)


@dataclass(slots=True)
class C5_HEIGHTBARRIER(LineFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C5_HEIGHTBARRIER"
    layer_attr: ClassVar[str] = "c5_heightbarrier"
    filename: ClassVar[str] = "C5_HEIGHTBARRIER.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("height_barrier",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "고가도로/교량",
        "2": "육교",
        "4": "기타 높이제한장애물",
    }

    polyline: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)
    link_id: str | None = ref_field("LinkID", 12, target_layer_attrs=("a2_link",), required=True)
    ref_lane: int = integer_field("Ref_Lane", required=True)


@dataclass(slots=True)
class C6_POSTPOINT(PointFeature, NGIIFeature):
    layer_name: ClassVar[str] = "C6_POSTPOINT"
    layer_attr: ClassVar[str] = "c6_postpoint"
    filename: ClassVar[str] = "C6_POSTPOINT.shp"
    id_max_length: ClassVar[int] = 12
    required_layer: ClassVar[bool] = False
    roles: ClassVar[tuple[str, ...]] = ("support_post",)
    TYPE_LABEL: ClassVar[dict[str, str]] = {"1": "신호기지주", "2": "교통표지지주"}

    point: NDArray[np.float64]
    admin_code: str = text_field("AdminCode", 3, required=True)
    maker: str = text_field("Maker", 20, required=True)
    update_date: str = text_field("UpdateDate", 8, required=False)
    version: str = text_field("Version", 4, required=True)
    remark: str = text_field("Remark", 30, required=False)
    hist_type: str = text_field("HistType", 5, required=False)
    hist_remark: str = text_field("HistRemark", 30, required=False)
    type: str = text_field("Type", 1, required=True, code_list=TYPE_LABEL)


LAYER_TYPES = (
    A1_NODE,
    A2_LINK,
    A3_DRIVEWAYSECTION,
    A4_SUBSIDIARYSECTION,
    A5_PARKINGLOT,
    B1_SAFETYSIGN,
    B2_SURFACELINEMARK,
    B3_SURFACEMARK,
    C1_TRAFFICLIGHT,
    C2_KILOPOST,
    C3_VEHICLEPROTECTIONSAFETY,
    C4_SPEEDBUMP,
    C5_HEIGHTBARRIER,
    C6_POSTPOINT,
)
