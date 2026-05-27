"""NGII 2025.12 layer registry for data model 2024."""

from __future__ import annotations

from ngii2xodr.ngii.data.schema import (
    LayerSpec,
    ReciprocalRelationshipRule,
    RelationshipRule,
    RoleFilter,
    RoleKey,
    SchemaDefinition,
    text_rule,
)
from ngii2xodr.ngii.data.v2025.layers import (
    NT1_NODE,
    NT2_LINK,
    PW1_PATHWAY,
    RM1_LANELINE,
    RM2_ROADMARKING,
    RM3_PARKINGLOT,
    RS1_ROADBORDER,
    RS2_ROADSTRUCTURE,
    RS3_SUBSIDIARYSECTION,
    SF1_BARRIER,
    SF2_TRAFFICSIGN,
    SF3_TRAFFICLIGHT,
    SF4_SUPPORTPOST,
    SF5_SPEEDBUMP,
)
from ngii2xodr.ngii.data.v2025.layers.nt1_node import make_feature as make_nt1
from ngii2xodr.ngii.data.v2025.layers.nt2_link import make_feature as make_nt2
from ngii2xodr.ngii.data.v2025.layers.pw1_pathway import make_feature as make_pw1
from ngii2xodr.ngii.data.v2025.layers.rm1_laneline import make_feature as make_rm1
from ngii2xodr.ngii.data.v2025.layers.rm2_roadmarking import make_feature as make_rm2
from ngii2xodr.ngii.data.v2025.layers.rm3_parkinglot import make_feature as make_rm3
from ngii2xodr.ngii.data.v2025.layers.rs1_roadborder import make_feature as make_rs1
from ngii2xodr.ngii.data.v2025.layers.rs2_roadstructure import make_feature as make_rs2
from ngii2xodr.ngii.data.v2025.layers.rs3_subsidiarysection import make_feature as make_rs3
from ngii2xodr.ngii.data.v2025.layers.sf1_barrier import make_feature as make_sf1
from ngii2xodr.ngii.data.v2025.layers.sf2_trafficsign import make_feature as make_sf2
from ngii2xodr.ngii.data.v2025.layers.sf3_trafficlight import make_feature as make_sf3
from ngii2xodr.ngii.data.v2025.layers.sf4_supportpost import make_feature as make_sf4
from ngii2xodr.ngii.data.v2025.layers.sf5_speedbump import make_feature as make_sf5

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

COMMON_FIELD_RULES = (
    text_rule("ID", 13, required=True),
    text_rule("SurveyDate", 8, required=True),
    text_rule("Version", 4, required=True),
    text_rule("Remark", 30, required=False),
    text_rule("HistType", 3, required=False, code_list=HIST_TYPE_LABEL),
)

LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        layer_name="NT1_NODE",
        python_attr="nt1_node",
        filename="NT1_NODE.shp",
        geometry_kinds=("point",),
        feature_type=NT1_NODE,
        factory=make_nt1,
        required_layer=True,
        roles=("node",),
        field_rules=(
            text_rule("ID", 13, required=True),
            text_rule("NodeType1", 3, required=True, code_list=NT1_NODE.NODE_TYPE_LABEL),
            text_rule("NodeType2", 3, required=False, code_list=NT1_NODE.NODE_TYPE_LABEL),
            text_rule("NodeType3", 3, required=False, code_list=NT1_NODE.NODE_TYPE_LABEL),
            text_rule("StartEnd1", 1, required=False, code_list=NT1_NODE.START_END_LABEL),
            text_rule("StartEnd2", 1, required=False, code_list=NT1_NODE.START_END_LABEL),
            text_rule("StartEnd3", 1, required=False, code_list=NT1_NODE.START_END_LABEL),
            text_rule("Pseudo", 1, required=True, code_list=NT1_NODE.PSEUDO_LABEL),
            text_rule("GroupID", 13, required=True),
            *COMMON_FIELD_RULES[1:],
        ),
    ),
    LayerSpec(
        layer_name="NT2_LINK",
        python_attr="nt2_link",
        filename="NT2_LINK.shp",
        geometry_kinds=("line",),
        feature_type=NT2_LINK,
        factory=make_nt2,
        required_layer=True,
        roles=("link",),
        field_rules=(
            text_rule("ID", 13, required=True),
            text_rule("RoadRank", 3, required=True, code_list=NT2_LINK.ROAD_RANK_LABEL),
            text_rule("RoadNo", 5, required=False),
            text_rule("AdminCode", 5, required=True),
            text_rule("RoadName", 7, required=True),
            text_rule("M_RoadRank", 3, required=False, code_list=NT2_LINK.ROAD_RANK_LABEL),
            text_rule("M_RoadNo", 5, required=False),
            text_rule("M_RoadName", 7, required=False),
            text_rule("RoadType", 3, required=True, code_list=NT2_LINK.ROAD_TYPE_LABEL),
            text_rule("MaxSpeed", 3, required=True),
            text_rule("Direction", 1, required=False, code_list=NT2_LINK.DIRECTION_LABEL),
            text_rule("LinkType", 3, required=True, code_list=NT2_LINK.LINK_TYPE_LABEL),
            text_rule("Turn", 1, required=False, code_list=NT2_LINK.TURN_LABEL),
            text_rule("R_LinkID", 13, required=False),
            text_rule("L_LinkID", 13, required=False, column_aliases=("L_LinKID",)),
            text_rule("FromNodeID", 13, required=True),
            text_rule("ToNodeID", 13, required=True),
            text_rule("RoadTypeID", 13, required=False),
            *COMMON_FIELD_RULES[1:],
        ),
        relationships=(
            RelationshipRule("FromNodeID", "from_node_id", ("nt1_node",), True),
            RelationshipRule("ToNodeID", "to_node_id", ("nt1_node",), True),
            RelationshipRule("R_LinkID", "r_link_id", ("nt2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("nt2_link",), False),
            RelationshipRule(
                "RoadTypeID",
                "road_type_id",
                ("rs2_roadstructure", "rs3_subsidiarysection"),
                False,
            ),
        ),
    ),
    LayerSpec(
        layer_name="RS1_ROADBORDER",
        python_attr="rs1_roadborder",
        filename="RS1_ROADBORDER.shp",
        geometry_kinds=("line",),
        feature_type=RS1_ROADBORDER,
        factory=make_rs1,
        required_layer=True,
        roles=("road_border",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Kerb", 1, required=True, code_list=BINARY_LABEL),
            text_rule(
                "TFCIsland",
                1,
                required=True,
                code_list=BINARY_LABEL,
                column_aliases=("TFCIsLand",),
            ),
            text_rule("R_LinkID", 13, required=False),
            text_rule("L_LinkID", 13, required=False, column_aliases=("L_LinKID",)),
            text_rule("PathwayID", 13, required=False),
        ),
        relationships=(
            RelationshipRule("R_LinkID", "r_link_id", ("nt2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("nt2_link",), False),
            RelationshipRule("PathwayID", "pathway_id", ("pw1_pathway",), False),
        ),
    ),
    LayerSpec(
        layer_name="RS2_ROADSTRUCTURE",
        python_attr="rs2_roadstructure",
        filename="RS2_ROADSTRUCTURE.shp",
        filename_aliases=("RS3_ROADSTRUCTURE.shp",),
        geometry_kinds=("polygon",),
        feature_type=RS2_ROADSTRUCTURE,
        factory=make_rs2,
        required_layer=False,
        roles=("road_structure",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("RSType", 3, required=True, code_list=RS2_ROADSTRUCTURE.RS_TYPE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="RS3_SUBSIDIARYSECTION",
        python_attr="rs3_subsidiarysection",
        filename="RS3_SUBSIDIARYSECTION.shp",
        geometry_kinds=("polygon",),
        feature_type=RS3_SUBSIDIARYSECTION,
        factory=make_rs3,
        required_layer=False,
        roles=("subsidiary_section",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule(
                "SubsType", 3, required=True, code_list=RS3_SUBSIDIARYSECTION.SUBS_TYPE_LABEL
            ),
            text_rule("GasStation", 1, required=False, code_list=PRESENCE_LABEL),
            text_rule("LPGStation", 1, required=False, code_list=PRESENCE_LABEL),
            text_rule("EVCharger", 1, required=False, code_list=PRESENCE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="PW1_PATHWAY",
        python_attr="pw1_pathway",
        filename="PW1_PATHWAY.shp",
        geometry_kinds=("polygon",),
        feature_type=PW1_PATHWAY,
        factory=make_pw1,
        required_layer=False,
        roles=("pathway",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("PathType", 3, required=True, code_list=PW1_PATHWAY.PATH_TYPE_LABEL),
            text_rule(
                "TFCIsland",
                1,
                required=True,
                code_list=BINARY_LABEL,
                column_aliases=("TFCIsLand",),
            ),
        ),
    ),
    LayerSpec(
        layer_name="RM1_LANELINE",
        python_attr="rm1_laneline",
        filename="RM1_LANELINE.shp",
        geometry_kinds=("line",),
        feature_type=RM1_LANELINE,
        factory=make_rm1,
        required_layer=False,
        roles=("lane_line",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("LineType", 3, required=True, code_list=RM1_LANELINE.LINE_TYPE_LABEL),
            text_rule("LineKind", 4, required=True, code_list=RM1_LANELINE.LINE_KIND_LABEL),
            text_rule("R_LinkID", 13, required=False, column_aliases=("R_linkID",)),
            text_rule("L_LinkID", 13, required=False, column_aliases=("L_linkID",)),
        ),
        relationships=(
            RelationshipRule("R_LinkID", "r_link_id", ("nt2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("nt2_link",), False),
        ),
    ),
    LayerSpec(
        layer_name="RM2_ROADMARKING",
        python_attr="rm2_roadmarking",
        filename="RM2_ROADMARKING.shp",
        geometry_kinds=("polygon",),
        feature_type=RM2_ROADMARKING,
        factory=make_rm2,
        required_layer=False,
        roles=("road_marking",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("MarkType", 3, required=True, code_list=RM2_ROADMARKING.MARK_TYPE_LABEL),
            text_rule("MarkKind", 4, required=True, code_list=RM2_ROADMARKING.MARK_KIND_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="RM3_PARKINGLOT",
        python_attr="rm3_parkinglot",
        filename="RM3_PARKINGLOT.shp",
        geometry_kinds=("polygon",),
        feature_type=RM3_PARKINGLOT,
        factory=make_rm3,
        required_layer=False,
        roles=("parking_lot",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("PLType", 3, required=True, code_list=RM3_PARKINGLOT.PL_TYPE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="SF1_BARRIER",
        python_attr="sf1_barrier",
        filename="SF1_BARRIER.shp",
        geometry_kinds=("line",),
        feature_type=SF1_BARRIER,
        factory=make_sf1,
        required_layer=False,
        roles=("barrier",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("BarrType", 3, required=True, code_list=SF1_BARRIER.BARR_TYPE_LABEL),
            text_rule("R_LinkID", 13, required=False),
            text_rule(
                "L_LinkID",
                13,
                required=False,
                column_aliases=("L_LinKID", "L_linkID"),
            ),
        ),
        relationships=(
            RelationshipRule("R_LinkID", "r_link_id", ("nt2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("nt2_link",), False),
        ),
    ),
    LayerSpec(
        layer_name="SF2_TRAFFICSIGN",
        python_attr="sf2_trafficsign",
        filename="SF2_TRAFFICSIGN.shp",
        geometry_kinds=("point",),
        feature_type=SF2_TRAFFICSIGN,
        factory=make_sf2,
        required_layer=False,
        roles=("traffic_sign",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("SignType", 3, required=True, code_list=SF2_TRAFFICSIGN.SIGN_TYPE_LABEL),
            text_rule("PostID", 13, required=False),
        ),
        relationships=(RelationshipRule("PostID", "post_id", ("sf4_supportpost",), False),),
    ),
    LayerSpec(
        layer_name="SF3_TRAFFICLIGHT",
        python_attr="sf3_trafficlight",
        filename="SF3_TRAFFICLIGHT.shp",
        geometry_kinds=("point",),
        feature_type=SF3_TRAFFICLIGHT,
        factory=make_sf3,
        required_layer=False,
        roles=("traffic_light",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("LightType", 3, required=True, code_list=SF3_TRAFFICLIGHT.LIGHT_TYPE_LABEL),
            text_rule("PostID", 13, required=False),
        ),
        relationships=(RelationshipRule("PostID", "post_id", ("sf4_supportpost",), False),),
    ),
    LayerSpec(
        layer_name="SF4_SUPPORTPOST",
        python_attr="sf4_supportpost",
        filename="SF4_SUPPORTPOST.shp",
        filename_aliases=("SF4_SUPPROTPOST.shp",),
        geometry_kinds=("point",),
        feature_type=SF4_SUPPORTPOST,
        factory=make_sf4,
        required_layer=False,
        roles=("support_post",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("PostType", 3, required=True, code_list=SF4_SUPPORTPOST.POST_TYPE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="SF5_SPEEDBUMP",
        python_attr="sf5_speedbump",
        filename="SF5_SPEEDBUMP.shp",
        geometry_kinds=("polygon",),
        feature_type=SF5_SPEEDBUMP,
        factory=make_sf5,
        required_layer=False,
        roles=("speed_bump",),
        field_rules=COMMON_FIELD_RULES,
    ),
)

SCHEMA = SchemaDefinition(
    version="2025.12",
    layer_specs=LAYER_SPECS,
    role_filters=(
        RoleFilter("junction_link", "link", "link_type", ("100", "102")),
        RoleFilter("ordinary_link", "link", "link_type", ("300",)),
        RoleFilter("pocket_link", "link", "turn", ("1", "3")),
        RoleFilter("uturn_link", "link", "turn", ("3",)),
        RoleFilter("uturn_marker", "lane_line", "line_kind", ("502",)),
        RoleFilter(
            "junction_node",
            "node",
            values=("100",),
            attr_names=("node_type1", "node_type2", "node_type3"),
        ),
    ),
    role_keys=(RoleKey("node_group", "node", "group_id"),),
    reciprocal_relationships=(
        ReciprocalRelationshipRule("nt2_link", "R_LinkID", "r_link_id", "L_LinkID", "l_link_id"),
        ReciprocalRelationshipRule("nt2_link", "L_LinkID", "l_link_id", "R_LinkID", "r_link_id"),
    ),
)
