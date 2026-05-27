"""NGII 2023.07 layer registry used by the loader and sanity checks."""

from __future__ import annotations

from ngii2xodr.ngii.data.schema import (
    LayerSpec,
    ReciprocalRelationshipRule,
    RelationshipRule,
    RoleFilter,
    SchemaDefinition,
    float_rule,
    integer_rule,
    text_rule,
)
from ngii2xodr.ngii.data.v2023.layers import (
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
from ngii2xodr.ngii.data.v2023.layers.a1_node import make_feature as make_a1
from ngii2xodr.ngii.data.v2023.layers.a2_link import make_feature as make_a2
from ngii2xodr.ngii.data.v2023.layers.a3_drivewaysection import make_feature as make_a3
from ngii2xodr.ngii.data.v2023.layers.a4_subsidiarysection import make_feature as make_a4
from ngii2xodr.ngii.data.v2023.layers.a5_parkinglot import make_feature as make_a5
from ngii2xodr.ngii.data.v2023.layers.b1_safetysign import make_feature as make_b1
from ngii2xodr.ngii.data.v2023.layers.b2_surfacelinemark import make_feature as make_b2
from ngii2xodr.ngii.data.v2023.layers.b3_surfacemark import make_feature as make_b3
from ngii2xodr.ngii.data.v2023.layers.c1_trafficlight import make_feature as make_c1
from ngii2xodr.ngii.data.v2023.layers.c2_kilopost import make_feature as make_c2
from ngii2xodr.ngii.data.v2023.layers.c3_vehicleprotectionsafety import make_feature as make_c3
from ngii2xodr.ngii.data.v2023.layers.c4_speedbump import make_feature as make_c4
from ngii2xodr.ngii.data.v2023.layers.c5_heightbarrier import make_feature as make_c5
from ngii2xodr.ngii.data.v2023.layers.c6_postpoint import make_feature as make_c6

COMMON_FIELD_RULES = (
    text_rule("ID", 12, required=True),
    text_rule("AdminCode", 3, required=True),
    text_rule("Maker", 20, required=True),
    text_rule("UpdateDate", 8, required=False),
    text_rule("Version", 4, required=True),
    text_rule("Remark", 30, required=False),
    text_rule("HistType", 5, required=False),
    text_rule("HistRemark", 30, required=False),
)

PRESENCE_LABEL = A4_SUBSIDIARYSECTION.PRESENCE_LABEL


LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        layer_name="A1_NODE",
        python_attr="a1_node",
        filename="A1_NODE.shp",
        geometry_kinds=("point",),
        feature_type=A1_NODE,
        factory=make_a1,
        required_layer=True,
        roles=("node",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("NodeType", 2, required=True, code_list=A1_NODE.NODE_TYPE_LABEL),
            text_rule("ITSNodeID", 10, required=False),
        ),
    ),
    LayerSpec(
        layer_name="A2_LINK",
        python_attr="a2_link",
        filename="A2_LINK.shp",
        geometry_kinds=("line",),
        feature_type=A2_LINK,
        factory=make_a2,
        required_layer=True,
        roles=("link",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("RoadRank", 1, required=True, code_list=A2_LINK.ROAD_RANK_LABEL),
            text_rule("RoadType", 1, required=True, code_list=A2_LINK.ROAD_TYPE_LABEL),
            text_rule("RoadNo", 5, required=False),
            text_rule("LinkType", 2, required=True, code_list=A2_LINK.LINK_TYPE_LABEL),
            integer_rule("LaneNo", required=False),
            text_rule("R_LinkID", 12, required=False),
            text_rule("L_LinkID", 12, required=False, column_aliases=("L_LinKID",)),
            text_rule("FromNodeID", 12, required=True),
            text_rule("ToNodeID", 12, required=True, column_aliases=("ToNodeId",)),
            text_rule("SectionID", 12, required=False),
            float_rule(
                "Length", required=False, attr_name="length_m", array_aliases=("lengths_m",)
            ),
            text_rule("ITSLinkID", 30, required=False),
        ),
        relationships=(
            RelationshipRule("FromNodeID", "from_node_id", ("a1_node",), True),
            RelationshipRule("ToNodeID", "to_node_id", ("a1_node",), True),
            RelationshipRule("R_LinkID", "r_link_id", ("a2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("a2_link",), False),
            RelationshipRule(
                "SectionID",
                "section_id",
                ("a3_drivewaysection", "a4_subsidiarysection"),
                False,
            ),
        ),
    ),
    LayerSpec(
        layer_name="A3_DRIVEWAYSECTION",
        python_attr="a3_drivewaysection",
        filename="A3_DRIVEWAYSECTION.shp",
        geometry_kinds=("polygon",),
        feature_type=A3_DRIVEWAYSECTION,
        factory=make_a3,
        required_layer=False,
        roles=("driveway_section",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Kind", 1, required=True, code_list=A3_DRIVEWAYSECTION.KIND_LABEL),
            text_rule("RoadType", 1, required=True, code_list=A3_DRIVEWAYSECTION.ROAD_TYPE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="A4_SUBSIDIARYSECTION",
        python_attr="a4_subsidiarysection",
        filename="A4_SUBSIDIARYSECTION.shp",
        geometry_kinds=("polygon",),
        feature_type=A4_SUBSIDIARYSECTION,
        factory=make_a4,
        required_layer=False,
        roles=("subsidiary_section",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule(
                "SubType",
                1,
                required=True,
                code_list=A4_SUBSIDIARYSECTION.SUBTYPE_LABEL,
                attr_name="subtype",
            ),
            text_rule("Name", 30, required=True),
            text_rule(
                "Direction", 1, required=True, code_list=A4_SUBSIDIARYSECTION.DIRECTION_LABEL
            ),
            text_rule("GasStation", 1, required=True, code_list=PRESENCE_LABEL),
            text_rule("LpgStation", 1, required=True, code_list=PRESENCE_LABEL),
            text_rule("EvCharger", 1, required=True, code_list=PRESENCE_LABEL),
            text_rule("Toilet", 1, required=True, code_list=PRESENCE_LABEL),
        ),
    ),
    LayerSpec(
        layer_name="A5_PARKINGLOT",
        python_attr="a5_parkinglot",
        filename="A5_PARKINGLOT.shp",
        geometry_kinds=("polygon",),
        feature_type=A5_PARKINGLOT,
        factory=make_a5,
        required_layer=False,
        roles=("parking_lot",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=A5_PARKINGLOT.TYPE_LABEL),
            text_rule("SectionID", 12, required=True),
        ),
        relationships=(
            RelationshipRule("SectionID", "section_id", ("a4_subsidiarysection",), True),
        ),
    ),
    LayerSpec(
        layer_name="B1_SAFETYSIGN",
        python_attr="b1_safetysign",
        filename="B1_SAFETYSIGN.shp",
        geometry_kinds=("point", "polygon"),
        feature_type=B1_SAFETYSIGN,
        factory=make_b1,
        required_layer=False,
        roles=("traffic_sign",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=B1_SAFETYSIGN.TYPE_LABEL),
            text_rule("LinkID", 12, required=True),
            integer_rule("Ref_Lane", required=True),
            text_rule("PostID", 12, required=False),
        ),
        relationships=(
            RelationshipRule("LinkID", "link_id", ("a2_link",), True),
            RelationshipRule("PostID", "post_id", ("c6_postpoint",), False),
        ),
    ),
    LayerSpec(
        layer_name="B2_SURFACELINEMARK",
        python_attr="b2_surfacelinemark",
        filename="B2_SURFACELINEMARK.shp",
        geometry_kinds=("line",),
        feature_type=B2_SURFACELINEMARK,
        factory=make_b2,
        required_layer=True,
        roles=("lane_line",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 3, required=True, code_list=B2_SURFACELINEMARK.TYPE_LABEL),
            text_rule("Kind", 5, required=True, code_list=B2_SURFACELINEMARK.KIND_LABEL),
            text_rule("R_LinkID", 12, required=False, column_aliases=("R_linkID",)),
            text_rule("L_LinkID", 12, required=False, column_aliases=("L_linkID",)),
        ),
        relationships=(
            RelationshipRule("R_LinkID", "r_link_id", ("a2_link",), False),
            RelationshipRule("L_LinkID", "l_link_id", ("a2_link",), False),
        ),
    ),
    LayerSpec(
        layer_name="B3_SURFACEMARK",
        python_attr="b3_surfacemark",
        filename="B3_SURFACEMARK.shp",
        geometry_kinds=("polygon",),
        feature_type=B3_SURFACEMARK,
        factory=make_b3,
        required_layer=False,
        roles=("road_marking",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=B3_SURFACEMARK.TYPE_LABEL),
            text_rule("Kind", 4, required=True, code_list=B3_SURFACEMARK.KIND_LABEL),
            text_rule("LinkID", 12, required=False),
        ),
        relationships=(RelationshipRule("LinkID", "link_id", ("a2_link",), False),),
    ),
    LayerSpec(
        layer_name="C1_TRAFFICLIGHT",
        python_attr="c1_trafficlight",
        filename="C1_TRAFFICLIGHT.shp",
        filename_aliases=("C1_TRAFFICLIGH.shp",),
        geometry_kinds=("point",),
        feature_type=C1_TRAFFICLIGHT,
        factory=make_c1,
        required_layer=False,
        roles=("traffic_light",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 2, required=True, code_list=C1_TRAFFICLIGHT.TYPE_LABEL),
            text_rule("LinkID", 12, required=True),
            integer_rule("Ref_Lane", required=True),
            text_rule("PostID", 12, required=False, column_aliases=("postID",)),
        ),
        relationships=(
            RelationshipRule("LinkID", "link_id", ("a2_link",), True),
            RelationshipRule("PostID", "post_id", ("c6_postpoint",), False),
        ),
    ),
    LayerSpec(
        layer_name="C2_KILOPOST",
        python_attr="c2_kilopost",
        filename="C2_KILOPOST.shp",
        geometry_kinds=("point",),
        feature_type=C2_KILOPOST,
        factory=make_c2,
        required_layer=False,
        roles=("kilopost",),
        field_rules=(
            *COMMON_FIELD_RULES,
            float_rule("Distance", required=True),
            text_rule("Origin", 30, required=False),
            text_rule("LinkID", 12, required=True),
            integer_rule("Ref_Lane", required=True),
        ),
        relationships=(RelationshipRule("LinkID", "link_id", ("a2_link",), True),),
    ),
    LayerSpec(
        layer_name="C3_VEHICLEPROTECTIONSAFETY",
        python_attr="c3_vehicleprotectionsafety",
        filename="C3_VEHICLEPROTECTIONSAFETY.shp",
        geometry_kinds=("line",),
        feature_type=C3_VEHICLEPROTECTIONSAFETY,
        factory=make_c3,
        required_layer=True,
        roles=("barrier",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 2, required=True, code_list=C3_VEHICLEPROTECTIONSAFETY.TYPE_LABEL),
            text_rule(
                "IsCentral",
                1,
                required=True,
                code_list=C3_VEHICLEPROTECTIONSAFETY.IS_CENTRAL_LABEL,
                column_aliases=("isCentral",),
                array_aliases=("is_central",),
            ),
            text_rule(
                "LowHigh",
                1,
                required=False,
                code_list=C3_VEHICLEPROTECTIONSAFETY.LOW_HIGH_LABEL,
                array_aliases=("low_high",),
            ),
            text_rule("Ref_ID", 12, required=False),
        ),
        relationships=(
            RelationshipRule("Ref_ID", "ref_id", ("c3_vehicleprotectionsafety",), False),
        ),
    ),
    LayerSpec(
        layer_name="C4_SPEEDBUMP",
        python_attr="c4_speedbump",
        filename="C4_SPEEDBUMP.shp",
        geometry_kinds=("polygon",),
        feature_type=C4_SPEEDBUMP,
        factory=make_c4,
        required_layer=False,
        roles=("speed_bump",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=C4_SPEEDBUMP.TYPE_LABEL),
            text_rule("LinkID", 12, required=True),
            integer_rule("Ref_Lane", required=True),
        ),
        relationships=(RelationshipRule("LinkID", "link_id", ("a2_link",), True),),
    ),
    LayerSpec(
        layer_name="C5_HEIGHTBARRIER",
        python_attr="c5_heightbarrier",
        filename="C5_HEIGHTBARRIER.shp",
        geometry_kinds=("line",),
        feature_type=C5_HEIGHTBARRIER,
        factory=make_c5,
        required_layer=False,
        roles=("height_barrier",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=C5_HEIGHTBARRIER.TYPE_LABEL),
            text_rule("LinkID", 12, required=True),
            integer_rule("Ref_Lane", required=True),
        ),
        relationships=(RelationshipRule("LinkID", "link_id", ("a2_link",), True),),
    ),
    LayerSpec(
        layer_name="C6_POSTPOINT",
        python_attr="c6_postpoint",
        filename="C6_POSTPOINT.shp",
        geometry_kinds=("point",),
        feature_type=C6_POSTPOINT,
        factory=make_c6,
        required_layer=False,
        roles=("support_post",),
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=C6_POSTPOINT.TYPE_LABEL),
        ),
    ),
)

SCHEMA = SchemaDefinition(
    version="2023.07",
    layer_specs=LAYER_SPECS,
    role_filters=(
        RoleFilter("junction_link", "link", "link_type", ("1",)),
        RoleFilter("ordinary_link", "link", "link_type", ("4", "5", "6")),
        RoleFilter("pocket_link", "link", "lane_no", numeric_min=90.0),
        RoleFilter("uturn_marker", "lane_line", "kind", ("502",)),
        RoleFilter("junction_node", "node", "node_type", ("1",)),
    ),
    reciprocal_relationships=(
        ReciprocalRelationshipRule("a2_link", "R_LinkID", "r_link_id", "L_LinkID", "l_link_id"),
        ReciprocalRelationshipRule("a2_link", "L_LinkID", "l_link_id", "R_LinkID", "r_link_id"),
        ReciprocalRelationshipRule(
            "c3_vehicleprotectionsafety", "Ref_ID", "ref_id", "Ref_ID", "ref_id"
        ),
    ),
)
