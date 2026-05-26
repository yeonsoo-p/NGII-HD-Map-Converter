"""NGII 2023.07 layer registry used by the loader and sanity checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ngii2xodr.ngii.data.features import FeatureRecord, NGIIFeature
from ngii2xodr.ngii.data.layers import (
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
from ngii2xodr.ngii.data.layers.a1_node import make_feature as make_a1
from ngii2xodr.ngii.data.layers.a2_link import make_feature as make_a2
from ngii2xodr.ngii.data.layers.a3_drivewaysection import make_feature as make_a3
from ngii2xodr.ngii.data.layers.a4_subsidiarysection import make_feature as make_a4
from ngii2xodr.ngii.data.layers.a5_parkinglot import make_feature as make_a5
from ngii2xodr.ngii.data.layers.b1_safetysign import make_feature as make_b1
from ngii2xodr.ngii.data.layers.b2_surfacelinemark import make_feature as make_b2
from ngii2xodr.ngii.data.layers.b3_surfacemark import make_feature as make_b3
from ngii2xodr.ngii.data.layers.c1_trafficlight import make_feature as make_c1
from ngii2xodr.ngii.data.layers.c2_kilopost import make_feature as make_c2
from ngii2xodr.ngii.data.layers.c3_vehicleprotectionsafety import make_feature as make_c3
from ngii2xodr.ngii.data.layers.c4_speedbump import make_feature as make_c4
from ngii2xodr.ngii.data.layers.c5_heightbarrier import make_feature as make_c5
from ngii2xodr.ngii.data.layers.c6_postpoint import make_feature as make_c6

GeometryKind = Literal["point", "line", "polygon"]
FieldType = Literal["text", "integer", "float"]


@dataclass(slots=True, frozen=True)
class FieldRule:
    name: str
    required: bool
    field_type: FieldType
    max_length: int | None = None
    code_list: dict[str, str] | None = None


@dataclass(slots=True, frozen=True)
class RelationshipRule:
    column_name: str
    source_attr: str
    target_attrs: tuple[str, ...]
    required: bool


@dataclass(slots=True, frozen=True)
class LayerSpec:
    layer_name: str
    python_attr: str
    filename: str
    geometry_kind: GeometryKind
    feature_type: type[NGIIFeature]
    factory: Callable[[FeatureRecord], NGIIFeature]
    required_layer: bool
    field_rules: tuple[FieldRule, ...]
    relationships: tuple[RelationshipRule, ...] = ()


def text_rule(
    name: str, max_length: int, *, required: bool, code_list: dict[str, str] | None = None
) -> FieldRule:
    return FieldRule(name, required, "text", max_length, code_list)


def integer_rule(name: str, *, required: bool) -> FieldRule:
    return FieldRule(name, required, "integer")


def float_rule(name: str, *, required: bool) -> FieldRule:
    return FieldRule(name, required, "float")


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
        geometry_kind="point",
        feature_type=A1_NODE,
        factory=make_a1,
        required_layer=True,
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
        geometry_kind="line",
        feature_type=A2_LINK,
        factory=make_a2,
        required_layer=True,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("RoadRank", 1, required=True, code_list=A2_LINK.ROAD_RANK_LABEL),
            text_rule("RoadType", 1, required=True, code_list=A2_LINK.ROAD_TYPE_LABEL),
            text_rule("RoadNo", 5, required=False),
            text_rule("LinkType", 2, required=True, code_list=A2_LINK.LINK_TYPE_LABEL),
            integer_rule("LaneNo", required=False),
            text_rule("R_LinkID", 12, required=False),
            text_rule("L_LinkID", 12, required=False),
            text_rule("FromNodeID", 12, required=True),
            text_rule("ToNodeID", 12, required=True),
            text_rule("SectionID", 12, required=False),
            float_rule("Length", required=False),
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
        geometry_kind="polygon",
        feature_type=A3_DRIVEWAYSECTION,
        factory=make_a3,
        required_layer=False,
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
        geometry_kind="polygon",
        feature_type=A4_SUBSIDIARYSECTION,
        factory=make_a4,
        required_layer=False,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("SubType", 1, required=True, code_list=A4_SUBSIDIARYSECTION.SUBTYPE_LABEL),
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
        geometry_kind="polygon",
        feature_type=A5_PARKINGLOT,
        factory=make_a5,
        required_layer=False,
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
        geometry_kind="polygon",
        feature_type=B1_SAFETYSIGN,
        factory=make_b1,
        required_layer=False,
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
        geometry_kind="line",
        feature_type=B2_SURFACELINEMARK,
        factory=make_b2,
        required_layer=True,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 3, required=True, code_list=B2_SURFACELINEMARK.TYPE_LABEL),
            text_rule("Kind", 5, required=True, code_list=B2_SURFACELINEMARK.KIND_LABEL),
            text_rule("R_LinkID", 12, required=False),
            text_rule("L_LinkID", 12, required=False),
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
        geometry_kind="polygon",
        feature_type=B3_SURFACEMARK,
        factory=make_b3,
        required_layer=False,
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
        geometry_kind="point",
        feature_type=C1_TRAFFICLIGHT,
        factory=make_c1,
        required_layer=False,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 2, required=True, code_list=C1_TRAFFICLIGHT.TYPE_LABEL),
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
        layer_name="C2_KILOPOST",
        python_attr="c2_kilopost",
        filename="C2_KILOPOST.shp",
        geometry_kind="point",
        feature_type=C2_KILOPOST,
        factory=make_c2,
        required_layer=False,
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
        geometry_kind="line",
        feature_type=C3_VEHICLEPROTECTIONSAFETY,
        factory=make_c3,
        required_layer=True,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 2, required=True, code_list=C3_VEHICLEPROTECTIONSAFETY.TYPE_LABEL),
            text_rule(
                "IsCentral",
                1,
                required=True,
                code_list=C3_VEHICLEPROTECTIONSAFETY.IS_CENTRAL_LABEL,
            ),
            text_rule(
                "LowHigh",
                1,
                required=False,
                code_list=C3_VEHICLEPROTECTIONSAFETY.LOW_HIGH_LABEL,
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
        geometry_kind="polygon",
        feature_type=C4_SPEEDBUMP,
        factory=make_c4,
        required_layer=False,
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
        geometry_kind="line",
        feature_type=C5_HEIGHTBARRIER,
        factory=make_c5,
        required_layer=False,
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
        geometry_kind="point",
        feature_type=C6_POSTPOINT,
        factory=make_c6,
        required_layer=False,
        field_rules=(
            *COMMON_FIELD_RULES,
            text_rule("Type", 1, required=True, code_list=C6_POSTPOINT.TYPE_LABEL),
        ),
    ),
)

SPECS_BY_FILENAME = {spec.filename.upper(): spec for spec in LAYER_SPECS}
SPECS_BY_LAYER_NAME = {spec.layer_name: spec for spec in LAYER_SPECS}
