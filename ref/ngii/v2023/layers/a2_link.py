from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Line3D
from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    optional_float,
    optional_int,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows

if TYPE_CHECKING:
    from ngii.v2023.layers import (
        A1_Data,
        A3_Data,
        A4_Data,
        B1_Data,
        B2_Data,
        B3_Data,
        C1_Data,
        C2_Data,
        C4_Data,
        C5_Data,
    )
    from ngii.v2023.section import Section

A1_LAYER_NAME = "A1_NODE"
A3_LAYER_NAME = "A3_DRIVEWAYSECTION"
A4_LAYER_NAME = "A4_SUBSIDIARYSECTION"


class RoadRank(StrEnum):
    고속도로 = "1"
    국도 = "2"
    특별광역시도 = "3"
    국가지원지방도 = "4"
    지방도 = "5"
    시도 = "6"
    군도 = "7"
    구도 = "8"
    기타도로 = "9"


class RoadType(StrEnum):
    일반도로 = "1"
    터널 = "2"
    교량 = "3"
    지하도로 = "4"
    고가도로 = "5"


class LinkType(StrEnum):
    교차로내주행경로 = "1"
    톨게이트차로_하이패스 = "2"
    톨게이트차로_비하이패스 = "3"
    버스전용차로 = "4"
    가변차선차로 = "5"
    일반주행차로 = "6"
    휴게소진입로 = "7"
    휴게소내주행경로 = "8"
    휴게소진출로 = "9"
    졸음쉼터진입로 = "10"
    졸음쉼터내주행경로 = "11"
    졸음쉼터진출로 = "12"
    교차로진입로 = "13"
    교차로진출로 = "14"
    기타차로 = "99"


@dataclass(slots=True, eq=False)
class A2_Data(BaseData):
    # Geometry
    geometry: Line3D

    # Fields defined by documentation
    road_rank: RoadRank | None
    road_type: RoadType | None
    road_no: str | None
    link_type: LinkType | None
    lane_no: int | None
    r_link: A2_Data | UnresolvedData | None
    l_link: A2_Data | UnresolvedData | None
    from_node: A1_Data | UnresolvedData
    to_node: A1_Data | UnresolvedData
    section: A3_Data | A4_Data | UnresolvedData | None
    length: float | None
    its_link_id: str | None

    # Fields defined by reference
    safety_signs: list[B1_Data] = field(default_factory=list, repr=False, compare=False)
    right_surface_line_marks: list[B2_Data] = field(default_factory=list, repr=False, compare=False)
    left_surface_line_marks: list[B2_Data] = field(default_factory=list, repr=False, compare=False)
    surface_marks: list[B3_Data] = field(default_factory=list, repr=False, compare=False)
    traffic_lights: list[C1_Data] = field(default_factory=list, repr=False, compare=False)
    kilo_posts: list[C2_Data] = field(default_factory=list, repr=False, compare=False)
    speed_bumps: list[C4_Data] = field(default_factory=list, repr=False, compare=False)
    height_barriers: list[C5_Data] = field(default_factory=list, repr=False, compare=False)


@dataclass(slots=True, eq=False)
class A2_Layer(BaseLayer[A2_Data]):
    layer_name: ClassVar[str] = "A2_LINK"
    file_name: ClassVar[str] = "A2_LINK.shp"
    korean_name: ClassVar[str] = "주행경로링크"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> A2_Layer | UnresolvedLayer:
        rows = read_layer_rows(section, cls, required=True)
        if rows is None:
            raise RuntimeError(f"{cls.layer_name} is required but no layer was loaded.")
        if not isinstance(rows, list):
            section.a2 = rows
            return rows

        layer = cls(section=section)
        section.a2 = layer
        side_link_refs: list[tuple[A2_Data, object, object]] = []
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "line", source_id)
            road_rank = parse_enum(
                RoadRank,
                row_value(row, "RoadRank"),
                source_id,
                "RoadRank",
            )
            road_type = parse_enum(
                RoadType,
                row_value(row, "RoadType"),
                source_id,
                "RoadType",
            )
            link_type = parse_enum(
                LinkType,
                row_value(row, "LinkType"),
                source_id,
                "LinkType",
            )
            if not isinstance(geometry, Line3D):
                continue
            from_node = cast(
                "A1_Data | UnresolvedData",
                section.resolve_required(
                    A1_LAYER_NAME,
                    section.a1,
                    row_value(row, "FromNodeID"),
                    source_id,
                    "from-node",
                ),
            )
            to_node = cast(
                "A1_Data | UnresolvedData",
                section.resolve_required(
                    A1_LAYER_NAME,
                    section.a1,
                    row_value(row, "ToNodeID"),
                    source_id,
                    "to-node",
                ),
            )
            road_section = cast(
                "A3_Data | A4_Data | UnresolvedData | None",
                section.resolve_section_ref(
                    A3_LAYER_NAME,
                    A4_LAYER_NAME,
                    row_value(row, "SectionID"),
                    source_id,
                ),
            )
            data = A2_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                road_rank=road_rank,
                road_type=road_type,
                road_no=optional_text(row_value(row, "RoadNo")),
                link_type=link_type,
                lane_no=optional_int(row_value(row, "LaneNo")),
                r_link=None,
                l_link=None,
                from_node=from_node,
                to_node=to_node,
                section=road_section,
                length=optional_float(row_value(row, "Length")),
                its_link_id=optional_text(row_value(row, "ITSLinkID")),
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if not isinstance(from_node, UnresolvedData):
                from_node.outgoing_links.append(data)
            if not isinstance(to_node, UnresolvedData):
                to_node.incoming_links.append(data)
            if road_section is not None and not isinstance(road_section, UnresolvedData):
                road_section.links.append(data)
            layer.data[data.id] = data
            side_link_refs.append(
                (
                    data,
                    row_value(row, "R_LinkID", "R_linkID"),
                    row_value(row, "L_LinkID", "L_linkID"),
                )
            )

        for data, right_raw, left_raw in side_link_refs:
            data.r_link = cast(
                A2_Data | UnresolvedData | None,
                section.resolve_optional(
                    cls.layer_name,
                    layer,
                    right_raw,
                    data.id,
                    "right-link",
                ),
            )
            data.l_link = cast(
                A2_Data | UnresolvedData | None,
                section.resolve_optional(
                    cls.layer_name,
                    layer,
                    left_raw,
                    data.id,
                    "left-link",
                ),
            )
        return layer
