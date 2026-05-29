from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Point3D
from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    int_or_zero,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows
from ngii.v2023.layers.a2_link import A2_Data, A2_Layer
from ngii.v2023.layers.c6_postpoint import C6_Data, C6_Layer

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class C1_Type(StrEnum):
    차량횡형_삼색등 = "1"
    차량횡형_사색등A = "2"
    차량횡형_사색등B = "3"
    차량횡형_화살표삼색등 = "4"
    차량종형_삼색등 = "5"
    차량종형_화살표삼색등 = "6"
    차량종형_사색등 = "7"
    버스삼색등 = "8"
    가변형_가변등 = "9"
    경보형_가변등 = "10"
    보행등 = "11"
    자전거종형_삼색등 = "12"
    자전거종형_이색등 = "13"
    차량보조등_종형삼색등 = "14"
    차량보조등_종형사색등 = "15"
    기타_신호등_유형 = "99"


@dataclass(slots=True, eq=False)
class C1_Data(BaseData):
    # Geometry
    geometry: Point3D

    # Fields defined by documentation
    c1_type: C1_Type | None
    link: A2_Data | UnresolvedData
    ref_lane: int
    post: C6_Data | UnresolvedData | None


@dataclass(slots=True, eq=False)
class C1_Layer(BaseLayer[C1_Data]):
    layer_name: ClassVar[str] = "C1_TRAFFICLIGHT"
    file_name: ClassVar[str] = "C1_TRAFFICLIGHT.shp"
    korean_name: ClassVar[str] = "신호등"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C1_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.c1 = rows
            return rows

        layer = cls(section=section)
        section.c1 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "point", source_id)
            c1_type = parse_enum(
                C1_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Point3D):
                continue
            link = cast(
                A2_Data | UnresolvedData,
                section.resolve_required(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "traffic-light link",
                ),
            )
            post = cast(
                C6_Data | UnresolvedData | None,
                section.resolve_optional(
                    C6_Layer.layer_name,
                    section.c6,
                    row_value(row, "PostID"),
                    source_id,
                    "traffic-light post",
                ),
            )
            data = C1_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                c1_type=c1_type,
                link=link,
                ref_lane=int_or_zero(row_value(row, "Ref_Lane")),
                maker=text(row, "Maker"),
                post=post,
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if isinstance(link, A2_Data):
                link.traffic_lights.append(data)
            if isinstance(post, C6_Data):
                post.traffic_lights.append(data)
            layer.data[data.id] = data
        return layer
