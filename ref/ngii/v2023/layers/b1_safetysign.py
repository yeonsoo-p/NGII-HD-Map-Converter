from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Point3D, Polygon3D
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


class B1_Type(StrEnum):
    주의표지 = "1"
    지시표지 = "2"
    규제표지 = "3"
    보조표지 = "4"


@dataclass(slots=True, eq=False)
class B1_Data(BaseData):
    # Geometry
    geometry: Point3D | Polygon3D

    # Fields defined by documentation
    b1_type: B1_Type | None
    link: A2_Data | UnresolvedData
    ref_lane: int
    post: C6_Data | UnresolvedData | None


@dataclass(slots=True, eq=False)
class B1_Layer(BaseLayer[B1_Data]):
    layer_name: ClassVar[str] = "B1_SAFETYSIGN"
    file_name: ClassVar[str] = "B1_SAFETYSIGN.shp"
    korean_name: ClassVar[str] = "안전표지"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> B1_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.b1 = rows
            return rows

        layer = cls(section=section)
        section.b1 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "point_or_polygon", source_id)
            b1_type = parse_enum(
                B1_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Point3D | Polygon3D):
                continue
            link = cast(
                A2_Data | UnresolvedData,
                section.resolve_required(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "sign link",
                ),
            )
            post = cast(
                C6_Data | UnresolvedData | None,
                section.resolve_optional(
                    C6_Layer.layer_name,
                    section.c6,
                    row_value(row, "PostID"),
                    source_id,
                    "sign post",
                ),
            )
            data = B1_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                b1_type=b1_type,
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
                link.safety_signs.append(data)
            if isinstance(post, C6_Data):
                post.safety_signs.append(data)
            layer.data[data.id] = data
        return layer
