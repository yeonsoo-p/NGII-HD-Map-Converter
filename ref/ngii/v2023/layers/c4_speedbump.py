from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Polygon3D
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

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class C4_Type(StrEnum):
    높이있는방지턱 = "1"
    높이없는방지턱표시 = "2"
    기타방지턱 = "3"


@dataclass(slots=True, eq=False)
class C4_Data(BaseData):
    # Geometry
    geometry: Polygon3D

    # Fields defined by documentation
    c4_type: C4_Type | None
    link: A2_Data | UnresolvedData
    ref_lane: int


@dataclass(slots=True, eq=False)
class C4_Layer(BaseLayer[C4_Data]):
    layer_name: ClassVar[str] = "C4_SPEEDBUMP"
    file_name: ClassVar[str] = "C4_SPEEDBUMP.shp"
    korean_name: ClassVar[str] = "과속방지턱"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C4_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.c4 = rows
            return rows

        layer = cls(section=section)
        section.c4 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "polygon", source_id)
            c4_type = parse_enum(
                C4_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Polygon3D):
                continue
            link = cast(
                A2_Data | UnresolvedData,
                section.resolve_required(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "speed-bump link",
                ),
            )
            data = C4_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                c4_type=c4_type,
                link=link,
                ref_lane=int_or_zero(row_value(row, "Ref_Lane")),
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if isinstance(link, A2_Data):
                link.speed_bumps.append(data)
            layer.data[data.id] = data
        return layer
