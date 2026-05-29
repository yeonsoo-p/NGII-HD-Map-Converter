from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Line3D
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


class C5_Type(StrEnum):
    고가도로_교량 = "1"
    육교 = "2"
    기타_높이제한장애물 = "4"


@dataclass(slots=True, eq=False)
class C5_Data(BaseData):
    # Geometry
    geometry: Line3D

    # Fields defined by documentation
    c5_type: C5_Type | None
    link: A2_Data | UnresolvedData
    ref_lane: int


@dataclass(slots=True, eq=False)
class C5_Layer(BaseLayer[C5_Data]):
    layer_name: ClassVar[str] = "C5_HEIGHTBARRIER"
    file_name: ClassVar[str] = "C5_HEIGHTBARRIER.shp"
    korean_name: ClassVar[str] = "높이장애물"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C5_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.c5 = rows
            return rows

        layer = cls(section=section)
        section.c5 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "line", source_id)
            c5_type = parse_enum(
                C5_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Line3D):
                continue
            link = cast(
                A2_Data | UnresolvedData,
                section.resolve_required(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "height-barrier link",
                ),
            )
            data = C5_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                c5_type=c5_type,
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
                link.height_barriers.append(data)
            layer.data[data.id] = data
        return layer
