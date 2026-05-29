from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Polygon3D
from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows
from ngii.v2023.layers.a4_subsidiarysection import A4_Data, A4_Layer

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class A5_Type(StrEnum):
    일반주차장 = "1"
    화물차전용주차장 = "2"
    장애인전용주차장 = "3"
    노인전용주차장 = "4"
    여성전용주차장_가족배려주차장 = "5"
    버스전용주차장 = "6"
    전기차전용주차장 = "7"
    기타주차장 = "9"


@dataclass(slots=True, eq=False)
class A5_Data(BaseData):
    # Geometry
    geometry: Polygon3D

    # Fields defined by documentation
    a5_type: A5_Type | None
    section: A4_Data | UnresolvedData


@dataclass(slots=True, eq=False)
class A5_Layer(BaseLayer[A5_Data]):
    layer_name: ClassVar[str] = "A5_PARKINGLOT"
    file_name: ClassVar[str] = "A5_PARKINGLOT.shp"
    korean_name: ClassVar[str] = "주차면"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> A5_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.a5 = rows
            return rows

        layer = cls(section=section)
        section.a5 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "polygon", source_id)
            a5_type = parse_enum(
                A5_Type,
                row_value(row, "Type", "PLType"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Polygon3D):
                continue
            parking_section = cast(
                A4_Data | UnresolvedData,
                section.resolve_required(
                    A4_Layer.layer_name,
                    section.a4,
                    row_value(row, "SectionID", "A4ID"),
                    source_id,
                    "parking-lot section",
                ),
            )
            data = A5_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                a5_type=a5_type,
                section=parking_section,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if isinstance(parking_section, A4_Data):
                parking_section.parking_lots.append(data)
            layer.data[data.id] = data
        return layer
