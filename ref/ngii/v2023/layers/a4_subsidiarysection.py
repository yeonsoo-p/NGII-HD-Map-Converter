from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from ngii.geometry import Polygon3D
from ngii.v2023.data import (
    BaseData,
    int_or_zero,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows

if TYPE_CHECKING:
    from ngii.v2023.layers import A2_Data, A5_Data
    from ngii.v2023.section import Section


class SubType(StrEnum):
    휴게소 = "1"
    졸음쉼터 = "2"
    보도 = "3"
    자전거도로 = "4"
    기타부속구간 = "9"


class Direction(StrEnum):
    상행 = "1"
    하행 = "2"
    양방향 = "3"


@dataclass(slots=True, eq=False)
class A4_Data(BaseData):
    # Geometry
    geometry: Polygon3D

    # Fields defined by documentation
    sub_type: SubType | None
    name: str
    direction: Direction | None
    gas_station: int
    lpg_station: int
    ev_charger: int
    toilet: int

    # Fields defined by reference
    links: list[A2_Data] = field(default_factory=list, repr=False, compare=False)
    parking_lots: list[A5_Data] = field(default_factory=list, repr=False, compare=False)


@dataclass(slots=True, eq=False)
class A4_Layer(BaseLayer[A4_Data]):
    layer_name: ClassVar[str] = "A4_SUBSIDIARYSECTION"
    file_name: ClassVar[str] = "A4_SUBSIDIARYSECTION.shp"
    korean_name: ClassVar[str] = "부속구간"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> A4_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.a4 = rows
            return rows

        layer = cls(section=section)
        section.a4 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "polygon", source_id)
            sub_type = parse_enum(
                SubType,
                row_value(row, "SubType"),
                source_id,
                "SubType",
            )
            direction = parse_enum(
                Direction,
                row_value(row, "Direction"),
                source_id,
                "Direction",
            )
            if not isinstance(geometry, Polygon3D):
                continue
            data = A4_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                sub_type=sub_type,
                name=text(row, "Name"),
                direction=direction,
                gas_station=int_or_zero(row_value(row, "GasStation")),
                lpg_station=int_or_zero(row_value(row, "LpgStation")),
                ev_charger=int_or_zero(row_value(row, "EvCharger")),
                toilet=int_or_zero(row_value(row, "Toilet")),
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            layer.data[data.id] = data
        return layer
