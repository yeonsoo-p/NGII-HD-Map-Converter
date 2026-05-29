from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from ngii.geometry import Point3D
from ngii.v2023.data import (
    BaseData,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows

if TYPE_CHECKING:
    from ngii.v2023.layers import B1_Data, C1_Data
    from ngii.v2023.section import Section


class C6_Type(StrEnum):
    신호기지주 = "1"
    교통표지지주 = "2"


@dataclass(slots=True, eq=False)
class C6_Data(BaseData):
    # Geometry
    geometry: Point3D

    # Fields defined by documentation
    c6_type: C6_Type | None

    # Fields defined by reference
    safety_signs: list[B1_Data] = field(default_factory=list, repr=False, compare=False)
    traffic_lights: list[C1_Data] = field(default_factory=list, repr=False, compare=False)


@dataclass(slots=True, eq=False)
class C6_Layer(BaseLayer[C6_Data]):
    layer_name: ClassVar[str] = "C6_POSTPOINT"
    file_name: ClassVar[str] = "C6_POSTPOINT.shp"
    korean_name: ClassVar[str] = "지주"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C6_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.c6 = rows
            return rows

        layer = cls(section=section)
        section.c6 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "point", source_id)
            c6_type = parse_enum(
                C6_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            if not isinstance(geometry, Point3D):
                continue
            data = C6_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                c6_type=c6_type,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            layer.data[data.id] = data
        return layer
