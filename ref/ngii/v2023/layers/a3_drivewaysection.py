from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from ngii.geometry import Polygon3D
from ngii.v2023.data import (
    BaseData,
    optional_text,
    parse_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows
from ngii.v2023.layers.a2_link import RoadType

if TYPE_CHECKING:
    from ngii.v2023.layers import A2_Data
    from ngii.v2023.section import Section


class A3_Kind(StrEnum):
    주행구간 = "1"
    자율주행금지구간 = "7"


@dataclass(slots=True, eq=False)
class A3_Data(BaseData):
    # Geometry
    geometry: Polygon3D

    # Fields defined by documentation
    a3_kind: A3_Kind | None
    road_type: RoadType | None

    # Fields defined by reference
    links: list[A2_Data] = field(default_factory=list, repr=False, compare=False)


@dataclass(slots=True, eq=False)
class A3_Layer(BaseLayer[A3_Data]):
    layer_name: ClassVar[str] = "A3_DRIVEWAYSECTION"
    file_name: ClassVar[str] = "A3_DRIVEWAYSECTION.shp"
    korean_name: ClassVar[str] = "구간"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> A3_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.a3 = rows
            return rows

        layer = cls(section=section)
        section.a3 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "polygon", source_id)
            kind = parse_enum(
                A3_Kind,
                row_value(row, "Kind"),
                source_id,
                "Kind",
            )
            road_type = parse_enum(
                RoadType,
                row_value(row, "RoadType"),
                source_id,
                "RoadType",
            )
            if not isinstance(geometry, Polygon3D):
                continue
            data = A3_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                a3_kind=kind,
                road_type=road_type,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            layer.data[data.id] = data
        return layer
