from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Point3D
from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    float_or_zero,
    int_or_zero,
    optional_text,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows
from ngii.v2023.layers.a2_link import A2_Data, A2_Layer

if TYPE_CHECKING:
    from ngii.v2023.section import Section


@dataclass(slots=True, eq=False)
class C2_Data(BaseData):
    # Geometry
    geometry: Point3D

    # Fields defined by documentation
    distance: float
    origin: str | None
    link: A2_Data | UnresolvedData
    ref_lane: int


@dataclass(slots=True, eq=False)
class C2_Layer(BaseLayer[C2_Data]):
    layer_name: ClassVar[str] = "C2_KILOPOST"
    file_name: ClassVar[str] = "C2_KILOPOST.shp"
    korean_name: ClassVar[str] = "킬로포스트"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C2_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.c2 = rows
            return rows

        layer = cls(section=section)
        section.c2 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "point", source_id)
            if not isinstance(geometry, Point3D):
                continue
            link = cast(
                A2_Data | UnresolvedData,
                section.resolve_required(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "kilopost link",
                ),
            )
            data = C2_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                distance=float_or_zero(row_value(row, "Distance")),
                origin=optional_text(row_value(row, "Origin")),
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
                link.kilo_posts.append(data)
            layer.data[data.id] = data
        return layer
