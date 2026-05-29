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
from ngii.v2023.layers.a2_link import A2_Data, A2_Layer

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class B3_Type(StrEnum):
    화살표 = "1"
    횡단보도 = "5"


class B3_Kind(StrEnum):
    횡단보도 = "5321"
    고원식횡단보도 = "533"
    자전거횡단보도 = "534"
    직진 = "5371"
    좌회전 = "5372"
    우회전 = "5373"
    좌우회전 = "5374"
    전방향 = "5379"
    직진_및_좌회전 = "5381"
    직진_및_우회전 = "5382"
    직진_및_유턴 = "5383"
    유턴 = "5391"
    좌회전_및_유턴 = "5392"
    차로변경_좌로합류 = "5431"
    차로변경_우로합류 = "5432"
    오르막경사면 = "544"
    기타_지시표시 = "599"


@dataclass(slots=True, eq=False)
class B3_Data(BaseData):
    # Geometry
    geometry: Polygon3D

    # Fields defined by documentation
    b3_type: B3_Type | None
    b3_kind: B3_Kind | None
    link: A2_Data | UnresolvedData | None


@dataclass(slots=True, eq=False)
class B3_Layer(BaseLayer[B3_Data]):
    layer_name: ClassVar[str] = "B3_SURFACEMARK"
    file_name: ClassVar[str] = "B3_SURFACEMARK.shp"
    korean_name: ClassVar[str] = "노면표시"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> B3_Layer | UnresolvedLayer | None:
        rows = read_layer_rows(section, cls, required=False)
        if not isinstance(rows, list):
            section.b3 = rows
            return rows

        layer = cls(section=section)
        section.b3 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "polygon", source_id)
            b3_type = parse_enum(
                B3_Type,
                row_value(row, "Type", "MarkType"),
                source_id,
                "Type",
            )
            b3_kind = parse_enum(
                B3_Kind,
                row_value(row, "Kind", "MarkKind"),
                source_id,
                "Kind",
            )
            if not isinstance(geometry, Polygon3D):
                continue
            link = cast(
                A2_Data | UnresolvedData | None,
                section.resolve_optional(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "LinkID"),
                    source_id,
                    "surface mark link",
                ),
            )
            data = B3_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                b3_type=b3_type,
                b3_kind=b3_kind,
                link=link,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if isinstance(link, A2_Data):
                link.surface_marks.append(data)
            layer.data[data.id] = data
        return layer
