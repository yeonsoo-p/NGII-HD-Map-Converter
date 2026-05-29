from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

from ngii.geometry import Line3D
from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    optional_text,
    parse_enum,
    parse_optional_enum,
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class C3_Type(StrEnum):
    가드레일 = "2"
    콘크리트방호벽 = "3"
    콘크리트연석 = "4"
    무단횡단방지시설 = "5"
    중앙분리대개구부 = "6"
    임시구조물 = "7"
    벽 = "8"
    기타 = "99"


class IsCentral(StrEnum):
    도로_가장자리 = "0"
    중앙분리대 = "1"


class LowHigh(StrEnum):
    상단 = "1"
    하단 = "2"


@dataclass(slots=True, eq=False)
class C3_Data(BaseData):
    # Geometry
    geometry: Line3D

    # Fields defined by documentation
    c3_type: C3_Type | None
    is_central: IsCentral | None
    low_high: LowHigh | None
    ref: C3_Data | UnresolvedData | None


@dataclass(slots=True, eq=False)
class C3_Layer(BaseLayer[C3_Data]):
    layer_name: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY"
    file_name: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY.shp"
    korean_name: ClassVar[str] = "차량방호안전시설"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> C3_Layer | UnresolvedLayer:
        rows = read_layer_rows(section, cls, required=True)
        if rows is None:
            raise RuntimeError(f"{cls.layer_name} is required but no layer was loaded.")
        if not isinstance(rows, list):
            section.c3 = rows
            return rows

        layer = cls(section=section)
        section.c3 = layer
        self_refs: list[tuple[C3_Data, object]] = []
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "line", source_id)
            c3_type = parse_enum(
                C3_Type,
                row_value(row, "Type"),
                source_id,
                "Type",
            )
            is_central = parse_enum(
                IsCentral,
                row_value(row, "IsCentral"),
                source_id,
                "IsCentral",
            )
            low_high = parse_optional_enum(LowHigh, row_value(row, "LowHigh"))
            if not isinstance(geometry, Line3D):
                continue
            data = C3_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                c3_type=c3_type,
                is_central=is_central,
                low_high=low_high,
                ref=None,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            layer.data[data.id] = data
            self_refs.append((data, row_value(row, "Ref_ID")))

        for data, ref_raw in self_refs:
            data.ref = cast(
                C3_Data | UnresolvedData | None,
                section.resolve_optional(
                    cls.layer_name,
                    layer,
                    ref_raw,
                    data.id,
                    "vehicle-protection reference",
                ),
            )
        return layer
