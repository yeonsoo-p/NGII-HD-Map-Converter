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
    row_id,
    row_value,
    text,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer, parse_geometry, read_layer_rows
from ngii.v2023.layers.a2_link import A2_Data, A2_Layer

if TYPE_CHECKING:
    from ngii.v2023.section import Section


class B2_Type(StrEnum):
    황색_단선_실선 = "111"
    황색_단선_점선 = "112"
    황색_단선_좌점혼선 = "113"
    황색_단선_우점혼선 = "114"
    황색_겹선_실선 = "121"
    황색_겹선_점선 = "122"
    황색_겹선_좌점혼선 = "123"
    황색_겹선_우점혼선 = "124"
    백색_단선_실선 = "211"
    백색_단선_점선 = "212"
    백색_단선_좌점혼선 = "213"
    백색_단선_우점혼선 = "214"
    백색_겹선_실선 = "221"
    백색_겹선_점선 = "222"
    백색_겹선_좌점혼선 = "223"
    백색_겹선_우점혼선 = "224"
    청색_단선_실선 = "311"
    청색_단선_점선 = "312"
    청색_단선_좌점혼선 = "313"
    청색_단선_우점혼선 = "314"
    청색_겹선_실선 = "321"
    청색_겹선_점선 = "322"
    청색_겹선_좌점혼선 = "323"
    청색_겹선_우점혼선 = "324"
    기타 = "999"


class B2_Kind(StrEnum):
    중앙선 = "501"
    가변차선 = "5011"
    유턴구역선 = "502"
    차선 = "503"
    버스전용차선 = "504"
    길가장자리구역선 = "505"
    진로변경제한선 = "506"
    주정차금지선 = "515"
    유도선 = "525"
    정지선 = "530"
    안전지대 = "531"
    자전거도로 = "535"
    기타선 = "599"


@dataclass(slots=True, eq=False)
class B2_Data(BaseData):
    # Geometry
    geometry: Line3D

    # Fields defined by documentation
    b2_type: B2_Type | None
    b2_kind: B2_Kind | None
    r_link: A2_Data | UnresolvedData | None
    l_link: A2_Data | UnresolvedData | None


@dataclass(slots=True, eq=False)
class B2_Layer(BaseLayer[B2_Data]):
    layer_name: ClassVar[str] = "B2_SURFACELINEMARK"
    file_name: ClassVar[str] = "B2_SURFACELINEMARK.shp"
    korean_name: ClassVar[str] = "노면선 표시"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> B2_Layer | UnresolvedLayer:
        rows = read_layer_rows(section, cls, required=True)
        if rows is None:
            raise RuntimeError(f"{cls.layer_name} is required but no layer was loaded.")
        if not isinstance(rows, list):
            section.b2 = rows
            return rows

        layer = cls(section=section)
        section.b2 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "line", source_id)
            b2_type = parse_enum(
                B2_Type,
                row_value(row, "Type", "LineType"),
                source_id,
                "Type",
            )
            b2_kind = parse_enum(
                B2_Kind,
                row_value(row, "Kind", "LineKind"),
                source_id,
                "Kind",
            )
            if not isinstance(geometry, Line3D):
                continue
            r_link = cast(
                A2_Data | UnresolvedData | None,
                section.resolve_optional(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "R_LinkID", "R_linkID"),
                    source_id,
                    "right link",
                ),
            )
            l_link = cast(
                A2_Data | UnresolvedData | None,
                section.resolve_optional(
                    A2_Layer.layer_name,
                    section.a2,
                    row_value(row, "L_LinkID", "L_linkID"),
                    source_id,
                    "left link",
                ),
            )
            data = B2_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                b2_type=b2_type,
                b2_kind=b2_kind,
                r_link=r_link,
                l_link=l_link,
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            if isinstance(r_link, A2_Data):
                r_link.right_surface_line_marks.append(data)
            if isinstance(l_link, A2_Data):
                l_link.left_surface_line_marks.append(data)
            layer.data[data.id] = data
        return layer
