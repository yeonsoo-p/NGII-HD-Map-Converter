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
    from ngii.v2023.layers import A2_Data
    from ngii.v2023.section import Section


class NodeType(StrEnum):
    평면교차로 = "1"
    입체교차로 = "2"
    터널시종점 = "3"
    교량시종점 = "4"
    지하차도시종점 = "5"
    고가차도시종점 = "6"
    도로차로수변화 = "7"
    톨게이트시종점 = "8"
    요금소 = "9"
    회전교차로 = "10"
    기타유형 = "99"


@dataclass(slots=True, eq=False)
class A1_Data(BaseData):
    # Geometry
    geometry: Point3D

    # Fields defined by documentation
    node_type: NodeType | None
    its_node_id: str | None

    # Fields defined by reference
    incoming_links: list[A2_Data] = field(default_factory=list, repr=False, compare=False)
    outgoing_links: list[A2_Data] = field(default_factory=list, repr=False, compare=False)


@dataclass(slots=True, eq=False)
class A1_Layer(BaseLayer[A1_Data]):
    layer_name: ClassVar[str] = "A1_NODE"
    file_name: ClassVar[str] = "A1_NODE.shp"
    korean_name: ClassVar[str] = "주행경로노드"

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> A1_Layer | UnresolvedLayer:
        rows = read_layer_rows(section, cls, required=True)
        if rows is None:
            raise RuntimeError(f"{cls.layer_name} is required but no layer was loaded.")
        if not isinstance(rows, list):
            section.a1 = rows
            return rows

        layer = cls(section=section)
        section.a1 = layer
        for row in rows:
            source_id = row_id(row)
            geometry = parse_geometry(row, "point", source_id)
            node_type = parse_enum(
                NodeType,
                row_value(row, "NodeType"),
                source_id,
                "NodeType",
            )
            if not isinstance(geometry, Point3D):
                continue

            data = A1_Data(
                id=source_id,
                layer=layer,
                geometry=geometry,
                admin_code=text(row, "AdminCode"),
                node_type=node_type,
                its_node_id=optional_text(row_value(row, "ITSNodeID")),
                maker=text(row, "Maker"),
                update_date=optional_text(row_value(row, "UpdateDate")),
                version=text(row, "Version"),
                remark=optional_text(row_value(row, "Remark")),
                hist_type=optional_text(row_value(row, "HistType")),
                hist_remark=optional_text(row_value(row, "HistRemark")),
            )
            layer.data[data.id] = data
        return layer
