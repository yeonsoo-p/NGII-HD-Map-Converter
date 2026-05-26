from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2023.features import V2023PolygonFeature, common_kwargs


@dataclass(slots=True)
class C4_SPEEDBUMP(V2023PolygonFeature):
    layer_name: ClassVar[str] = "C4_SPEEDBUMP"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "높이있는방지턱",
        "2": "높이없는방지턱표시",
        "3": "기타방지턱",
    }

    type: str
    link_id: str | None
    ref_lane: int


def make_feature(record: FeatureRecord) -> C4_SPEEDBUMP:
    link_id = record.text("LinkID")
    return C4_SPEEDBUMP(
        **common_kwargs(record),
        ring=record.geometry,
        type=record.text("Type"),
        link_id=link_id or None,
        ref_lane=record.integer("Ref_Lane"),
    )
