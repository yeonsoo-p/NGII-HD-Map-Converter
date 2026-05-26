from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2023.features import V2023LineFeature, common_kwargs


@dataclass(slots=True)
class C5_HEIGHTBARRIER(V2023LineFeature):
    layer_name: ClassVar[str] = "C5_HEIGHTBARRIER"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "고가도로/교량",
        "2": "육교",
        "4": "기타 높이제한장애물",
    }

    type: str
    link_id: str | None
    ref_lane: int


def make_feature(record: FeatureRecord) -> C5_HEIGHTBARRIER:
    link_id = record.text("LinkID")
    return C5_HEIGHTBARRIER(
        **common_kwargs(record),
        polyline=record.geometry,
        type=record.text("Type"),
        link_id=link_id or None,
        ref_lane=record.integer("Ref_Lane"),
    )
