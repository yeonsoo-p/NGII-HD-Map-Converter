from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, relation_property
from ngii2xodr.ngii.data.v2023.features import V2023PointFeature, common_kwargs


@dataclass(slots=True)
class C2_KILOPOST(V2023PointFeature):
    layer_name: ClassVar[str] = "C2_KILOPOST"

    distance: float
    origin: str
    link_id: str | None
    ref_lane: int

    link = relation_property("LinkID")


def make_feature(record: FeatureRecord) -> C2_KILOPOST:
    link_id = record.text("LinkID")
    return C2_KILOPOST(
        **common_kwargs(record),
        point=record.geometry,
        distance=record.floating("Distance"),
        origin=record.text("Origin"),
        link_id=link_id or None,
        ref_lane=record.integer("Ref_Lane"),
    )
