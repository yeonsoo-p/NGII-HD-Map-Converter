from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PointFeature, common_kwargs


@dataclass(slots=True)
class C2_KILOPOST(PointFeature):
    layer_name: ClassVar[str] = "C2_KILOPOST"

    distance: float
    origin: str
    link_id: str | None
    ref_lane: int

    @property
    def link(self) -> object | None:
        return self.resolve_relation("LinkID")


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
