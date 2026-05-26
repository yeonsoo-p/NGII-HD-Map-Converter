from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PointFeature, common_kwargs
from ngii2xodr.ngii.data.layers.a2_link import A2_LINK


@dataclass(slots=True)
class C2_KILOPOST(PointFeature):
    layer_name: ClassVar[str] = "C2_KILOPOST"

    distance: float
    origin: str
    link_id: str | None
    ref_lane: int

    @property
    def link(self) -> A2_LINK | None:
        if self.link_id is None:
            return None
        return self._require_dataset().a2_link.get(self.link_id)


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
