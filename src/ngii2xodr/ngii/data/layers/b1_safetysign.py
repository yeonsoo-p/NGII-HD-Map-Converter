from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PointOrPolygonFeature, common_kwargs
from ngii2xodr.ngii.data.layers.a2_link import A2_LINK
from ngii2xodr.ngii.data.layers.c6_postpoint import C6_POSTPOINT


@dataclass(slots=True)
class B1_SAFETYSIGN(PointOrPolygonFeature):
    """Safety sign geometry.

    The 2023 manual declares B1 as PolygonZ, but real deliverables include
    point rows. Keep the row canonical and represent the observed geometry
    shape without duplicating the feature.
    """

    layer_name: ClassVar[str] = "B1_SAFETYSIGN"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "주의표지",
        "2": "지시표지",
        "3": "규제표지",
        "4": "보조표지",
    }

    type: str
    link_id: str | None
    ref_lane: int
    post_id: str | None

    @property
    def link(self) -> A2_LINK | None:
        if self.link_id is None:
            return None
        return self._require_dataset().a2_link.get(self.link_id)

    @property
    def post(self) -> C6_POSTPOINT | None:
        if self.post_id is None:
            return None
        return self._require_dataset().c6_postpoint.get(self.post_id)


def make_feature(record: FeatureRecord) -> B1_SAFETYSIGN:
    link_id = record.text("LinkID")
    post_id = record.text("PostID")
    return B1_SAFETYSIGN(
        **common_kwargs(record),
        point=record.geometry if record.geometry_kind == "point" else None,
        ring=record.geometry if record.geometry_kind == "polygon" else None,
        type=record.text("Type"),
        link_id=link_id or None,
        ref_lane=record.integer("Ref_Lane"),
        post_id=post_id or None,
    )
