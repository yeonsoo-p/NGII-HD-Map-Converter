from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, relation_property
from ngii2xodr.ngii.data.v2023.features import V2023PointOrPolygonFeature, common_kwargs


@dataclass(slots=True)
class B1_SAFETYSIGN(V2023PointOrPolygonFeature):
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

    link = relation_property("LinkID")
    post = relation_property("PostID")


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
