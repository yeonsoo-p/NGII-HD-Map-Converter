from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PointFeature, common_kwargs


@dataclass(slots=True)
class SF2_TRAFFICSIGN(V2025PointFeature):
    layer_name: ClassVar[str] = "SF2_TRAFFICSIGN"
    SIGN_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "주의표지",
        "200": "규제표지",
        "300": "지시표지",
        "400": "보조표지",
    }

    sign_type: str
    post_id: str | None


def make_feature(record: FeatureRecord) -> SF2_TRAFFICSIGN:
    post_id = record.text("PostID")
    return SF2_TRAFFICSIGN(
        **common_kwargs(record),
        point=record.geometry,
        sign_type=record.text("SignType"),
        post_id=post_id or None,
    )
