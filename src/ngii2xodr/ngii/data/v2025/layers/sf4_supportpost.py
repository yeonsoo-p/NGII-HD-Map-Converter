from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PointFeature, common_kwargs


@dataclass(slots=True)
class SF4_SUPPORTPOST(V2025PointFeature):
    layer_name: ClassVar[str] = "SF4_SUPPORTPOST"
    POST_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "100": "교통 및 보행 신호기 지주",
        "200": "교통안전표지 지주",
    }

    post_type: str


def make_feature(record: FeatureRecord) -> SF4_SUPPORTPOST:
    return SF4_SUPPORTPOST(
        **common_kwargs(record),
        point=record.geometry,
        post_type=record.text("PostType"),
    )
