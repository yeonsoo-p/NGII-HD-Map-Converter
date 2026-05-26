from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2023.features import V2023PointFeature, common_kwargs


@dataclass(slots=True)
class C6_POSTPOINT(V2023PointFeature):
    layer_name: ClassVar[str] = "C6_POSTPOINT"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "신호기지주",
        "2": "교통표지지주",
    }

    type: str


def make_feature(record: FeatureRecord) -> C6_POSTPOINT:
    return C6_POSTPOINT(
        **common_kwargs(record),
        point=record.geometry,
        type=record.text("Type"),
    )
