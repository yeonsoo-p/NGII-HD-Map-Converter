from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PolygonFeature, common_kwargs


@dataclass(slots=True)
class PW1_PATHWAY(V2025PolygonFeature):
    layer_name: ClassVar[str] = "PW1_PATHWAY"
    PATH_TYPE_LABEL: ClassVar[dict[str, str]] = {"100": "보도", "200": "자전거도로"}
    BINARY_LABEL: ClassVar[dict[str, str]] = {"0": "아님", "1": "해당"}

    path_type: str
    tfc_island: str


def make_feature(record: FeatureRecord) -> PW1_PATHWAY:
    return PW1_PATHWAY(
        **common_kwargs(record),
        ring=record.geometry,
        path_type=record.text("PathType"),
        tfc_island=record.text("TFCIsland"),
    )
