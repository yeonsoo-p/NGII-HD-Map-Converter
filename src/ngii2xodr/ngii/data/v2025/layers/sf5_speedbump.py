from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025PolygonFeature, common_kwargs


@dataclass(slots=True)
class SF5_SPEEDBUMP(V2025PolygonFeature):
    layer_name: ClassVar[str] = "SF5_SPEEDBUMP"


def make_feature(record: FeatureRecord) -> SF5_SPEEDBUMP:
    return SF5_SPEEDBUMP(
        **common_kwargs(record),
        ring=record.geometry,
    )
