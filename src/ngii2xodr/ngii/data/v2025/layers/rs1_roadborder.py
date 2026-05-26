from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, LineFeature, common_kwargs


@dataclass(slots=True)
class RS1_ROADBORDER(LineFeature):
    layer_name: ClassVar[str] = "RS1_ROADBORDER"
    BINARY_LABEL: ClassVar[dict[str, str]] = {"0": "아님", "1": "해당"}

    kerb: str
    tfc_island: str
    r_link_id: str | None
    l_link_id: str | None
    pathway_id: str | None


def _optional_id(value: str) -> str | None:
    return value or None


def make_feature(record: FeatureRecord) -> RS1_ROADBORDER:
    return RS1_ROADBORDER(
        **common_kwargs(record),
        polyline=record.geometry,
        kerb=record.text("Kerb"),
        tfc_island=record.text("TFCIsland"),
        r_link_id=_optional_id(record.text("R_LinkID")),
        l_link_id=_optional_id(record.text("L_LinkID")),
        pathway_id=_optional_id(record.text("PathwayID")),
    )
