from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PolygonFeature, common_kwargs


@dataclass(slots=True)
class A5_PARKINGLOT(PolygonFeature):
    layer_name: ClassVar[str] = "A5_PARKINGLOT"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반주차장",
        "2": "화물차전용주차장",
        "3": "장애인전용주차장",
        "4": "노인전용주차장",
        "5": "여성전용주차장/가족배려주차장",
        "6": "버스전용주차장",
        "7": "전기차전용주차장",
        "9": "기타주차장",
    }

    type: str
    section_id: str | None

    @property
    def section(self) -> object | None:
        if self.section_id is None:
            return None
        dataset = self._require_dataset()
        return dataset.a4_subsidiarysection.get(self.section_id)


def make_feature(record: FeatureRecord) -> A5_PARKINGLOT:
    section_id = record.text("SectionID")
    return A5_PARKINGLOT(
        **common_kwargs(record),
        ring=record.geometry,
        type=record.text("Type"),
        section_id=section_id or None,
    )
