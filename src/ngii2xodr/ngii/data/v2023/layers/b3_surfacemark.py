from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord, PolygonFeature, common_kwargs
from ngii2xodr.ngii.data.v2023.layers.a2_link import A2_LINK


@dataclass(slots=True)
class B3_SURFACEMARK(PolygonFeature):
    layer_name: ClassVar[str] = "B3_SURFACEMARK"
    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "화살표",
        "5": "횡단보도",
    }
    KIND_LABEL: ClassVar[dict[str, str]] = {
        "5321": "횡단보도",
        "533": "고원식횡단보도",
        "534": "자전거횡단보도",
        "5371": "직진",
        "5372": "좌회전",
        "5373": "우회전",
        "5374": "좌우회전",
        "5379": "전방향",
        "5381": "직진 및 좌회전",
        "5382": "직진 및 우회전",
        "5383": "직진 및 유턴",
        "5391": "유턴",
        "5392": "좌회전 및 유턴",
        "5431": "차로변경(좌로합류)",
        "5432": "차로변경(우로합류)",
        "544": "오르막경사면",
        "599": "기타 지시표시",
    }

    type: str
    kind: str
    link_id: str | None

    @property
    def link(self) -> A2_LINK | None:
        if self.link_id is None:
            return None
        return self._require_dataset().a2_link.get(self.link_id)


def make_feature(record: FeatureRecord) -> B3_SURFACEMARK:
    link_id = record.text("LinkID")
    return B3_SURFACEMARK(
        **common_kwargs(record),
        ring=record.geometry,
        type=record.text("Type"),
        kind=record.text("Kind"),
        link_id=link_id or None,
    )
