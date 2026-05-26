from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ngii2xodr.ngii.data.features import FeatureRecord
from ngii2xodr.ngii.data.v2025.features import V2025LineFeature, common_kwargs


@dataclass(slots=True)
class RM1_LANELINE(V2025LineFeature):
    layer_name: ClassVar[str] = "RM1_LANELINE"
    LINE_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "111": "황색-단선-실선",
        "112": "황색-단선-점선",
        "113": "황색-단선-좌점혼선",
        "114": "황색-단선-우점혼선",
        "121": "황색-겹선-실선",
        "122": "황색-겹선-점선",
        "123": "황색-겹선-좌점혼선",
        "124": "황색-겹선-우점혼선",
        "211": "백색-단선-실선",
        "212": "백색-단선-점선",
        "213": "백색-단선-좌점혼선",
        "214": "백색-단선-우점혼선",
        "221": "백색-겹선-실선",
        "222": "백색-겹선-점선",
        "223": "백색-겹선-좌점혼선",
        "224": "백색-겹선-우점혼선",
        "311": "청색-단선-실선",
        "312": "청색-단선-점선",
        "313": "청색-단선-좌점혼선",
        "314": "청색-단선-우점혼선",
        "321": "청색-겹선-실선",
        "322": "청색-겹선-점선",
        "323": "청색-겹선-좌점혼선",
        "324": "청색-겹선-우점혼선",
        "999": "기타",
    }
    LINE_KIND_LABEL: ClassVar[dict[str, str]] = {
        "501": "중앙선",
        "5011": "가변차선",
        "502": "유턴구역선",
        "503": "차선",
        "504": "버스전용차선",
        "505": "길가장자리구역선",
        "506": "진로변경제한선",
        "525": "유도선",
        "530": "정지선",
        "531": "안전지대",
        "535": "자전거도로",
        "599": "기타선",
    }

    line_type: str
    line_kind: str
    r_link_id: str | None
    l_link_id: str | None


def _optional_id(value: str) -> str | None:
    return value or None


def make_feature(record: FeatureRecord) -> RM1_LANELINE:
    return RM1_LANELINE(
        **common_kwargs(record),
        polyline=record.geometry,
        line_type=record.text("LineType"),
        line_kind=record.text("LineKind"),
        r_link_id=_optional_id(record.text("R_LinkID")),
        l_link_id=_optional_id(record.text("L_LinkID")),
    )
