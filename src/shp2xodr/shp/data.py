"""SHP loaders for NGII HD-map (정밀도로지도) layers.

Each NGII layer is a frozen :mod:`dataclasses` record carrying both the raw
arrays the codebase reads and the NGII domain code → Korean label maps from
the 2023.07 manual. Construct one with::

    A1Data(section_dir)        # section directory; SHP_FILENAME resolved

Same shape for `A2Data` / `A3Data` / `A4Data` / `B2Data` / `C3Data`. The
constructor handles cp949 mojibake and NGII column-name drift internally;
downstream consumers see only the typed fields.

Code-list ClassVars (``NODE_TYPE_LABEL``, ``LINK_TYPE_LABEL``, ``KIND_LABEL``
etc.) encode the manual's tables (table 9.12, 9.16, 9.17, 9.18, 9.22, 9.23,
9.45, 9.61, …) — domain knowledge that belongs next to the data, not in the
rendering layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_COLUMN_ALIASES: dict[str, str] = {
    "L_LinKID": "L_LinkID",
}


# ---- cp949 / column-alias handling ---------------------------------------------


def _looks_like_cp949_mojibake(gdf: gpd.GeoDataFrame) -> bool:
    """CP949 bytes decoded as UTF-8 land in the half-width katakana block.

    NGII datasets routinely omit (or lie in) the .cpg sidecar, so pyogrio
    defaults to UTF-8 and silently mojibakes Korean DBF fields. Half-width
    katakana in any object column is the giveaway since real NGII text is
    Hangul + ASCII.
    """
    obj_cols = gdf.select_dtypes(include=["object", "str"]).columns.drop(
        "geometry", errors="ignore"
    )
    if obj_cols.empty:
        return False
    return bool(gdf[obj_cols].stack().astype(str).str.contains(r"[｡-ﾟ]", regex=True).any())


def _load(shp_path: Path) -> gpd.GeoDataFrame:
    if not shp_path.is_file():
        raise FileNotFoundError(shp_path)
    try:
        gdf = gpd.read_file(shp_path)
        retry_reason = "mojibake heuristic" if _looks_like_cp949_mojibake(gdf) else None
    except UnicodeDecodeError:
        retry_reason = "UTF-8 decode failure"
    if retry_reason is not None:
        log.debug("retrying %s with cp949 after %s", shp_path.name, retry_reason)
        gdf = gpd.read_file(shp_path, encoding="cp949")
    rename = {src: dst for src, dst in _COLUMN_ALIASES.items() if src in gdf.columns}
    if rename:
        log.debug("normalized columns in %s: %s", shp_path.name, rename)
        gdf = gdf.rename(columns=rename)
    return gdf


# ---- Shared column / geometry extractors ---------------------------------------


def _ids(gdf: gpd.GeoDataFrame) -> NDArray[np.str_]:
    return np.asarray(gdf["ID"].astype(str).to_numpy(), dtype=np.str_)


def _points_from_gdf(gdf: gpd.GeoDataFrame) -> NDArray[np.float64]:
    return np.array([(g.x, g.y, g.z) for g in gdf.geometry], dtype=np.float64)


def _polylines_from_gdf(gdf: gpd.GeoDataFrame) -> list[NDArray[np.float64]]:
    return [np.asarray(g.coords, dtype=np.float64) for g in gdf.geometry]


def _outer_rings_from_gdf(gdf: gpd.GeoDataFrame) -> list[NDArray[np.float64]]:
    return [np.asarray(g.exterior.coords, dtype=np.float64) for g in gdf.geometry]


def _str_col(gdf: gpd.GeoDataFrame, col: str, *, fillna: str | None = None) -> NDArray[np.str_]:
    series = gdf[col].fillna(fillna) if fillna is not None else gdf[col]
    return np.asarray(series.astype(str).to_numpy(), dtype=np.str_)


def _int_col(gdf: gpd.GeoDataFrame, col: str, *, fillna: int) -> NDArray[np.int32]:
    series = pd.to_numeric(gdf[col], errors="coerce").fillna(fillna)
    return np.asarray(series.to_numpy(dtype=np.int32), dtype=np.int32)


def _float_col(gdf: gpd.GeoDataFrame, col: str) -> NDArray[np.float64]:
    series = pd.to_numeric(gdf[col], errors="coerce")
    return np.asarray(series.to_numpy(dtype=np.float64), dtype=np.float64)


# ---- Geometry-typed base records -----------------------------------------------


@dataclass(slots=True, frozen=True, init=False)
class PointLayerData:
    """Base record for any NGII point-geometry layer (A1 nodes, future B1
    signs, C1 lights, …). ``points`` has shape ``(N, 3)`` in source CRS units.

    Construct with ``Subclass(section_dir)`` — ``SHP_FILENAME`` is resolved
    against the directory.
    """

    SHP_FILENAME: ClassVar[str]
    ids: NDArray[np.str_]
    points: NDArray[np.float64]

    def __init__(self, shp_dir: Path) -> None:
        if not shp_dir.is_dir():
            raise NotADirectoryError(shp_dir)
        gdf = _load(shp_dir / self.SHP_FILENAME)
        self._populate(gdf)

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        # Subclasses override and call PointLayerData._populate(self, gdf)
        # explicitly first - super() is broken under @dataclass(slots=True)
        # inheritance because the decorator rebuilds the class object.
        object.__setattr__(self, "ids", _ids(gdf))
        object.__setattr__(self, "points", _points_from_gdf(gdf))


@dataclass(slots=True, frozen=True, init=False)
class LineLayerData:
    """Base record for any NGII polyline-geometry layer (A2 links, B2 paint,
    C3 barriers, …). ``polylines`` is a list of ``(N_i, 3)`` XYZ arrays, one
    per row, parallel to ``ids``.
    """

    SHP_FILENAME: ClassVar[str]
    ids: NDArray[np.str_]
    polylines: list[NDArray[np.float64]]

    def __init__(self, shp_dir: Path) -> None:
        if not shp_dir.is_dir():
            raise NotADirectoryError(shp_dir)
        gdf = _load(shp_dir / self.SHP_FILENAME)
        self._populate(gdf)

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        object.__setattr__(self, "ids", _ids(gdf))
        object.__setattr__(self, "polylines", _polylines_from_gdf(gdf))


@dataclass(slots=True, frozen=True, init=False)
class PolygonLayerData:
    """Base record for any NGII polygon-geometry layer (A3 driveway sections,
    A4 subsidiary sections, future B3 marks, …). ``rings`` is a list of
    outer-ring ``(N_i, 3)`` arrays — interior rings are absent in NGII sample
    data and would be dropped here.
    """

    SHP_FILENAME: ClassVar[str]
    ids: NDArray[np.str_]
    rings: list[NDArray[np.float64]]

    def __init__(self, shp_dir: Path) -> None:
        if not shp_dir.is_dir():
            raise NotADirectoryError(shp_dir)
        gdf = _load(shp_dir / self.SHP_FILENAME)
        self._populate(gdf)

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        object.__setattr__(self, "ids", _ids(gdf))
        object.__setattr__(self, "rings", _outer_rings_from_gdf(gdf))


# ---- NGII-specific layer records -----------------------------------------------


@dataclass(slots=True, frozen=True, init=False)
class A1Data(PointLayerData):
    """A1_NODE (주행경로노드). Manual §9.4.1, tables 9.10-9.12.

    ``node_types`` is the ``NodeType`` code (1-10 + 99 per :attr:`NODE_TYPE_LABEL`);
    ``its_node_ids`` references the ITS 표준노드 (empty when not provided).
    """

    SHP_FILENAME: ClassVar[str] = "A1_NODE.shp"

    NODE_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "평면교차로",
        "2": "입체교차로",
        "3": "터널시종점",
        "4": "교량시종점",
        "5": "지하차도시종점",
        "6": "고가차도시종점",
        "7": "도로차로수변화",
        "8": "톨게이트시종점",
        "9": "요금소",
        "10": "회전교차로",
        "99": "기타유형",
    }

    node_types: NDArray[np.str_]
    its_node_ids: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        # Explicit base-class call instead of super(): @dataclass(slots=True)
        # constructs a new class object, breaking the implicit super() MRO
        # lookup that bakes in the pre-decoration class reference.
        PointLayerData._populate(self, gdf)
        object.__setattr__(self, "node_types", _str_col(gdf, "NodeType"))
        object.__setattr__(self, "its_node_ids", _str_col(gdf, "ITSNodeID", fillna=""))


@dataclass(slots=True, frozen=True, init=False)
class A2Data(LineLayerData):
    """A2_LINK (주행경로링크). Manual §9.4.2, tables 9.14-9.18.

    Carries every column needed to drive segmentation: ``link_types`` (lane
    role per :attr:`LINK_TYPE_LABEL`), ``from_node_ids`` / ``to_node_ids``
    (A1 references), ``r_link_ids`` / ``l_link_ids`` (lateral neighbours,
    empty when absent), plus road-classification metadata
    (:attr:`ROAD_RANK_LABEL`, :attr:`ROAD_TYPE_LABEL`).
    """

    SHP_FILENAME: ClassVar[str] = "A2_LINK.shp"

    ROAD_RANK_LABEL: ClassVar[dict[str, str]] = {
        "1": "고속도로",
        "2": "국도",
        "3": "특별광역시도",
        "4": "국가지원지방도",
        "5": "지방도",
        "6": "시도",
        "7": "군도",
        "8": "구도",
        "9": "기타도로",
    }
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반도로",
        "2": "터널",
        "3": "교량",
        "4": "지하도로",
        "5": "고가도로",
    }
    LINK_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "교차로내주행경로",
        "2": "톨게이트차로(하이패스)",
        "3": "톨게이트차로(비하이패스)",
        "4": "버스전용차로",
        "5": "가변차선차로",
        "6": "일반주행차로",
        "7": "휴게소진입로",
        "8": "휴게소내주행경로",
        "9": "휴게소진출로",
        "10": "졸음쉼터진입로",
        "11": "졸음쉼터내주행경로",
        "12": "졸음쉼터진출로",
        "13": "교차로진입로",
        "14": "교차로진출로",
        "99": "기타차로",
    }

    road_ranks: NDArray[np.str_]
    road_types: NDArray[np.str_]
    road_nos: NDArray[np.str_]
    link_types: NDArray[np.str_]
    lane_nos: NDArray[np.int32]
    r_link_ids: NDArray[np.str_]
    l_link_ids: NDArray[np.str_]
    from_node_ids: NDArray[np.str_]
    to_node_ids: NDArray[np.str_]
    section_ids: NDArray[np.str_]
    lengths_m: NDArray[np.float64]
    its_link_ids: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        LineLayerData._populate(self, gdf)
        object.__setattr__(self, "road_ranks", _str_col(gdf, "RoadRank"))
        object.__setattr__(self, "road_types", _str_col(gdf, "RoadType"))
        object.__setattr__(self, "road_nos", _str_col(gdf, "RoadNo", fillna=""))
        object.__setattr__(self, "link_types", _str_col(gdf, "LinkType"))
        object.__setattr__(self, "lane_nos", _int_col(gdf, "LaneNo", fillna=-1))
        object.__setattr__(self, "r_link_ids", _str_col(gdf, "R_LinkID", fillna=""))
        object.__setattr__(self, "l_link_ids", _str_col(gdf, "L_LinkID", fillna=""))
        object.__setattr__(self, "from_node_ids", _str_col(gdf, "FromNodeID"))
        object.__setattr__(self, "to_node_ids", _str_col(gdf, "ToNodeID"))
        object.__setattr__(self, "section_ids", _str_col(gdf, "SectionID", fillna=""))
        object.__setattr__(self, "lengths_m", _float_col(gdf, "Length"))
        object.__setattr__(self, "its_link_ids", _str_col(gdf, "ITSLinkID", fillna=""))


@dataclass(slots=True, frozen=True, init=False)
class A3Data(PolygonLayerData):
    """A3_DRIVEWAYSECTION (구간). Manual §9.4.3, tables 9.20-9.23.

    ``kinds`` ∈ ``{'1' 주행구간, '7' 자율주행금지구간}``; ``road_types`` ∈
    ``{'1' 일반도로, '2' 터널, '3' 교량, '4' 지하도로, '5' 고가도로}``.
    ``remarks`` carries the 보호 대상 specifier (어린이/노인/마을주민/장애인)
    when ``Kind=7``.
    """

    SHP_FILENAME: ClassVar[str] = "A3_DRIVEWAYSECTION.shp"

    KIND_LABEL: ClassVar[dict[str, str]] = {
        "1": "주행구간",
        "7": "자율주행금지구간",
    }
    ROAD_TYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "일반도로",
        "2": "터널",
        "3": "교량",
        "4": "지하도로",
        "5": "고가도로",
    }

    kinds: NDArray[np.str_]
    road_types: NDArray[np.str_]
    remarks: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        PolygonLayerData._populate(self, gdf)
        object.__setattr__(self, "kinds", _str_col(gdf, "Kind"))
        object.__setattr__(self, "road_types", _str_col(gdf, "RoadType"))
        object.__setattr__(self, "remarks", _str_col(gdf, "Remark", fillna=""))


@dataclass(slots=True, frozen=True, init=False)
class A4Data(PolygonLayerData):
    """A4_SUBSIDIARYSECTION (부속구간). Manual §9.4.4, table 9.26.

    The manual prose (§9.4.4) lists 휴게소 / 졸음쉼터 / 보도 / 자전거도로 /
    과적검문소 in that order; the SubType code maps follow. ``Name`` is the
    human-readable label; ``Direction`` is the travel-direction flag for
    휴게시설 entries. ``GasStation`` / ``LpgStation`` / ``EvCharger`` /
    ``Toilet`` are per-row amenity flags ('Y'/'N').
    """

    SHP_FILENAME: ClassVar[str] = "A4_SUBSIDIARYSECTION.shp"

    SUBTYPE_LABEL: ClassVar[dict[str, str]] = {
        "1": "휴게소",
        "2": "졸음쉼터",
        "3": "보도",
        "4": "자전거도로",
        "5": "과적검문소",
    }

    subtypes: NDArray[np.str_]
    names: NDArray[np.str_]
    directions: NDArray[np.str_]
    gas_stations: NDArray[np.str_]
    lpg_stations: NDArray[np.str_]
    ev_chargers: NDArray[np.str_]
    toilets: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        PolygonLayerData._populate(self, gdf)
        object.__setattr__(self, "subtypes", _str_col(gdf, "SubType"))
        object.__setattr__(self, "names", _str_col(gdf, "Name"))
        object.__setattr__(self, "directions", _str_col(gdf, "Direction", fillna=""))
        object.__setattr__(self, "gas_stations", _str_col(gdf, "GasStation", fillna=""))
        object.__setattr__(self, "lpg_stations", _str_col(gdf, "LpgStation", fillna=""))
        object.__setattr__(self, "ev_chargers", _str_col(gdf, "EvCharger", fillna=""))
        object.__setattr__(self, "toilets", _str_col(gdf, "Toilet", fillna=""))


@dataclass(slots=True, frozen=True, init=False)
class B2Data(LineLayerData):
    """B2_SURFACELINEMARK (노면선표시). Manual §9.4.7, tables 9.42-9.45.

    ``Type`` is the 3-digit paint code: ``color(1) + 겹수(2) + 형태(3)``.
    :attr:`TYPE_COLOR_LABEL` decodes the first digit; full enumeration is
    in the manual's table 9.44. ``Kind`` is the semantic class
    (:attr:`KIND_LABEL`). ``r_link_ids`` / ``l_link_ids`` reference the
    A2_LINK on either side of the painted line in driving direction;
    empty string when the line bounds the road on one side only.
    """

    SHP_FILENAME: ClassVar[str] = "B2_SURFACELINEMARK.shp"

    KIND_LABEL: ClassVar[dict[str, str]] = {
        "501": "중앙선",
        "5011": "가변차선",
        "502": "유턴구역선",
        "503": "차선",
        "504": "버스전용차선",
        "505": "길가장자리구역선",
        "506": "진로변경제한선",
        "515": "주정차금지선",
        "525": "유도선",
        "530": "정지선",
        "531": "안전지대",
        "535": "자전거도로",
        "599": "기타선",
    }
    TYPE_COLOR_LABEL: ClassVar[dict[str, str]] = {
        "1": "황색",
        "2": "백색",
        "3": "청색",
        "9": "기타",
    }

    types: NDArray[np.str_]
    kinds: NDArray[np.str_]
    r_link_ids: NDArray[np.str_]
    l_link_ids: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        LineLayerData._populate(self, gdf)
        object.__setattr__(self, "types", _str_col(gdf, "Type"))
        object.__setattr__(self, "kinds", _str_col(gdf, "Kind"))
        object.__setattr__(self, "r_link_ids", _str_col(gdf, "R_LinkID", fillna=""))
        object.__setattr__(self, "l_link_ids", _str_col(gdf, "L_LinkID", fillna=""))


@dataclass(slots=True, frozen=True, init=False)
class C3Data(LineLayerData):
    """C3_VEHICLEPROTECTIONSAFETY (차량방호안전시설). Manual §9.4.11,
    tables 9.60-9.63.

    ``Type`` is the facility class (:attr:`TYPE_LABEL`); ``IsCentral`` is
    ``"0"`` road-edge or ``"1"`` central median; ``LowHigh`` is ``"1"`` 상단
    (top of barrier) or ``"2"`` 하단 (base); ``Ref_ID`` pairs the upper /
    lower polylines of the same physical barrier.

    The NGII manual table 9.60 spells the column ``isCentral``, but real
    NGII SHPs ship it as ``IsCentral`` — we read the SHP name as-is rather
    than aliasing capitalization differences (the column-alias table is
    reserved for actual mojibake typos like ``L_LinKID``).
    """

    SHP_FILENAME: ClassVar[str] = "C3_VEHICLEPROTECTIONSAFETY.shp"

    TYPE_LABEL: ClassVar[dict[str, str]] = {
        "2": "가드레일",
        "3": "콘크리트방호벽",
        "4": "콘크리트연석",
        "5": "무단횡단방지시설",
        "6": "중앙분리대개구부",
        "7": "임시구조물",
        "8": "벽",
        "99": "기타",
    }
    IS_CENTRAL_LABEL: ClassVar[dict[str, str]] = {
        "0": "도로 가장자리",
        "1": "중앙분리대",
    }
    LOW_HIGH_LABEL: ClassVar[dict[str, str]] = {
        "1": "상단",
        "2": "하단",
    }

    types: NDArray[np.str_]
    is_central: NDArray[np.str_]
    low_high: NDArray[np.str_]
    ref_ids: NDArray[np.str_]

    def _populate(self, gdf: gpd.GeoDataFrame) -> None:
        LineLayerData._populate(self, gdf)
        object.__setattr__(self, "types", _str_col(gdf, "Type"))
        object.__setattr__(self, "is_central", _str_col(gdf, "IsCentral"))
        object.__setattr__(self, "low_high", _str_col(gdf, "LowHigh"))
        object.__setattr__(self, "ref_ids", _str_col(gdf, "Ref_ID", fillna=""))
