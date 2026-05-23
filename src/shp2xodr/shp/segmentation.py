"""Small staged segmentation pipeline for A2_LINK-derived entities.

Each stage is an explicit class with a tiny public contract: it reads raw
typed layer arrays, emits typed entities, and derives a per-A2 membership
array for fast visualization. The GUI builds its segmentation-level selector
from ``SEGMENTATION_STAGES`` so adding or removing a stage is a code-local
change.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, Self

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import nearest_points

from shp2xodr.shp.data import A2Data, B2Data

log = logging.getLogger(__name__)

_UTURN_LINK_TYPE = "6"
_LATERAL_GROUP_LINK_TYPE = "6"
_JUNCTION_LINK_TYPE = "1"
_UTURN_MARKER_KIND = "502"


@dataclass(slots=True, frozen=True)
class SegmentationConfig:
    """Configuration shared by geometry-based segmentation stages."""

    z_intersection_tol_m: float = 2.0
    junction_proximity_merge_dist_m: float = 5.0


@dataclass(slots=True, frozen=True)
class SelectedField:
    """One selected-table field, with one or more display values."""

    name: str
    values: tuple[str, ...]

    @classmethod
    def scalar(cls, name: str, value: str | int | float) -> Self:
        return cls(name=name, values=(str(value),))

    @classmethod
    def list(cls, name: str, values: tuple[str, ...]) -> Self:
        return cls(name=name, values=values if values else ("-",))


@dataclass(slots=True, frozen=True)
class SelectedSection:
    """A titled group of selected-table fields."""

    title: str
    rows: tuple[SelectedField, ...]


@dataclass(slots=True, frozen=True)
class SegmentationInput:
    """Raw arrays needed by segmentation stages."""

    a2_ids: NDArray[np.str_]
    a2_link_types: NDArray[np.str_]
    a2_r_link_ids: NDArray[np.str_]
    a2_l_link_ids: NDArray[np.str_]
    a2_polylines: tuple[NDArray[np.float64], ...]
    b2_ids: NDArray[np.str_]
    b2_kinds: NDArray[np.str_]
    b2_polylines: tuple[NDArray[np.float64], ...]
    cfg: SegmentationConfig

    @classmethod
    def from_layers(cls, a2: A2Data, b2: B2Data, cfg: SegmentationConfig) -> Self:
        return cls(
            a2_ids=a2.ids,
            a2_link_types=a2.link_types,
            a2_r_link_ids=a2.r_link_ids,
            a2_l_link_ids=a2.l_link_ids,
            a2_polylines=tuple(a2.polylines),
            b2_ids=b2.ids,
            b2_kinds=b2.kinds,
            b2_polylines=tuple(b2.polylines),
            cfg=cfg,
        )

    @property
    def n_links(self) -> int:
        return len(self.a2_ids)


class SegmentEntity(Protocol):
    """Class interface shared by all staged segmentation entities."""

    @property
    def id(self) -> int: ...

    @property
    def link_indices(self) -> tuple[int, ...]: ...

    @property
    def link_ids(self) -> tuple[str, ...]: ...

    def selected_fields(self) -> tuple[SelectedField, ...]: ...


@dataclass(slots=True, frozen=True)
class UTurn:
    """One U-turn entity: a single Type=6 A2 link plus B2 502 marker(s)."""

    id: int
    link_index: int
    link_id: str
    marker_indices: tuple[int, ...]
    marker_ids: tuple[str, ...]

    @property
    def link_indices(self) -> tuple[int, ...]:
        return (self.link_index,)

    @property
    def link_ids(self) -> tuple[str, ...]:
        return (self.link_id,)

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("U-turn", self.id),
            SelectedField.list("U-turn marker ID", self.marker_ids),
        )


@dataclass(slots=True, frozen=True)
class Junction:
    """One junction entity: a connected component of Type=1 A2 links."""

    id: int
    link_indices: tuple[int, ...]
    link_ids: tuple[str, ...]

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Junction", self.id),
            SelectedField.list("Junction link ID", self.link_ids),
        )


@dataclass(slots=True, frozen=True)
class LateralGroup:
    """One lateral bundle of ordinary Type=6 A2 links."""

    id: int
    link_indices: tuple[int, ...]
    link_ids: tuple[str, ...]

    def selected_fields(self) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Lateral group", self.id),
            SelectedField.list("Lateral group link ID", self.link_ids),
        )


@dataclass(slots=True, frozen=True)
class StageResult:
    """Output of one segmentation stage."""

    stage_id: str
    label: str
    entity_label: str
    entities: tuple[SegmentEntity, ...]
    entity_id_per_link: NDArray[np.int32]

    def selected_fields_for_link(self, link_index: int) -> tuple[SelectedField, ...]:
        entity_id = int(self.entity_id_per_link[link_index])
        if entity_id < 0:
            return ()
        return self.entities[entity_id].selected_fields()


class SegmentationStage:
    """Base class for code-registered segmentation stages."""

    id: ClassVar[str]
    label: ClassVar[str]
    entity_label: ClassVar[str]

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        raise NotImplementedError


class UTurnStage(SegmentationStage):
    """Detect U-turn links from Type=6 A2 links crossing B2 Kind=502 markers."""

    id = "uturn"
    label = "U-turns"
    entity_label = "U-turn"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        marker_rows = [
            i
            for i, kind in enumerate(data.b2_kinds)
            if kind == _UTURN_MARKER_KIND and _can_make_line(data.b2_polylines[i])
        ]
        if not marker_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        marker_lines = [_line_xy(data.b2_polylines[i]) for i in marker_rows]
        tree = shapely.STRtree(marker_lines)
        entities: list[UTurn] = []

        for link_i, link_type in enumerate(data.a2_link_types):
            if link_type != _UTURN_LINK_TYPE or not _can_make_line(data.a2_polylines[link_i]):
                continue
            link_line = _line_xy(data.a2_polylines[link_i])
            matched_marker_rows: list[int] = []
            for marker_pos_raw in tree.query(link_line, predicate="intersects"):
                marker_pos = int(marker_pos_raw)
                marker_i = marker_rows[marker_pos]
                if _intersects_within_z_tol(
                    data.a2_polylines[link_i],
                    data.b2_polylines[marker_i],
                    data.cfg.z_intersection_tol_m,
                ):
                    matched_marker_rows.append(marker_i)
            if not matched_marker_rows:
                continue
            entity_id = len(entities)
            entity_id_per_link[link_i] = np.int32(entity_id)
            marker_indices = tuple(sorted(set(matched_marker_rows)))
            entities.append(
                UTurn(
                    id=entity_id,
                    link_index=link_i,
                    link_id=str(data.a2_ids[link_i]),
                    marker_indices=marker_indices,
                    marker_ids=tuple(str(data.b2_ids[i]) for i in marker_indices),
                )
            )

        if entities:
            log.info("UTurnStage: detected %d U-turn link(s)", len(entities))
        return StageResult(
            self.id, self.label, self.entity_label, tuple(entities), entity_id_per_link
        )


class JunctionStage(SegmentationStage):
    """Detect junctions as connected components of Type=1 A2 links."""

    id = "junction"
    label = "Junctions"
    entity_label = "Junction"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        del previous_results
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        candidate_rows = _junction_candidate_rows(data)
        if not candidate_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)

        _union_junction_link_refs(data, candidate_rows, row_to_candidate, parent)
        _union_junction_intersections(data, candidate_rows, row_to_candidate, parent)
        if data.cfg.junction_proximity_merge_dist_m > 0.0:
            _union_junction_components_by_proximity(data, candidate_rows, row_to_candidate, parent)

        root_to_entity: dict[int, int] = {}
        rows_by_entity: dict[int, list[int]] = {}
        for row_i in candidate_rows:
            root = _uf_find(parent, row_to_candidate[row_i])
            entity_id = root_to_entity.get(root)
            if entity_id is None:
                entity_id = len(root_to_entity)
                root_to_entity[root] = entity_id
            entity_id_per_link[row_i] = np.int32(entity_id)
            rows_by_entity.setdefault(entity_id, []).append(row_i)

        entities = tuple(
            Junction(
                id=entity_id,
                link_indices=tuple(rows_by_entity[entity_id]),
                link_ids=tuple(str(data.a2_ids[i]) for i in rows_by_entity[entity_id]),
            )
            for entity_id in range(len(rows_by_entity))
        )
        if entities:
            log.info("JunctionStage: detected %d junction component(s)", len(entities))
        return StageResult(
            self.id, self.label, self.entity_label, entities, entity_id_per_link
        )


class LateralGroupStage(SegmentationStage):
    """Group ordinary Type=6 A2 links by lateral R/L link references."""

    id = "lateral_group"
    label = "Lateral groups"
    entity_label = "Lateral group"

    def run(
        self,
        data: SegmentationInput,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        entity_id_per_link = np.full(data.n_links, -1, dtype=np.int32)
        candidate_rows = _lateral_group_candidate_rows(data, previous_results)
        if not candidate_rows:
            return StageResult(self.id, self.label, self.entity_label, (), entity_id_per_link)

        row_to_candidate = {row_i: candidate_i for candidate_i, row_i in enumerate(candidate_rows)}
        parent = np.arange(len(candidate_rows), dtype=np.int32)

        _union_lateral_group_link_refs(data, candidate_rows, row_to_candidate, parent)

        root_to_entity: dict[int, int] = {}
        rows_by_entity: dict[int, list[int]] = {}
        for row_i in candidate_rows:
            root = _uf_find(parent, row_to_candidate[row_i])
            entity_id = root_to_entity.get(root)
            if entity_id is None:
                entity_id = len(root_to_entity)
                root_to_entity[root] = entity_id
            entity_id_per_link[row_i] = np.int32(entity_id)
            rows_by_entity.setdefault(entity_id, []).append(row_i)

        entities = tuple(
            LateralGroup(
                id=entity_id,
                link_indices=tuple(rows_by_entity[entity_id]),
                link_ids=tuple(str(data.a2_ids[i]) for i in rows_by_entity[entity_id]),
            )
            for entity_id in range(len(rows_by_entity))
        )
        if entities:
            log.info("LateralGroupStage: detected %d lateral group(s)", len(entities))
        return StageResult(
            self.id, self.label, self.entity_label, entities, entity_id_per_link
        )


SEGMENTATION_STAGES: tuple[type[SegmentationStage], ...] = (
    UTurnStage,
    JunctionStage,
    LateralGroupStage,
)


@dataclass(slots=True, frozen=True)
class Segmentation:
    """All stage outputs for one loaded section."""

    stage_results: tuple[StageResult, ...]

    @classmethod
    def from_layers(cls, a2: A2Data, b2: B2Data, cfg: SegmentationConfig) -> Self:
        data = SegmentationInput.from_layers(a2, b2, cfg)
        results: list[StageResult] = []
        by_id: dict[str, StageResult] = {}
        for stage_cls in SEGMENTATION_STAGES:
            result = stage_cls().run(data, by_id)
            results.append(result)
            by_id[result.stage_id] = result
        return cls(stage_results=tuple(results))

    @property
    def level_labels(self) -> tuple[str, ...]:
        return segmentation_level_labels()

    def active_results(self, level: int) -> tuple[StageResult, ...]:
        if level < 0 or level > len(self.stage_results):
            raise ValueError(level)
        return self.stage_results[:level]

    def result(self, stage_id: str) -> StageResult:
        for result in self.stage_results:
            if result.stage_id == stage_id:
                return result
        raise KeyError(stage_id)

    def entity_id_per_link(self, stage_id: str) -> NDArray[np.int32]:
        return self.result(stage_id).entity_id_per_link

    def selected_fields_for_link(self, link_index: int) -> tuple[SelectedField, ...]:
        fields: list[SelectedField] = []
        for result in self.stage_results:
            fields.extend(result.selected_fields_for_link(link_index))
        return tuple(fields)


def segmentation_level_labels() -> tuple[str, ...]:
    """Labels for GUI selector levels: raw plus one entry per stage."""
    return ("Raw", *(stage.label for stage in SEGMENTATION_STAGES))


def _can_make_line(polyline: NDArray[np.float64]) -> bool:
    return len(polyline) >= 2


def _junction_candidate_rows(data: SegmentationInput) -> list[int]:
    return [
        i for i, link_type in enumerate(data.a2_link_types) if link_type == _JUNCTION_LINK_TYPE
    ]


def _lateral_group_candidate_rows(
    data: SegmentationInput,
    previous_results: Mapping[str, StageResult],
) -> list[int]:
    uturn_result = previous_results.get(UTurnStage.id)
    uturn_entity_id_per_link = (
        uturn_result.entity_id_per_link if uturn_result is not None else None
    )
    return [
        i
        for i, link_type in enumerate(data.a2_link_types)
        if link_type == _LATERAL_GROUP_LINK_TYPE
        and (
            uturn_entity_id_per_link is None
            or int(uturn_entity_id_per_link[i]) < 0
        )
    ]


def _union_junction_link_refs(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    id_to_row = {str(link_id): i for i, link_id in enumerate(data.a2_ids)}
    for row_i in candidate_rows:
        candidate_i = row_to_candidate[row_i]
        for neighbour_id in (data.a2_r_link_ids[row_i], data.a2_l_link_ids[row_i]):
            neighbour_row = id_to_row.get(str(neighbour_id))
            if neighbour_row is None or neighbour_row not in row_to_candidate:
                continue
            _uf_union(parent, candidate_i, row_to_candidate[neighbour_row])


def _union_lateral_group_link_refs(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    id_to_row = {str(link_id): i for i, link_id in enumerate(data.a2_ids)}
    for row_i in candidate_rows:
        candidate_i = row_to_candidate[row_i]
        for neighbour_id in (data.a2_r_link_ids[row_i], data.a2_l_link_ids[row_i]):
            neighbour_row = id_to_row.get(str(neighbour_id))
            if neighbour_row is None or neighbour_row not in row_to_candidate:
                continue
            _uf_union(parent, candidate_i, row_to_candidate[neighbour_row])


def _union_junction_intersections(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    geometry_rows = [
        row_i for row_i in candidate_rows if _can_make_line(data.a2_polylines[row_i])
    ]
    if len(geometry_rows) < 2:
        return

    lines = [_line_xy(data.a2_polylines[row_i]) for row_i in geometry_rows]
    tree = shapely.STRtree(lines)
    for a_pos, line in enumerate(lines):
        a_row = geometry_rows[a_pos]
        for b_pos_raw in tree.query(line, predicate="intersects"):
            b_pos = int(b_pos_raw)
            if b_pos <= a_pos:
                continue
            b_row = geometry_rows[b_pos]
            if _intersects_within_z_tol(
                data.a2_polylines[a_row],
                data.a2_polylines[b_row],
                data.cfg.z_intersection_tol_m,
            ):
                _uf_union(parent, row_to_candidate[a_row], row_to_candidate[b_row])


def _line_xy(polyline: NDArray[np.float64]) -> shapely.LineString:
    return shapely.LineString(polyline[:, :2])


def _intersects_within_z_tol(
    poly_a: NDArray[np.float64],
    poly_b: NDArray[np.float64],
    z_tol_m: float,
) -> bool:
    if not _can_make_line(poly_a) or not _can_make_line(poly_b):
        return False
    line_a = _line_xy(poly_a)
    line_b = _line_xy(poly_b)
    if not line_a.intersects(line_b):
        return False
    intersection = line_a.intersection(line_b)
    for point in _sample_intersection_points(intersection):
        z_a = _z_at_xy(poly_a, point.x, point.y)
        z_b = _z_at_xy(poly_b, point.x, point.y)
        if abs(z_a - z_b) <= z_tol_m:
            return True
    return False


def _union_junction_components_by_proximity(
    data: SegmentationInput,
    candidate_rows: list[int],
    row_to_candidate: dict[int, int],
    parent: NDArray[np.int32],
) -> None:
    rows_by_root: dict[int, list[int]] = {}
    for row_i in candidate_rows:
        root = _uf_find(parent, row_to_candidate[row_i])
        rows_by_root.setdefault(root, []).append(row_i)

    components: list[tuple[int, list[int], tuple[NDArray[np.float64], ...], shapely.Geometry]] = []
    for root, rows in rows_by_root.items():
        polylines = tuple(
            data.a2_polylines[row_i]
            for row_i in rows
            if _can_make_line(data.a2_polylines[row_i])
        )
        if not polylines:
            continue
        geom = shapely.unary_union([_line_xy(polyline) for polyline in polylines])
        components.append((root, rows, polylines, geom))

    max_distance = data.cfg.junction_proximity_merge_dist_m
    n_unioned = 0
    for i, (root_a, rows_a, polylines_a, geom_a) in enumerate(components):
        for root_b, rows_b, polylines_b, geom_b in components[i + 1 :]:
            if _uf_find(parent, row_to_candidate[rows_a[0]]) == _uf_find(
                parent, row_to_candidate[rows_b[0]]
            ):
                continue
            if geom_a.distance(geom_b) > max_distance:
                continue
            if not _nearest_points_within_z_tol(
                polylines_a,
                geom_a,
                polylines_b,
                geom_b,
                data.cfg.z_intersection_tol_m,
            ):
                continue
            _uf_union(parent, root_a, root_b)
            n_unioned += 1
    if n_unioned:
        log.info(
            "JunctionStage: proximity-merged %d junction component pair(s) within %.2f m",
            n_unioned,
            max_distance,
        )


def _nearest_points_within_z_tol(
    polylines_a: tuple[NDArray[np.float64], ...],
    geom_a: shapely.Geometry,
    polylines_b: tuple[NDArray[np.float64], ...],
    geom_b: shapely.Geometry,
    z_tol_m: float,
) -> bool:
    point_a, point_b = nearest_points(geom_a, geom_b)
    z_a = _z_at_nearest_polyline(polylines_a, point_a)
    z_b = _z_at_nearest_polyline(polylines_b, point_b)
    return abs(z_a - z_b) <= z_tol_m


def _z_at_nearest_polyline(
    polylines: tuple[NDArray[np.float64], ...],
    point: shapely.Point,
) -> float:
    best_polyline = min(polylines, key=lambda polyline: _line_xy(polyline).distance(point))
    return _z_at_xy(best_polyline, point.x, point.y)


def _sample_intersection_points(geometry: object) -> list[shapely.Point]:
    points: list[shapely.Point] = []
    if not isinstance(geometry, shapely.Geometry):
        return points
    if geometry.is_empty:
        return points
    if isinstance(geometry, shapely.Point):
        points.append(geometry)
    elif isinstance(geometry, shapely.MultiPoint):
        points.extend(geometry.geoms)
    elif isinstance(geometry, shapely.LineString):
        points.append(geometry.interpolate(0.5, normalized=True))
    elif isinstance(geometry, shapely.MultiLineString):
        points.extend(line.interpolate(0.5, normalized=True) for line in geometry.geoms)
    elif isinstance(geometry, shapely.GeometryCollection):
        for part in geometry.geoms:
            points.extend(_sample_intersection_points(part))
    return points


def _z_at_xy(poly_xyz: NDArray[np.float64], x: float, y: float) -> float:
    if len(poly_xyz) == 0:
        return 0.0
    if len(poly_xyz) == 1:
        return float(poly_xyz[0, 2])
    xy = poly_xyz[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    line = shapely.LineString(xy)
    s = float(line.project(shapely.Point(x, y)))
    idx = int(np.searchsorted(cum, s) - 1)
    idx = max(0, min(idx, len(poly_xyz) - 2))
    seg_len = float(seg_lens[idx])
    if seg_len <= 0.0:
        return float(poly_xyz[idx, 2])
    t = (s - cum[idx]) / seg_len
    t = max(0.0, min(1.0, t))
    return float((1.0 - t) * poly_xyz[idx, 2] + t * poly_xyz[idx + 1, 2])


def _uf_find(parent: NDArray[np.int32], x: int) -> int:
    root = x
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[x]) != root:
        parent[x], x = np.int32(root), int(parent[x])
    return root


def _uf_union(parent: NDArray[np.int32], a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[rb] = np.int32(ra)
