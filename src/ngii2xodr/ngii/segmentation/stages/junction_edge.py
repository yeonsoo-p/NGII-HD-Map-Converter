"""Junction-edge segmentation stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.ngii.segmentation.context import SegmentationContext
from ngii2xodr.ngii.segmentation.model import (
    Junction,
    JunctionConnection,
    JunctionEdge,
    JunctionReference,
    StageResult,
)
from ngii2xodr.ngii.segmentation.stage import empty_result
from ngii2xodr.ngii.segmentation.stages.junction import JunctionStage
from ngii2xodr.ngii.segmentation.stages.junction_connection import JunctionConnectionStage
from ngii2xodr.ngii.segmentation.stages.junction_reference import JunctionReferenceStage


class JunctionEdgeStage:
    id: ClassVar[str] = "junction_edge"
    label: ClassVar[str] = "Junction edges"
    entity_label: ClassVar[str] = "Junction edge"
    enabled_attr: ClassVar[str] = "enable_junction_edge"
    requires: ClassVar[tuple[str, ...]] = (
        JunctionStage.id,
        JunctionConnectionStage.id,
        JunctionReferenceStage.id,
    )

    def run(
        self,
        context: SegmentationContext,
        previous_results: Mapping[str, StageResult],
    ) -> StageResult:
        junction_result = previous_results[JunctionStage.id]
        connection_result = previous_results[JunctionConnectionStage.id]
        reference_result = previous_results[JunctionReferenceStage.id]

        junctions = {
            junction.id: junction
            for junction in junction_result.entities
            if isinstance(junction, Junction)
        }
        connections = tuple(
            connection
            for connection in connection_result.entities
            if isinstance(connection, JunctionConnection)
        )
        references = {
            reference.connection_id: reference
            for reference in reference_result.entities
            if isinstance(reference, JunctionReference)
        }
        if not junctions or not connections or not references:
            return empty_result(self)

        junction_centroids = {
            junction_id: centroid
            for junction_id, junction in junctions.items()
            if (centroid := _junction_centroid_xy(context, junction)) is not None
        }

        entities: list[JunctionEdge] = []
        entity_ids_by_ref: dict[FeatureRef, list[int]] = defaultdict(list)
        for connection in connections:
            reference = references.get(connection.id)
            junction_centroid_xy = junction_centroids.get(connection.junction_id)
            if reference is None or junction_centroid_xy is None:
                continue
            entity = _junction_edge_for(
                context,
                connection,
                reference,
                junction_centroid_xy,
                entity_id=len(entities),
            )
            if entity is None:
                continue
            entities.append(entity)
            for ref in (
                entity.farthest_node_ref,
                entity.reference_link_ref,
                *entity.link_refs,
            ):
                entity_ids_by_ref[ref].append(entity.id)

        return StageResult(
            stage_id=self.id,
            label=self.label,
            entity_label=self.entity_label,
            entities=tuple(entities),
            entity_id_by_ref={},
            entity_ids_by_ref={ref: tuple(ids) for ref, ids in entity_ids_by_ref.items()},
        )


@dataclass(slots=True, frozen=True)
class _ProjectedNode:
    ref: FeatureRef
    xyz: tuple[float, float, float]
    projection_m: float


@dataclass(slots=True, frozen=True)
class _ProjectedPoint:
    xyz: tuple[float, float, float]
    lateral_m: float
    link_ref: FeatureRef | None


@dataclass(slots=True, frozen=True)
class _JunctionEdgeSpan:
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    link_refs: tuple[FeatureRef, ...]


def _junction_edge_for(
    context: SegmentationContext,
    connection: JunctionConnection,
    reference: JunctionReference,
    junction_centroid_xy: tuple[float, float],
    *,
    entity_id: int,
) -> JunctionEdge | None:
    outward_xy = _outward_direction_from_reference(reference, junction_centroid_xy)
    anchor_xy = (reference.anchor_xyz[0], reference.anchor_xyz[1])
    farthest_node = _farthest_node_along(
        context,
        connection.node_refs,
        anchor_xy,
        outward_xy,
    )
    if farthest_node is None:
        return None

    perpendicular_xy = (-outward_xy[1], outward_xy[0])
    span = _link_cross_section_span(
        context,
        connection.link_refs,
        farthest_node.xyz,
        outward_xy,
        perpendicular_xy,
    )
    if span is None:
        span = _node_projection_span(
            context,
            connection.node_refs,
            connection.link_refs,
            farthest_node.xyz,
            perpendicular_xy,
        )
    if span is None:
        return None

    return JunctionEdge(
        id=entity_id,
        connection_id=connection.id,
        junction_reference_id=reference.id,
        junction_id=connection.junction_id,
        farthest_node_ref=farthest_node.ref,
        reference_link_ref=reference.link_ref,
        anchor_xyz=farthest_node.xyz,
        perpendicular_xy=perpendicular_xy,
        segment_start_xyz=span.start_xyz,
        segment_end_xyz=span.end_xyz,
        link_refs=span.link_refs,
    )


def _outward_direction_from_reference(
    reference: JunctionReference,
    junction_centroid_xy: tuple[float, float],
) -> tuple[float, float]:
    tx, ty = reference.tangent_xy
    ax = reference.anchor_xyz[0] - junction_centroid_xy[0]
    ay = reference.anchor_xyz[1] - junction_centroid_xy[1]
    if tx * ax + ty * ay >= 0.0:
        return (tx, ty)
    return (-tx, -ty)


def _farthest_node_along(
    context: SegmentationContext,
    node_refs: tuple[FeatureRef, ...],
    anchor_xy: tuple[float, float],
    direction_xy: tuple[float, float],
) -> _ProjectedNode | None:
    best: _ProjectedNode | None = None
    for node_ref in node_refs:
        xyz = context.node_xyz_for_ref(node_ref)
        if xyz is None:
            continue
        projection_m = _dot_xy(
            (xyz[0] - anchor_xy[0], xyz[1] - anchor_xy[1]),
            direction_xy,
        )
        candidate = _ProjectedNode(node_ref, xyz, projection_m)
        if (
            best is None
            or projection_m > best.projection_m
            or (projection_m == best.projection_m and node_ref.feature_id < best.ref.feature_id)
        ):
            best = candidate
    return best


def _link_cross_section_span(
    context: SegmentationContext,
    link_refs: tuple[FeatureRef, ...],
    anchor_xyz: tuple[float, float, float],
    forward_xy: tuple[float, float],
    perpendicular_xy: tuple[float, float],
) -> _JunctionEdgeSpan | None:
    projected: list[_ProjectedPoint] = []
    anchor_xy = (anchor_xyz[0], anchor_xyz[1])
    for link_ref in link_refs:
        polyline = context.link_polyline_for_ref(link_ref)
        if polyline is None or len(polyline) < 2:
            continue
        for point_xyz in _station_crossings(polyline, anchor_xy, forward_xy):
            projected.append(
                _ProjectedPoint(
                    xyz=point_xyz,
                    lateral_m=_lateral_projection_m(point_xyz, anchor_xy, perpendicular_xy),
                    link_ref=link_ref,
                )
            )
    return _span_from_projected_points(projected, fallback_link_refs=())


def _node_projection_span(
    context: SegmentationContext,
    node_refs: tuple[FeatureRef, ...],
    link_refs: tuple[FeatureRef, ...],
    anchor_xyz: tuple[float, float, float],
    perpendicular_xy: tuple[float, float],
) -> _JunctionEdgeSpan | None:
    projected: list[_ProjectedPoint] = []
    anchor_xy = (anchor_xyz[0], anchor_xyz[1])
    for node_ref in node_refs:
        xyz = context.node_xyz_for_ref(node_ref)
        if xyz is None:
            continue
        lateral_m = _lateral_projection_m(xyz, anchor_xy, perpendicular_xy)
        projected.append(
            _ProjectedPoint(
                xyz=_xyz_on_perpendicular(anchor_xyz, perpendicular_xy, lateral_m, xyz[2]),
                lateral_m=lateral_m,
                link_ref=None,
            )
        )
    return _span_from_projected_points(projected, fallback_link_refs=link_refs)


def _span_from_projected_points(
    projected: list[_ProjectedPoint],
    *,
    fallback_link_refs: tuple[FeatureRef, ...],
) -> _JunctionEdgeSpan | None:
    if len(projected) < 2:
        return None
    start = min(projected, key=lambda point: point.lateral_m)
    end = max(projected, key=lambda point: point.lateral_m)
    if start.lateral_m == end.lateral_m:
        return None
    link_refs = _unique_refs(point.link_ref for point in projected if point.link_ref is not None)
    return _JunctionEdgeSpan(
        start_xyz=start.xyz,
        end_xyz=end.xyz,
        link_refs=link_refs if link_refs else fallback_link_refs,
    )


def _station_crossings(
    polyline: NDArray[np.float64],
    anchor_xy: tuple[float, float],
    forward_xy: tuple[float, float],
) -> tuple[tuple[float, float, float], ...]:
    crossings: list[tuple[float, float, float]] = []
    stations = (polyline[:, :2] - np.asarray(anchor_xy, dtype=np.float64)) @ np.asarray(
        forward_xy, dtype=np.float64
    )
    for index in range(len(polyline) - 1):
        start_station = float(stations[index])
        end_station = float(stations[index + 1])
        start_xyz = polyline[index]
        end_xyz = polyline[index + 1]
        if start_station == 0.0 and end_station == 0.0:
            crossings.append(_xyz_tuple(start_xyz))
            crossings.append(_xyz_tuple(end_xyz))
        elif start_station == 0.0:
            crossings.append(_xyz_tuple(start_xyz))
        elif end_station == 0.0:
            crossings.append(_xyz_tuple(end_xyz))
        elif (start_station < 0.0 < end_station) or (end_station < 0.0 < start_station):
            ratio = -start_station / (end_station - start_station)
            crossings.append(_xyz_tuple(start_xyz + (end_xyz - start_xyz) * ratio))
    return tuple(crossings)


def _junction_centroid_xy(
    context: SegmentationContext,
    junction: Junction,
) -> tuple[float, float] | None:
    weighted_sum = np.zeros(2, dtype=np.float64)
    total_length_m = 0.0
    for link_ref in junction.link_refs:
        polyline = context.link_polyline_for_ref(link_ref)
        if polyline is None or len(polyline) < 2:
            continue
        xy = polyline[:, :2]
        segments = xy[1:] - xy[:-1]
        lengths_m = np.hypot(segments[:, 0], segments[:, 1])
        valid = lengths_m > 0.0
        if not np.any(valid):
            continue
        midpoints = (xy[1:] + xy[:-1]) * 0.5
        valid_lengths_m = lengths_m[valid]
        weighted_sum += np.sum(midpoints[valid] * valid_lengths_m[:, None], axis=0)
        total_length_m += float(np.sum(valid_lengths_m))
    if total_length_m > 0.0:
        centroid = weighted_sum / total_length_m
        return (float(centroid[0]), float(centroid[1]))
    return _node_centroid_xy(context, junction.endpoint_node_refs)


def _node_centroid_xy(
    context: SegmentationContext,
    node_refs: tuple[FeatureRef, ...],
) -> tuple[float, float] | None:
    points = [xyz for ref in node_refs if (xyz := context.node_xyz_for_ref(ref)) is not None]
    if not points:
        return None
    xy = np.asarray([(point[0], point[1]) for point in points], dtype=np.float64)
    centroid = np.mean(xy, axis=0)
    return (float(centroid[0]), float(centroid[1]))


def _lateral_projection_m(
    xyz: tuple[float, float, float],
    anchor_xy: tuple[float, float],
    perpendicular_xy: tuple[float, float],
) -> float:
    return _dot_xy(
        (xyz[0] - anchor_xy[0], xyz[1] - anchor_xy[1]),
        perpendicular_xy,
    )


def _xyz_on_perpendicular(
    anchor_xyz: tuple[float, float, float],
    perpendicular_xy: tuple[float, float],
    lateral_m: float,
    z_m: float,
) -> tuple[float, float, float]:
    return (
        anchor_xyz[0] + lateral_m * perpendicular_xy[0],
        anchor_xyz[1] + lateral_m * perpendicular_xy[1],
        z_m,
    )


def _dot_xy(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _xyz_tuple(point: NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def _unique_refs(refs: Iterable[FeatureRef | None]) -> tuple[FeatureRef, ...]:
    output: list[FeatureRef] = []
    seen: set[FeatureRef] = set()
    for ref in refs:
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        output.append(ref)
    return tuple(output)
