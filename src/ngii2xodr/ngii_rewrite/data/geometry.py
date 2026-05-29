"""Geometry conversion helpers for NGII rewrite rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.ops
from numpy import float64
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ngii2xodr.ngii_rewrite.data.features import FeatureGeometryKind
    from ngii2xodr.ngii_rewrite.data.metadata import LayerType


def _ensure_xyz(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    if coords.ndim != 2:
        msg = f"expected 2D coordinate array, got shape {coords.shape}"
        raise ValueError(msg)
    if coords.shape[1] == 3:
        return coords
    if coords.shape[1] == 2:
        z = np.zeros((coords.shape[0], 1), dtype=np.float64)
        return np.hstack((coords, z))
    msg = f"expected XY or XYZ coordinates, got shape {coords.shape}"
    raise ValueError(msg)


def point_xyz(g: shapely.geometry.base.BaseGeometry, *, row_id: str) -> NDArray[np.float64]:
    if not isinstance(g, shapely.Point):
        msg = f"row ID={row_id!r}: unexpected geometry {type(g).__name__}; expected Point"
        raise TypeError(msg)
    coords = _ensure_xyz(np.asarray(g.coords, dtype=float64))
    return np.asarray(coords[0], dtype=np.float64)


def polyline_xyz(
    g: shapely.geometry.base.BaseGeometry,
    *,
    row_id: str,
    multipart_snap_tolerance_m: float,
) -> NDArray[np.float64]:
    line = _as_single_linestring(
        g, row_id=row_id, multipart_snap_tolerance_m=multipart_snap_tolerance_m
    )
    return _ensure_xyz(np.asarray(line.coords, dtype=float64))


def polygon_outer_ring_xyz(
    g: shapely.geometry.base.BaseGeometry, *, row_id: str
) -> NDArray[np.float64]:
    return _ensure_xyz(
        np.asarray(_as_single_polygon(g, row_id=row_id).exterior.coords, dtype=float64)
    )


def convert_geometry(
    layer_type: LayerType,
    geometry: shapely.geometry.base.BaseGeometry,
    *,
    row_id: str,
    multipart_snap_tolerance_m: float,
) -> tuple[FeatureGeometryKind, NDArray[np.float64]]:
    from ngii2xodr.ngii_rewrite.data.metadata import geometry_kinds_for_layer  # noqa: PLC0415

    geometry_kind = _geometry_kind_for(geometry, row_id)
    accepted = geometry_kinds_for_layer(layer_type)
    if geometry_kind not in accepted:
        expected = ", ".join(accepted)
        msg = (
            f"row ID={row_id!r}: {layer_type.layer_name} does not accept "
            f"{type(geometry).__name__}; expected {expected}"
        )
        raise TypeError(msg)
    if geometry_kind == "point":
        return geometry_kind, point_xyz(geometry, row_id=row_id)
    if geometry_kind == "line":
        return geometry_kind, polyline_xyz(
            geometry,
            row_id=row_id,
            multipart_snap_tolerance_m=multipart_snap_tolerance_m,
        )
    return geometry_kind, polygon_outer_ring_xyz(geometry, row_id=row_id)


def xy_line(polyline: NDArray[np.float64]) -> shapely.LineString:
    if len(polyline) < 2:
        return shapely.LineString()
    return shapely.LineString(polyline[:, :2])


def xy_distance(a_xyz: NDArray[np.float64], b_xyz: NDArray[np.float64]) -> float:
    dx = float(a_xyz[0] - b_xyz[0])
    dy = float(a_xyz[1] - b_xyz[1])
    return float(np.hypot(dx, dy))


def _geometry_kind_for(
    geometry: shapely.geometry.base.BaseGeometry, row_id: str
) -> FeatureGeometryKind:
    if isinstance(geometry, shapely.Point):
        return "point"
    if isinstance(geometry, (shapely.LineString, shapely.MultiLineString)):
        return "line"
    if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):
        return "polygon"
    msg = f"row ID={row_id!r}: unsupported geometry {type(geometry).__name__}"
    raise TypeError(msg)


def _as_single_linestring(
    g: shapely.geometry.base.BaseGeometry,
    *,
    row_id: str,
    multipart_snap_tolerance_m: float,
) -> shapely.LineString:
    if isinstance(g, shapely.LineString):
        return g
    if isinstance(g, shapely.MultiLineString):
        if len(g.geoms) == 1:
            return g.geoms[0]
        merged = shapely.ops.linemerge(g)
        if isinstance(merged, shapely.LineString):
            return merged
        snapped = shapely.snap(g, g, multipart_snap_tolerance_m)
        merged = shapely.ops.linemerge(shapely.ops.unary_union(snapped))
        if isinstance(merged, shapely.LineString):
            return merged
        msg = (
            f"row ID={row_id!r}: MultiLineString with {len(g.geoms)} parts could not be "
            f"merged into one polyline at {multipart_snap_tolerance_m} m tolerance"
        )
        raise ValueError(msg)
    msg = f"row ID={row_id!r}: unexpected geometry {type(g).__name__}; expected LineString"
    raise TypeError(msg)


def _as_single_polygon(g: shapely.geometry.base.BaseGeometry, *, row_id: str) -> shapely.Polygon:
    if isinstance(g, shapely.Polygon):
        return g
    if isinstance(g, shapely.MultiPolygon):
        if len(g.geoms) == 1:
            return g.geoms[0]
        msg = f"row ID={row_id!r}: MultiPolygon with {len(g.geoms)} parts is not supported"
        raise ValueError(msg)
    msg = f"row ID={row_id!r}: unexpected geometry {type(g).__name__}; expected Polygon"
    raise TypeError(msg)
