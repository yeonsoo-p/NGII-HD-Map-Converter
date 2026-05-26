"""Geometry and graph helpers for segmentation stages."""

from __future__ import annotations

import numpy as np
import shapely
from numpy.typing import NDArray


def can_make_line(polyline: NDArray[np.float64]) -> bool:
    return len(polyline) >= 2


def uf_find(parent: NDArray[np.int32], i: int) -> int:
    root = i
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[i]) != i:
        next_i = int(parent[i])
        parent[i] = np.int32(root)
        i = next_i
    return root


def uf_union(parent: NDArray[np.int32], a: int, b: int) -> None:
    root_a = uf_find(parent, a)
    root_b = uf_find(parent, b)
    if root_a != root_b:
        parent[root_b] = np.int32(root_a)


def intersects_within_z_tol(
    poly_a: NDArray[np.float64],
    line_a: shapely.LineString,
    poly_b: NDArray[np.float64],
    line_b: shapely.LineString,
    z_tol_m: float,
) -> bool:
    if not can_make_line(poly_a) or not can_make_line(poly_b):
        return False
    if not line_a.intersects(line_b):
        return False
    intersection = line_a.intersection(line_b)
    for point in sample_intersection_points(intersection):
        z_a = z_at_xy(poly_a, point.x, point.y)
        z_b = z_at_xy(poly_b, point.x, point.y)
        if abs(z_a - z_b) <= z_tol_m:
            return True
    return False


def sample_intersection_points(geometry: object) -> list[shapely.Point]:
    points: list[shapely.Point] = []
    if not isinstance(geometry, shapely.Geometry) or geometry.is_empty:
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
            points.extend(sample_intersection_points(part))
    return points


def z_at_xy(poly_xyz: NDArray[np.float64], x: float, y: float) -> float:
    if len(poly_xyz) == 0:
        return 0.0
    if len(poly_xyz) == 1:
        return float(poly_xyz[0, 2])
    xy = poly_xyz[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    s = float(shapely.LineString(xy).project(shapely.Point(x, y)))
    idx = int(np.searchsorted(cum, s) - 1)
    idx = max(0, min(idx, len(poly_xyz) - 2))
    seg_len = float(seg_lens[idx])
    if seg_len <= 0.0:
        return float(poly_xyz[idx, 2])
    t = (s - float(cum[idx])) / seg_len
    return float(poly_xyz[idx, 2] + t * (poly_xyz[idx + 1, 2] - poly_xyz[idx, 2]))
