from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import shapely
import shapely.ops
import vtk
from numpy.typing import NDArray
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon

from ngii.geometry.definitions import Geometry, Line3D, Point3D, Polygon3D
from ngii.geometry.operations import open_ring


def build_point_polydata(geometries: Iterable[Geometry]) -> Any | None:
    points = vtk.vtkPoints()
    for geometry in geometries:
        if not isinstance(geometry, Point3D):
            continue
        xyz = geometry.xyz
        if len(xyz) != 1:
            continue
        points.InsertNextPoint(float(xyz[0, 0]), float(xyz[0, 1]), float(xyz[0, 2]))

    if points.GetNumberOfPoints() == 0:
        return None

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    return polydata


def build_line_polydata(geometries: Iterable[Geometry]) -> Any | None:
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    point_id = 0

    for geometry in geometries:
        if not isinstance(geometry, Line3D):
            continue
        xyz = geometry.xyz
        if len(xyz) < 2:
            continue
        cells.InsertNextCell(len(xyz))
        for row in xyz:
            points.InsertNextPoint(float(row[0]), float(row[1]), float(row[2]))
            cells.InsertCellPoint(point_id)
            point_id += 1

    if points.GetNumberOfPoints() == 0:
        return None

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)
    return polydata


def build_polygon_polydata(geometries: Iterable[Geometry]) -> Any | None:
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    point_id = 0

    for geometry in geometries:
        if not isinstance(geometry, Polygon3D):
            continue
        for triangle in triangulated_polygon_xyz(geometry):
            cells.InsertNextCell(3)
            for row in triangle:
                points.InsertNextPoint(float(row[0]), float(row[1]), float(row[2]))
                cells.InsertCellPoint(point_id)
                point_id += 1

    if points.GetNumberOfPoints() == 0:
        return None

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    if polydata.GetNumberOfPolys() == 0:
        return None
    return polydata


def triangulated_polygon_xyz(polygon: Polygon3D) -> tuple[NDArray[np.float64], ...]:
    shapely_polygon = _shapely_polygon(polygon)
    if shapely_polygon is None:
        return ()

    boundary = _boundary_xyz(polygon)
    triangles: list[NDArray[np.float64]] = []
    for triangle_xy in _triangulated_polygon_xy(shapely_polygon):
        triangle_z = np.asarray(
            [_nearest_boundary_z(boundary, x, y) for x, y in triangle_xy],
            dtype=np.float64,
        )
        triangles.append(np.column_stack([triangle_xy, triangle_z]))
    return tuple(triangles)


def _shapely_polygon(polygon: Polygon3D) -> ShapelyPolygon | None:
    exterior = _valid_open_ring(polygon.xyz)
    if exterior is None:
        return None

    interiors: list[NDArray[np.float64]] = []
    for interior in polygon.interiors:
        interior_ring = _valid_open_ring(interior)
        if interior_ring is not None:
            interiors.append(interior_ring[:, :2])

    return ShapelyPolygon(exterior[:, :2], holes=interiors)


def _valid_open_ring(xyz: NDArray[np.float64]) -> NDArray[np.float64] | None:
    finite = xyz[np.isfinite(xyz).all(axis=1)]
    ring = open_ring(finite)
    return ring if len(ring) >= 3 else None


def _triangulated_polygon_xy(
    polygon: ShapelyPolygon,
) -> tuple[NDArray[np.float64], ...]:
    triangles = _constrained_triangle_xys(polygon)
    if triangles:
        return tuple(triangles)

    repaired_triangles: list[NDArray[np.float64]] = []
    for repaired in _polygonal_repair_candidates(polygon):
        constrained = _constrained_triangle_xys(repaired)
        if constrained:
            repaired_triangles.extend(constrained)
        else:
            repaired_triangles.extend(_unconstrained_triangle_xys(repaired))
    return tuple(repaired_triangles)


def _constrained_triangle_xys(polygon: ShapelyPolygon) -> list[NDArray[np.float64]]:
    if polygon.is_empty or polygon.area <= 0.0:
        return []
    try:
        triangles = shapely.constrained_delaunay_triangles(polygon)
    except (GEOSException, ValueError):
        return []
    return [
        np.asarray(triangle.exterior.coords[:3], dtype=np.float64)
        for triangle in triangles.geoms
        if isinstance(triangle, ShapelyPolygon) and triangle.area > 0.0
    ]


def _unconstrained_triangle_xys(polygon: ShapelyPolygon) -> list[NDArray[np.float64]]:
    if polygon.is_empty or polygon.area <= 0.0:
        return []

    triangles: list[NDArray[np.float64]] = []
    for triangle in shapely.ops.triangulate(polygon):
        if polygon.covers(triangle.representative_point()) and triangle.area > 0.0:
            triangles.append(np.asarray(triangle.exterior.coords[:3], dtype=np.float64))
    return triangles


def _polygonal_repair_candidates(polygon: ShapelyPolygon) -> list[ShapelyPolygon]:
    candidates: list[ShapelyPolygon] = []
    for repaired in (shapely.make_valid(polygon), polygon.buffer(0)):
        candidates.extend(_polygonal_parts(repaired))
    return candidates


def _polygonal_parts(geometry: object) -> list[ShapelyPolygon]:
    if isinstance(geometry, ShapelyPolygon):
        return [geometry] if geometry.area > 0.0 else []
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if part.area > 0.0]
    if isinstance(geometry, GeometryCollection):
        parts: list[ShapelyPolygon] = []
        for part in geometry.geoms:
            parts.extend(_polygonal_parts(part))
        return parts
    return []


def _boundary_xyz(polygon: Polygon3D) -> NDArray[np.float64]:
    rings = [polygon.xyz, *polygon.interiors]
    arrays = [ring[np.isfinite(ring).all(axis=1)] for ring in rings]
    arrays = [array for array in arrays if len(array)]
    if not arrays:
        return np.zeros((1, 3), dtype=np.float64)
    return np.vstack(arrays)


def _nearest_boundary_z(boundary: NDArray[np.float64], x: float, y: float) -> float:
    xy = boundary[:, :2]
    diff = xy - np.asarray((x, y), dtype=np.float64)
    index = int(np.argmin(np.sum(diff * diff, axis=1)))
    return float(boundary[index, 2])
