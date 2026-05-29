from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeGuard

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.geometry import Polygon as ShapelyPolygon

from ngii.geometry.definitions import Geometry, Line3D, Point3D, Polygon3D

type GeometryKind = Literal["point", "line", "polygon", "point_or_polygon"]
type GeometryName = Literal["point", "line", "polygon"]


@dataclass(slots=True, frozen=True)
class GeometryParseResult:
    geometry: Geometry | None
    issues: tuple[str, ...] = ()


def is_geometry(value: object) -> TypeGuard[Geometry]:
    return isinstance(value, Point3D | Line3D | Polygon3D)


def geometry_name(geometry: object) -> GeometryName | None:
    if isinstance(geometry, Point3D):
        return "point"
    if isinstance(geometry, Line3D):
        return "line"
    if isinstance(geometry, Polygon3D):
        return "polygon"
    return None


def parse_shapely_geometry(geometry: object, kind: GeometryKind) -> GeometryParseResult:
    if bool(getattr(geometry, "is_empty", False)):
        return GeometryParseResult(None, ("Skipped row with empty geometry.",))

    if kind == "point" and isinstance(geometry, Point):
        return _parse_point(geometry)
    if kind == "line" and isinstance(geometry, LineString):
        return _parse_line(geometry)
    if kind == "polygon" and isinstance(geometry, ShapelyPolygon | MultiPolygon):
        return _parse_polygon(geometry)
    if kind == "point_or_polygon":
        if isinstance(geometry, Point):
            return _parse_point(geometry)
        if isinstance(geometry, ShapelyPolygon | MultiPolygon):
            return _parse_polygon(geometry)

    geometry_type = str(getattr(geometry, "geom_type", type(geometry).__name__))
    return GeometryParseResult(
        None,
        (f"Skipped unsupported {geometry_type or 'unknown'} geometry.",),
    )


def translated_geometry(
    geometry: Geometry,
    render_origin: NDArray[np.float64],
) -> Geometry:
    if isinstance(geometry, Point3D):
        return Point3D(xyz=geometry.xyz - render_origin)
    if isinstance(geometry, Line3D):
        return Line3D(xyz=geometry.xyz - render_origin)
    return Polygon3D(
        xyz=geometry.xyz - render_origin,
        interiors=tuple(interior - render_origin for interior in geometry.interiors),
    )


def geometry_bounds(
    geometries: Iterable[Geometry],
) -> tuple[float, float, float, float, float, float] | None:
    arrays = [array for geometry in geometries for array in _geometry_arrays(geometry)]
    if not arrays:
        return None
    xyz = np.concatenate(arrays, axis=0)
    mins = xyz.min(axis=0)
    maxes = xyz.max(axis=0)
    return (
        float(mins[0]),
        float(maxes[0]),
        float(mins[1]),
        float(maxes[1]),
        float(mins[2]),
        float(maxes[2]),
    )


def open_ring(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(xyz) > 1 and np.allclose(xyz[0], xyz[-1]):
        return xyz[:-1]
    return xyz


def coords_to_xyz(coords: object) -> NDArray[np.float64] | None:
    if not isinstance(coords, Iterable) or isinstance(coords, str | bytes):
        return None

    rows: list[tuple[float, float, float]] = []
    for coord in coords:
        if not isinstance(coord, Sequence) or isinstance(coord, str | bytes):
            return None
        if len(coord) < 2:
            return None
        x_value = coord[0]
        y_value = coord[1]
        z_value = coord[2] if len(coord) > 2 else 0.0
        if not (
            isinstance(x_value, Real) and isinstance(y_value, Real) and isinstance(z_value, Real)
        ):
            return None
        rows.append((float(x_value), float(y_value), float(z_value)))

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _parse_point(geometry: Point) -> GeometryParseResult:
    xyz = coords_to_xyz(geometry.coords)
    if xyz is None or len(xyz) != 1:
        return GeometryParseResult(None, ("Skipped malformed point geometry.",))
    return GeometryParseResult(Point3D(xyz=xyz))


def _parse_line(geometry: LineString) -> GeometryParseResult:
    xyz = coords_to_xyz(geometry.coords)
    if xyz is None or len(xyz) < 2:
        return GeometryParseResult(None, ("Skipped malformed line geometry.",))
    return GeometryParseResult(Line3D(xyz=xyz))


def _parse_polygon(geometry: ShapelyPolygon | MultiPolygon) -> GeometryParseResult:
    polygon = _single_polygon(geometry)
    if isinstance(polygon, str):
        return GeometryParseResult(None, (polygon,))

    xyz = coords_to_xyz(polygon.exterior.coords)
    if xyz is None or len(xyz) < 4:
        return GeometryParseResult(None, ("Skipped malformed polygon exterior ring.",))

    issues: list[str] = []
    interiors: list[NDArray[np.float64]] = []
    for interior in polygon.interiors:
        interior_xyz = coords_to_xyz(interior.coords)
        if interior_xyz is None or len(interior_xyz) < 4:
            issues.append("Ignored malformed polygon interior ring.")
            continue
        interiors.append(interior_xyz)

    return GeometryParseResult(
        Polygon3D(xyz=xyz, interiors=tuple(interiors)),
        tuple(issues),
    )


def _single_polygon(geometry: ShapelyPolygon | MultiPolygon) -> ShapelyPolygon | str:
    if isinstance(geometry, ShapelyPolygon):
        return geometry
    if len(geometry.geoms) == 1:
        return geometry.geoms[0]
    return f"Skipped unsupported MultiPolygon geometry with {len(geometry.geoms)} parts."


def _geometry_arrays(geometry: Geometry) -> tuple[NDArray[np.float64], ...]:
    if isinstance(geometry, Polygon3D):
        return (geometry.xyz, *geometry.interiors)
    return (geometry.xyz,)
