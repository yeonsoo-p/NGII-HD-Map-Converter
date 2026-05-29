from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Geometry3D:
    xyz: NDArray[np.float64]


@dataclass(slots=True)
class Point3D(Geometry3D):
    pass


@dataclass(slots=True)
class Line3D(Geometry3D):
    pass


@dataclass(slots=True)
class Polygon3D(Geometry3D):
    interiors: tuple[NDArray[np.float64], ...] = ()


type Geometry = Point3D | Line3D | Polygon3D
