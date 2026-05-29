from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import geopandas as gpd

from ngii.geometry import (
    GeometryKind,
    Line3D,
    Point3D,
    Polygon3D,
    parse_shapely_geometry,
)
from ngii.v2023.data import BaseData, UnresolvedData, is_null, log_row_issue, row_value

if TYPE_CHECKING:
    from ngii.v2023.section import Section

logger = logging.getLogger(__name__)


@dataclass(slots=True, eq=False)
class BaseLayer[T: BaseData]:
    layer_name: ClassVar[str] = "-"
    file_name: ClassVar[str] = "-"
    korean_name: ClassVar[str] = "-"

    section: Section | None = field(default=None, repr=False, compare=False)
    data: dict[str, T] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        section: Section,
    ) -> Self | UnresolvedLayer | None:
        raise NotImplementedError


@dataclass(slots=True, eq=False)
class UnresolvedLayer(BaseLayer[UnresolvedData]):
    layer_name: ClassVar[str] = "UNRESOLVED"
    file_name: ClassVar[str] = "-"
    korean_name: ClassVar[str] = "미해결 레이어"

    expected_layer_name: str = "-"
    expected_file_name: str = "-"
    expected_korean_name: str = "-"
    reason: str = ""


def read_layer_rows(
    section: Section,
    layer_type: type[BaseLayer[Any]],
    required: bool,
) -> list[Any] | UnresolvedLayer | None:
    path = section.coordinate_dir / layer_type.file_name
    if not path.exists():
        if required:
            return unresolved_layer(
                layer_type,
                section,
                reason=f"Missing required layer file: {layer_type.file_name}",
                level="error",
            )
        log_layer_issue(
            "info",
            section.name,
            layer_type.layer_name,
            f"Missing optional layer file: {layer_type.file_name}",
        )
        return None

    frame = read_geodataframe(path)
    return list(iter_rows(frame))


def read_geodataframe(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gpd.read_file(path)


def iter_rows(frame: Any) -> Iterator[Any]:
    for _, row in frame.iterrows():
        yield row


def unresolved_layer(
    layer_type: type[BaseLayer[Any]],
    section: Section,
    reason: str,
    level: Literal["warning", "error"],
) -> UnresolvedLayer:
    log_layer_issue(level, section.name, layer_type.layer_name, reason)
    layer = UnresolvedLayer(
        section=section,
        expected_layer_name=layer_type.layer_name,
        expected_file_name=layer_type.file_name,
        expected_korean_name=layer_type.korean_name,
        reason=reason,
    )
    unresolved = UnresolvedData(
        layer=layer,
        layer_name=layer_type.layer_name,
        reason=reason,
    )
    layer.data[unresolved.id] = unresolved
    return layer


def parse_geometry(
    row: Any,
    kind: GeometryKind,
    source_id: str,
) -> Point3D | Line3D | Polygon3D | None:
    geometry = row_value(row, "geometry")
    if is_null(geometry) or bool(getattr(geometry, "is_empty", False)):
        log_row_issue(
            "warning",
            source_id,
            "Skipped row with empty geometry.",
        )
        return None

    result = parse_shapely_geometry(geometry, kind)
    for issue in result.issues:
        log_row_issue("warning", source_id, issue)
    return result.geometry


def log_layer_issue(
    level: Literal["info", "warning", "error"],
    section_name: str,
    layer_name: str,
    message: str,
) -> None:
    formatted = f"{section_name} {layer_name}: {message}"
    if level == "error":
        logger.error(formatted)
    elif level == "warning":
        logger.warning(formatted)
    else:
        logger.info(formatted)
