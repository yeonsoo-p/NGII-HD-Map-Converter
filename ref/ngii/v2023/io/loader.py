from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ngii.v2023.dataset import DataSet
from ngii.v2023.io.discovery import discover_dataset
from ngii.v2023.layers import (
    A1_Layer,
    A2_Layer,
    A3_Layer,
    A4_Layer,
    A5_Layer,
    B1_Layer,
    B2_Layer,
    B3_Layer,
    BaseLayer,
    C1_Layer,
    C2_Layer,
    C3_Layer,
    C4_Layer,
    C5_Layer,
    C6_Layer,
    UnresolvedLayer,
)
from ngii.v2023.section import Section

logger = logging.getLogger(__name__)

type LayerInstance = BaseLayer[Any] | UnresolvedLayer | None


def load_dataset(dataset_root: Path, coordinate: str) -> DataSet:
    dataset = discover_dataset(dataset_root, coordinate)

    if not dataset.sections:
        logger.warning("No SEC*/%s directories were found.", coordinate)

    for layer_type in _LAYER_LOAD_ORDER:
        _load_layer_across_sections(layer_type, dataset.sections)

    _warn_if_no_compatible_layers(dataset)
    return dataset


def _warn_if_no_compatible_layers(dataset: DataSet) -> None:
    if not dataset.has_loaded_layer_data():
        logger.warning("No compatible 2023 layer data was loaded.")


_LAYER_LOAD_ORDER: tuple[type[BaseLayer[Any]], ...] = (
    A1_Layer,
    A3_Layer,
    A4_Layer,
    C6_Layer,
    C3_Layer,
    A2_Layer,
    A5_Layer,
    B1_Layer,
    B2_Layer,
    B3_Layer,
    C1_Layer,
    C2_Layer,
    C4_Layer,
    C5_Layer,
)


def _load_layer_across_sections(
    layer_type: type[BaseLayer[Any]],
    sections: list[Section],
) -> None:
    logger.info("Loading %s across all sections", layer_type.layer_name)
    for section in sections:
        layer = layer_type.load(section)
        _log_layer_load(section, layer_type, layer)


def _log_layer_load(
    section: Section,
    layer_type: type[BaseLayer[Any]],
    layer: LayerInstance,
) -> None:
    if layer is None:
        return
    elif isinstance(layer, UnresolvedLayer):
        logger.info("%s: %s: unresolved layer", section.name, layer_type.layer_name)
    else:
        logger.info(
            "%s: %s: loaded %d objects",
            section.name,
            layer_type.layer_name,
            len(layer.data),
        )
