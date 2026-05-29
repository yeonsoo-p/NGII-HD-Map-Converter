from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ngii.v2023.layer import BaseLayer, UnresolvedLayer
from ngii.v2023.section import Section


@dataclass(slots=True)
class DataSet:
    path: Path
    coordinate: str
    sections: list[Section]

    def iter_layers(self) -> Iterator[BaseLayer[Any] | UnresolvedLayer]:
        for section in self.sections:
            yield from section.iter_layers()

    def has_loaded_layer_data(self) -> bool:
        for layer in self.iter_layers():
            if not isinstance(layer, UnresolvedLayer) and layer.data:
                return True
        return False
