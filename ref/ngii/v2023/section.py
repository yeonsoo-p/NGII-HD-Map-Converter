from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ngii.v2023.data import (
    BaseData,
    UnresolvedData,
    log_row_issue,
    optional_code,
)
from ngii.v2023.layer import BaseLayer, UnresolvedLayer

if TYPE_CHECKING:
    from ngii.v2023.dataset import DataSet
    from ngii.v2023.layers import (
        A1_Layer,
        A2_Layer,
        A3_Layer,
        A4_Layer,
        A5_Layer,
        B1_Layer,
        B2_Layer,
        B3_Layer,
        C1_Layer,
        C2_Layer,
        C3_Layer,
        C4_Layer,
        C5_Layer,
        C6_Layer,
    )


@dataclass(slots=True)
class Section:
    dataset: DataSet = field(repr=False, compare=False)
    name: str
    path: Path
    coordinate_dir: Path
    a1: A1_Layer | UnresolvedLayer | None = None
    a2: A2_Layer | UnresolvedLayer | None = None
    a3: A3_Layer | UnresolvedLayer | None = None
    a4: A4_Layer | UnresolvedLayer | None = None
    a5: A5_Layer | UnresolvedLayer | None = None
    b1: B1_Layer | UnresolvedLayer | None = None
    b2: B2_Layer | UnresolvedLayer | None = None
    b3: B3_Layer | UnresolvedLayer | None = None
    c1: C1_Layer | UnresolvedLayer | None = None
    c2: C2_Layer | UnresolvedLayer | None = None
    c3: C3_Layer | UnresolvedLayer | None = None
    c4: C4_Layer | UnresolvedLayer | None = None
    c5: C5_Layer | UnresolvedLayer | None = None
    c6: C6_Layer | UnresolvedLayer | None = None

    def iter_layers(self) -> Iterator[BaseLayer[Any] | UnresolvedLayer]:
        for layer in (
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.a5,
            self.b1,
            self.b2,
            self.b3,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
            self.c5,
            self.c6,
        ):
            if layer is not None:
                yield layer

    def resolve_required(
        self,
        target_layer_name: str,
        same_section_layer: BaseLayer[Any] | UnresolvedLayer | None,
        raw_id: object,
        source_id: str,
        role: str,
    ) -> BaseData | UnresolvedData:
        target_id = optional_code(raw_id)
        if target_id is None:
            return self._unresolved_ref(
                target_layer_name=target_layer_name,
                target_layer=same_section_layer,
                target_id="-",
                source_id=source_id,
                reason=f"Missing required {role} reference.",
            )
        return self._resolve_known_id(
            target_layer_name,
            same_section_layer,
            target_id,
            source_id,
            role,
        )

    def resolve_optional(
        self,
        target_layer_name: str,
        same_section_layer: BaseLayer[Any] | UnresolvedLayer | None,
        raw_id: object,
        source_id: str,
        role: str,
    ) -> BaseData | UnresolvedData | None:
        target_id = optional_code(raw_id)
        if target_id is None:
            return None
        return self._resolve_known_id(
            target_layer_name,
            same_section_layer,
            target_id,
            source_id,
            role,
        )

    def resolve_section_ref(
        self,
        a3_layer_name: str,
        a4_layer_name: str,
        raw_id: object,
        source_id: str,
    ) -> BaseData | UnresolvedData | None:
        target_id = optional_code(raw_id)
        if target_id is None:
            return None

        matches = self._lookup_ref_matches(self.a3, target_id) + self._lookup_ref_matches(
            self.a4, target_id
        )
        if len(matches) == 1:
            return matches[0]

        reason = (
            f"Ambiguous section reference to {target_id}."
            if len(matches) > 1
            else f"Missing section reference to {target_id}."
        )
        return self._unresolved_ref(
            target_layer_name=f"{a3_layer_name}/{a4_layer_name}",
            target_layer=None,
            target_id=target_id,
            source_id=source_id,
            reason=reason,
        )

    def _resolve_known_id(
        self,
        target_layer_name: str,
        same_section_layer: BaseLayer[Any] | UnresolvedLayer | None,
        target_id: str,
        source_id: str,
        role: str,
    ) -> BaseData | UnresolvedData:
        same_section = self._same_section_lookup(same_section_layer, target_id)
        if same_section is not None:
            return same_section

        return self._unresolved_ref(
            target_layer_name=target_layer_name,
            target_layer=same_section_layer,
            target_id=target_id,
            source_id=source_id,
            reason=f"Missing {role} reference to {target_layer_name}:{target_id}.",
        )

    def _unresolved_ref(
        self,
        target_layer_name: str,
        target_layer: BaseLayer[Any] | UnresolvedLayer | None,
        target_id: str,
        source_id: str,
        reason: str,
    ) -> UnresolvedData:
        log_row_issue("warning", source_id, reason)
        return UnresolvedData(
            id=target_id,
            layer=target_layer,
            layer_name=target_layer_name,
            reason=reason,
            source_id=source_id,
        )

    def _lookup_ref_matches(
        self,
        layer: BaseLayer[Any] | UnresolvedLayer | None,
        target_id: str,
    ) -> list[BaseData]:
        same_section = self._same_section_lookup(layer, target_id)
        if same_section is not None:
            return [same_section]
        return []

    def _same_section_lookup(
        self,
        layer: BaseLayer[Any] | UnresolvedLayer | None,
        target_id: str,
    ) -> BaseData | None:
        actual_layer = self._actual_layer_or_none(layer)
        if actual_layer is None:
            return None
        return cast(BaseData | None, actual_layer.data.get(target_id))

    def _actual_layer_or_none(
        self,
        layer: BaseLayer[Any] | UnresolvedLayer | None,
    ) -> BaseLayer[Any] | None:
        if layer is None or isinstance(layer, UnresolvedLayer):
            return None
        return layer
