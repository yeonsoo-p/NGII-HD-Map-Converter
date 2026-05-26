"""Shared segmentation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Self

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.profile import PerformanceProfile


@dataclass(slots=True, frozen=True)
class SegmentationConfig:
    z_intersection_tol_m: float
    junction_proximity_merge_dist_m: float
    junction_connection_node_merge_dist_m: float
    connection_perpendicular_half_length_m: float
    enable_node_link_relations: bool
    enable_uturn: bool
    enable_lateral_link_group: bool
    enable_lateral_node_group: bool
    enable_junction: bool
    enable_junction_connection: bool
    enable_connection_reference: bool
    enable_connection_perpendicular: bool


@dataclass(slots=True, frozen=True)
class SelectedField:
    name: str
    values: tuple[str, ...]
    refs: tuple[FeatureRef | None, ...]

    @classmethod
    def scalar(cls, name: str, value: str | int | float, ref: FeatureRef | None = None) -> Self:
        return cls(name=name, values=(str(value),), refs=(ref,))

    @classmethod
    def list(
        cls,
        name: str,
        values: tuple[str, ...],
        refs: tuple[FeatureRef | None, ...] = (),
    ) -> Self:
        if not values:
            return cls(name=name, values=("-",), refs=(None,))
        normalized_refs = refs if refs else tuple(None for _ in values)
        return cls(name=name, values=values, refs=normalized_refs)


class SegmentEntity(Protocol):
    @property
    def id(self) -> int: ...

    def selected_fields(
        self, selected_ref: FeatureRef | None = None
    ) -> tuple[SelectedField, ...]: ...


@dataclass(slots=True, frozen=True)
class UTurn:
    id: int
    link_ref: FeatureRef
    marker_refs: tuple[FeatureRef, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref is not None and selected_ref.layer_attr == "a2_link":
            return (SelectedField.scalar("U-turn", self.id),)
        return (
            SelectedField.scalar("U-turn", self.id),
            SelectedField.scalar("U-turn link ID", self.link_ref.feature_id, self.link_ref),
            SelectedField.list(
                "U-turn marker ID",
                tuple(ref.feature_id for ref in self.marker_refs),
                self.marker_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class NodeLinkRelation:
    id: int
    node_ref: FeatureRef
    incoming_link_refs: tuple[FeatureRef, ...]
    outgoing_link_refs: tuple[FeatureRef, ...]

    def selected_fields(self, _selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        return (
            SelectedField.list(
                "Incoming link ID",
                tuple(ref.feature_id for ref in self.incoming_link_refs),
                self.incoming_link_refs,
            ),
            SelectedField.list(
                "Outgoing link ID",
                tuple(ref.feature_id for ref in self.outgoing_link_refs),
                self.outgoing_link_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class Junction:
    id: int
    link_refs: tuple[FeatureRef, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref is not None and selected_ref.layer_attr == "a2_link":
            return (SelectedField.scalar("Junction", self.id),)
        return (
            SelectedField.scalar("Junction", self.id),
            SelectedField.list(
                "Junction link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class LateralLinkGroup:
    id: int
    link_refs: tuple[FeatureRef, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref is not None and selected_ref.layer_attr == "a2_link":
            neighbor_refs = tuple(ref for ref in self.link_refs if ref != selected_ref)
            return (
                SelectedField.scalar("Lateral link group", self.id),
                SelectedField.list(
                    "Lateral link neighbor ID",
                    tuple(ref.feature_id for ref in neighbor_refs),
                    neighbor_refs,
                ),
            )
        return (
            SelectedField.scalar("Lateral link group", self.id),
            SelectedField.list(
                "Lateral link group link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class LateralNodeGroup:
    id: int
    lateral_link_group_id: int
    side: str
    link_refs: tuple[FeatureRef, ...]
    node_refs: tuple[FeatureRef, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref is not None and selected_ref.layer_attr == "a1_node":
            return (
                SelectedField.list(
                    "Lateral node group node ID",
                    tuple(ref.feature_id for ref in self.node_refs),
                    self.node_refs,
                ),
            )
        if selected_ref is not None and selected_ref.layer_attr == "a2_link":
            return (
                SelectedField.scalar(f"Lateral {self.side} node group", self.id),
                SelectedField.list(
                    f"Lateral {self.side} node ID",
                    tuple(ref.feature_id for ref in self.node_refs),
                    self.node_refs,
                ),
            )
        return (
            SelectedField.scalar("Lateral node group", self.id),
            SelectedField.scalar("Endpoint side", self.side),
            SelectedField.scalar("Lateral link group ID", self.lateral_link_group_id),
            SelectedField.list(
                "Lateral link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
            SelectedField.list(
                "Lateral node ID",
                tuple(ref.feature_id for ref in self.node_refs),
                self.node_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class JunctionConnection:
    id: int
    node_refs: tuple[FeatureRef, ...]

    def selected_fields(self, _selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Junction connection", self.id),
            SelectedField.list(
                "Connection node ID",
                tuple(ref.feature_id for ref in self.node_refs),
                self.node_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class ConnectionReference:
    id: int
    link_ref: FeatureRef

    def selected_fields(self, _selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Connection reference", self.id),
            SelectedField.scalar("Reference link ID", self.link_ref.feature_id, self.link_ref),
        )


@dataclass(slots=True, frozen=True)
class ConnectionPerpendicular:
    id: int
    node_ref: FeatureRef

    def selected_fields(self, _selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Connection perpendicular", self.id),
            SelectedField.scalar(
                "Perpendicular node ID",
                self.node_ref.feature_id,
                self.node_ref,
            ),
        )


@dataclass(slots=True, frozen=True)
class StageResult:
    stage_id: str
    label: str
    entity_label: str
    entities: tuple[SegmentEntity, ...]
    entity_id_by_ref: dict[FeatureRef, int]
    entity_ids_by_ref: dict[FeatureRef, tuple[int, ...]] = field(default_factory=dict)
    skipped_reason: str = ""

    def selected_fields_for_ref(self, ref: FeatureRef) -> tuple[SelectedField, ...]:
        entity_ids = self.entity_ids_by_ref.get(ref)
        if entity_ids is None:
            entity_id = self.entity_id_by_ref.get(ref)
            entity_ids = () if entity_id is None else (entity_id,)
        fields: list[SelectedField] = []
        for entity_id in entity_ids:
            if 0 <= entity_id < len(self.entities):
                fields.extend(self.entities[entity_id].selected_fields(ref))
        return tuple(fields)


@dataclass(slots=True)
class SegmentationResult:
    stage_results: tuple[StageResult, ...]
    profile: PerformanceProfile

    def active_results(self, level: int) -> tuple[StageResult, ...]:
        if level < 0 or level > len(self.stage_results):
            raise ValueError(level)
        return self.stage_results[:level]

    def selected_fields_for_ref(self, ref: FeatureRef) -> tuple[SelectedField, ...]:
        fields: list[SelectedField] = []
        for result in self.stage_results:
            fields.extend(result.selected_fields_for_ref(ref))
        return tuple(fields)

    def result_or_none(self, stage_id: str) -> StageResult | None:
        for result in self.stage_results:
            if result.stage_id == stage_id:
                return result
        return None

    @property
    def level_labels(self) -> tuple[str, ...]:
        return ("Raw", *(result.label for result in self.stage_results))
