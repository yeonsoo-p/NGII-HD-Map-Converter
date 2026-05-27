"""Shared segmentation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Self

from ngii2xodr.ngii.app import FeatureRef
from ngii2xodr.profile import PerformanceProfile

EndpointSide = Literal["from", "to"]


@dataclass(slots=True, frozen=True)
class SegmentationConfig:
    z_intersection_tol_m: float
    junction_connection_node_merge_dist_m: float
    junction_connection_opposite_direction_dot_min: float
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
        if selected_ref == self.link_ref:
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
    endpoint_node_refs: tuple[FeatureRef, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref in self.link_refs or selected_ref in self.endpoint_node_refs:
            return (SelectedField.scalar("Junction", self.id),)
        return (
            SelectedField.scalar("Junction", self.id),
            SelectedField.list(
                "Junction link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
            SelectedField.list(
                "Junction node ID",
                tuple(ref.feature_id for ref in self.endpoint_node_refs),
                self.endpoint_node_refs,
            ),
        )


@dataclass(slots=True, frozen=True)
class LateralLinkGroup:
    id: int
    link_refs: tuple[FeatureRef, ...]
    pocket_link_refs: tuple[FeatureRef, ...] = ()
    reference_link_ref: FeatureRef | None = None
    ordering_source: str = ""
    ordering_warning: str = ""

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref in self.link_refs:
            neighbor_refs = tuple(ref for ref in self.link_refs if ref != selected_ref)
            fields = [
                SelectedField.scalar("Lateral link group", self.id),
                SelectedField.list(
                    "Lateral link neighbor ID",
                    tuple(ref.feature_id for ref in neighbor_refs),
                    neighbor_refs,
                ),
            ]
            if selected_ref in self.pocket_link_refs:
                fields.append(SelectedField.scalar("Pocket link", "yes"))
            if self.reference_link_ref == selected_ref:
                fields.append(SelectedField.scalar("Reference-side link", "yes"))
            if self.ordering_warning:
                fields.append(SelectedField.scalar("Ordering warning", self.ordering_warning))
            return tuple(fields)
        fields = [
            SelectedField.scalar("Lateral link group", self.id),
            SelectedField.list(
                "Lateral link group link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
        ]
        if self.reference_link_ref is not None:
            fields.append(
                SelectedField.scalar(
                    "Reference-side link ID",
                    self.reference_link_ref.feature_id,
                    self.reference_link_ref,
                )
            )
        fields.append(SelectedField.scalar("Ordering source", self.ordering_source or "-"))
        if self.ordering_warning:
            fields.append(SelectedField.scalar("Ordering warning", self.ordering_warning))
        return tuple(fields)


@dataclass(slots=True, frozen=True)
class LateralNodeGroup:
    id: int
    lateral_link_group_id: int
    side: EndpointSide
    link_refs: tuple[FeatureRef, ...]
    node_refs: tuple[FeatureRef, ...]
    node_group_keys: tuple[str, ...] = ()

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref in self.node_refs:
            return (
                SelectedField.list(
                    "Lateral node group node ID",
                    tuple(ref.feature_id for ref in self.node_refs),
                    self.node_refs,
                ),
            )
        if selected_ref in self.link_refs:
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
            SelectedField.list("Node group key", self.node_group_keys),
        )


@dataclass(slots=True, frozen=True)
class JunctionConnection:
    id: int
    junction_id: int
    lateral_link_group_ids: tuple[int, ...]
    endpoint_sides: tuple[EndpointSide, ...]
    node_refs: tuple[FeatureRef, ...]
    junction_node_refs: tuple[FeatureRef, ...]
    link_refs: tuple[FeatureRef, ...]
    match_methods: tuple[str, ...]

    def selected_fields(self, selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        if selected_ref in self.node_refs:
            return (
                SelectedField.scalar("Junction connection", self.id),
                SelectedField.list(
                    "Connection node ID",
                    tuple(ref.feature_id for ref in self.node_refs),
                    self.node_refs,
                ),
            )
        return (
            SelectedField.scalar("Junction connection", self.id),
            SelectedField.scalar("Junction", self.junction_id),
            SelectedField.list(
                "Lateral link group",
                tuple(str(group_id) for group_id in self.lateral_link_group_ids),
            ),
            SelectedField.list("Endpoint side", self.endpoint_sides),
            SelectedField.list(
                "Connection node ID",
                tuple(ref.feature_id for ref in self.node_refs),
                self.node_refs,
            ),
            SelectedField.list(
                "Junction node ID",
                tuple(ref.feature_id for ref in self.junction_node_refs),
                self.junction_node_refs,
            ),
            SelectedField.list(
                "Connection link ID",
                tuple(ref.feature_id for ref in self.link_refs),
                self.link_refs,
            ),
            SelectedField.list("Match method", self.match_methods),
        )


@dataclass(slots=True, frozen=True)
class ConnectionReference:
    id: int
    connection_id: int
    junction_id: int
    endpoint_side: EndpointSide
    link_ref: FeatureRef
    anchor_xyz: tuple[float, float, float]
    tangent_xy: tuple[float, float]
    reversed_from_source: bool
    selection_source: str

    def selected_fields(self, _selected_ref: FeatureRef | None = None) -> tuple[SelectedField, ...]:
        return (
            SelectedField.scalar("Connection reference", self.id),
            SelectedField.scalar("Junction connection", self.connection_id),
            SelectedField.scalar("Junction", self.junction_id),
            SelectedField.scalar("Endpoint side", self.endpoint_side),
            SelectedField.scalar("Reference link ID", self.link_ref.feature_id, self.link_ref),
            SelectedField.scalar("Reversed from source", self.reversed_from_source),
            SelectedField.scalar("Selection source", self.selection_source),
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
