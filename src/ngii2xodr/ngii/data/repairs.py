"""Version-agnostic NGII topology sanity checks and repairs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii.data.config import NGIIConfig, check_repairs, check_reports
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import (
    FeatureRef,
    NGIIFeature,
    optional_text,
    same_feature,
)
from ngii2xodr.ngii.data.geometry import xy_distance, xy_line
from ngii2xodr.ngii.data.sanity import SanityReport
from ngii2xodr.ngii.data.schema import ReciprocalReferenceRule

SanityHook = Callable[[NGIIDataset, SanityReport, NGIIConfig], None]
_REPLACEMENT_CHAR = "\ufffd"


@dataclass(slots=True, frozen=True)
class _RemovalCause:
    code: str
    message: str
    before: dict[str, Any]
    after: dict[str, Any]


def merge_features(
    store: LayerStore[Any],
    features: list[NGIIFeature],
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    for feature in features:
        if not feature.id:
            if check_reports(cfg.sanity.checks.feature_id_missing):
                sanity.warn(
                    "feature-id-missing",
                    f"{feature.layer_name} row has an empty ID and cannot be globally indexed",
                    layer_name=feature.layer_name,
                    source_path=feature.source_path,
                )
            continue
        existing = store.by_id.get(feature.id)
        if existing is None:
            store.id_to_index[feature.id] = len(store.features)
            store.by_id[feature.id] = feature
            store.features.append(feature)
            continue
        if same_feature(existing, feature):
            if check_reports(cfg.sanity.checks.feature_id_duplicate_identical):
                sanity.warn(
                    "feature-id-duplicate-identical",
                    f"{feature.layer_name} ID {feature.id!r} appears more than once "
                    "with identical data",
                    layer_name=feature.layer_name,
                    feature_id=feature.id,
                    source_path=feature.source_path,
                )
            continue
        before = {
            "kept_source": str(existing.source_path),
            "dropped_source": str(feature.source_path),
        }
        after = {"canonical_source": str(existing.source_path)}
        if check_repairs(cfg.sanity.checks.feature_id_duplicate_conflicting):
            sanity.action(
                "feature-id-duplicate-conflicting-dropped",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row dropped",
                before=before,
                after=after,
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )
        elif check_reports(cfg.sanity.checks.feature_id_duplicate_conflicting):
            sanity.warn(
                "feature-id-duplicate-conflicting",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row dropped",
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )


def check_text_values(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    if check_reports(cfg.sanity.checks.text_mojibake):
        _check_mojibake_text(
            dataset,
            sanity,
            repair=check_repairs(cfg.sanity.checks.text_mojibake),
        )
    if check_reports(cfg.sanity.checks.text_replacement_char):
        _warn_unrepaired_replacement_chars(dataset, sanity)


def _check_mojibake_text(dataset: NGIIDataset, sanity: SanityReport, *, repair: bool) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in store.iter_text_fields(feature):
                repaired = _repair_cp949_latin1_mojibake(value)
                if repaired is None:
                    continue
                if repair:
                    store.set_column(feature, field_name, repaired)
                    sanity.action(
                        "text-mojibake-repaired",
                        f"{feature.layer_name} {feature.id} {field_name} repaired by "
                        "latin1-to-cp949 mojibake rule",
                        before={field_name: value},
                        after={field_name: repaired},
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )
                else:
                    sanity.warn(
                        "text-mojibake",
                        f"{feature.layer_name} {feature.id} {field_name} looks like "
                        "latin1-to-cp949 mojibake",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def _repair_cp949_latin1_mojibake(value: str) -> str | None:
    if not value or _REPLACEMENT_CHAR in value:
        return None
    try:
        candidate = value.encode("latin1").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if candidate == value or _REPLACEMENT_CHAR in candidate:
        return None
    original_hangul = _hangul_count(value)
    candidate_hangul = _hangul_count(candidate)
    if candidate_hangul <= original_hangul:
        return None
    if _mojibake_marker_count(value) == 0 and candidate_hangul < 2:
        return None
    return candidate


def _hangul_count(value: str) -> int:
    return sum(1 for char in value if "\uac00" <= char <= "\ud7a3")


def _mojibake_marker_count(value: str) -> int:
    return sum(1 for char in value if "\u00a1" <= char <= "\u00ff" or "\uff61" <= char <= "\uff9f")


def _warn_unrepaired_replacement_chars(dataset: NGIIDataset, sanity: SanityReport) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in store.iter_text_fields(feature):
                if _REPLACEMENT_CHAR in value:
                    sanity.warn(
                        "text-replacement-char",
                        f"{feature.layer_name} {feature.id}: {field_name} contains "
                        "Unicode replacement characters",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def check_link_too_short(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    link_store = dataset.store_for_role("link")
    link_attr = dataset.schema.attr_for_role("link")
    removal_causes: dict[FeatureRef, _RemovalCause] = {}
    threshold_m = cfg.sanity.link_min_length_m
    for link in link_store.features:
        length_m = float(xy_line(link.polyline).length)
        if length_m >= threshold_m:
            continue
        if check_repairs(cfg.sanity.checks.link_too_short):
            removal_causes[FeatureRef(link_attr, link.id)] = _RemovalCause(
                code="link-too-short-removed",
                message=(
                    f"{link.layer_name} {link.id} geometry length {length_m:.3f} m is "
                    f"shorter than {threshold_m:.3f} m"
                ),
                before={"length_m": length_m, "threshold_m": threshold_m},
                after={"removed": True},
            )
        elif check_reports(cfg.sanity.checks.link_too_short):
            sanity.warn(
                "link-too-short",
                f"{link.layer_name} {link.id} geometry length {length_m:.3f} m is shorter "
                f"than {threshold_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
    _remove_features_with_cascade(dataset, sanity, removal_causes)


def check_link_endpoint_isolated(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    node_store = dataset.store_for_role("node")
    link_attr = dataset.schema.attr_for_role("link")
    removal_causes: dict[FeatureRef, _RemovalCause] = {}
    for link in link_store.features:
        from_node_id = optional_text(getattr(link, "from_node_id", None))
        to_node_id = optional_text(getattr(link, "to_node_id", None))
        if not from_node_id or not to_node_id:
            continue
        if node_store.get(from_node_id) is None or node_store.get(to_node_id) is None:
            continue
        if _other_link_uses_node(link_store, link.id, from_node_id) or _other_link_uses_node(
            link_store, link.id, to_node_id
        ):
            continue
        if check_repairs(cfg.sanity.checks.link_endpoint_isolated):
            removal_causes[FeatureRef(link_attr, link.id)] = _RemovalCause(
                code="link-endpoint-isolated-removed",
                message=(
                    f"{link.layer_name} {link.id} is isolated from other links at both "
                    f"endpoint nodes {from_node_id!r} and {to_node_id!r}"
                ),
                before={"from_node_id": from_node_id, "to_node_id": to_node_id},
                after={"removed": True},
            )
        elif check_reports(cfg.sanity.checks.link_endpoint_isolated):
            sanity.warn(
                "link-endpoint-isolated",
                f"{link.layer_name} {link.id} is isolated from other links at both endpoint "
                f"nodes {from_node_id!r} and {to_node_id!r}",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
    _remove_features_with_cascade(dataset, sanity, removal_causes)


def check_reference_unresolved(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    removal_causes: dict[FeatureRef, _RemovalCause] = {}
    optional_unresolved: list[tuple[FeatureRef, str, str, str]] = []
    for store in dataset.layer_stores:
        for feature in store.features:
            feature_ref = FeatureRef(store.spec.python_attr, feature.id)
            for reference in store.spec.references:
                value = optional_text(getattr(feature, reference.source_attr, None))
                if not value and not reference.required:
                    continue
                if value and _reference_resolves(dataset, reference.target_attrs, value):
                    continue
                if check_repairs(cfg.sanity.checks.reference_unresolved):
                    if reference.required:
                        target_layer = _reference_target_label(dataset, reference.target_attrs)
                        removal_causes.setdefault(
                            feature_ref,
                            _RemovalCause(
                                code="reference-required-unresolved-removed",
                                message=(
                                    f"{feature.layer_name} {feature.id} "
                                    f"{reference.column_name}={value!r} does not resolve "
                                    f"to required {target_layer}"
                                ),
                                before={
                                    reference.source_attr: getattr(feature, reference.source_attr)
                                },
                                after={"removed": True},
                            ),
                        )
                    elif value:
                        optional_unresolved.append(
                            (
                                feature_ref,
                                reference.source_attr,
                                reference.column_name,
                                value,
                            )
                        )
                elif check_reports(cfg.sanity.checks.reference_unresolved):
                    target_layer = _reference_target_label(dataset, reference.target_attrs)
                    sanity.warn(
                        "reference-unresolved",
                        f"{feature.layer_name} {feature.id} {reference.column_name}={value!r} "
                        f"does not resolve to {target_layer}",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )
    removed_refs = _remove_features_with_cascade(dataset, sanity, removal_causes)
    for feature_ref, source_attr, column_name, value in optional_unresolved:
        if feature_ref in removed_refs:
            continue
        store = dataset.store_for_attr(feature_ref.layer_attr)
        feature = store.get(feature_ref.feature_id)
        if feature is None or optional_text(getattr(feature, source_attr, None)) != value:
            continue
        setattr(feature, source_attr, None)
        sanity.action(
            "reference-optional-unresolved-cleared",
            f"{feature.layer_name} {feature.id} {column_name}={value!r} does not resolve and "
            "was cleared",
            before={source_attr: value},
            after={source_attr: None},
            layer_name=feature.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )


def check_node_unreferenced(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    link_store = dataset.store_for_role("link")
    node_store = dataset.store_for_role("node")
    node_attr = dataset.schema.attr_for_role("node")
    used_node_ids = {
        node_id
        for link in link_store.features
        for node_id in (link.from_node_id, link.to_node_id)
        if node_id
    }
    removal_causes: dict[FeatureRef, _RemovalCause] = {}
    for node in node_store.features:
        if node.id in used_node_ids:
            continue
        if check_repairs(cfg.sanity.checks.node_unreferenced):
            removal_causes[FeatureRef(node_attr, node.id)] = _RemovalCause(
                code="node-unreferenced-removed",
                message=(
                    f"{node.layer_name} {node.id} is not referenced by any remaining link endpoint"
                ),
                before={"referenced_by_link_endpoint": False},
                after={"removed": True},
            )
        elif check_reports(cfg.sanity.checks.node_unreferenced):
            sanity.warn(
                "node-unreferenced",
                f"{node.layer_name} {node.id} is not referenced by any remaining link endpoint, "
                "but node removal is disabled",
                layer_name=node.layer_name,
                feature_id=node.id,
                source_path=node.source_path,
            )
    _remove_features_with_cascade(dataset, sanity, removal_causes)


def check_link_endpoint_unresolved(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    node_store = dataset.store_for_role("node")
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in dataset.store_for_role("link").features:
        if len(link.polyline) < 2:
            continue
        _repair_endpoint(
            dataset,
            sanity,
            node_store,
            link,
            "from_node_id",
            "FromNodeID",
            link.polyline[0],
            cfg,
            tolerance_m,
        )
        _repair_endpoint(
            dataset,
            sanity,
            node_store,
            link,
            "to_node_id",
            "ToNodeID",
            link.polyline[-1],
            cfg,
            tolerance_m,
        )


def _repair_endpoint(
    dataset: NGIIDataset,
    sanity: SanityReport,
    node_store: LayerStore[Any],
    link: Any,
    attr_name: str,
    column_name: str,
    endpoint_xyz: NDArray[np.float64],
    cfg: NGIIConfig,
    tolerance_m: float,
) -> None:
    current_id = getattr(link, attr_name)
    if current_id is not None and node_store.get(current_id) is not None:
        return
    nearby = [
        node for node in node_store.features if xy_distance(endpoint_xyz, node.point) <= tolerance_m
    ]
    before = {attr_name: current_id}
    if len(nearby) == 1:
        repaired_id = nearby[0].id
        if check_repairs(cfg.sanity.checks.link_endpoint_unresolved):
            setattr(link, attr_name, repaired_id)
            sanity.action(
                "link-endpoint-unresolved-assigned",
                f"{link.layer_name} {link.id} {column_name} repaired to nearby "
                f"{node_store.layer_name} {repaired_id}",
                before=before,
                after={attr_name: repaired_id},
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif check_reports(cfg.sanity.checks.link_endpoint_unresolved):
            sanity.warn(
                "link-endpoint-unresolved",
                f"{link.layer_name} {link.id} {column_name} could be repaired to nearby "
                f"{node_store.layer_name} {repaired_id}",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if len(nearby) > 1 and check_reports(cfg.sanity.checks.link_endpoint_unresolved):
        sanity.warn(
            "link-endpoint-unresolved-ambiguous",
            f"{link.layer_name} {link.id} {column_name} has {len(nearby)} nearby "
            f"{node_store.layer_name} candidates",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    if current_id is None:
        if not nearby and check_reports(cfg.sanity.checks.link_endpoint_unresolved):
            sanity.warn(
                "link-endpoint-unresolved",
                f"{link.layer_name} {link.id} {column_name} is missing and has no nearby "
                f"{node_store.layer_name} candidate",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if check_repairs(cfg.sanity.checks.link_endpoint_unresolved):
        setattr(link, attr_name, None)
        sanity.action(
            "link-endpoint-unresolved-cleared",
            f"{link.layer_name} {link.id} {column_name} could not be resolved and was cleared",
            before=before,
            after={attr_name: None},
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif check_reports(cfg.sanity.checks.link_endpoint_unresolved):
        sanity.warn(
            "link-endpoint-unresolved",
            f"{link.layer_name} {link.id} {column_name} could not be resolved",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


@dataclass(slots=True)
class _DisjointSet:
    parent: dict[str, str]

    @classmethod
    def from_ids(cls, item_ids: tuple[str, ...]) -> _DisjointSet:
        return cls(parent={item_id: item_id for item_id in item_ids})

    def find(self, item_id: str) -> str:
        parent = self.parent[item_id]
        if parent != item_id:
            parent = self.find(parent)
            self.parent[item_id] = parent
        return parent

    def union(self, left_id: str, right_id: str) -> None:
        left_root = self.find(left_id)
        right_root = self.find(right_id)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(slots=True, frozen=True)
class _OrientationDecision:
    node_ids_reversed: bool
    polyline_reversed: bool


def check_link_orientation_reversed(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    node_store = dataset.store_for_role("node")
    links_by_node = _links_by_node(dataset)
    seed_votes = {
        link.id: vote
        for link in link_store.features
        if (vote := _one_sided_topology_vote(link, links_by_node)) is not None
    }
    for group in _link_orientation_groups(dataset):
        group_seed_votes = tuple(seed_votes[link.id] for link in group if link.id in seed_votes)
        group_vote = _majority_node_reversal_vote(group_seed_votes)
        if group_vote is None:
            reason = "no one-sided topology seed" if not group_seed_votes else "tied seed votes"
            _warn_orientation_group(sanity, cfg, link_store, group, reason)
            continue
        for link in group:
            node_ids_reversed = seed_votes.get(link.id, group_vote)
            decision = _orientation_decision(
                node_store,
                link,
                node_ids_reversed,
                sanity,
                cfg,
            )
            if decision is not None:
                _apply_or_warn_orientation(link, decision, sanity, cfg)


def _link_orientation_groups(dataset: NGIIDataset) -> tuple[tuple[Any, ...], ...]:
    link_store = dataset.store_for_role("link")
    disjoint = _DisjointSet.from_ids(tuple(link.id for link in link_store.features))
    _union_lateral_link_groups(disjoint, link_store)
    _union_node_group_links(disjoint, dataset, link_store)

    groups_by_root: dict[str, list[Any]] = {}
    for link in link_store.features:
        groups_by_root.setdefault(disjoint.find(link.id), []).append(link)
    return tuple(tuple(group) for group in groups_by_root.values())


def _union_lateral_link_groups(disjoint: _DisjointSet, link_store: LayerStore[Any]) -> None:
    for link in link_store.features:
        for attr_name in ("r_link_id", "l_link_id"):
            target_id = optional_text(getattr(link, attr_name, None))
            if not target_id:
                continue
            target = link_store.get(target_id)
            if target is not None:
                disjoint.union(link.id, target.id)


def _union_node_group_links(
    disjoint: _DisjointSet,
    dataset: NGIIDataset,
    link_store: LayerStore[Any],
) -> None:
    node_group_by_id = {
        node.id: group_id
        for node in dataset.store_for_role("node").features
        if (group_id := optional_text(getattr(node, "group_id", None)))
    }
    links_by_node_group: dict[str, list[str]] = defaultdict(list)
    for link in link_store.features:
        group_ids = {
            node_group_by_id[node_id]
            for node_id in (
                optional_text(getattr(link, "from_node_id", None)),
                optional_text(getattr(link, "to_node_id", None)),
            )
            if node_id in node_group_by_id
        }
        for group_id in group_ids:
            links_by_node_group[group_id].append(link.id)
    for link_ids in links_by_node_group.values():
        if len(link_ids) < 2:
            continue
        first_id = link_ids[0]
        for link_id in link_ids[1:]:
            disjoint.union(first_id, link_id)


def _one_sided_topology_vote(link: Any, by_node: dict[str, list[Any]]) -> bool | None:
    from_node_id = optional_text(getattr(link, "from_node_id", None))
    to_node_id = optional_text(getattr(link, "to_node_id", None))
    if not from_node_id or not to_node_id:
        return None
    from_connected = _node_has_other_link(by_node, from_node_id, link.id)
    to_connected = _node_has_other_link(by_node, to_node_id, link.id)
    if from_connected == to_connected:
        return None
    return from_connected


def _links_by_node(dataset: NGIIDataset) -> dict[str, list[Any]]:
    by_node: dict[str, list[Any]] = defaultdict(list)
    for link in dataset.store_for_role("link").features:
        from_node_id = optional_text(getattr(link, "from_node_id", None))
        to_node_id = optional_text(getattr(link, "to_node_id", None))
        if from_node_id:
            by_node[from_node_id].append(link)
        if to_node_id:
            by_node[to_node_id].append(link)
    return by_node


def _node_has_other_link(by_node: dict[str, list[Any]], node_id: str, link_id: str) -> bool:
    return any(link.id != link_id for link in by_node.get(node_id, ()))


def _majority_node_reversal_vote(votes: tuple[bool, ...]) -> bool | None:
    true_count = sum(1 for vote in votes if vote)
    false_count = len(votes) - true_count
    if true_count == false_count:
        return None
    return true_count > false_count


def _orientation_decision(
    node_store: LayerStore[Any],
    link: Any,
    node_ids_reversed: bool,
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> _OrientationDecision | None:
    if len(link.polyline) < 2:
        if check_reports(cfg.sanity.checks.link_orientation_reversed):
            sanity.warn(
                "link-orientation-geometry-invalid",
                f"{link.layer_name} {link.id} has too few polyline points for orientation repair",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return None
    current_from_id = optional_text(getattr(link, "from_node_id", None))
    current_to_id = optional_text(getattr(link, "to_node_id", None))
    desired_from_id = current_to_id if node_ids_reversed else current_from_id
    desired_to_id = current_from_id if node_ids_reversed else current_to_id
    polyline_reversed = _polyline_reversed_for_desired_endpoints(
        node_store,
        link,
        desired_from_id,
        desired_to_id,
        sanity,
        cfg,
    )
    if polyline_reversed is None:
        return None
    return _OrientationDecision(
        node_ids_reversed=node_ids_reversed,
        polyline_reversed=polyline_reversed,
    )


def _polyline_reversed_for_desired_endpoints(
    node_store: LayerStore[Any],
    link: Any,
    desired_from_id: str,
    desired_to_id: str,
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> bool | None:
    from_node = node_store.get(desired_from_id)
    to_node = node_store.get(desired_to_id)
    if from_node is None or to_node is None:
        if check_reports(cfg.sanity.checks.link_orientation_reversed):
            sanity.warn(
                "link-orientation-endpoint-unresolved",
                f"{link.layer_name} {link.id} cannot resolve desired orientation endpoints",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return None

    start = link.polyline[0]
    end = link.polyline[-1]
    tolerance_m = cfg.sanity.node_match_tolerance_m
    normal = (
        xy_distance(start, from_node.point) <= tolerance_m
        and xy_distance(end, to_node.point) <= tolerance_m
    )
    reversed_alignment = (
        xy_distance(start, to_node.point) <= tolerance_m
        and xy_distance(end, from_node.point) <= tolerance_m
    )
    if normal and not reversed_alignment:
        return False
    if reversed_alignment and not normal:
        return True
    if normal and reversed_alignment:
        return None
    if check_reports(cfg.sanity.checks.link_endpoint_misaligned):
        sanity.warn(
            "link-endpoint-misaligned",
            f"{link.layer_name} {link.id} endpoint nodes do not match polyline endpoints "
            f"within {tolerance_m:.3f} m",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    return None


def _apply_or_warn_orientation(
    link: Any,
    decision: _OrientationDecision,
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    if not decision.node_ids_reversed and not decision.polyline_reversed:
        return
    operation = _orientation_operation_label(decision)
    if check_repairs(cfg.sanity.checks.link_orientation_reversed):
        before = _orientation_snapshot(link)
        if decision.node_ids_reversed:
            link.from_node_id, link.to_node_id = link.to_node_id, link.from_node_id
        if decision.polyline_reversed:
            link.polyline = link.polyline[::-1].copy()
        sanity.action(
            "link-orientation-reversed-repaired",
            f"{link.layer_name} {link.id} orientation repaired: {operation}",
            before=before,
            after=_orientation_snapshot(link),
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif check_reports(cfg.sanity.checks.link_orientation_reversed):
        sanity.warn(
            "link-orientation-reversed",
            f"{link.layer_name} {link.id} orientation appears reversed: {operation}",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _orientation_operation_label(decision: _OrientationDecision) -> str:
    if decision.node_ids_reversed and decision.polyline_reversed:
        return "swap FromNodeID/ToNodeID and reverse polyline"
    if decision.node_ids_reversed:
        return "swap FromNodeID/ToNodeID"
    return "reverse polyline"


def _orientation_snapshot(link: Any) -> dict[str, Any]:
    return {
        "from_node_id": getattr(link, "from_node_id", None),
        "to_node_id": getattr(link, "to_node_id", None),
        "polyline_start": _point_tuple(link.polyline[0]),
        "polyline_end": _point_tuple(link.polyline[-1]),
    }


def _point_tuple(point: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in point)


def _warn_orientation_group(
    sanity: SanityReport,
    cfg: NGIIConfig,
    link_store: LayerStore[Any],
    group: tuple[Any, ...],
    reason: str,
) -> None:
    if not check_reports(cfg.sanity.checks.link_orientation_reversed):
        return
    examples = ", ".join(link.id for link in group[:8])
    sanity.warn(
        "link-orientation-reversed-ambiguous",
        f"{link_store.layer_name} orientation group skipped: {reason}; links={examples}",
        layer_name=link_store.layer_name,
    )


def check_link_side_reference_longitudinal(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    for link in link_store.features:
        _clear_longitudinal_side_reference(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="l_link_id",
            source_column="L_LinkID",
            reciprocal_attr="r_link_id",
        )
        _clear_longitudinal_side_reference(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="r_link_id",
            source_column="R_LinkID",
            reciprocal_attr="l_link_id",
        )


def _clear_longitudinal_side_reference(
    sanity: SanityReport,
    cfg: NGIIConfig,
    link_store: LayerStore[Any],
    link: Any,
    *,
    source_attr: str,
    source_column: str,
    reciprocal_attr: str,
) -> None:
    target_id = optional_text(getattr(link, source_attr, None))
    if not target_id:
        return
    target = link_store.get(target_id)
    if target is None:
        return
    shared_node_id = _shared_endpoint_node_id(link, target)
    if not shared_node_id:
        return

    source_value = getattr(link, source_attr, None)
    reciprocal_value = getattr(target, reciprocal_attr, None)
    clears_reciprocal = optional_text(reciprocal_value) == link.id
    before = {source_attr: source_value}
    after = {source_attr: None}
    if clears_reciprocal:
        reciprocal_key = f"{target.id}.{reciprocal_attr}"
        before[reciprocal_key] = reciprocal_value
        after[reciprocal_key] = None

    if check_repairs(cfg.sanity.checks.link_side_reference_longitudinal):
        setattr(link, source_attr, None)
        if clears_reciprocal:
            setattr(target, reciprocal_attr, None)
        sanity.action(
            "link-side-reference-longitudinal-cleared",
            f"{link.layer_name} {link.id} {source_column}={target_id!r} points to "
            f"longitudinally connected {target.layer_name} {target.id} through node "
            f"{shared_node_id!r}",
            before=before,
            after=after,
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif check_reports(cfg.sanity.checks.link_side_reference_longitudinal):
        sanity.warn(
            "link-side-reference-longitudinal",
            f"{link.layer_name} {link.id} {source_column}={target_id!r} points to "
            f"longitudinally connected {target.layer_name} {target.id} through node "
            f"{shared_node_id!r}",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _shared_endpoint_node_id(link: Any, target: Any) -> str:
    link_from_node_id = optional_text(getattr(link, "from_node_id", None))
    link_to_node_id = optional_text(getattr(link, "to_node_id", None))
    target_from_node_id = optional_text(getattr(target, "from_node_id", None))
    target_to_node_id = optional_text(getattr(target, "to_node_id", None))
    if link_to_node_id and link_to_node_id == target_from_node_id:
        return link_to_node_id
    if link_from_node_id and link_from_node_id == target_to_node_id:
        return link_from_node_id
    return ""


def check_link_side_reference_nonreciprocal(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    links_by_left_ref = _links_by_side_ref(link_store, "l_link_id")
    links_by_right_ref = _links_by_side_ref(link_store, "r_link_id")
    for link in link_store.features:
        _check_nonreciprocal_side_reference(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="l_link_id",
            source_column="L_LinkID",
            reciprocal_attr="r_link_id",
            reciprocal_column="R_LinkID",
            inverse_links_by_ref=links_by_right_ref,
        )
        _check_nonreciprocal_side_reference(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="r_link_id",
            source_column="R_LinkID",
            reciprocal_attr="l_link_id",
            reciprocal_column="L_LinkID",
            inverse_links_by_ref=links_by_left_ref,
        )


def _links_by_side_ref(
    link_store: LayerStore[Any],
    attr_name: str,
) -> dict[str, list[Any]]:
    links_by_ref: dict[str, list[Any]] = defaultdict(list)
    for link in link_store.features:
        ref_id = optional_text(getattr(link, attr_name, None))
        if ref_id:
            links_by_ref[ref_id].append(link)
    return links_by_ref


def _check_nonreciprocal_side_reference(
    sanity: SanityReport,
    cfg: NGIIConfig,
    link_store: LayerStore[Any],
    link: Any,
    *,
    source_attr: str,
    source_column: str,
    reciprocal_attr: str,
    reciprocal_column: str,
    inverse_links_by_ref: dict[str, list[Any]],
) -> None:
    current_id = optional_text(getattr(link, source_attr, None))
    if not current_id:
        return
    current = link_store.get(current_id)
    if current is None:
        return
    current_reciprocal_id = optional_text(getattr(current, reciprocal_attr, None))
    if current_reciprocal_id == link.id:
        return
    candidates = tuple(inverse_links_by_ref.get(link.id, ()))
    if len(candidates) != 1:
        _warn_nonreciprocal_side_reference(
            sanity,
            cfg,
            link,
            source_column,
            reciprocal_column,
            current_id,
            current_reciprocal_id,
            tuple(candidate.id for candidate in candidates),
        )
        return
    repaired_id = candidates[0].id
    before = {source_attr: getattr(link, source_attr, None)}
    after = {source_attr: repaired_id}
    if check_repairs(cfg.sanity.checks.link_side_reference_nonreciprocal):
        setattr(link, source_attr, repaired_id)
        sanity.action(
            "link-side-reference-nonreciprocal-redirected",
            f"{link.layer_name} {link.id} {source_column} repaired from {current_id!r} "
            f"to {repaired_id!r} by inverse {reciprocal_column} evidence",
            before=before,
            after=after,
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif check_reports(cfg.sanity.checks.link_side_reference_nonreciprocal):
        sanity.warn(
            "link-side-reference-nonreciprocal",
            f"{link.layer_name} {link.id} {source_column}={current_id!r} does not point "
            f"reciprocally through {reciprocal_column}; inverse evidence suggests "
            f"{repaired_id!r}",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _warn_nonreciprocal_side_reference(
    sanity: SanityReport,
    cfg: NGIIConfig,
    link: Any,
    source_column: str,
    reciprocal_column: str,
    current_id: str,
    current_reciprocal_id: str,
    candidate_ids: tuple[str, ...],
) -> None:
    if not check_reports(cfg.sanity.checks.link_side_reference_nonreciprocal):
        return
    candidate_label = ", ".join(candidate_ids) if candidate_ids else "-"
    sanity.warn(
        "link-side-reference-nonreciprocal-ambiguous",
        f"{link.layer_name} {link.id} {source_column}={current_id!r} does not point "
        f"reciprocally through {reciprocal_column}={current_reciprocal_id!r}; "
        f"found {len(candidate_ids)} inverse candidate(s): {candidate_label}",
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
    )


def check_reciprocal_references(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    for rule in dataset.schema.reciprocal_references:
        store = dataset.store_for_attr(rule.layer_attr)
        for feature in store.features:
            target_id = optional_text(getattr(feature, rule.source_attr, None))
            if not target_id:
                continue
            target = store.get(target_id)
            if target is None:
                continue
            reciprocal_id = optional_text(getattr(target, rule.reciprocal_attr, None))
            if reciprocal_id == feature.id:
                continue
            if not reciprocal_id:
                _fill_missing_reciprocal_reference(sanity, cfg, rule, feature, target)
                continue
            if check_reports(cfg.sanity.checks.reciprocal_reference_conflict):
                sanity.warn(
                    "reciprocal-reference-conflict",
                    f"{target.layer_name} {target.id} {rule.reciprocal_column}="
                    f"{reciprocal_id!r} does not point back to {feature.layer_name} "
                    f"{feature.id} from {rule.source_column}",
                    layer_name=target.layer_name,
                    feature_id=target.id,
                    source_path=target.source_path,
                )


def _fill_missing_reciprocal_reference(
    sanity: SanityReport,
    cfg: NGIIConfig,
    rule: ReciprocalReferenceRule,
    feature: Any,
    target: Any,
) -> None:
    before = {rule.reciprocal_attr: getattr(target, rule.reciprocal_attr, None)}
    after = {rule.reciprocal_attr: feature.id}
    if check_repairs(cfg.sanity.checks.reciprocal_reference_missing):
        setattr(target, rule.reciprocal_attr, feature.id)
        sanity.action(
            "reciprocal-reference-missing-filled",
            f"{target.layer_name} {target.id} {rule.reciprocal_column} filled to point "
            f"back to {feature.layer_name} {feature.id}",
            before=before,
            after=after,
            layer_name=target.layer_name,
            feature_id=target.id,
            source_path=target.source_path,
        )
    elif check_reports(cfg.sanity.checks.reciprocal_reference_missing):
        sanity.warn(
            "reciprocal-reference-missing",
            f"{target.layer_name} {target.id} {rule.reciprocal_column} could point back to "
            f"{feature.layer_name} {feature.id}",
            layer_name=target.layer_name,
            feature_id=target.id,
            source_path=target.source_path,
        )


def _remove_features_with_cascade(
    dataset: NGIIDataset,
    sanity: SanityReport,
    seed_causes: dict[FeatureRef, _RemovalCause],
) -> set[FeatureRef]:
    removal_causes = dict(seed_causes)
    queue = list(removal_causes)
    while queue:
        queue.pop(0)
        for dependent_ref, cause in _required_dependents_for_removed_refs(
            dataset, frozenset(removal_causes)
        ).items():
            if dependent_ref in removal_causes:
                continue
            removal_causes[dependent_ref] = cause
            queue.append(dependent_ref)

    removal_refs = set(removal_causes)
    if not removal_refs:
        return set()

    for feature_ref, cause in removal_causes.items():
        feature = dataset.store_for_attr(feature_ref.layer_attr).get(feature_ref.feature_id)
        if feature is None:
            continue
        sanity.action(
            cause.code,
            cause.message,
            before=cause.before,
            after=cause.after,
            layer_name=feature.layer_name,
            feature_id=feature.id,
            source_path=feature.source_path,
        )

    _clear_optional_references_to_removed(dataset, sanity, frozenset(removal_refs))
    for layer_attr, store in dataset.layer_items:
        ids_to_remove = {
            feature_ref.feature_id
            for feature_ref in removal_refs
            if feature_ref.layer_attr == layer_attr
        }
        store.remove_feature_ids(ids_to_remove)
    return removal_refs


def _required_dependents_for_removed_refs(
    dataset: NGIIDataset,
    removal_refs: frozenset[FeatureRef],
) -> dict[FeatureRef, _RemovalCause]:
    dependents: dict[FeatureRef, _RemovalCause] = {}
    for layer_attr, store in dataset.layer_items:
        for feature in store.features:
            source_ref = FeatureRef(layer_attr, feature.id)
            if source_ref in removal_refs:
                continue
            for reference in store.spec.references:
                if not reference.required:
                    continue
                value = optional_text(getattr(feature, reference.source_attr, None))
                if not value or not _reference_targets_removed_feature(
                    dataset, reference.target_attrs, value, removal_refs
                ):
                    continue
                if _reference_resolves_outside_removal(
                    dataset, reference.target_attrs, value, removal_refs
                ):
                    continue
                target_label = _reference_removed_target_label(
                    dataset, reference.target_attrs, value, removal_refs
                )
                dependents.setdefault(
                    source_ref,
                    _RemovalCause(
                        code="required-reference-dependent-removed",
                        message=(
                            f"{feature.layer_name} {feature.id} "
                            f"{reference.column_name}={value!r} depends on removed "
                            f"{target_label}"
                        ),
                        before={reference.source_attr: getattr(feature, reference.source_attr)},
                        after={"removed": True},
                    ),
                )
    return dependents


def _clear_optional_references_to_removed(
    dataset: NGIIDataset,
    sanity: SanityReport,
    removal_refs: frozenset[FeatureRef],
) -> None:
    for layer_attr, store in dataset.layer_items:
        for feature in store.features:
            source_ref = FeatureRef(layer_attr, feature.id)
            if source_ref in removal_refs:
                continue
            for reference in store.spec.references:
                if reference.required:
                    continue
                value = optional_text(getattr(feature, reference.source_attr, None))
                if not value or not _reference_targets_removed_feature(
                    dataset, reference.target_attrs, value, removal_refs
                ):
                    continue
                if _reference_resolves_outside_removal(
                    dataset, reference.target_attrs, value, removal_refs
                ):
                    continue
                before = {reference.source_attr: getattr(feature, reference.source_attr)}
                setattr(feature, reference.source_attr, None)
                sanity.action(
                    "optional-dependent-reference-cleared",
                    f"{feature.layer_name} {feature.id} {reference.column_name}={value!r} "
                    "depended on a removed feature and was cleared",
                    before=before,
                    after={reference.source_attr: None},
                    layer_name=feature.layer_name,
                    feature_id=feature.id,
                    source_path=feature.source_path,
                )


def _reference_targets_removed_feature(
    dataset: NGIIDataset,
    target_attrs: tuple[str, ...],
    feature_id: str,
    removal_refs: frozenset[FeatureRef],
) -> bool:
    return any(FeatureRef(target_attr, feature_id) in removal_refs for target_attr in target_attrs)


def _reference_resolves_outside_removal(
    dataset: NGIIDataset,
    target_attrs: tuple[str, ...],
    feature_id: str,
    removal_refs: frozenset[FeatureRef],
) -> bool:
    return any(
        FeatureRef(target_attr, feature_id) not in removal_refs
        and dataset.store_for_attr(target_attr).get(feature_id) is not None
        for target_attr in target_attrs
    )


def _reference_removed_target_label(
    dataset: NGIIDataset,
    target_attrs: tuple[str, ...],
    feature_id: str,
    removal_refs: frozenset[FeatureRef],
) -> str:
    labels = [
        f"{dataset.store_for_attr(target_attr).layer_name} {feature_id}"
        for target_attr in target_attrs
        if FeatureRef(target_attr, feature_id) in removal_refs
    ]
    return "/".join(labels)


def _other_link_uses_node(link_store: LayerStore[Any], link_id: str, node_id: str) -> bool:
    return any(
        link.id != link_id and node_id in {link.from_node_id, link.to_node_id}
        for link in link_store.features
    )


def _reference_resolves(
    dataset: NGIIDataset, target_attrs: tuple[str, ...], feature_id: str
) -> bool:
    return any(
        dataset.store_for_attr(target_attr).get(feature_id) is not None
        for target_attr in target_attrs
    )


def _reference_target_label(dataset: NGIIDataset, target_attrs: tuple[str, ...]) -> str:
    return "/".join(dataset.store_for_attr(target_attr).layer_name for target_attr in target_attrs)


DEFAULT_SANITY_HOOKS: tuple[tuple[str, SanityHook], ...] = (
    ("link_too_short", check_link_too_short),
    ("link_endpoint_isolated", check_link_endpoint_isolated),
    ("link_endpoint_unresolved", check_link_endpoint_unresolved),
    ("reference_unresolved", check_reference_unresolved),
    ("link_side_reference_longitudinal", check_link_side_reference_longitudinal),
    ("link_side_reference_nonreciprocal", check_link_side_reference_nonreciprocal),
    ("link_orientation_reversed", check_link_orientation_reversed),
    ("node_unreferenced", check_node_unreferenced),
    ("reciprocal_reference", check_reciprocal_references),
)
