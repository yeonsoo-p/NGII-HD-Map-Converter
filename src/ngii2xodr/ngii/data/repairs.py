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
from ngii2xodr.ngii.data.features import FeatureRef, NGIIFeature, same_feature
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
        from_node_id = _optional_text(getattr(link, "from_node_id", None))
        to_node_id = _optional_text(getattr(link, "to_node_id", None))
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
                value = _optional_text(getattr(feature, reference.source_attr, None))
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
        if feature is None or _optional_text(getattr(feature, source_attr, None)) != value:
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


def check_link_endpoint_reversed(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    node_store = dataset.store_for_role("node")
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in link_store.features:
        from_node = node_store.get(link.from_node_id)
        to_node = node_store.get(link.to_node_id)
        if from_node is None or to_node is None or len(link.polyline) < 2:
            continue
        start = link.polyline[0]
        end = link.polyline[-1]
        normal = (
            xy_distance(start, from_node.point) <= tolerance_m
            and xy_distance(end, to_node.point) <= tolerance_m
        )
        reversed_alignment = (
            xy_distance(start, to_node.point) <= tolerance_m
            and xy_distance(end, from_node.point) <= tolerance_m
        )
        if reversed_alignment and not normal:
            _swap_endpoint_ids(
                dataset,
                sanity,
                link,
                "link-endpoint-reversed-swapped",
                "had reversed FromNodeID/ToNodeID relative to geometry order",
                enabled=check_repairs(cfg.sanity.checks.link_endpoint_reversed),
                warn_disabled=check_reports(cfg.sanity.checks.link_endpoint_reversed),
            )
        elif (
            reversed_alignment
            and normal
            and check_reports(cfg.sanity.checks.link_endpoint_order_ambiguous)
        ):
            sanity.warn(
                "link-endpoint-order-ambiguous",
                f"{link.layer_name} {link.id} endpoints match both normal and reversed "
                "node ordering",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif not normal and check_reports(cfg.sanity.checks.link_endpoint_misaligned):
            sanity.warn(
                "link-endpoint-misaligned",
                f"{link.layer_name} {link.id} endpoint nodes do not match polyline endpoints "
                f"within {tolerance_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )


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


def check_link_flow_reversed(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    desired_flip = _desired_flips_from_leaf_flow(dataset)
    candidates: list[Any] = []
    for link in dataset.store_for_role("link").features:
        if not desired_flip.get(link.id, False):
            continue
        if _has_opposing_same_direction_neighbour(
            dataset, link, cfg.sanity.direction_parallel_dot_min
        ):
            candidates.append(link)

    if not candidates:
        return
    link_store = dataset.store_for_role("link")
    if len(candidates) > 1:
        if check_reports(cfg.sanity.checks.link_flow_reversed):
            sanity.warn(
                "link-flow-reversed-ambiguous",
                "topology flow and R/L same-direction evidence found multiple possible "
                f"backward {link_store.layer_name} features: "
                f"{', '.join(link.id for link in candidates[:8])}",
                layer_name=link_store.layer_name,
            )
        return
    link = candidates[0]
    if check_repairs(cfg.sanity.checks.link_flow_reversed):
        _swap_endpoint_ids(
            dataset,
            sanity,
            link,
            "link-flow-reversed-swapped",
            "was reversed by topology flow and R/L same-direction evidence",
            enabled=True,
            warn_disabled=False,
        )
    elif check_reports(cfg.sanity.checks.link_flow_reversed):
        sanity.warn(
            "link-flow-reversed",
            f"{link.layer_name} {link.id} appears reversed by topology flow and R/L "
            "same-direction evidence",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _desired_flips_from_leaf_flow(dataset: NGIIDataset) -> dict[str, bool]:
    by_node = _links_by_node(dataset)
    votes: dict[str, set[bool]] = defaultdict(set)
    queue = _leaf_flow_queue(by_node)
    seen: set[tuple[str, bool]] = set()
    while queue:
        link, flip = queue.pop(0)
        state = (link.id, flip)
        if state in seen:
            continue
        seen.add(state)
        votes[link.id].add(flip)
        _queue_downstream_links(queue, by_node, link, flip)
    return {
        link_id: next(iter(link_votes))
        for link_id, link_votes in votes.items()
        if len(link_votes) == 1
    }


def _links_by_node(dataset: NGIIDataset) -> dict[str, list[Any]]:
    by_node: dict[str, list[Any]] = defaultdict(list)
    for link in dataset.store_for_role("link").features:
        if link.from_node_id:
            by_node[link.from_node_id].append(link)
        if link.to_node_id:
            by_node[link.to_node_id].append(link)
    return by_node


def _leaf_flow_queue(by_node: dict[str, list[Any]]) -> list[tuple[Any, bool]]:
    leaves = {node_id for node_id, links in by_node.items() if len(links) == 1}
    queue: list[tuple[Any, bool]] = []
    for node_id in sorted(leaves):
        link = by_node[node_id][0]
        if link.from_node_id == node_id:
            queue.append((link, False))
        elif link.to_node_id == node_id:
            queue.append((link, True))
    return queue


def _queue_downstream_links(
    queue: list[tuple[Any, bool]],
    by_node: dict[str, list[Any]],
    link: Any,
    flip: bool,
) -> None:
    upstream = link.to_node_id if flip else link.from_node_id
    downstream = link.from_node_id if flip else link.to_node_id
    if upstream is None or downstream is None:
        return
    for next_link in by_node.get(downstream, ()):
        if next_link.id == link.id:
            continue
        if next_link.from_node_id == downstream:
            queue.append((next_link, False))
        elif next_link.to_node_id == downstream:
            queue.append((next_link, True))


def _has_opposing_same_direction_neighbour(
    dataset: NGIIDataset, link: Any, parallel_dot_min: float
) -> bool:
    link_vec = _unit_vector(link)
    if link_vec is None:
        return False
    link_store = dataset.store_for_role("link")
    for neighbour_id in (link.r_link_id, link.l_link_id):
        neighbour = link_store.get(neighbour_id)
        if neighbour is None:
            continue
        neighbour_vec = _unit_vector(neighbour)
        if (
            neighbour_vec is not None
            and float(np.dot(link_vec, neighbour_vec)) <= -parallel_dot_min
        ):
            return True
    return False


def _unit_vector(link: Any) -> NDArray[np.float64] | None:
    if len(link.polyline) < 2:
        return None
    vec = link.polyline[-1, :2] - link.polyline[0, :2]
    norm = float(np.linalg.norm(vec))
    return None if norm <= 0.0 else vec / norm


def _swap_endpoint_ids(
    dataset: NGIIDataset,
    sanity: SanityReport,
    link: Any,
    code: str,
    reason: str,
    *,
    enabled: bool,
    warn_disabled: bool,
) -> None:
    before = {"from_node_id": link.from_node_id, "to_node_id": link.to_node_id}
    if not enabled:
        if warn_disabled:
            sanity.warn(
                f"{code}-disabled",
                f"{link.layer_name} {link.id} {reason}, but repair is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    link.from_node_id, link.to_node_id = link.to_node_id, link.from_node_id
    sanity.action(
        code,
        f"{link.layer_name} {link.id} {reason}",
        before=before,
        after={"from_node_id": link.from_node_id, "to_node_id": link.to_node_id},
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
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
    target_id = _optional_text(getattr(link, source_attr, None))
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
    clears_reciprocal = _optional_text(reciprocal_value) == link.id
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
    link_from_node_id = _optional_text(getattr(link, "from_node_id", None))
    link_to_node_id = _optional_text(getattr(link, "to_node_id", None))
    target_from_node_id = _optional_text(getattr(target, "from_node_id", None))
    target_to_node_id = _optional_text(getattr(target, "to_node_id", None))
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
        ref_id = _optional_text(getattr(link, attr_name, None))
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
    current_id = _optional_text(getattr(link, source_attr, None))
    if not current_id:
        return
    current = link_store.get(current_id)
    if current is None:
        return
    current_reciprocal_id = _optional_text(getattr(current, reciprocal_attr, None))
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
            target_id = _optional_text(getattr(feature, rule.source_attr, None))
            if not target_id:
                continue
            target = store.get(target_id)
            if target is None:
                continue
            reciprocal_id = _optional_text(getattr(target, rule.reciprocal_attr, None))
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
                value = _optional_text(getattr(feature, reference.source_attr, None))
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
                value = _optional_text(getattr(feature, reference.source_attr, None))
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


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    ("link_endpoint_reversed", check_link_endpoint_reversed),
    ("link_flow_reversed", check_link_flow_reversed),
    ("node_unreferenced", check_node_unreferenced),
    ("link_side_reference_longitudinal", check_link_side_reference_longitudinal),
    ("link_side_reference_nonreciprocal", check_link_side_reference_nonreciprocal),
    ("reciprocal_reference", check_reciprocal_references),
)
