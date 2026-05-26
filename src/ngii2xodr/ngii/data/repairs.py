"""Version-agnostic NGII topology sanity repairs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import (
    NGIIFeature,
    iter_feature_text_fields,
    same_feature,
    set_feature_column,
)
from ngii2xodr.ngii.data.geometry import xy_distance
from ngii2xodr.ngii.data.sanity import SanityReport

RepairHook = Callable[[NGIIDataset, NGIIConfig], None]
_REPLACEMENT_CHAR = "\ufffd"


def merge_features(
    store: LayerStore[Any],
    features: list[NGIIFeature],
    sanity: SanityReport,
    cfg: NGIIConfig,
) -> None:
    for feature in features:
        if not feature.id:
            if cfg.sanity.warnings.manual_field_rules:
                sanity.warn(
                    "missing-feature-id",
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
            if cfg.sanity.warnings.duplicate_identical_ids:
                sanity.warn(
                    "duplicate-identical-id",
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
        if cfg.sanity.repairs.duplicate_conflicting_id_drop:
            sanity.action(
                "duplicate-conflicting-id-dropped",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row dropped",
                before=before,
                after=after,
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )
        elif cfg.sanity.warnings.duplicate_conflicting_ids:
            sanity.warn(
                "duplicate-conflicting-id-drop-disabled",
                f"{feature.layer_name} ID {feature.id!r} conflicts with an earlier row; "
                "later row could not become canonical because duplicate repair is disabled",
                layer_name=feature.layer_name,
                feature_id=feature.id,
                source_path=feature.source_path,
            )


def apply_text_repairs(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    if not cfg.text_repair.enabled:
        return
    if cfg.text_repair.repair_mojibake:
        _repair_mojibake_text(dataset)
    if cfg.text_repair.warn_unrepaired_replacement_chars:
        _warn_unrepaired_replacement_chars(dataset)


def _repair_mojibake_text(dataset: NGIIDataset) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in iter_feature_text_fields(feature):
                repaired = _repair_cp949_latin1_mojibake(value)
                if repaired is None:
                    continue
                set_feature_column(feature, field_name, repaired)
                dataset.sanity.action(
                    "text-mojibake-repaired",
                    f"{feature.layer_name} {feature.id} {field_name} repaired by "
                    "latin1-to-cp949 mojibake rule",
                    before={field_name: value},
                    after={field_name: repaired},
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


def _warn_unrepaired_replacement_chars(dataset: NGIIDataset) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in iter_feature_text_fields(feature):
                if _REPLACEMENT_CHAR in value:
                    dataset.sanity.warn(
                        "corrupt-text-unrepaired",
                        f"{feature.layer_name} {feature.id}: {field_name} contains "
                        "Unicode replacement characters",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def repair_reversed_link_endpoints(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
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
                link,
                "link-direction-swapped",
                "had reversed FromNodeID/ToNodeID relative to geometry order",
                enabled=cfg.sanity.repairs.link_endpoint_direction_swap,
                warn_disabled=cfg.sanity.warnings.link_endpoint_alignment,
            )
        elif reversed_alignment and normal and cfg.sanity.warnings.link_direction_ambiguous:
            dataset.sanity.warn(
                "link-direction-ambiguous",
                f"{link.layer_name} {link.id} endpoints match both normal and reversed "
                "node ordering",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif not normal and cfg.sanity.warnings.link_endpoint_alignment:
            dataset.sanity.warn(
                "link-endpoint-alignment-mismatch",
                f"{link.layer_name} {link.id} endpoint nodes do not match polyline endpoints "
                f"within {tolerance_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )


def repair_missing_link_node_refs(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    node_store = dataset.store_for_role("node")
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in dataset.store_for_role("link").features:
        if len(link.polyline) < 2:
            continue
        _repair_endpoint(
            dataset,
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
        if cfg.sanity.repairs.link_missing_node_ref_nearest:
            setattr(link, attr_name, repaired_id)
            dataset.sanity.action(
                "link-node-ref-nearest",
                f"{link.layer_name} {link.id} {column_name} repaired to nearby "
                f"{node_store.layer_name} {repaired_id}",
                before=before,
                after={attr_name: repaired_id},
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif cfg.sanity.warnings.link_endpoint_alignment:
            dataset.sanity.warn(
                "link-node-ref-nearest-disabled",
                f"{link.layer_name} {link.id} {column_name} could be repaired to nearby "
                f"{node_store.layer_name} {repaired_id}, but nearest-node repair is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if len(nearby) > 1 and cfg.sanity.warnings.link_endpoint_alignment:
        dataset.sanity.warn(
            "link-node-ref-ambiguous",
            f"{link.layer_name} {link.id} {column_name} has {len(nearby)} nearby "
            f"{node_store.layer_name} candidates",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    if cfg.sanity.repairs.link_missing_node_ref_remove:
        setattr(link, attr_name, None)
        dataset.sanity.action(
            "link-node-ref-removed",
            f"{link.layer_name} {link.id} {column_name} could not be resolved and was removed",
            before=before,
            after={attr_name: None},
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.link_endpoint_alignment:
        dataset.sanity.warn(
            "link-node-ref-remove-disabled",
            f"{link.layer_name} {link.id} {column_name} could not be resolved, but relation "
            "removal is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def repair_link_topology_direction(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
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
        if cfg.sanity.warnings.link_topology_direction:
            dataset.sanity.warn(
                "link-topology-direction-ambiguous",
                "topology flow and R/L same-direction evidence found multiple possible "
                f"backward {link_store.layer_name} features: "
                f"{', '.join(link.id for link in candidates[:8])}",
                layer_name=link_store.layer_name,
            )
        return
    link = candidates[0]
    if cfg.sanity.repairs.link_topology_direction_swap:
        _swap_endpoint_ids(
            dataset,
            link,
            "link-topology-direction-swapped",
            "was reversed by topology flow and R/L same-direction evidence",
            enabled=True,
            warn_disabled=False,
        )
    elif cfg.sanity.warnings.link_topology_direction:
        dataset.sanity.warn(
            "link-topology-direction-swap-disabled",
            f"{link.layer_name} {link.id} appears reversed by topology flow and R/L "
            "same-direction evidence, but topology repair is disabled",
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
            dataset.sanity.warn(
                f"{code}-disabled",
                f"{link.layer_name} {link.id} {reason}, but direction swap is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    link.from_node_id, link.to_node_id = link.to_node_id, link.from_node_id
    dataset.sanity.action(
        code,
        f"{link.layer_name} {link.id} {reason}",
        before=before,
        after={"from_node_id": link.from_node_id, "to_node_id": link.to_node_id},
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
    )


DEFAULT_REPAIR_HOOKS: tuple[tuple[str, RepairHook], ...] = (
    ("link_endpoint_direction", repair_reversed_link_endpoints),
    ("link_missing_node_refs", repair_missing_link_node_refs),
    ("link_topology_direction", repair_link_topology_direction),
)
