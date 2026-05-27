"""Version-agnostic NGII topology sanity repairs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import LayerStore, NGIIDataset
from ngii2xodr.ngii.data.features import NGIIFeature, same_feature
from ngii2xodr.ngii.data.geometry import xy_distance, xy_line
from ngii2xodr.ngii.data.sanity import SanityReport
from ngii2xodr.ngii.data.schema import ReciprocalRelationshipRule

RepairHook = Callable[[NGIIDataset, SanityReport, NGIIConfig], None]
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


def apply_text_repairs(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    if not cfg.text_repair.enabled:
        return
    if cfg.text_repair.repair_mojibake:
        _repair_mojibake_text(dataset, sanity)
    if cfg.text_repair.warn_unrepaired_replacement_chars:
        _warn_unrepaired_replacement_chars(dataset, sanity)


def _repair_mojibake_text(dataset: NGIIDataset, sanity: SanityReport) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for field_name, value in store.iter_text_fields(feature):
                repaired = _repair_cp949_latin1_mojibake(value)
                if repaired is None:
                    continue
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
                        "corrupt-text-unrepaired",
                        f"{feature.layer_name} {feature.id}: {field_name} contains "
                        "Unicode replacement characters",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def repair_too_short_links(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    link_store = dataset.store_for_role("link")
    link_ids_to_remove: list[str] = []
    threshold_m = cfg.sanity.link_min_length_m
    for link in link_store.features:
        length_m = float(xy_line(link.polyline).length)
        if length_m >= threshold_m:
            continue
        if cfg.sanity.repairs.link_too_short_remove:
            link_ids_to_remove.append(link.id)
            sanity.action(
                "link-too-short-removed",
                f"{link.layer_name} {link.id} geometry length {length_m:.3f} m is shorter "
                f"than {threshold_m:.3f} m",
                before={"length_m": length_m, "threshold_m": threshold_m},
                after={"removed": True},
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif cfg.sanity.warnings.too_short_links:
            sanity.warn(
                "link-too-short-remove-disabled",
                f"{link.layer_name} {link.id} geometry length {length_m:.3f} m is shorter "
                f"than {threshold_m:.3f} m, but short-link removal is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
    link_store.remove_feature_ids(link_ids_to_remove)


def repair_dangling_relationships(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    for store in dataset.layer_stores:
        for feature in store.features:
            for relationship in store.spec.relationships:
                value = _optional_text(getattr(feature, relationship.source_attr, None))
                if not value or _relationship_resolves(dataset, relationship.target_attrs, value):
                    continue
                before = {relationship.source_attr: getattr(feature, relationship.source_attr)}
                if cfg.sanity.repairs.dangling_relationship_remove:
                    setattr(feature, relationship.source_attr, None)
                    sanity.action(
                        "dangling-relationship-removed",
                        f"{feature.layer_name} {feature.id} {relationship.column_name}={value!r} "
                        "does not resolve and was removed",
                        before=before,
                        after={relationship.source_attr: None},
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )
                elif cfg.sanity.warnings.dangling_relationships:
                    target_layer = _relationship_target_label(dataset, relationship.target_attrs)
                    sanity.warn(
                        "dangling-relationship-remove-disabled",
                        f"{feature.layer_name} {feature.id} {relationship.column_name}={value!r} "
                        f"does not resolve to {target_layer}, but dangling-reference removal "
                        "is disabled",
                        layer_name=feature.layer_name,
                        feature_id=feature.id,
                        source_path=feature.source_path,
                    )


def repair_dangling_nodes(dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig) -> None:
    link_store = dataset.store_for_role("link")
    node_store = dataset.store_for_role("node")
    used_node_ids = {
        node_id
        for link in link_store.features
        for node_id in (link.from_node_id, link.to_node_id)
        if node_id
    }
    node_ids_to_remove: list[str] = []
    for node in node_store.features:
        if node.id in used_node_ids:
            continue
        if cfg.sanity.repairs.dangling_node_remove:
            node_ids_to_remove.append(node.id)
            sanity.action(
                "dangling-node-removed",
                f"{node.layer_name} {node.id} is not referenced by any remaining link endpoint",
                before={"referenced_by_link_endpoint": False},
                after={"removed": True},
                layer_name=node.layer_name,
                feature_id=node.id,
                source_path=node.source_path,
            )
        elif cfg.sanity.warnings.dangling_nodes:
            sanity.warn(
                "dangling-node-remove-disabled",
                f"{node.layer_name} {node.id} is not referenced by any remaining link endpoint, "
                "but dangling-node removal is disabled",
                layer_name=node.layer_name,
                feature_id=node.id,
                source_path=node.source_path,
            )
    node_store.remove_feature_ids(node_ids_to_remove)


def repair_reversed_link_endpoints(
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
                "link-direction-swapped",
                "had reversed FromNodeID/ToNodeID relative to geometry order",
                enabled=cfg.sanity.repairs.link_endpoint_direction_swap,
                warn_disabled=cfg.sanity.warnings.link_endpoint_alignment,
            )
        elif reversed_alignment and normal and cfg.sanity.warnings.link_direction_ambiguous:
            sanity.warn(
                "link-direction-ambiguous",
                f"{link.layer_name} {link.id} endpoints match both normal and reversed "
                "node ordering",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif not normal and cfg.sanity.warnings.link_endpoint_alignment:
            sanity.warn(
                "link-endpoint-alignment-mismatch",
                f"{link.layer_name} {link.id} endpoint nodes do not match polyline endpoints "
                f"within {tolerance_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )


def repair_missing_link_node_refs(
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
        if cfg.sanity.repairs.link_missing_node_ref_nearest:
            setattr(link, attr_name, repaired_id)
            sanity.action(
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
            sanity.warn(
                "link-node-ref-nearest-disabled",
                f"{link.layer_name} {link.id} {column_name} could be repaired to nearby "
                f"{node_store.layer_name} {repaired_id}, but nearest-node repair is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if len(nearby) > 1 and cfg.sanity.warnings.link_endpoint_alignment:
        sanity.warn(
            "link-node-ref-ambiguous",
            f"{link.layer_name} {link.id} {column_name} has {len(nearby)} nearby "
            f"{node_store.layer_name} candidates",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    if current_id is None:
        return
    if cfg.sanity.repairs.link_missing_node_ref_remove:
        setattr(link, attr_name, None)
        sanity.action(
            "link-node-ref-removed",
            f"{link.layer_name} {link.id} {column_name} could not be resolved and was removed",
            before=before,
            after={attr_name: None},
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.link_endpoint_alignment:
        sanity.warn(
            "link-node-ref-remove-disabled",
            f"{link.layer_name} {link.id} {column_name} could not be resolved, but relation "
            "removal is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def repair_link_topology_direction(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
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
            sanity.warn(
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
            sanity,
            link,
            "link-topology-direction-swapped",
            "was reversed by topology flow and R/L same-direction evidence",
            enabled=True,
            warn_disabled=False,
        )
    elif cfg.sanity.warnings.link_topology_direction:
        sanity.warn(
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
                f"{link.layer_name} {link.id} {reason}, but direction swap is disabled",
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


def repair_link_lateral_longitudinal_conflicts(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    for link in link_store.features:
        _clear_lateral_longitudinal_conflict(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="l_link_id",
            source_column="L_LinkID",
            reciprocal_attr="r_link_id",
        )
        _clear_lateral_longitudinal_conflict(
            sanity,
            cfg,
            link_store,
            link,
            source_attr="r_link_id",
            source_column="R_LinkID",
            reciprocal_attr="l_link_id",
        )


def _clear_lateral_longitudinal_conflict(
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

    if cfg.sanity.repairs.link_lateral_longitudinal_conflict_clear:
        setattr(link, source_attr, None)
        if clears_reciprocal:
            setattr(target, reciprocal_attr, None)
        sanity.action(
            "link-lateral-longitudinal-conflict-cleared",
            f"{link.layer_name} {link.id} {source_column}={target_id!r} points to "
            f"longitudinally connected {target.layer_name} {target.id} through node "
            f"{shared_node_id!r}",
            before=before,
            after=after,
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.link_lateral_longitudinal_conflict:
        sanity.warn(
            "link-lateral-longitudinal-conflict-clear-disabled",
            f"{link.layer_name} {link.id} {source_column}={target_id!r} points to "
            f"longitudinally connected {target.layer_name} {target.id} through node "
            f"{shared_node_id!r}, but lateral longitudinal conflict repair is disabled",
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


def repair_link_lateral_reciprocal_conflicts(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    link_store = dataset.store_for_role("link")
    links_by_left_ref = _links_by_lateral_ref(link_store, "l_link_id")
    links_by_right_ref = _links_by_lateral_ref(link_store, "r_link_id")
    for link in link_store.features:
        _repair_lateral_reciprocal_conflict(
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
        _repair_lateral_reciprocal_conflict(
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


def _links_by_lateral_ref(
    link_store: LayerStore[Any],
    attr_name: str,
) -> dict[str, list[Any]]:
    links_by_ref: dict[str, list[Any]] = defaultdict(list)
    for link in link_store.features:
        ref_id = _optional_text(getattr(link, attr_name, None))
        if ref_id:
            links_by_ref[ref_id].append(link)
    return links_by_ref


def _repair_lateral_reciprocal_conflict(
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
        _warn_lateral_reciprocal_conflict(
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
    if cfg.sanity.repairs.link_lateral_reciprocal_conflict_repair:
        setattr(link, source_attr, repaired_id)
        sanity.action(
            "link-lateral-reciprocal-conflict-repaired",
            f"{link.layer_name} {link.id} {source_column} repaired from {current_id!r} "
            f"to {repaired_id!r} by inverse {reciprocal_column} evidence",
            before=before,
            after=after,
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.link_lateral_reciprocal_conflict:
        sanity.warn(
            "link-lateral-reciprocal-conflict-repair-disabled",
            f"{link.layer_name} {link.id} {source_column}={current_id!r} does not point "
            f"reciprocally through {reciprocal_column}; inverse evidence suggests "
            f"{repaired_id!r}, but lateral reciprocal repair is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _warn_lateral_reciprocal_conflict(
    sanity: SanityReport,
    cfg: NGIIConfig,
    link: Any,
    source_column: str,
    reciprocal_column: str,
    current_id: str,
    current_reciprocal_id: str,
    candidate_ids: tuple[str, ...],
) -> None:
    if not cfg.sanity.warnings.link_lateral_reciprocal_conflict:
        return
    candidate_label = ", ".join(candidate_ids) if candidate_ids else "-"
    sanity.warn(
        "link-lateral-reciprocal-conflict-ambiguous",
        f"{link.layer_name} {link.id} {source_column}={current_id!r} does not point "
        f"reciprocally through {reciprocal_column}={current_reciprocal_id!r}; "
        f"found {len(candidate_ids)} inverse candidate(s): {candidate_label}",
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
    )


def repair_reciprocal_relationships(
    dataset: NGIIDataset, sanity: SanityReport, cfg: NGIIConfig
) -> None:
    for rule in dataset.schema.reciprocal_relationships:
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
                _fill_missing_reciprocal_relationship(sanity, cfg, rule, feature, target)
                continue
            if cfg.sanity.warnings.reciprocal_relationships:
                sanity.warn(
                    "reciprocal-relation-conflict",
                    f"{target.layer_name} {target.id} {rule.reciprocal_column}="
                    f"{reciprocal_id!r} does not point back to {feature.layer_name} "
                    f"{feature.id} from {rule.source_column}",
                    layer_name=target.layer_name,
                    feature_id=target.id,
                    source_path=target.source_path,
                )


def _fill_missing_reciprocal_relationship(
    sanity: SanityReport,
    cfg: NGIIConfig,
    rule: ReciprocalRelationshipRule,
    feature: Any,
    target: Any,
) -> None:
    before = {rule.reciprocal_attr: getattr(target, rule.reciprocal_attr, None)}
    after = {rule.reciprocal_attr: feature.id}
    if cfg.sanity.repairs.reciprocal_relationship_fill:
        setattr(target, rule.reciprocal_attr, feature.id)
        sanity.action(
            "reciprocal-relation-filled",
            f"{target.layer_name} {target.id} {rule.reciprocal_column} filled to point "
            f"back to {feature.layer_name} {feature.id}",
            before=before,
            after=after,
            layer_name=target.layer_name,
            feature_id=target.id,
            source_path=target.source_path,
        )
    elif cfg.sanity.warnings.reciprocal_relationships:
        sanity.warn(
            "reciprocal-relation-fill-disabled",
            f"{target.layer_name} {target.id} {rule.reciprocal_column} could point back to "
            f"{feature.layer_name} {feature.id}, but reciprocal repair is disabled",
            layer_name=target.layer_name,
            feature_id=target.id,
            source_path=target.source_path,
        )


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _relationship_resolves(
    dataset: NGIIDataset, target_attrs: tuple[str, ...], feature_id: str
) -> bool:
    return any(
        dataset.store_for_attr(target_attr).get(feature_id) is not None
        for target_attr in target_attrs
    )


def _relationship_target_label(dataset: NGIIDataset, target_attrs: tuple[str, ...]) -> str:
    return "/".join(dataset.store_for_attr(target_attr).layer_name for target_attr in target_attrs)


DEFAULT_REPAIR_HOOKS: tuple[tuple[str, RepairHook], ...] = (
    ("link_too_short_removal", repair_too_short_links),
    ("dangling_relationships", repair_dangling_relationships),
    ("link_missing_node_refs", repair_missing_link_node_refs),
    ("link_endpoint_direction", repair_reversed_link_endpoints),
    ("link_topology_direction", repair_link_topology_direction),
    ("dangling_nodes", repair_dangling_nodes),
    ("link_lateral_longitudinal_conflicts", repair_link_lateral_longitudinal_conflicts),
    ("link_lateral_reciprocal_conflicts", repair_link_lateral_reciprocal_conflicts),
    ("reciprocal_relationships", repair_reciprocal_relationships),
)
