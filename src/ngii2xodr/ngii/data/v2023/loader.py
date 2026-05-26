"""NGII 2023.07 loader hooks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ngii2xodr.ngii.data.config import NGIIConfig
from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.data.engine import load_schema
from ngii2xodr.ngii.data.geometry import xy_distance
from ngii2xodr.ngii.data.v2023.definitions import SCHEMA


def load_ngii(root: Path, coordinate: str, cfg: NGIIConfig) -> NGIIDataset:
    """Load one 2023 NGII coordinate product."""
    return load_schema(
        root,
        coordinate,
        cfg,
        SCHEMA,
        repair_hooks=(
            ("a2_endpoint_direction", _repair_reversed_a2_links),
            ("a2_missing_node_refs", _repair_missing_a2_node_refs),
            ("a2_topology_direction", _repair_topology_direction),
        ),
    )


def _repair_reversed_a2_links(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    link_store = dataset.store_for_attr("a2_link")
    node_store = dataset.store_for_attr("a1_node")
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
                "a2-direction-swapped",
                "had reversed FromNodeID/ToNodeID relative to geometry order",
                enabled=cfg.sanity.repairs.a2_endpoint_direction_swap,
                warn_disabled=cfg.sanity.warnings.a2_endpoint_alignment,
            )
        elif reversed_alignment and normal and cfg.sanity.warnings.a2_direction_ambiguous:
            dataset.sanity.warn(
                "a2-direction-ambiguous",
                f"A2_LINK {link.id} endpoints match both normal and reversed node ordering",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif not normal and cfg.sanity.warnings.a2_endpoint_alignment:
            dataset.sanity.warn(
                "a2-endpoint-alignment-mismatch",
                f"A2_LINK {link.id} endpoint nodes do not match polyline endpoints within "
                f"{tolerance_m:.3f} m",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )


def _repair_missing_a2_node_refs(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    tolerance_m = cfg.sanity.node_match_tolerance_m
    for link in dataset.store_for_attr("a2_link").features:
        if len(link.polyline) < 2:
            continue
        _repair_endpoint(
            dataset, link, "from_node_id", "FromNodeID", link.polyline[0], cfg, tolerance_m
        )
        _repair_endpoint(
            dataset, link, "to_node_id", "ToNodeID", link.polyline[-1], cfg, tolerance_m
        )


def _repair_endpoint(
    dataset: NGIIDataset,
    link: Any,
    attr_name: str,
    column_name: str,
    endpoint_xyz: np.ndarray,
    cfg: NGIIConfig,
    tolerance_m: float,
) -> None:
    node_store = dataset.store_for_attr("a1_node")
    current_id = getattr(link, attr_name)
    if current_id is not None and node_store.get(current_id) is not None:
        return
    nearby = [
        node for node in node_store.features if xy_distance(endpoint_xyz, node.point) <= tolerance_m
    ]
    before = {attr_name: current_id}
    if len(nearby) == 1:
        repaired_id = nearby[0].id
        if cfg.sanity.repairs.a2_missing_node_ref_nearest:
            setattr(link, attr_name, repaired_id)
            dataset.sanity.action(
                "a2-node-ref-nearest",
                f"A2_LINK {link.id} {column_name} repaired to nearby A1_NODE {repaired_id}",
                before=before,
                after={attr_name: repaired_id},
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        elif cfg.sanity.warnings.a2_endpoint_alignment:
            dataset.sanity.warn(
                "a2-node-ref-nearest-disabled",
                f"A2_LINK {link.id} {column_name} could be repaired to nearby "
                f"A1_NODE {repaired_id}, but nearest-node repair is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    if len(nearby) > 1 and cfg.sanity.warnings.a2_endpoint_alignment:
        dataset.sanity.warn(
            "a2-node-ref-ambiguous",
            f"A2_LINK {link.id} {column_name} has {len(nearby)} nearby A1_NODE candidates",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    if cfg.sanity.repairs.a2_missing_node_ref_remove:
        setattr(link, attr_name, None)
        dataset.sanity.action(
            "a2-node-ref-removed",
            f"A2_LINK {link.id} {column_name} could not be resolved and was removed",
            before=before,
            after={attr_name: None},
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )
    elif cfg.sanity.warnings.a2_endpoint_alignment:
        dataset.sanity.warn(
            "a2-node-ref-remove-disabled",
            f"A2_LINK {link.id} {column_name} could not be resolved, but relation "
            "removal is disabled",
            layer_name=link.layer_name,
            feature_id=link.id,
            source_path=link.source_path,
        )


def _repair_topology_direction(dataset: NGIIDataset, cfg: NGIIConfig) -> None:
    desired_flip = _desired_flips_from_leaf_flow(dataset)
    candidates: list[Any] = []
    for link in dataset.store_for_attr("a2_link").features:
        if not desired_flip.get(link.id, False):
            continue
        if _has_opposing_same_direction_neighbour(
            dataset, link, cfg.sanity.direction_parallel_dot_min
        ):
            candidates.append(link)

    if not candidates:
        return
    if len(candidates) > 1:
        if cfg.sanity.warnings.a2_topology_direction:
            dataset.sanity.warn(
                "a2-topology-direction-ambiguous",
                "topology flow and R/L same-direction evidence found multiple possible "
                f"backward A2 links: {', '.join(link.id for link in candidates[:8])}",
                layer_name="A2_LINK",
            )
        return
    link = candidates[0]
    if cfg.sanity.repairs.a2_topology_direction_swap:
        _swap_endpoint_ids(
            dataset,
            link,
            "a2-topology-direction-swapped",
            "was reversed by topology flow and R/L same-direction evidence",
            enabled=True,
            warn_disabled=False,
        )
    elif cfg.sanity.warnings.a2_topology_direction:
        dataset.sanity.warn(
            "a2-topology-direction-swap-disabled",
            f"A2_LINK {link.id} appears reversed by topology flow and R/L "
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
    for link in dataset.store_for_attr("a2_link").features:
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
    link_store = dataset.store_for_attr("a2_link")
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


def _unit_vector(link: Any) -> np.ndarray | None:
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
                f"A2_LINK {link.id} {reason}, but direction swap is disabled",
                layer_name=link.layer_name,
                feature_id=link.id,
                source_path=link.source_path,
            )
        return
    link.from_node_id, link.to_node_id = link.to_node_id, link.from_node_id
    dataset.sanity.action(
        code,
        f"A2_LINK {link.id} {reason}",
        before=before,
        after={"from_node_id": link.from_node_id, "to_node_id": link.to_node_id},
        layer_name=link.layer_name,
        feature_id=link.id,
        source_path=link.source_path,
    )
