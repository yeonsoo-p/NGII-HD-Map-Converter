"""Group A2_LINK records into road bundles per the NGII → OpenDRIVE rules.

A bundle is a maximal set of A2_LINK lanes that should become one OpenDRIVE
``<road>`` element. Same-direction lanes are joined laterally via
``R_LinkID``/``L_LinkID``; longitudinal continuations are joined through any
A1_NODE that is *not* a segmentation cut.

Bidirectional merging (opposite-direction pairs into one ``<road>``) is
intentionally deferred — each direction stays its own bundle here.
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


class NodeRole(Enum):
    """Why an A1_NODE exists, in terms that drive segmentation."""

    JUNCTION = "junction"
    ROAD_BREAK = "road_break"
    LANE_SECTION = "lane_section"
    IGNORE = "ignore"


# NGII A1_NODE NodeType code-list → segmentation role.
# 1 평면교차로, 2 입체교차로, 8 톨게이트, 9 요금소, 10 회전교차로
_JUNCTION_NODE_TYPES = frozenset({"1", "2", "8", "9", "10"})
# 3 터널, 4 교량, 5 지하차도, 6 고가차도 시·종점
_ROAD_BREAK_NODE_TYPES = frozenset({"3", "4", "5", "6"})
# 7 도로 차로 수 변화 — kept within the same road as a future laneSection break
_LANE_SECTION_NODE_TYPES = frozenset({"7"})
# 99 기타 — resolved by node degree (fan-out > 1 ⇒ JUNCTION, else LANE_SECTION)
_AMBIGUOUS_NODE_TYPE = "99"


def classify_nodes(a1: gpd.GeoDataFrame, a2: gpd.GeoDataFrame) -> dict[str, NodeRole]:
    """Map every A1_NODE.ID to its segmentation role.

    NodeType=99 is split by node degree: a real diverge/merge (degree > 2)
    counts as a junction; a 1-to-1 feature break counts as a lane-section
    break.
    """
    fan_in: Counter[str] = Counter(a2["ToNodeID"].dropna().astype(str))
    fan_out: Counter[str] = Counter(a2["FromNodeID"].dropna().astype(str))

    roles: dict[str, NodeRole] = {}
    for nid, ntype in zip(a1["ID"].astype(str), a1["NodeType"].astype(str), strict=True):
        if ntype in _JUNCTION_NODE_TYPES:
            roles[nid] = NodeRole.JUNCTION
        elif ntype in _ROAD_BREAK_NODE_TYPES:
            roles[nid] = NodeRole.ROAD_BREAK
        elif ntype in _LANE_SECTION_NODE_TYPES:
            roles[nid] = NodeRole.LANE_SECTION
        elif ntype == _AMBIGUOUS_NODE_TYPE:
            if fan_in.get(nid, 0) > 1 or fan_out.get(nid, 0) > 1:
                roles[nid] = NodeRole.JUNCTION
            else:
                roles[nid] = NodeRole.LANE_SECTION
        else:
            roles[nid] = NodeRole.IGNORE
    return roles


def _is_cut(role: NodeRole) -> bool:
    return role in (NodeRole.JUNCTION, NodeRole.ROAD_BREAK)


def bundle_links(a1: gpd.GeoDataFrame, a2: gpd.GeoDataFrame) -> NDArray[np.int32]:
    """Assign a bundle id to each A2_LINK row, aligned with the input order.

    Two A2_LINKs share a bundle iff they are reachable via:
        * lateral edges  — ``a.R_LinkID == b.ID`` or ``a.L_LinkID == b.ID``,
        * longitudinal edges — ``a.ToNodeID == b.FromNodeID`` and that shared
          node is not a JUNCTION / ROAD_BREAK cut.

    Bundle ids are dense (``0..k-1``) in first-occurrence order.
    """
    node_role = classify_nodes(a1, a2)

    n = len(a2)
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        # Iterative with path compression.
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = np.int32(root), int(parent[x])
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = np.int32(ra)

    link_ids = a2["ID"].astype(str).tolist()
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}

    # Lateral merges via R/L_LinkID.
    r_ids = a2["R_LinkID"].fillna("").astype(str).tolist()
    l_ids = a2["L_LinkID"].fillna("").astype(str).tolist()
    for i in range(n):
        for nb_id in (r_ids[i], l_ids[i]):
            j = id_to_idx.get(nb_id)
            if j is not None:
                union(i, j)

    # Longitudinal merges through non-cut nodes.
    from_nodes = a2["FromNodeID"].astype(str).tolist()
    to_nodes = a2["ToNodeID"].astype(str).tolist()
    by_from: dict[str, list[int]] = {}
    for i, fn in enumerate(from_nodes):
        by_from.setdefault(fn, []).append(i)

    for i in range(n):
        tn = to_nodes[i]
        role = node_role.get(tn, NodeRole.IGNORE)
        if _is_cut(role):
            continue
        for j in by_from.get(tn, ()):
            union(i, j)

    # Densify roots into 0..k-1 in first-occurrence order.
    bundle_ids = np.empty(n, dtype=np.int32)
    root_to_bid: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        bid = root_to_bid.get(root)
        if bid is None:
            bid = len(root_to_bid)
            root_to_bid[root] = bid
        bundle_ids[i] = bid

    n_bundles = len(root_to_bid)
    n_cuts = sum(1 for r in node_role.values() if _is_cut(r))
    log.info(
        "segmentation: %d links → %d bundles (cut at %d / %d nodes)",
        n,
        n_bundles,
        n_cuts,
        len(node_role),
    )
    return bundle_ids
