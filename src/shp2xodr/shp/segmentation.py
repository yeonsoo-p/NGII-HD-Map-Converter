"""Group A2_LINK records into road bundles and junction components, and
bind B2_SURFACELINEMARK lines onto the resulting bundles.

Entry point: :meth:`Segmentation.from_shp_dir`. The pipeline lives as
private classmethods on :class:`Segmentation` itself - that's where the
result lives, so that's where the construction logic lives. All steps
consume typed :class:`A1Data` / :class:`A2Data` / :class:`B2Data` records
(no raw GeoDataFrame column access here).

A *bundle* is a maximal set of A2_LINK lanes that should become one OpenDRIVE
``<road>``. Same-direction lanes are joined laterally via ``R_LinkID`` /
``L_LinkID``; longitudinal continuations are joined through any A1_NODE that
is *not* a segmentation cut.

A *junction component* is a maximal connected set of JUNCTION-role A1
nodes reachable through ``LinkType=1`` interior links - including via
lateral adjacency, so two interior links that are R/L_LinkID neighbours
belong to one component even when no single link joins their endpoints.
Each component becomes one OpenDRIVE ``<junction>``; each bundle whose
rows are interior to it becomes a connecting road with
``road@junction = <jid>``.

Each B2_SURFACELINEMARK row resolves its R/L_LinkID references to the A2
bundle that lane belongs to. Bundle ids are dense and shared between the A2
and B2 sides, so a B2 line and the lanes it bounds are always comparable.

Bidirectional merging (opposite-direction pairs into one ``<road>``) is
intentionally deferred - each direction stays its own bundle here.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from shp2xodr.shp.data import A1Data, A2Data, B2Data

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

# NGII A2_LINK LinkType: 1 = 교차로내주행경로 (intersection interior path).
# Any other value is a mainline lane — even if it happens to span two junction
# nodes, it's a road between junctions, not interior to one.
_INTERIOR_LINK_TYPE = "1"


@dataclass(slots=True, frozen=True)
class Segmentation:
    """Bundles, junction topology, and B2 line bindings for one section.

    Build one with :meth:`from_shp_dir`. Arrays carry different lengths
    depending on what they index:

    * ``bundle_id`` / ``junction_id`` — parallel to the A2_LINK input row order.
    * ``node_junction_id`` — parallel to the A1_NODE input row order.
    * ``bundle_junction`` / ``bundle_pred_junctions`` / ``bundle_succ_junctions``
      — indexed by bundle id ``0..n_bundles-1``.
    * ``b2_*`` — parallel to the B2_SURFACELINEMARK input row order.

    A *mainline* bundle has ``bundle_junction[b] == -1`` and connects on each
    side to the set of junctions reachable through its boundary nodes (empty
    set = dangling). NGII data does allow laterally-merged lanes whose
    pred/succ endpoints sit in different junctions, so each side is modeled
    as a set rather than a single id. Interior bundles have empty pred/succ
    sets (their connectivity is implied by ``bundle_junction``).

    The ``b2_*`` arrays carry ``-1`` on a side when that B2 row has no A2
    reference in the loaded section. The two sides are kept independent on
    purpose: a centerline (중앙선, Kind 501) separates two opposite-direction
    bundles, so its R/L sides resolve to *different* bundle ids — we don't
    want to force them to merge. Regular lane dividers have
    ``b2_r_bundle == b2_l_bundle``; outer edge lines have one side ``-1``.
    """

    bundle_id: NDArray[np.int32]
    junction_id: NDArray[np.int32]
    node_junction_id: NDArray[np.int32]
    bundle_junction: NDArray[np.int32]
    bundle_pred_junctions: tuple[frozenset[int], ...]
    bundle_succ_junctions: tuple[frozenset[int], ...]
    b2_r_link_idx: NDArray[np.int32]
    b2_l_link_idx: NDArray[np.int32]
    b2_r_bundle: NDArray[np.int32]
    b2_l_bundle: NDArray[np.int32]

    # ---- Construction ---------------------------------------------------------

    @classmethod
    def from_shp_dir(
        cls,
        shp_dir: Path,
        junction_merge_dist_m: float = 0.0,
    ) -> Self:
        """Build a :class:`Segmentation` from one section's raw SHP layers.

        ``junction_merge_dist_m`` controls a final spatial pass that fuses
        graph-disjoint junction components whose nodes are closer than the
        threshold; set to ``0`` to disable.

        NGII makes B2_SURFACELINEMARK a mandatory layer for every section,
        so the B2 resolution always runs alongside the A2 pass; there is no
        A2-only mode.
        """
        a1 = A1Data(shp_dir)
        a2 = A2Data(shp_dir)
        b2 = B2Data(shp_dir)
        node_role = cls._classify_nodes(a1, a2)

        bundle_id = cls._bundle_links(a2, node_role)
        junction_id, nid_to_jid = cls._cluster_junctions(
            a1, a2, node_role, bundle_id, junction_merge_dist_m=junction_merge_dist_m
        )
        bundle_junction, bundle_pred, bundle_succ = cls._resolve_bundle_endpoints(
            a2, bundle_id, junction_id, nid_to_jid
        )
        assert len(bundle_pred) == len(bundle_succ) == len(bundle_junction)

        node_junction_id = cls._node_junction_array(a1, nid_to_jid)
        b2_r_link, b2_l_link, b2_r_bundle, b2_l_bundle = cls._resolve_b2_to_bundles(
            a2, b2, bundle_id
        )

        cls._log_summary(a2, bundle_id, node_junction_id, node_role, b2_r_link, b2_l_link, b2)
        return cls(
            bundle_id=bundle_id,
            junction_id=junction_id,
            node_junction_id=node_junction_id,
            bundle_junction=bundle_junction,
            bundle_pred_junctions=bundle_pred,
            bundle_succ_junctions=bundle_succ,
            b2_r_link_idx=b2_r_link,
            b2_l_link_idx=b2_l_link,
            b2_r_bundle=b2_r_bundle,
            b2_l_bundle=b2_l_bundle,
        )

    # ---- Pipeline steps -------------------------------------------------------

    @staticmethod
    def _classify_nodes(a1: A1Data, a2: A2Data) -> dict[str, NodeRole]:
        """Map every A1_NODE.ID to its segmentation role.

        NodeType=99 is split by node degree: a real diverge/merge (degree > 2)
        counts as a junction; a 1-to-1 feature break counts as a lane-section
        break.
        """
        fan_in: Counter[str] = Counter(n for n in a2.to_node_ids if n)
        fan_out: Counter[str] = Counter(n for n in a2.from_node_ids if n)

        roles: dict[str, NodeRole] = {}
        for nid, ntype in zip(a1.ids, a1.node_types, strict=True):
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

    @staticmethod
    def _bundle_links(a2: A2Data, node_role: dict[str, NodeRole]) -> NDArray[np.int32]:
        """Assign a dense bundle id to each A2_LINK row.

        Lateral unions: R/L_LinkID neighbours. Longitudinal unions: any
        shared A1 node that isn't a JUNCTION (which is the only segmentation
        cut — ROAD_BREAK passes through as a ``<bridge>``/``<tunnel>``,
        LANE_SECTION becomes a ``<laneSection>`` break within the same road).
        """
        link_ids = a2.ids
        n = len(link_ids)
        parent = np.arange(n, dtype=np.int32)

        def find(x: int) -> int:
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

        id_to_idx = {lid: i for i, lid in enumerate(link_ids)}

        r_ids = a2.r_link_ids
        l_ids = a2.l_link_ids
        for i in range(n):
            for nb_id in (r_ids[i], l_ids[i]):
                j = id_to_idx.get(nb_id)
                if j is not None:
                    union(i, j)

        from_nodes = a2.from_node_ids
        to_nodes = a2.to_node_ids
        by_from: dict[str, list[int]] = {}
        for i in range(n):
            by_from.setdefault(from_nodes[i], []).append(i)

        for i in range(n):
            tn = to_nodes[i]
            if node_role.get(tn, NodeRole.IGNORE) is NodeRole.JUNCTION:
                continue
            for j in by_from.get(tn, ()):
                union(i, j)

        bundle_ids = np.empty(n, dtype=np.int32)
        root_to_bid: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            bid = root_to_bid.get(root)
            if bid is None:
                bid = len(root_to_bid)
                root_to_bid[root] = bid
            bundle_ids[i] = bid
        return bundle_ids

    @staticmethod
    def _cluster_junctions(
        a1: A1Data,
        a2: A2Data,
        node_role: dict[str, NodeRole],
        bundle_id: NDArray[np.int32],
        junction_merge_dist_m: float = 0.0,
    ) -> tuple[NDArray[np.int32], dict[str, int]]:
        """Cluster JUNCTION-role A1 nodes into junction components.

        An interior link is one with ``LinkType=1`` (intersection interior path)
        and both endpoints junction-role. Junction nodes are unioned if they
        share an interior link, *or* if their interior links sit in the same
        bundle — laterally adjacent connecting lanes belong to one physical
        intersection even when no single link directly joins their endpoints.
        When ``junction_merge_dist_m > 0``, a final pass also unions any two
        components whose closest junction nodes lie within that planimetric
        distance — this catches channelized turns and free-flow paths that
        are physically inside one intersection but topologically disjoint
        from the rest of its connecting lanes.

        Mainline links that happen to span two adjacent junction nodes are
        *not* interior — they're a road between junctions. Returns
        ``(junction_id_per_link, nid_to_jid)``:

          * ``junction_id_per_link`` — shape ``(n_links,)``, parallel to A2 input;
            ``-1`` for non-interior rows.
          * ``nid_to_jid`` — A1 node id → junction id, only for JUNCTION-role
            nodes.

        Junction ids are dense ``0..k-1`` in first-occurrence order.
        """
        junction_nids = [nid for nid in a1.ids if node_role.get(nid) == NodeRole.JUNCTION]
        nid_to_idx = {nid: i for i, nid in enumerate(junction_nids)}
        m = len(nid_to_idx)
        parent = np.arange(m, dtype=np.int32)

        def find(x: int) -> int:
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

        from_ids = a2.from_node_ids
        to_ids = a2.to_node_ids
        link_types = a2.link_types
        n_links = len(a2.ids)
        interior = np.zeros(n_links, dtype=bool)
        bundle_junction_idxs: dict[int, list[int]] = {}
        for i in range(n_links):
            if link_types[i] != _INTERIOR_LINK_TYPE:
                continue
            fi = nid_to_idx.get(from_ids[i])
            ti = nid_to_idx.get(to_ids[i])
            if fi is None or ti is None:
                continue
            union(fi, ti)
            interior[i] = True
            bundle_junction_idxs.setdefault(int(bundle_id[i]), []).extend((fi, ti))
        for idxs in bundle_junction_idxs.values():
            first = idxs[0]
            for j in idxs[1:]:
                union(first, j)

        root_to_jid: dict[int, int] = {}
        idx_to_jid = np.empty(m, dtype=np.int32)
        for nid in junction_nids:
            idx = nid_to_idx[nid]
            root = find(idx)
            jid = root_to_jid.get(root)
            if jid is None:
                jid = len(root_to_jid)
                root_to_jid[root] = jid
            idx_to_jid[idx] = jid

        if junction_merge_dist_m > 0:
            idx_to_jid = Segmentation._merge_junctions_by_proximity(
                a1, junction_nids, nid_to_idx, idx_to_jid, junction_merge_dist_m
            )

        nid_to_jid = {nid: int(idx_to_jid[nid_to_idx[nid]]) for nid in junction_nids}
        junction_id_per_link = np.full(n_links, -1, dtype=np.int32)
        for i in range(n_links):
            if interior[i]:
                junction_id_per_link[i] = nid_to_jid[from_ids[i]]
        return junction_id_per_link, nid_to_jid

    @staticmethod
    def _merge_junctions_by_proximity(
        a1: A1Data,
        junction_nids: list[str],
        nid_to_idx: dict[str, int],
        idx_to_jid: NDArray[np.int32],
        max_distance_m: float,
    ) -> NDArray[np.int32]:
        """Re-densify ``idx_to_jid`` after fusing any two junction components
        whose closest nodes lie within ``max_distance_m`` (planimetric, XY only).

        Z is ignored — NGII puts bridges/underpasses under NodeType 3-6
        (ROAD_BREAK), so two distinct JUNCTION-role components stacked
        vertically would be unusual.
        """
        m = len(junction_nids)
        if m < 2:
            return idx_to_jid

        # XY lookup by A1_NODE id, from the typed A1Data.points (N, 3) array.
        a1_id_to_xy: dict[str, tuple[float, float]] = {
            nid: (float(a1.points[i, 0]), float(a1.points[i, 1])) for i, nid in enumerate(a1.ids)
        }
        coords = np.empty((m, 2), dtype=np.float64)
        for nid in junction_nids:
            coords[nid_to_idx[nid]] = a1_id_to_xy[nid]

        n_jids = int(idx_to_jid.max()) + 1
        jparent = np.arange(n_jids, dtype=np.int32)

        def jfind(x: int) -> int:
            root = x
            while jparent[root] != root:
                root = int(jparent[root])
            while jparent[x] != root:
                jparent[x], x = np.int32(root), int(jparent[x])
            return root

        def junion(a: int, b: int) -> None:
            ra, rb = jfind(a), jfind(b)
            if ra != rb:
                jparent[rb] = np.int32(ra)

        threshold_sq = max_distance_m * max_distance_m
        for i in range(m - 1):
            diff = coords[i + 1 :] - coords[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            ji = int(idx_to_jid[i])
            for off in np.flatnonzero(d2 <= threshold_sq):
                jj = int(idx_to_jid[i + 1 + int(off)])
                if ji != jj:
                    junion(ji, jj)

        old_to_new: dict[int, int] = {}
        merged = np.empty(m, dtype=np.int32)
        for i in range(m):
            root = jfind(int(idx_to_jid[i]))
            new = old_to_new.get(root)
            if new is None:
                new = len(old_to_new)
                old_to_new[root] = new
            merged[i] = new
        return merged

    @staticmethod
    def _resolve_bundle_endpoints(
        a2: A2Data,
        bundle_id: NDArray[np.int32],
        junction_id: NDArray[np.int32],
        nid_to_jid: dict[str, int],
    ) -> tuple[
        NDArray[np.int32],
        tuple[frozenset[int], ...],
        tuple[frozenset[int], ...],
    ]:
        """Per-bundle: which junction it is interior to (or ``-1`` for mainline),
        plus the set of junctions on its predecessor and successor sides.

        A bundle must be fully interior or fully mainline — this is asserted.
        Mainline bundles whose pred/succ endpoints span multiple junction
        components are accepted; that's a real NGII pattern (e.g. two lanes
        laterally merged but entering from different intersection nodes), and we
        record the multi-junction set rather than collapsing it.
        """
        n_bundles = int(bundle_id.max()) + 1 if len(bundle_id) else 0
        bundle_junction = np.full(n_bundles, -1, dtype=np.int32)
        pred_sets: list[frozenset[int]] = [frozenset()] * n_bundles
        succ_sets: list[frozenset[int]] = [frozenset()] * n_bundles

        rows_by_bundle: dict[int, list[int]] = {}
        for i, b in enumerate(bundle_id):
            rows_by_bundle.setdefault(int(b), []).append(i)

        from_ids = a2.from_node_ids
        to_ids = a2.to_node_ids

        for b, rows in rows_by_bundle.items():
            row_jids = {int(junction_id[i]) for i in rows}
            if -1 not in row_jids:
                assert len(row_jids) == 1, (
                    f"bundle {b} is interior but spans junction ids {row_jids}"
                )
                bundle_junction[b] = row_jids.pop()
                continue
            assert row_jids == {-1}, f"bundle {b} mixes mainline and interior rows: {row_jids}"
            froms = {from_ids[i] for i in rows}
            tos = {to_ids[i] for i in rows}
            pred_sets[b] = frozenset(nid_to_jid[nid] for nid in froms - tos if nid in nid_to_jid)
            succ_sets[b] = frozenset(nid_to_jid[nid] for nid in tos - froms if nid in nid_to_jid)
        return bundle_junction, tuple(pred_sets), tuple(succ_sets)

    @staticmethod
    def _node_junction_array(a1: A1Data, nid_to_jid: dict[str, int]) -> NDArray[np.int32]:
        """Build ``(n_a1,)`` int array of junction ids, ``-1`` for non-junction rows."""
        n = len(a1.ids)
        node_junction_id = np.full(n, -1, dtype=np.int32)
        for i, nid in enumerate(a1.ids):
            jid = nid_to_jid.get(nid)
            if jid is not None:
                node_junction_id[i] = jid
        return node_junction_id

    @staticmethod
    def _resolve_b2_to_bundles(
        a2: A2Data,
        b2: B2Data,
        bundle_id: NDArray[np.int32],
    ) -> tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ]:
        """Resolve each B2 row's R/L_LinkID into A2 row indices and bundle ids.

        Empty / unknown A2 references yield ``-1`` on that side. Pure look-up:
        bundle topology is already fixed by the A2 pass.
        """
        a2_id_to_idx = {lid: i for i, lid in enumerate(a2.ids)}
        r_ids = b2.r_link_ids
        l_ids = b2.l_link_ids
        n = len(b2.ids)
        r_link = np.full(n, -1, dtype=np.int32)
        l_link = np.full(n, -1, dtype=np.int32)
        for i in range(n):
            ri = a2_id_to_idx.get(r_ids[i])
            li = a2_id_to_idx.get(l_ids[i])
            if ri is not None:
                r_link[i] = np.int32(ri)
            if li is not None:
                l_link[i] = np.int32(li)
        r_bundle = np.where(r_link >= 0, bundle_id[r_link], -1).astype(np.int32)
        l_bundle = np.where(l_link >= 0, bundle_id[l_link], -1).astype(np.int32)
        return r_link, l_link, r_bundle, l_bundle

    @staticmethod
    def _log_summary(
        a2: A2Data,
        bundle_id: NDArray[np.int32],
        node_junction_id: NDArray[np.int32],
        node_role: dict[str, NodeRole],
        b2_r_link: NDArray[np.int32],
        b2_l_link: NDArray[np.int32],
        b2: B2Data,
    ) -> None:
        n_a2 = len(a2.ids)
        n_b2 = len(b2.ids)
        n_bundles = int(bundle_id.max()) + 1 if len(bundle_id) else 0
        n_junctions = int(node_junction_id.max()) + 1 if len(node_junction_id) else 0
        n_cuts = sum(1 for r in node_role.values() if r is NodeRole.JUNCTION)
        n_b2_bound = int(((b2_r_link >= 0) | (b2_l_link >= 0)).sum())
        log.info(
            "segmentation: %d links → %d bundles, %d junctions (cut at %d / %d nodes); "
            "%d / %d B2 lines bound",
            n_a2,
            n_bundles,
            n_junctions,
            n_cuts,
            len(node_role),
            n_b2_bound,
            n_b2,
        )
