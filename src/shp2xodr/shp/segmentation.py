"""Group A2_LINK records into lane-bundles and cluster A1_NODE records into
junction components.

Entry point: :meth:`Segmentation.from_shp_dir`. The pipeline lives as
private classmethods on :class:`Segmentation` itself - that's where the
result lives, so that's where the construction logic lives. All steps
consume typed :class:`A1Data` / :class:`A2Data` records (no raw
GeoDataFrame column access here).

A *group* is a maximal set of A2_LINK lanes that share the same lane-
bundle. Same-direction lanes are joined laterally via ``R_LinkID`` /
``L_LinkID``; longitudinal continuations are joined through any A1_NODE
that is *not* a segmentation cut.

A *junction component* is a maximal connected set of JUNCTION-role A1
nodes reachable through ``LinkType=1`` interior links - including via
lateral adjacency, so two interior links that are R/L_LinkID neighbours
belong to one component even when no single link joins their endpoints.

The result is a flat set of per-A2-link, per-A1-node, and per-group
arrays. Mainline groups are assigned dense "road" ids via
``road_id_per_link`` for fast viz coloring; no object graph is built
here.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import shapely
from numpy.typing import NDArray

from shp2xodr.shp.data import A1Data, A2Data

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

# NGII A2_LINK LinkType: 6 = 일반주행차로 (general driving lane). The only
# mainline LinkType the within-junction promotion rule fires for — bus
# (4), variable (5), toll (2/3), rest-area (7-12) and ramp (13/14) lanes
# carry domain semantics that would be silently erased by promoting them
# to junction-interior. Manual §9.4.2 documents the case explicitly under
# "교차로 내 예외(일반주행차로)": a Type=6 lane traversing a 평면교차로.
_PROMOTABLE_LINK_TYPE = "6"


@dataclass(slots=True, frozen=True)
class SegmentationConfig:
    """Hydra-managed tuning knobs for the segmentation pipeline.

    Values live in ``conf/config.yaml`` under ``segmentation:``; the entry
    point builds an instance and hands it to :meth:`Segmentation.from_shp_dir`.
    Tests construct directly.
    """

    junction_merge_dist_m: float
    junction_crossing_z_tol_m: float


@dataclass(slots=True, frozen=True)
class Segmentation:
    """Per-row and per-group arrays for one section.

    Build one with :meth:`from_shp_dir`. Arrays carry different lengths
    depending on what they index:

    * ``group_id`` / ``junction_id`` / ``road_id_per_link`` — parallel to
      the A2_LINK input row order.
    * ``node_junction_id`` — parallel to the A1_NODE input row order.
    * ``group_junction`` / ``group_pred_junctions`` / ``group_succ_junctions``
      — indexed by group id ``0..n_groups-1``.

    A *mainline* group has ``group_junction[b] == -1`` and connects on each
    side to the set of junctions reachable through its boundary nodes (empty
    set = dangling). NGII data does allow laterally-merged lanes whose
    pred/succ endpoints sit in different junctions, so each side is modeled
    as a set rather than a single id. Interior groups have empty pred/succ
    sets (their connectivity is implied by ``group_junction``).

    ``road_id_per_link`` carries a dense mainline-only id ``0..n_main-1`` on
    each mainline row and ``-1`` on interior rows — a flat projection of
    ``group_id`` for the viz's road-palette layer. There is no object graph;
    consumers that need typed entities build them from these arrays.
    """

    group_id: NDArray[np.int32]
    junction_id: NDArray[np.int32]
    node_junction_id: NDArray[np.int32]
    group_junction: NDArray[np.int32]
    group_pred_junctions: tuple[frozenset[int], ...]
    group_succ_junctions: tuple[frozenset[int], ...]
    road_id_per_link: NDArray[np.int32]

    # ---- Disjoint-set helpers -----------------------------------------------
    # Stateless: each caller owns its own ``parent = np.arange(n, np.int32)``.
    # Path-compressing find + arbitrary-root union. Used by every pass that
    # builds connected components (link grouping, junction clustering,
    # proximity merge, crossing-link union).

    @staticmethod
    def _uf_find(parent: NDArray[np.int32], x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = np.int32(root), int(parent[x])
        return root

    @staticmethod
    def _uf_union(parent: NDArray[np.int32], a: int, b: int) -> None:
        ra = Segmentation._uf_find(parent, a)
        rb = Segmentation._uf_find(parent, b)
        if ra != rb:
            parent[rb] = np.int32(ra)

    # ---- Construction ---------------------------------------------------------

    @classmethod
    def from_shp_dir(cls, shp_dir: Path, cfg: SegmentationConfig) -> Self:
        """Build a :class:`Segmentation` from one section's A1/A2 layers.

        ``cfg.junction_merge_dist_m`` controls a final spatial pass that
        fuses graph-disjoint junction components whose nodes are closer
        than the threshold; set to ``0`` to disable.

        ``cfg.junction_crossing_z_tol_m`` controls the geometric-crossing
        union: two ``LinkType=1`` interior polylines that cross in plan
        with a |Δz| at the crossing point within this tolerance are
        treated as belonging to one physical intersection. Different floors
        of an overpass keep their distinct junctions because the z gap at
        the planar crossing exceeds the tolerance.
        """
        a1 = A1Data(shp_dir)
        a2 = A2Data(shp_dir)
        node_role = cls._classify_nodes(a1, a2)

        group_id = cls._group_links(a2, node_role)
        junction_id, nid_to_jid = cls._cluster_junctions(
            a1,
            a2,
            node_role,
            group_id,
            junction_merge_dist_m=cfg.junction_merge_dist_m,
            junction_crossing_z_tol_m=cfg.junction_crossing_z_tol_m,
        )
        group_junction, group_pred, group_succ = cls._resolve_group_endpoints(
            a2, group_id, junction_id, nid_to_jid
        )
        junction_id, group_junction, group_pred, group_succ = (
            cls._promote_within_junction_mainline_groups(
                a2, group_id, junction_id, group_junction, group_pred, group_succ
            )
        )

        node_junction_id = cls._node_junction_array(a1, nid_to_jid)

        # Dense mainline-only "road" id per link: each mainline group gets
        # 0..n_main-1 in ascending group-id order; interior rows stay at -1.
        n_groups = len(group_junction)
        group_to_road = np.full(n_groups, -1, dtype=np.int32)
        mainline_groups = np.flatnonzero(group_junction == -1).astype(np.int32)
        group_to_road[mainline_groups] = np.arange(len(mainline_groups), dtype=np.int32)
        road_id_per_link = np.where(
            group_junction[group_id] == -1, group_to_road[group_id], -1
        ).astype(np.int32)

        cls._log_summary(a2, group_id, road_id_per_link, node_junction_id, node_role)
        return cls(
            group_id=group_id,
            junction_id=junction_id,
            node_junction_id=node_junction_id,
            group_junction=group_junction,
            group_pred_junctions=group_pred,
            group_succ_junctions=group_succ,
            road_id_per_link=road_id_per_link,
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
    def _group_links(a2: A2Data, node_role: dict[str, NodeRole]) -> NDArray[np.int32]:
        """Assign a dense group id to each A2_LINK row.

        Lateral unions: R/L_LinkID neighbours, but *only* when both sides
        share the interior-vs-mainline status. NGII sometimes lists a
        LinkType=1 (intersection-interior path) and an adjacent LinkType=6
        (ordinary lane through the same intersection footprint) as R/L
        neighbours; the segmentation invariant downstream requires a group
        to be fully one or the other, so we must not fuse them here.

        Longitudinal unions: any shared A1 node that isn't a JUNCTION
        (which is the only segmentation cut — ROAD_BREAK passes through as
        a ``<bridge>``/``<tunnel>``, LANE_SECTION becomes a
        ``<laneSection>`` break within the same road). The same
        interior-vs-mainline gate applies — NGII occasionally places a
        LANE_SECTION node at the lane-stripe boundary between an
        intersection-interior path (1) and the mainline lane (6) feeding
        into it, and we must not let that shared node fuse them.
        """
        link_ids = a2.ids
        n = len(link_ids)
        parent = np.arange(n, dtype=np.int32)

        id_to_idx = {lid: i for i, lid in enumerate(link_ids)}

        r_ids = a2.r_link_ids
        l_ids = a2.l_link_ids
        link_types = a2.link_types
        for i in range(n):
            i_interior = link_types[i] == _INTERIOR_LINK_TYPE
            for nb_id in (r_ids[i], l_ids[i]):
                j = id_to_idx.get(nb_id)
                if j is None:
                    continue
                if (link_types[j] == _INTERIOR_LINK_TYPE) != i_interior:
                    continue
                Segmentation._uf_union(parent, i, j)

        from_nodes = a2.from_node_ids
        to_nodes = a2.to_node_ids
        by_from: dict[str, list[int]] = {}
        for i in range(n):
            by_from.setdefault(from_nodes[i], []).append(i)

        for i in range(n):
            tn = to_nodes[i]
            if node_role.get(tn, NodeRole.IGNORE) is NodeRole.JUNCTION:
                continue
            i_interior = link_types[i] == _INTERIOR_LINK_TYPE
            for j in by_from.get(tn, ()):
                if (link_types[j] == _INTERIOR_LINK_TYPE) != i_interior:
                    continue
                Segmentation._uf_union(parent, i, j)

        group_ids = np.empty(n, dtype=np.int32)
        root_to_bid: dict[int, int] = {}
        for i in range(n):
            root = Segmentation._uf_find(parent, i)
            bid = root_to_bid.get(root)
            if bid is None:
                bid = len(root_to_bid)
                root_to_bid[root] = bid
            group_ids[i] = bid
        return group_ids

    @staticmethod
    def _cluster_junctions(
        a1: A1Data,
        a2: A2Data,
        node_role: dict[str, NodeRole],
        group_id: NDArray[np.int32],
        junction_merge_dist_m: float = 0.0,
        junction_crossing_z_tol_m: float = 0.0,
    ) -> tuple[NDArray[np.int32], dict[str, int]]:
        """Cluster JUNCTION-role A1 nodes into junction components.

        An interior link is any A2 row tagged ``LinkType=1`` (intersection
        interior path, per NGII spec table 9.18). The spec guarantees both
        endpoints reference junction-class A1 nodes, but real NGII exports
        occasionally ship rows whose ``ToNodeID`` / ``FromNodeID`` refers to
        an A1 row that doesn't exist (residue of the deprecated dummy-node
        mechanism). We still trust the ``LinkType=1`` tag and attach the
        row to whichever junction component its known endpoint belongs to.

        Junction nodes are unioned if they share an interior link, *or* if
        their interior links sit in the same group — laterally adjacent
        connecting lanes belong to one physical intersection even when no
        single link directly joins their endpoints. When
        ``junction_crossing_z_tol_m > 0``, a geometric pass also unions any
        two interior links that cross in plan whose z-values at the
        crossing point differ by at most this tolerance — catching turn
        paths that physically intersect inside the same intersection but
        share no A1 endpoint or group, while keeping different floors of an
        overpass apart by the z gap at the crossing. When
        ``junction_merge_dist_m > 0``, a final pass also unions any two
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

        interior, group_junction_idxs = Segmentation._mark_interior_links(
            a2, nid_to_idx, group_id, parent
        )
        for idxs in group_junction_idxs.values():
            first = idxs[0]
            for j in idxs[1:]:
                Segmentation._uf_union(parent, first, j)

        if junction_crossing_z_tol_m > 0:
            Segmentation._union_crossing_interior_links(
                a2, interior, nid_to_idx, parent, junction_crossing_z_tol_m
            )

        root_to_jid: dict[int, int] = {}
        idx_to_jid = np.empty(m, dtype=np.int32)
        for nid in junction_nids:
            idx = nid_to_idx[nid]
            root = Segmentation._uf_find(parent, idx)
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
        junction_id_per_link = Segmentation._assign_link_junction_ids(
            a2, interior, nid_to_jid, idx_to_jid, group_id, group_junction_idxs
        )
        return junction_id_per_link, nid_to_jid

    @staticmethod
    def _mark_interior_links(
        a2: A2Data,
        nid_to_idx: dict[str, int],
        group_id: NDArray[np.int32],
        parent: NDArray[np.int32],
    ) -> tuple[NDArray[np.bool_], dict[int, list[int]]]:
        """Tag every ``LinkType=1`` row as interior and union its junction
        endpoints.

        Rows whose ``FromNodeID`` / ``ToNodeID`` reference an A1 row that
        doesn't exist (data-quality bug in NGII exports) still count as
        interior — the ``LinkType=1`` tag is dispositive per spec — and
        get attached at the assignment step via either their surviving
        endpoint or a junction-bearing peer in their group.
        """
        from_ids = a2.from_node_ids
        to_ids = a2.to_node_ids
        link_types = a2.link_types
        n_links = len(a2.ids)
        interior = np.zeros(n_links, dtype=bool)
        group_junction_idxs: dict[int, list[int]] = {}
        n_orphan_endpoint = 0
        n_orphan_both = 0
        for i in range(n_links):
            if link_types[i] != _INTERIOR_LINK_TYPE:
                continue
            interior[i] = True
            fi = nid_to_idx.get(from_ids[i])
            ti = nid_to_idx.get(to_ids[i])
            known = tuple(int(x) for x in (fi, ti) if x is not None)
            if len(known) == 2:
                Segmentation._uf_union(parent, known[0], known[1])
            elif len(known) == 1:
                n_orphan_endpoint += 1
            else:
                n_orphan_both += 1
                log.warning(
                    "A2 row %s: LinkType=1 has no JUNCTION-role endpoint "
                    "(from=%s, to=%s); will try to inherit junction from "
                    "group %d peers, else demote to mainline",
                    a2.ids[i],
                    from_ids[i],
                    to_ids[i],
                    int(group_id[i]),
                )
            if known:
                group_junction_idxs.setdefault(int(group_id[i]), []).extend(known)
        if n_orphan_endpoint or n_orphan_both:
            log.info(
                "_cluster_junctions: %d interior rows with one missing endpoint, "
                "%d with both missing (data-quality bugs in A2 FK references)",
                n_orphan_endpoint,
                n_orphan_both,
            )
        return interior, group_junction_idxs

    @staticmethod
    def _assign_link_junction_ids(
        a2: A2Data,
        interior: NDArray[np.bool_],
        nid_to_jid: dict[str, int],
        idx_to_jid: NDArray[np.int32],
        group_id: NDArray[np.int32],
        group_junction_idxs: dict[int, list[int]],
    ) -> NDArray[np.int32]:
        """Per-row junction-id for interior rows. Picks from whichever
        endpoint is JUNCTION-role; falls back to a junction-bearing peer
        in the same group when neither endpoint is. If neither path
        resolves (truly orphaned LinkType=1 stub), demotes the row to
        mainline (``junction_id = -1``) so the rest of the section can
        still load — the operator already saw the WARNING upstream and
        can repair the SHP."""
        from_ids = a2.from_node_ids
        to_ids = a2.to_node_ids
        n_links = len(a2.ids)
        junction_id_per_link = np.full(n_links, -1, dtype=np.int32)
        for i in range(n_links):
            if not interior[i]:
                continue
            if from_ids[i] in nid_to_jid:
                junction_id_per_link[i] = nid_to_jid[from_ids[i]]
                continue
            if to_ids[i] in nid_to_jid:
                junction_id_per_link[i] = nid_to_jid[to_ids[i]]
                continue
            peers = group_junction_idxs.get(int(group_id[i]), [])
            if peers:
                junction_id_per_link[i] = int(idx_to_jid[peers[0]])
                continue
            log.warning(
                "A2 row %s: LinkType=1 has no JUNCTION-role endpoint and no "
                "junction-bearing peer in group %d; demoting to mainline. "
                "Geometry preserved as a single-link road. Repair the SHP "
                "(FromNodeID / ToNodeID must reference existing A1 rows) to "
                "restore the interior classification.",
                a2.ids[i],
                int(group_id[i]),
            )
        return junction_id_per_link

    @staticmethod
    def _union_crossing_interior_links(
        a2: A2Data,
        interior: NDArray[np.bool_],
        nid_to_idx: dict[str, int],
        parent: NDArray[np.int32],
        z_tol_m: float,
    ) -> None:
        """Union junction-node components whose interior polylines geometrically
        cross at the same elevation.

        Two ``LinkType=1`` connecting lanes that cross in plan are part of
        one physical intersection — typical example: a right-turn slip and
        a left-turn path inside one large intersection that share no A1
        endpoint and aren't laterally adjacent. The z-tolerance keeps
        stacked but topologically distinct overpasses apart: NGII puts the
        bridge/underpass nodes themselves under NodeType 3-6 (ROAD_BREAK),
        so genuine same-grade crossings sit at nearly identical z (sub-
        decimeter), while overpasses clear ~4.5 m vertically.

        Mutates ``parent`` in place; no return.
        """
        interior_rows: NDArray[np.intp] = np.flatnonzero(interior)
        if len(interior_rows) < 2:
            return

        lines: list[shapely.LineString] = []
        anchors: list[int] = []  # parallel to lines: a junction-node idx for each row
        keep_rows: list[int] = []
        for row_i in interior_rows:
            anchor = Segmentation._row_anchor_junction_idx(int(row_i), a2, nid_to_idx)
            if anchor is None:
                # Truly orphaned LinkType=1 — no JUNCTION endpoint we can
                # union toward. Skip; _assign_link_junction_ids will still
                # try to inherit from a group peer.
                continue
            poly = a2.polylines[int(row_i)]
            if len(poly) < 2:
                continue
            lines.append(shapely.LineString(poly[:, :2]))
            anchors.append(anchor)
            keep_rows.append(int(row_i))

        if len(lines) < 2:
            return

        tree = shapely.STRtree(lines)
        n_unioned = 0
        for a_idx in range(len(lines)):
            anchor_a = anchors[a_idx]
            for b_raw in tree.query(lines[a_idx], predicate="crosses"):
                b_idx = int(b_raw)
                if b_idx <= a_idx:
                    continue
                anchor_b = anchors[b_idx]
                if Segmentation._uf_find(parent, anchor_a) == Segmentation._uf_find(
                    parent, anchor_b
                ):
                    continue
                intersection = lines[a_idx].intersection(lines[b_idx])
                if not Segmentation._crossing_within_z_tol(
                    intersection,
                    a2.polylines[keep_rows[a_idx]],
                    a2.polylines[keep_rows[b_idx]],
                    z_tol_m,
                ):
                    continue
                Segmentation._uf_union(parent, anchor_a, anchor_b)
                n_unioned += 1
        if n_unioned:
            log.info(
                "_union_crossing_interior_links: unioned %d junction component pair(s) "
                "via geometric crossings within z-tolerance %.2f m",
                n_unioned,
                z_tol_m,
            )

    @staticmethod
    def _row_anchor_junction_idx(row_i: int, a2: A2Data, nid_to_idx: dict[str, int]) -> int | None:
        """Return a junction-node index for the given A2 row, or ``None``
        when neither endpoint is a JUNCTION-role node (orphaned FK)."""
        fi = nid_to_idx.get(a2.from_node_ids[row_i])
        if fi is not None:
            return int(fi)
        ti = nid_to_idx.get(a2.to_node_ids[row_i])
        if ti is not None:
            return int(ti)
        return None

    @staticmethod
    def _crossing_within_z_tol(
        intersection: shapely.geometry.base.BaseGeometry,
        poly_a: NDArray[np.float64],
        poly_b: NDArray[np.float64],
        z_tol_m: float,
    ) -> bool:
        """True if at least one crossing point has |Δz| ≤ ``z_tol_m`` between
        the two XYZ polylines. ``intersection`` is the shapely intersection
        of the two XY LineStrings — typically a Point or MultiPoint when the
        ``crosses`` predicate matched.
        """
        points: list[shapely.Point] = []
        if isinstance(intersection, shapely.Point):
            points.append(intersection)
        elif isinstance(intersection, shapely.MultiPoint):
            points.extend(intersection.geoms)
        else:
            # crosses can also yield LineString (collinear overlap). Sample
            # the midpoint — same-grade overlaps satisfy the tolerance, and
            # an overpass that overlaps in plan is so unusual that the
            # midpoint test is good enough.
            if intersection.is_empty:
                return False
            mid = intersection.interpolate(0.5, normalized=True)
            if isinstance(mid, shapely.Point):
                points.append(mid)
        for pt in points:
            z_a = Segmentation._z_at_xy(poly_a, pt.x, pt.y)
            z_b = Segmentation._z_at_xy(poly_b, pt.x, pt.y)
            if abs(z_a - z_b) <= z_tol_m:
                return True
        return False

    @staticmethod
    def _z_at_xy(poly_xyz: NDArray[np.float64], x: float, y: float) -> float:
        """Linear-interp z at the polyline's nearest point to ``(x, y)``.

        Uses planimetric arc-length to find the containing segment and
        interpolates z within it. Falls back to the nearest vertex for
        degenerate zero-length segments.
        """
        xy = poly_xyz[:, :2]
        diffs = np.diff(xy, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        line = shapely.LineString(xy)
        s = float(line.project(shapely.Point(x, y)))
        idx = int(np.searchsorted(cum, s) - 1)
        idx = max(0, min(idx, len(poly_xyz) - 2))
        seg_len = float(seg_lens[idx])
        if seg_len <= 0.0:
            return float(poly_xyz[idx, 2])
        t = (s - cum[idx]) / seg_len
        t = max(0.0, min(1.0, t))
        return float((1.0 - t) * poly_xyz[idx, 2] + t * poly_xyz[idx + 1, 2])

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

        # STRtree on junction-node points, then dwithin-query per node. Same
        # idiom as the bidirectional Pass B on B2 centerlines.
        points = [shapely.Point(float(coords[i, 0]), float(coords[i, 1])) for i in range(m)]
        tree = shapely.STRtree(points)
        for i in range(m):
            ji = int(idx_to_jid[i])
            for j in tree.query(points[i], predicate="dwithin", distance=max_distance_m):
                if int(j) <= i:
                    continue
                jj = int(idx_to_jid[int(j)])
                if ji != jj:
                    Segmentation._uf_union(jparent, ji, jj)

        old_to_new: dict[int, int] = {}
        merged = np.empty(m, dtype=np.int32)
        for i in range(m):
            root = Segmentation._uf_find(jparent, int(idx_to_jid[i]))
            new = old_to_new.get(root)
            if new is None:
                new = len(old_to_new)
                old_to_new[root] = new
            merged[i] = new
        return merged

    @staticmethod
    def _resolve_group_endpoints(
        a2: A2Data,
        group_id: NDArray[np.int32],
        junction_id: NDArray[np.int32],
        nid_to_jid: dict[str, int],
    ) -> tuple[
        NDArray[np.int32],
        tuple[frozenset[int], ...],
        tuple[frozenset[int], ...],
    ]:
        """Per-group: which junction it is interior to (or ``-1`` for mainline),
        plus the set of junctions on its predecessor and successor sides.

        A group must be fully interior or fully mainline — this is asserted.
        Mainline groups whose pred/succ endpoints span multiple junction
        components are accepted; that's a real NGII pattern (e.g. two lanes
        laterally merged but entering from different intersection nodes), and we
        record the multi-junction set rather than collapsing it.
        """
        n_groups = int(group_id.max()) + 1 if len(group_id) else 0
        group_junction = np.full(n_groups, -1, dtype=np.int32)
        pred_sets: list[frozenset[int]] = [frozenset()] * n_groups
        succ_sets: list[frozenset[int]] = [frozenset()] * n_groups

        rows_by_group: dict[int, list[int]] = {}
        for i, b in enumerate(group_id):
            rows_by_group.setdefault(int(b), []).append(i)

        from_ids = a2.from_node_ids
        to_ids = a2.to_node_ids

        for b, rows in rows_by_group.items():
            row_jids = {int(junction_id[i]) for i in rows}
            if -1 not in row_jids:
                if len(row_jids) != 1:
                    msg = f"group {b} is interior but spans junction ids {row_jids}"
                    raise ValueError(msg)
                group_junction[b] = row_jids.pop()
                continue
            if row_jids != {-1}:
                msg = f"group {b} mixes mainline and interior rows: {row_jids}"
                raise ValueError(msg)
            froms = {from_ids[i] for i in rows}
            tos = {to_ids[i] for i in rows}
            pred_sets[b] = frozenset(nid_to_jid[nid] for nid in froms - tos if nid in nid_to_jid)
            succ_sets[b] = frozenset(nid_to_jid[nid] for nid in tos - froms if nid in nid_to_jid)
        return group_junction, tuple(pred_sets), tuple(succ_sets)

    @staticmethod
    def _promote_within_junction_mainline_groups(
        a2: A2Data,
        group_id: NDArray[np.int32],
        junction_id: NDArray[np.int32],
        group_junction: NDArray[np.int32],
        group_pred: tuple[frozenset[int], ...],
        group_succ: tuple[frozenset[int], ...],
    ) -> tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        tuple[frozenset[int], ...],
        tuple[frozenset[int], ...],
    ]:
        """Reclassify Type=6 mainline groups whose pred and succ touch the same
        junction as interior of that junction.

        NGII manual §9.4.2 documents the "exception lane" case explicitly
        ("교차로 내 예외(일반주행차로)" page 151): a 일반주행차로 (LinkType=6)
        traversing a 평면교차로. Topologically these rows enter and exit the
        same junction via two of its boundary nodes, so
        ``group_pred[b] ∩ group_succ[b]`` is non-empty for exactly the
        offending group. Promoting them keeps the intersection a single
        :class:`Junction` instead of fracturing it into two roads + one
        spurious connector.

        Only LinkType=6 rows are promoted; other LinkTypes (bus 4, variable
        5, toll 2/3, rest-area 7-12, ramp 13/14) carry domain semantics
        that would be silently erased by junction-interior reclassification.

        Returns the four arrays/tuples updated in place semantics:
        ``junction_id`` and ``group_junction`` gain the promoted ids;
        ``group_pred`` / ``group_succ`` are emptied for promoted groups
        (interior groups carry no pred/succ — connectivity is implied by
        ``group_junction``).
        """
        n_groups = len(group_junction)
        rows_by_group: dict[int, list[int]] = {}
        for i, b in enumerate(group_id):
            rows_by_group.setdefault(int(b), []).append(int(i))

        pred_list = list(group_pred)
        succ_list = list(group_succ)
        link_types = a2.link_types
        n_promoted = 0
        for b in range(n_groups):
            if int(group_junction[b]) != -1:
                continue
            rows = rows_by_group.get(b, [])
            if not rows:
                continue
            if not all(link_types[i] == _PROMOTABLE_LINK_TYPE for i in rows):
                continue
            shared = pred_list[b] & succ_list[b]
            if not shared:
                continue
            j = min(shared)
            for i in rows:
                junction_id[i] = np.int32(j)
            group_junction[b] = np.int32(j)
            pred_list[b] = frozenset()
            succ_list[b] = frozenset()
            n_promoted += 1
            log.info(
                "_promote_within_junction_mainline_groups: group %d (Type=6, %d row(s)) "
                "promoted to interior of junction %d (pred ∩ succ = %s)",
                b,
                len(rows),
                j,
                sorted(shared),
            )
        if n_promoted:
            log.info(
                "_promote_within_junction_mainline_groups: promoted %d mainline group(s) "
                "to junction interior",
                n_promoted,
            )
        return junction_id, group_junction, tuple(pred_list), tuple(succ_list)

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
    def _log_summary(
        a2: A2Data,
        group_id: NDArray[np.int32],
        road_id_per_link: NDArray[np.int32],
        node_junction_id: NDArray[np.int32],
        node_role: dict[str, NodeRole],
    ) -> None:
        n_a2 = len(a2.ids)
        n_groups = int(group_id.max()) + 1 if len(group_id) else 0
        n_mainline_groups = int(road_id_per_link.max()) + 1 if (road_id_per_link >= 0).any() else 0
        n_junctions = int(node_junction_id.max()) + 1 if len(node_junction_id) else 0
        n_cuts = sum(1 for r in node_role.values() if r is NodeRole.JUNCTION)
        log.info(
            "segmentation: %d links -> %d groups (%d mainline), %d junctions "
            "(cut at %d / %d nodes)",
            n_a2,
            n_groups,
            n_mainline_groups,
            n_junctions,
            n_cuts,
            len(node_role),
        )
