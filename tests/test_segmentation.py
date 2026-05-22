"""Segmentation invariants against the grouped NGII Jeju sample dataset.

These pin down behaviour that is not visible from the code alone and that
real NGII data has tripped over during development.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shp2xodr.shp.data import A1Data, A2Data, B2Data
from shp2xodr.shp.segmentation import Segmentation

_SAMPLE_SHP_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "SEC001_제주첨단과학기술단지"
    / "HDMap_UTMK_정표고"
)


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    if not _SAMPLE_SHP_DIR.is_dir():
        pytest.skip(f"sample dataset not present at {_SAMPLE_SHP_DIR}")
    return _SAMPLE_SHP_DIR


def test_interior_link_iff_link_type_1(sample_dir: Path) -> None:
    a2 = A2Data(sample_dir)
    seg = Segmentation.from_shp_dir(sample_dir)
    is_interior = seg.junction_id >= 0
    assert np.array_equal(is_interior, a2.link_types == "1")


def test_group_junction_uniform(sample_dir: Path) -> None:
    seg = Segmentation.from_shp_dir(sample_dir)
    n_groups = int(seg.group_id.max()) + 1
    for b in range(n_groups):
        rows = seg.group_id == b
        row_jids = set(seg.junction_id[rows].tolist())
        bj = int(seg.group_junction[b])
        if bj < 0:
            assert row_jids == {-1}, f"group {b} mainline but rows={row_jids}"
        else:
            assert row_jids == {bj}, f"group {b} interior to {bj} but rows={row_jids}"


def test_interior_links_at_junction_node_share_junction(sample_dir: Path) -> None:
    """For a 평면교차로 (NodeType=1) node, all touching LinkType=1 links
    resolve to the same junction id, and that id matches the node's own
    junction id.
    """
    a1 = A1Data(sample_dir)
    a2 = A2Data(sample_dir)
    seg = Segmentation.from_shp_dir(sample_dir)

    plane_xings = a1.node_types == "1"

    checked = 0
    for idx in np.where(plane_xings)[0]:
        nid = a1.ids[idx]
        touch = ((a2.from_node_ids == nid) | (a2.to_node_ids == nid)) & (a2.link_types == "1")
        if not touch.any():
            continue
        node_jid = int(seg.node_junction_id[idx])
        assert node_jid >= 0, f"node {nid} is NodeType=1 but has no junction id"
        row_jids = set(seg.junction_id[touch].tolist())
        assert row_jids == {node_jid}, (
            f"node {nid} has touching LinkType=1 links with junction_ids "
            f"{row_jids}, expected {{{node_jid}}}"
        )
        checked += 1
    assert checked > 0, "no 평면교차로 nodes had touching LinkType=1 links"


def test_road_break_does_not_cut_group(sample_dir: Path) -> None:
    """Mainline links sharing a ROAD_BREAK node (tunnel/bridge/under- or
    overpass start-end point) belong to the same group — in OpenDRIVE the
    road runs continuously and the structure is a ``<bridge>``/``<tunnel>``
    span on it.
    """
    a2 = A2Data(sample_dir)
    seg = Segmentation.from_shp_dir(sample_dir)
    # 014391 and 014392 share node A123AI014623 (NodeType=4, 교량).
    left = int(np.where(a2.ids == "A223AI014391")[0][0])
    right = int(np.where(a2.ids == "A223AI014392")[0][0])
    assert seg.group_id[left] == seg.group_id[right]


def test_proximity_merge_fuses_split_intersection(sample_dir: Path) -> None:
    """Two junction-role A1 nodes that the graph-only pass left in disjoint
    components — but that sit ~4 m apart inside one physical intersection
    — must end up in one junction once proximity merging is enabled.
    """
    a1 = A1Data(sample_dir)
    # See the group 306 / junction 46-vs-83 case from interactive picks: the
    # 4-node group around A123AI014562 was split off from the 6-node group
    # around A123AI014208, with ~3.8 m between their nearest nodes.
    left_idx = int(np.where(a1.ids == "A123AI014208")[0][0])
    right_idx = int(np.where(a1.ids == "A123AI014562")[0][0])

    graph_only = Segmentation.from_shp_dir(sample_dir)
    assert graph_only.node_junction_id[left_idx] != graph_only.node_junction_id[right_idx]

    merged = Segmentation.from_shp_dir(sample_dir, junction_merge_dist_m=5.0)
    left_jid = int(merged.node_junction_id[left_idx])
    right_jid = int(merged.node_junction_id[right_idx])
    assert left_jid >= 0
    assert left_jid == right_jid


def test_b2_two_sided_mainline_rows_share_group(sample_dir: Path) -> None:
    """A B2 row with both sides bound to *mainline* A2 lanes must resolve to
    the same group on both sides — mainline lanes that share a B2 divider
    are laterally adjacent by definition, and the segmentation pass unions
    them via R/L_LinkID. (Interior LinkType=1 connecting lanes inside a
    junction each group alone, so a B2 line straddling two of them is
    legitimately cross-group and is excluded here.)
    """
    a2 = A2Data(sample_dir)
    seg = Segmentation.from_shp_dir(sample_dir)
    both_bound = (seg.b2_r_link_idx >= 0) & (seg.b2_l_link_idx >= 0)
    r_main = both_bound & (a2.link_types[np.clip(seg.b2_r_link_idx, 0, None)] != "1")
    l_main = both_bound & (a2.link_types[np.clip(seg.b2_l_link_idx, 0, None)] != "1")
    mask = r_main & l_main
    assert mask.any(), "no B2 row has both sides bound to mainline lanes"
    mismatched = int((mask & (seg.b2_r_group != seg.b2_l_group)).sum())
    assert mismatched == 0, f"{mismatched} two-sided mainline B2 rows resolve to different groups"


def test_group_side_junctions_well_formed(sample_dir: Path) -> None:
    """Interior groups carry empty pred/succ sets; mainline groups whose
    pred/succ side is non-empty must reference valid junction ids.
    """
    seg = Segmentation.from_shp_dir(sample_dir)
    n_junctions = int(seg.node_junction_id.max()) + 1
    valid_jids = set(range(n_junctions))
    n_groups = int(seg.group_id.max()) + 1
    for b in range(n_groups):
        bj = int(seg.group_junction[b])
        pred = seg.group_pred_junctions[b]
        succ = seg.group_succ_junctions[b]
        if bj >= 0:
            assert pred == frozenset() and succ == frozenset(), (
                f"interior group {b} should carry empty pred/succ sets"
            )
            continue
        assert pred.issubset(valid_jids), f"group {b} pred={pred}"
        assert succ.issubset(valid_jids), f"group {b} succ={succ}"


# ---- Road / Junction graph invariants -----------------------------------------


def test_road_id_per_link_partitions_against_junction_id(sample_dir: Path) -> None:
    """Every A2 link is either mainline (road_id >= 0, junction_id == -1) or
    junction-interior (road_id == -1, junction_id >= 0) — never both, never
    neither.
    """
    seg = Segmentation.from_shp_dir(sample_dir)
    mainline = seg.junction_id == -1
    assert (seg.road_id_per_link[mainline] >= 0).all()
    assert (seg.road_id_per_link[~mainline] == -1).all()


def test_road_and_junction_ids_are_dense(sample_dir: Path) -> None:
    seg = Segmentation.from_shp_dir(sample_dir)
    assert tuple(r.id for r in seg.roads) == tuple(range(len(seg.roads)))
    assert tuple(j.id for j in seg.junctions) == tuple(range(len(seg.junctions)))


def test_road_junction_back_references(sample_dir: Path) -> None:
    """Every road's touching junction lists that road, and vice versa."""
    seg = Segmentation.from_shp_dir(sample_dir)
    for road in seg.roads:
        for junction in road.junctions:
            assert road in junction.roads
    for junction in seg.junctions:
        for road in junction.roads:
            assert junction in road.junctions


def test_centerline_b2_both_sides_consistent_when_bound(sample_dir: Path) -> None:
    """Pass-A invariant: any B2 Kind=501 row whose R and L sides both bind
    mainline lanes must resolve to the same road on both sides.

    The Jeju fixture digitizes every centerline one-sided (R bound, L empty
    facing the median), so the predicate is typically vacuous here — pass-B
    geometric pairing is what drives the merge for this dataset. The
    assertion still pins the invariant for fixtures that include two-sided
    centerlines.
    """
    seg = Segmentation.from_shp_dir(sample_dir)
    b2 = B2Data(sample_dir)
    is_centerline = b2.kinds == "501"
    both_mainline = is_centerline & (seg.b2_r_road >= 0) & (seg.b2_l_road >= 0)
    assert np.array_equal(seg.b2_r_road[both_mainline], seg.b2_l_road[both_mainline])


def test_bidirectional_merge_engages_via_passb_in_sample(sample_dir: Path) -> None:
    """The Jeju fixture has divided road segments whose two directions are
    only paired via pass-B geometric proximity. Verify the merge engages:
    fewer mainline roads than mainline groups, and at least one road
    composed of multiple groups (the merge product).
    """
    seg = Segmentation.from_shp_dir(sample_dir)
    n_mainline_groups = int((seg.group_junction == -1).sum())
    assert len(seg.roads) < n_mainline_groups
    multi_group_roads = [r for r in seg.roads if len(r.group_ids) > 1]
    assert multi_group_roads, "no merged roads — bidirectional merge did not engage"
