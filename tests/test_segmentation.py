"""Segmentation invariants against the grouped NGII Jeju sample dataset.

These pin down behaviour that is not visible from the code alone and that
real NGII data has tripped over during development.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shp2xodr.shp.data import A1Data, A2Data, B2Data
from shp2xodr.shp.segmentation import Segmentation, SegmentationConfig

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


def _seg_cfg(
    *,
    junction_merge_dist_m: float = 0.0,
    bidirectional_merge_max_separation_m: float = 15.0,
) -> SegmentationConfig:
    """Test helper. Defaults match the production behaviour each test relied
    on under the old float-arg signature: no junction-proximity merge unless
    a test explicitly enables it, bidirectional centerline merge always on.
    """
    return SegmentationConfig(
        junction_merge_dist_m=junction_merge_dist_m,
        bidirectional_merge_max_separation_m=bidirectional_merge_max_separation_m,
    )


# ---- Shared input fixtures -----------------------------------------------------
# Module-scoped so each SHP layer is read once per pytest session for the Jeju
# fixture - frozen dataclasses and a Segmentation built with frozen=True make
# sharing safe across tests.


@pytest.fixture(scope="module")
def a1_data(sample_dir: Path) -> A1Data:
    return A1Data(sample_dir)


@pytest.fixture(scope="module")
def a2_data(sample_dir: Path) -> A2Data:
    return A2Data(sample_dir)


@pytest.fixture(scope="module")
def b2_data(sample_dir: Path) -> B2Data:
    return B2Data(sample_dir)


@pytest.fixture(scope="module")
def segmentation(sample_dir: Path) -> Segmentation:
    return Segmentation.from_shp_dir(sample_dir, _seg_cfg())


@pytest.fixture(scope="module")
def segmentation_with_proximity_merge(sample_dir: Path) -> Segmentation:
    return Segmentation.from_shp_dir(sample_dir, _seg_cfg(junction_merge_dist_m=5.0))


def test_interior_link_iff_link_type_1(a2_data: A2Data, segmentation: Segmentation) -> None:
    is_interior = segmentation.junction_id >= 0
    assert np.array_equal(is_interior, a2_data.link_types == "1")


def test_group_junction_uniform(segmentation: Segmentation) -> None:
    n_groups = int(segmentation.group_id.max()) + 1
    for b in range(n_groups):
        rows = segmentation.group_id == b
        row_jids = set(segmentation.junction_id[rows].tolist())
        bj = int(segmentation.group_junction[b])
        if bj < 0:
            assert row_jids == {-1}, f"group {b} mainline but rows={row_jids}"
        else:
            assert row_jids == {bj}, f"group {b} interior to {bj} but rows={row_jids}"


def test_interior_links_at_junction_node_share_junction(
    a1_data: A1Data, a2_data: A2Data, segmentation: Segmentation
) -> None:
    """For a 평면교차로 (NodeType=1) node, all touching LinkType=1 links
    resolve to the same junction id, and that id matches the node's own
    junction id.
    """
    plane_xings = a1_data.node_types == "1"

    checked = 0
    for idx in np.where(plane_xings)[0]:
        nid = a1_data.ids[idx]
        touch = ((a2_data.from_node_ids == nid) | (a2_data.to_node_ids == nid)) & (
            a2_data.link_types == "1"
        )
        if not touch.any():
            continue
        node_jid = int(segmentation.node_junction_id[idx])
        assert node_jid >= 0, f"node {nid} is NodeType=1 but has no junction id"
        row_jids = set(segmentation.junction_id[touch].tolist())
        assert row_jids == {node_jid}, (
            f"node {nid} has touching LinkType=1 links with junction_ids "
            f"{row_jids}, expected {{{node_jid}}}"
        )
        checked += 1
    assert checked > 0, "no 평면교차로 nodes had touching LinkType=1 links"


def test_road_break_does_not_cut_group(a2_data: A2Data, segmentation: Segmentation) -> None:
    """Mainline links sharing a ROAD_BREAK node (tunnel/bridge/under- or
    overpass start-end point) belong to the same group — in OpenDRIVE the
    road runs continuously and the structure is a ``<bridge>``/``<tunnel>``
    span on it.
    """
    # 014391 and 014392 share node A123AI014623 (NodeType=4, 교량).
    left = int(np.where(a2_data.ids == "A223AI014391")[0][0])
    right = int(np.where(a2_data.ids == "A223AI014392")[0][0])
    assert segmentation.group_id[left] == segmentation.group_id[right]


def test_proximity_merge_fuses_split_intersection(
    a1_data: A1Data,
    segmentation: Segmentation,
    segmentation_with_proximity_merge: Segmentation,
) -> None:
    """Two junction-role A1 nodes that the graph-only pass left in disjoint
    components — but that sit ~4 m apart inside one physical intersection
    — must end up in one junction once proximity merging is enabled.
    """
    # See the group 306 / junction 46-vs-83 case from interactive picks: the
    # 4-node group around A123AI014562 was split off from the 6-node group
    # around A123AI014208, with ~3.8 m between their nearest nodes.
    left_idx = int(np.where(a1_data.ids == "A123AI014208")[0][0])
    right_idx = int(np.where(a1_data.ids == "A123AI014562")[0][0])

    assert segmentation.node_junction_id[left_idx] != segmentation.node_junction_id[right_idx]

    merged = segmentation_with_proximity_merge
    left_jid = int(merged.node_junction_id[left_idx])
    right_jid = int(merged.node_junction_id[right_idx])
    assert left_jid >= 0
    assert left_jid == right_jid


def test_b2_two_sided_mainline_rows_share_group(
    a2_data: A2Data, segmentation: Segmentation
) -> None:
    """A B2 row with both sides bound to *mainline* A2 lanes must resolve to
    the same group on both sides — mainline lanes that share a B2 divider
    are laterally adjacent by definition, and the segmentation pass unions
    them via R/L_LinkID. (Interior LinkType=1 connecting lanes inside a
    junction each group alone, so a B2 line straddling two of them is
    legitimately cross-group and is excluded here.)
    """
    both_bound = (segmentation.b2_r_link_idx >= 0) & (segmentation.b2_l_link_idx >= 0)
    r_main = both_bound & (a2_data.link_types[np.clip(segmentation.b2_r_link_idx, 0, None)] != "1")
    l_main = both_bound & (a2_data.link_types[np.clip(segmentation.b2_l_link_idx, 0, None)] != "1")
    mask = r_main & l_main
    assert mask.any(), "no B2 row has both sides bound to mainline lanes"
    mismatched = int((mask & (segmentation.b2_r_group != segmentation.b2_l_group)).sum())
    assert mismatched == 0, f"{mismatched} two-sided mainline B2 rows resolve to different groups"


def test_group_side_junctions_well_formed(segmentation: Segmentation) -> None:
    """Interior groups carry empty pred/succ sets; mainline groups whose
    pred/succ side is non-empty must reference valid junction ids.
    """
    n_junctions = int(segmentation.node_junction_id.max()) + 1
    valid_jids = set(range(n_junctions))
    n_groups = int(segmentation.group_id.max()) + 1
    for b in range(n_groups):
        bj = int(segmentation.group_junction[b])
        pred = segmentation.group_pred_junctions[b]
        succ = segmentation.group_succ_junctions[b]
        if bj >= 0:
            assert pred == frozenset() and succ == frozenset(), (
                f"interior group {b} should carry empty pred/succ sets"
            )
            continue
        assert pred.issubset(valid_jids), f"group {b} pred={pred}"
        assert succ.issubset(valid_jids), f"group {b} succ={succ}"


# ---- Road / Junction graph invariants -----------------------------------------


def test_road_id_per_link_partitions_against_junction_id(segmentation: Segmentation) -> None:
    """Every A2 link is either mainline (road_id >= 0, junction_id == -1) or
    junction-interior (road_id == -1, junction_id >= 0) — never both, never
    neither.
    """
    mainline = segmentation.junction_id == -1
    assert (segmentation.road_id_per_link[mainline] >= 0).all()
    assert (segmentation.road_id_per_link[~mainline] == -1).all()


def test_road_and_junction_ids_are_dense(segmentation: Segmentation) -> None:
    assert tuple(r.id for r in segmentation.roads) == tuple(range(len(segmentation.roads)))
    assert tuple(j.id for j in segmentation.junctions) == tuple(range(len(segmentation.junctions)))


def test_road_junction_back_references(segmentation: Segmentation) -> None:
    """Every road's touching junction lists that road, and vice versa."""
    for road in segmentation.roads:
        for junction in road.junctions:
            assert road in junction.roads
    for junction in segmentation.junctions:
        for road in junction.roads:
            assert junction in road.junctions


def test_centerline_b2_both_sides_consistent_when_bound(
    b2_data: B2Data, segmentation: Segmentation
) -> None:
    """Pass-A invariant: any B2 Kind=501 row whose R and L sides both bind
    mainline lanes must resolve to the same road on both sides.

    The Jeju fixture digitizes every centerline one-sided (R bound, L empty
    facing the median), so the predicate is typically vacuous here — pass-B
    geometric pairing is what drives the merge for this dataset. The
    assertion still pins the invariant for fixtures that include two-sided
    centerlines.
    """
    is_centerline = b2_data.kinds == "501"
    both_mainline = is_centerline & (segmentation.b2_r_road >= 0) & (segmentation.b2_l_road >= 0)
    assert np.array_equal(
        segmentation.b2_r_road[both_mainline], segmentation.b2_l_road[both_mainline]
    )


def test_bidirectional_merge_engages_via_passb_in_sample(segmentation: Segmentation) -> None:
    """The Jeju fixture has divided road segments whose two directions are
    only paired via pass-B geometric proximity. Verify the merge engages:
    fewer mainline roads than mainline groups, and at least one road
    composed of multiple groups (the merge product).
    """
    n_mainline_groups = int((segmentation.group_junction == -1).sum())
    assert len(segmentation.roads) < n_mainline_groups
    multi_group_roads = [r for r in segmentation.roads if len(r.group_ids) > 1]
    assert multi_group_roads, "no merged roads — bidirectional merge did not engage"
