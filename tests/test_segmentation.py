"""Segmentation invariants against the bundled NGII Jeju sample dataset.

These pin down behaviour that is not visible from the code alone and that
real NGII data has tripped over during development.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shp2xodr.shp.io import load_a1_nodes, load_a2_links
from shp2xodr.shp.segmentation import segment_links

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
    a2 = load_a2_links(sample_dir)
    seg = segment_links(sample_dir)
    link_types = a2["LinkType"].astype(str).to_numpy()
    is_interior = seg.junction_id >= 0
    assert np.array_equal(is_interior, link_types == "1")


def test_bundle_junction_uniform(sample_dir: Path) -> None:
    seg = segment_links(sample_dir)
    n_bundles = int(seg.bundle_id.max()) + 1
    for b in range(n_bundles):
        rows = seg.bundle_id == b
        row_jids = set(seg.junction_id[rows].tolist())
        bj = int(seg.bundle_junction[b])
        if bj < 0:
            assert row_jids == {-1}, f"bundle {b} mainline but rows={row_jids}"
        else:
            assert row_jids == {bj}, f"bundle {b} interior to {bj} but rows={row_jids}"


def test_interior_links_at_junction_node_share_junction(sample_dir: Path) -> None:
    """For a 평면교차로 (NodeType=1) node, all touching LinkType=1 links
    resolve to the same junction id, and that id matches the node's own
    junction id.
    """
    a1 = load_a1_nodes(sample_dir)
    a2 = load_a2_links(sample_dir)
    seg = segment_links(sample_dir)

    a1_nids = a1["ID"].astype(str).to_numpy()
    froms = a2["FromNodeID"].astype(str).to_numpy()
    tos = a2["ToNodeID"].astype(str).to_numpy()
    link_types = a2["LinkType"].astype(str).to_numpy()
    plane_xings = a1["NodeType"].astype(str).to_numpy() == "1"

    checked = 0
    for idx in np.where(plane_xings)[0]:
        nid = a1_nids[idx]
        touch = ((froms == nid) | (tos == nid)) & (link_types == "1")
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


def test_road_break_does_not_cut_bundle(sample_dir: Path) -> None:
    """Mainline links sharing a ROAD_BREAK node (tunnel/bridge/under- or
    overpass start-end point) belong to the same bundle — in OpenDRIVE the
    road runs continuously and the structure is a ``<bridge>``/``<tunnel>``
    span on it.
    """
    a2 = load_a2_links(sample_dir)
    seg = segment_links(sample_dir)
    ids = a2["ID"].astype(str).to_numpy()
    # 014391 and 014392 share node A123AI014623 (NodeType=4, 교량).
    left = int(np.where(ids == "A223AI014391")[0][0])
    right = int(np.where(ids == "A223AI014392")[0][0])
    assert seg.bundle_id[left] == seg.bundle_id[right]


def test_proximity_merge_fuses_split_intersection(sample_dir: Path) -> None:
    """Two junction-role A1 nodes that the graph-only pass left in disjoint
    components — but that sit ~4 m apart inside one physical intersection
    — must end up in one junction once proximity merging is enabled.
    """
    a1 = load_a1_nodes(sample_dir)
    a1_nids = a1["ID"].astype(str).to_numpy()
    # See the bundle 306 / junction 46-vs-83 case from interactive picks: the
    # 4-node group around A123AI014562 was split off from the 6-node group
    # around A123AI014208, with ~3.8 m between their nearest nodes.
    left_idx = int(np.where(a1_nids == "A123AI014208")[0][0])
    right_idx = int(np.where(a1_nids == "A123AI014562")[0][0])

    graph_only = segment_links(sample_dir)
    assert graph_only.node_junction_id[left_idx] != graph_only.node_junction_id[right_idx]

    merged = segment_links(sample_dir, junction_merge_dist_m=5.0)
    left_jid = int(merged.node_junction_id[left_idx])
    right_jid = int(merged.node_junction_id[right_idx])
    assert left_jid >= 0
    assert left_jid == right_jid


def test_bundle_side_junctions_well_formed(sample_dir: Path) -> None:
    """Interior bundles carry empty pred/succ sets; mainline bundles whose
    pred/succ side is non-empty must reference valid junction ids.
    """
    seg = segment_links(sample_dir)
    n_junctions = int(seg.node_junction_id.max()) + 1
    valid_jids = set(range(n_junctions))
    n_bundles = int(seg.bundle_id.max()) + 1
    for b in range(n_bundles):
        bj = int(seg.bundle_junction[b])
        pred = seg.bundle_pred_junctions[b]
        succ = seg.bundle_succ_junctions[b]
        if bj >= 0:
            assert pred == frozenset() and succ == frozenset(), (
                f"interior bundle {b} should carry empty pred/succ sets"
            )
            continue
        assert pred.issubset(valid_jids), f"bundle {b} pred={pred}"
        assert succ.issubset(valid_jids), f"bundle {b} succ={succ}"
