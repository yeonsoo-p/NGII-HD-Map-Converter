# shp2xodr — NGII HD-map → OpenDRIVE

Converter for the Korean National Geographic Information Institute (NGII)
정밀도로지도 (precise road map) SHP layers into ASAM OpenDRIVE 1.7. The
companion Qt + PyVista inspector renders every loaded NGII layer in 3D and
exposes the segmentation result at four abstraction levels (raw / group /
junction / road).

## Run

```shell
uv sync
uv run python -m shp2xodr                          # opens empty; File → Open SHP folder…
uv run python -m shp2xodr shp_dir=/path/to/SEC###  # auto-load a section
```

Configuration lives in [conf/config.yaml](conf/config.yaml); every magic
constant is exposed there.

## Segmentation pipeline

`Segmentation.from_shp_dir` in
[src/shp2xodr/shp/segmentation.py](src/shp2xodr/shp/segmentation.py)
turns the raw A1/A2/B2 layers into a typed graph of `Road`s and
`Junction`s. The steps run in this order:

1. **Load layers** — A1_NODE (points), A2_LINK (polylines), and
   B2_SURFACELINEMARK (polylines) via the typed records in
   [src/shp2xodr/shp/data.py](src/shp2xodr/shp/data.py). cp949 mojibake,
   single-part MultiLineString promotion, and column-capitalization drift
   are handled at this boundary so downstream code sees only canonical
   names and numpy arrays.
2. **Classify A1 nodes** ([`_classify_nodes`](src/shp2xodr/shp/segmentation.py)) —
   each node is JUNCTION (NodeType 1/2/8/9/10), ROAD_BREAK (3-6 — tunnel,
   bridge, under/overpass start/end), LANE_SECTION (7), or IGNORE.
   NodeType 99 (기타) is split by fan-in/out degree: a real diverge/merge
   counts as JUNCTION, a 1-to-1 feature break as LANE_SECTION.
3. **Group A2 links into lane-bundles**
   ([`_group_links`](src/shp2xodr/shp/segmentation.py)) — union-find on
   R/L_LinkID for lateral lanes and shared non-JUNCTION endpoints for
   longitudinal continuations. Interior (`LinkType=1`) and mainline rows
   are kept in separate groups even when NGII lists them as lateral
   neighbours.
4. **Cluster junctions**
   ([`_cluster_junctions`](src/shp2xodr/shp/segmentation.py)):
   1. Mark every `LinkType=1` row as interior and union its two
      junction-node endpoints.
   2. Union junction nodes whose interior links sit in the same A2 group
      (laterally adjacent connecting lanes are one intersection).
   3. **Geometric crossing union** — two interior polylines that cross in
      plan with `|Δz| ≤ junction_crossing_z_tol_m` at the crossing point
      are unioned. Catches turn paths that physically cross inside one
      large intersection but share no endpoint; the z gate keeps stacked
      overpasses apart.
   4. Optional proximity merge — components whose nearest nodes lie within
      `junction_merge_dist_m` are fused (channelized turns, free-flow
      paths).
5. **Resolve group endpoints**
   ([`_resolve_group_endpoints`](src/shp2xodr/shp/segmentation.py)) — each
   group is fully interior (one junction id) or fully mainline (pred/succ
   junction sets collected from its boundary A1 nodes).
6. **Promote within-junction mainline groups**
   ([`_promote_within_junction_mainline_groups`](src/shp2xodr/shp/segmentation.py)) —
   any `LinkType=6 일반주행차로` group whose pred and succ both touch the
   *same* junction is the NGII "exception lane" case (manual §9.4.2,
   "교차로 내 예외(일반주행차로)"). Reclassify it as interior of that
   junction instead of letting it split the intersection into two roads.
7. **Resolve B2 rows to groups**
   ([`_resolve_b2_to_groups`](src/shp2xodr/shp/segmentation.py)) — each
   B2 row's R/L_LinkID → A2 row → group id, on each side. Missing FKs
   yield `-1` on that side.
8. **Bidirectional group merge**
   ([`_merge_groups_bidirectional`](src/shp2xodr/shp/segmentation.py)):
   1. **Pass A — same-row pairing.** For every centerline-like B2 row
      (`Kind=501` 중앙선 or `Kind=503` 주행선) whose R and L sides both
      bind mainline groups, union them. Covers the typical undivided
      road.
   2. **Pass B — proximity pairing.** For every pair of centerline-like
      B2 rows that each bind one mainline group and lie within
      `bidirectional_merge_max_separation_m`, union the two bound
      groups. Covers divided roads with one centerline per direction.
   3. **Single-link junction-bridge veto.** Both passes skip the union
      when a single `LinkType=1` interior link directly bridges the two
      mainline groups — i.e., the link's `FromNodeID` is the terminus of
      one group and its `ToNodeID` is the start of the other. Such a
      pair is two different roads connected through a junction's
      interior path, not opposing carriageways. Opposing carriageways of
      a divided road meeting an intersection are *not* caught by this
      veto because no single interior link goes between their boundary
      nodes.
9. **Resolve B2 to roads & build object graph**
   ([`_resolve_b2_to_roads`](src/shp2xodr/shp/segmentation.py),
   [`_build_graph`](src/shp2xodr/shp/segmentation.py)) — flat per-link
   `road_id_per_link` arrays and the `Road` / `Junction` dataclass graph
   with back-patched mutual references. The viz reads both views; the
   OpenDRIVE emit side will consume the object graph.

## Key tuning knobs ([conf/config.yaml](conf/config.yaml))

| Knob | What it protects against |
| --- | --- |
| `junction_merge_dist_m` (10 m) | Channelized turns / free-flow paths topologically disjoint from the rest of one intersection. |
| `junction_crossing_z_tol_m` (2 m) | Turn paths that cross in plan inside one large intersection; the z gate keeps overpasses apart. |
| `bidirectional_merge_max_separation_m` (15 m) | Divided roads with one centerline per carriageway; default leaves headroom for wide expressway medians. |

## Toolchain

Python 3.13 + `uv` for deps. `ruff format` + `ruff check` for style;
`mypy --strict` for types; See [CLAUDE.md](CLAUDE.md) for the full coding rules.
