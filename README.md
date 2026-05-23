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
groups A2_LINK rows into lane-bundles, clusters A1_NODE rows into
junction components, and exposes per-row / per-group arrays the viz
reads. The steps run in this order:

1. **Load layers** — A1_NODE (points) and A2_LINK (polylines) via the
   typed records in
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
   *same* group is the NGII "exception lane" case (manual §9.4.2,
   "교차로 내 예외(일반주행차로)"). Reclassify it as interior of that
   junction instead.
