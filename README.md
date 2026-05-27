# ngii2xodr - NGII HD-map -> OpenDRIVE

Converter workspace for Korean National Geographic Information Institute (NGII)
precise road map SHP layers and ASAM OpenDRIVE 1.7 experiments.

The current application is a Qt + PyVista inspector for NGII section folders.
It loads one configured coordinate product into a canonical `NGIIDataset`,
renders every supported NGII layer by geometry type, supports layer visibility
toggles, and lets you shift-click or search by ID to inspect canonical feature
objects plus staged A2 segmentation results.

## Run

```shell
uv sync
uv run python -m ngii2xodr                          # opens empty; File -> Open NGII folder...
uv run python -m ngii2xodr ngii_dir=/path/to/SEC### # auto-load a section
```

Configuration lives in [conf/config.yaml](conf/config.yaml). Relative
`ngii_dir` values resolve against the invocation cwd, not Hydra's run directory.
`ngii.coordinate` selects the one coordinate product to load.

## Data Loading, Sanity, And Segmentation

`HdMapViz` in [src/ngii2xodr/ngii/viz.py](src/ngii2xodr/ngii/viz.py) loads typed
records from [src/ngii2xodr/ngii/data](src/ngii2xodr/ngii/data) and builds a
generic render registry from `dataset.layer_stores`. Point, line, and polygon
layers are rendered according to per-layer defaults in
[conf/config.yaml](conf/config.yaml).

The inspector dock in [src/ngii2xodr/gui/window.py](src/ngii2xodr/gui/window.py)
has three tabs: `Layers`, `Items`, and `Selected`. Both shift-click selection
and item search resolve through `FeatureRef(layer_attr, feature_id)`, so the
3D highlight, item tree, and selected table all point at the same canonical
feature object.

NGII data code is split between common contracts in
[src/ngii2xodr/ngii/data](src/ngii2xodr/ngii/data) and version-specific
definitions/loaders in
[src/ngii2xodr/ngii/data/v2023](src/ngii2xodr/ngii/data/v2023) and
[src/ngii2xodr/ngii/data/v2025](src/ngii2xodr/ngii/data/v2025). The public
loader remains `load_ngii(path, coordinate, cfg.ngii)` and dispatches from the
requested coordinate product.

The loader builds one canonical `NGIIDataset` in this order:

- Detect the implemented manual schema from SHP filenames under the selected
  coordinate product. Missing required layers, missing `.dbf` / `.shx` / `.prj`
  sidecars, and unknown SHP layers are reported through `SanityReport`.
- Read each supported layer with GeoPandas, normalize documented column
  capitalization and aliases, warn for missing or unknown manual columns, and
  convert supported Point, LineString/MultiLineString, and Polygon/MultiPolygon
  geometries into XYZ numpy arrays. Multi-part lines are line-merged directly or
  after configured snapping; multi-part polygons with more than one polygon are
  rejected for that row.
- Decode text defensively. A CP949 retry is used for UTF-8 failures or likely
  mojibake, `.cpg` UTF-8 DBF text can be overlaid when safe, latin1-to-CP949
  mojibake is repaired when it increases Hangul content, and remaining Unicode
  replacement characters are warned.
- Merge rows into layer stores by feature ID. Empty IDs are skipped, duplicate
  identical IDs are warned, and duplicate conflicting IDs keep the first row;
  the later row is reported as dropped when that check is in `repair` mode.
- Bind global and per-layer indexes, run configured sanity hooks, bind again,
  rebuild resolved reference edges, validate remaining references, and
  log grouped warning/action summaries.

Sanity checks are configured under `ngii.sanity.checks` in
[conf/config.yaml](conf/config.yaml). Each check mode is `null`, `warn`, or
`repair`: `null` disables reporting for that condition, `warn` reports without
changing data, and `repair` applies deterministic repairs while logging both
repair actions and residual warnings. They currently cover:

- Manual field conformance: required fields, max text length, integer/float
  parseability, code lists, and `HistType` format.
- Reference conformance: unresolved required and optional references,
  cross-layer global ID collisions, unresolved reference cleanup, unreferenced
  nodes, and cascading removal of rows whose required references depend on
  removed rows.
- Link endpoint topology: too-short link removal, endpoint-isolated
  link removal, missing endpoint references repaired to the only nearby node
  within `node_match_tolerance_m`, unresolved endpoint references removed,
  reversed `FromNodeID`/`ToNodeID` swapped when the geometry clearly points the
  other way, and ambiguous/misaligned endpoints warned.
- Link flow direction: a leaf-to-leaf topology walk proposes reversed links,
  then R/L same-direction neighbors must support the reversal with an opposing
  geometry vector before a single unambiguous candidate is swapped.
- Side-reference topology: R/L references that point to a head-to-tail longitudinal
  neighbor are cleared, including the reciprocal pointer when it points back.
  Sharing only the same `from` node or only the same `to` node is not treated as
  a longitudinal conflict. Non-reciprocal R/L references are repaired only when
  exactly one inverse candidate exists; otherwise they are warned.
- Schema reciprocal references: missing reciprocal pointers, such as link
  `R_LinkID`/`L_LinkID` pairs, are filled when the target slot is empty; targets
  already pointing somewhere else are warned as conflicts.

Segmentation stages live under
[src/ngii2xodr/ngii/segmentation](src/ngii2xodr/ngii/segmentation). The pipeline
builds a `SegmentationContext` once, then runs enabled stages in dependency
order. The context caches the directed node graph from `FromNodeID`/`ToNodeID`,
explicit and inverse lateral R/L link maps, Shapely line trees, schema
`RoleFilter` rows, and schema `RoleKey` semantic keys.

The current stage order is:

- `UTurnStage`: emits U-turn link entities. Schemas with a direct `uturn_link`
  role, currently 2025 `Turn=3`, use it directly. Otherwise ordinary links are
  matched against U-turn lane-line markers, currently 2023/2025 kind `502`, by
  XY intersection plus `z_intersection_tol_m`.
- `LateralLinkGroupStage`: takes ordinary links except U-turn links and builds
  connected components from explicit and inverse R/L references. Pocket links are
  marked from the schema `pocket_link` role. A reference-side link is chosen from
  the unique non-pocket `LaneNo=1` link when present, then falls back to the
  unique non-pocket link with no non-pocket left neighbor.
- `LateralNodeGroupStage`: materializes each lateral link group's `from` and
  `to` endpoint node sets. For 2025 data, node `GroupID` is retained as a
  `node_group` key and later gives connection pairing priority.
- `JunctionStage`: seeds components from schema `junction_link` rows, promotes
  direction-compatible adjacent links and laterally related ordinary groups, adds
  bounded lateral-node bridge rows, then unions links by R/L pairs, exact shared
  endpoint nodes, z-filtered geometry intersections, promotion pairs, and bridge
  pairs. It does not merge nearby but topologically separate junctions solely
  because their endpoint nodes are close.
- `JunctionConnectionStage`: attaches external lateral node groups to the best
  junction by exact shared node first, then graph adjacency. Link groups already
  inside any junction are skipped. Per junction, opposite-side `to`/`from`
  endpoint groups are greedily paired when their nearest endpoint nodes are
  within `junction_connection_node_merge_dist_m` and their reference tangents are
  opposite by `junction_connection_opposite_direction_dot_min`; unpaired groups
  remain one-sided connections.
- `JunctionReferenceStage`: chooses one reference link for each connection from
  the connection's lateral link groups. It uses the selected group reference
  link, keeps only links whose endpoint belongs to the connection, allows
  outbound `from` references only for outbound-only one-sided connections, and
  prefers lower `LaneNo`, then lower group ID, then feature ID. Reference
  tangents keep the source geometry direction.
- `JunctionEdgeStage`: turns a junction reference into a cross-section segment.
  It orients the reference tangent outward from the junction centroid, finds the
  farthest connection node in that outward direction, intersects connection link
  geometries with the perpendicular station line, and falls back to projected
  endpoint nodes when link crossings are unavailable.

Version shortcuts are declared by the version schema. Data model 2023 uses
`FromNodeID` / `ToNodeID`, `R/L_LinkID`, `LaneNo`, `NodeType`, `LinkType`, and
lane-line `Kind`. Data model 2025 uses `FromNodeID` / `ToNodeID`,
`R/L_LinkID`, `Turn`, expanded `LinkType`, multi-`NodeType`, lane-line
`LineKind`, and node `GroupID`. `ITSNodeID` is loaded as source data only and is
not used for segmentation.

Renderer note: the current PyVista/VTK viewport remains the default. Viewport
interaction profiling is log-only and configured under `viz.profiling`; use
those timings before evaluating a dedicated 2D renderer.
