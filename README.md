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

## Inspector And Segmentation

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

Segmentation stages live under
[src/ngii2xodr/ngii/segmentation](src/ngii2xodr/ngii/segmentation). The pipeline
builds road semantics in layers:

- `NodeLinkRelationsStage`: turns `FromNodeID` / `ToNodeID` into the directed
  graph at every node: which lane-center links enter the point and which leave
  it.
- `UTurnStage`: marks links whose geometry is a U-turn. Data model 2025 can use
  the link `Turn` attribute directly; data model 2023 falls back to links
  crossing U-turn lane-line markers.
- `LateralLinkGroupStage`: groups ordinary lane-center links that run side by
  side in the same travel direction using `R/L_LinkID`. Pockets are excluded
  from reference-lane selection; 2023 uses `LaneNo`, while 2025 uses `Turn`.
- `LateralNodeGroupStage`: materializes each lateral link group's `from` and
  `to` endpoint nodes. These are the physical mouths where a lane bundle enters
  or leaves a junction area.
- `JunctionStage`: finds the junction interior by grouping junction-type links
  that touch laterally, share endpoint nodes, meet the same lateral endpoint
  mouth, or cross in the same plane. Each one-way lateral endpoint mouth is
  allowed to connect to at most one junction.
- `JunctionConnectionStage`: attaches lateral endpoint mouths to a junction.
  Opposite-direction `to`/`from` mouths are paired into one bidirectional
  connection only when their endpoint link tangents face opposite directions;
  unmatched mouths remain visible as one-sided connections.
- `ConnectionReferenceStage`: emits reference arrows only from inbound endpoint
  links whose native source geometry already points toward the junction. Outbound
  one-sided connections are visible, but they do not get reversed arrows.

Version shortcuts are declared by the version schema. Data model 2023 uses
`FromNodeID` / `ToNodeID`, `R/L_LinkID`, `LaneNo`, `NodeType`, and `LinkType`.
Data model 2025 additionally provides reliable `GroupID`, multi-`NodeType`,
`Turn`, and expanded `LinkType` semantics. `ITSNodeID` is loaded as source data
only and is not used for segmentation.

Renderer note: the current PyVista/VTK viewport remains the default. Viewport
interaction profiling is log-only and configured under `viz.profiling`; use
those timings before evaluating a dedicated 2D renderer.
