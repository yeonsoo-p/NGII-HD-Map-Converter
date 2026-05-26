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

The inspector dock in [src/ngii2xodr/ngii/gui.py](src/ngii2xodr/ngii/gui.py)
has three tabs: `Layers`, `Items`, and `Selected`. Both shift-click selection
and item search resolve through `FeatureRef(layer_attr, feature_id)`, so the
3D highlight, item tree, and selected table all point at the same canonical
feature object.

Segmentation stages live under
[src/ngii2xodr/ngii/segmentation](src/ngii2xodr/ngii/segmentation). The pipeline
currently enables:

- `NodeLinkRelationsStage`: A1 nodes annotated with incoming and outgoing A2
  link references.
- `UTurnStage`: Type 6 A2 links intersecting B2 Kind 502 markers within the
  configured z tolerance.
- `LateralLinkGroupStage`: ordinary Type 6 A2 links grouped by R/L_LinkID
  lateral references, excluding links already classified as U-turns.
- `LateralNodeGroupStage`: separate from/to A1 endpoint groups for each lateral
  link group.
- `JunctionStage`: connected components of Type 1 A2 links, joined by
  R/L_LinkID references or same-plane geometric intersections.

Post-junction stages are explicit pipeline stages but default to disabled in
Hydra until their dataset-native semantics are ready.

Renderer note: the current PyVista/VTK viewport remains the default for this
pass. If 2D inspection becomes the bottleneck, profile viewport interaction
first and introduce a small viewport interface before evaluating a dedicated
2D renderer.
