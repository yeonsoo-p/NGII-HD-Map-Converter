# shp2xodr - NGII HD-map -> OpenDRIVE

Converter workspace for Korean National Geographic Information Institute (NGII)
precise road map SHP layers and ASAM OpenDRIVE 1.7 experiments.

The current application is a Qt + PyVista inspector for NGII section folders.
It renders loaded A1 / A2 / A3 / A4 / B2 / C3 layers in 3D, supports layer
visibility toggles, and lets you shift-click features to inspect source
attributes plus staged A2 segmentation results.

## Run

```shell
uv sync
uv run python -m shp2xodr                          # opens empty; File -> Open SHP folder...
uv run python -m shp2xodr shp_dir=/path/to/SEC###  # auto-load a section
```

Configuration lives in [conf/config.yaml](conf/config.yaml). Relative
`shp_dir` values resolve against the invocation cwd, not Hydra's run directory.

## Inspector And Segmentation

`HdMapViz` in [src/shp2xodr/shp/viz.py](src/shp2xodr/shp/viz.py) loads typed
records from [src/shp2xodr/shp/data.py](src/shp2xodr/shp/data.py) and builds
selectable PyVista actors:

- A1 nodes render as black points.
- A2 links render in one neutral color at level 0, then gain cumulative
  segmentation coloring by stage.
- B2 surface line marks render by paint color.
- C3 safety fixtures render by facility type.
- A3 / A4 polygons render by their NGII attribute color tables when present.

The inspector dock in [src/shp2xodr/shp/gui.py](src/shp2xodr/shp/gui.py)
contains layer toggles, a segmentation level selector, and a selected-feature
table. The selected table separates raw NGII fields from A2 segmentation
details; list values such as junction member links are shown as repeated rows.

Segmentation stages live in
[src/shp2xodr/shp/segmentation.py](src/shp2xodr/shp/segmentation.py). The
code registry currently runs:

- `UTurnStage`: Type 6 A2 links intersecting B2 Kind 502 markers within the
  configured z tolerance.
- `JunctionStage`: connected components of Type 1 A2 links, joined by
  R/L_LinkID references, same-plane geometric intersections, or near same-plane
  component proximity within the configured gap distance.

Adding or removing a stage is done in the code registry; the GUI selector is
generated from that registry.
