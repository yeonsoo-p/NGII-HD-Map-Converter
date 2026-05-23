# shp2xodr - NGII HD-map -> OpenDRIVE

Converter workspace for Korean National Geographic Information Institute (NGII)
precise road map SHP layers and ASAM OpenDRIVE 1.7 experiments.

The current application is a raw Qt + PyVista inspector for NGII section
folders. It renders loaded A1 / A2 / A3 / A4 / B2 / C3 layers in 3D, supports
layer visibility toggles, and lets you shift-click features to inspect their
source attributes.

## Run

```shell
uv sync
uv run python -m shp2xodr                          # opens empty; File -> Open SHP folder...
uv run python -m shp2xodr shp_dir=/path/to/SEC###  # auto-load a section
```

Configuration lives in [conf/config.yaml](conf/config.yaml). Relative
`shp_dir` values resolve against the invocation cwd, not Hydra's run directory.

## Raw Inspector

`HdMapViz` in [src/shp2xodr/shp/viz.py](src/shp2xodr/shp/viz.py) loads typed
records from [src/shp2xodr/shp/data.py](src/shp2xodr/shp/data.py) and builds
pickable PyVista actors:

- A1 nodes render as black points.
- A2 links render in one neutral color for raw link inspection.
- B2 surface line marks render by paint color.
- C3 safety fixtures render by facility type.
- A3 / A4 polygons render by their NGII attribute color tables when present.

The inspector dock in [src/shp2xodr/shp/gui.py](src/shp2xodr/shp/gui.py)
contains layer toggles and a picked-feature table. No segmentation, grouping,
junction, road, or U-turn model is active in this version.
