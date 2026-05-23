from __future__ import annotations

import importlib.util

from omegaconf import OmegaConf

from shp2xodr.__main__ import _build_viz_cfg


def test_segmentation_module_is_removed() -> None:
    assert importlib.util.find_spec("shp2xodr.shp.segmentation") is None


def test_viz_config_has_only_raw_inspector_fields() -> None:
    cfg = OmegaConf.create(
        {
            "viz": {
                "node_point_size": 3.0,
                "poly_opacity": 0.85,
                "poly_depth_offset_factor": 2.0,
                "poly_depth_offset_units": 2.0,
                "line_width_a2": 2.5,
                "line_width_b2": 1.4,
                "line_width_c3": 1.8,
                "line_width_highlight": 4.5,
                "picker_tol_a1": 0.01,
                "picker_tol_a2": 0.005,
                "picker_tol_thin": 0.012,
                "picker_tol_poly": 0.0,
                "background_color": [0.86, 0.88, 0.90],
                "highlight_rgb": [255, 30, 200],
                "a2_uniform_rgb": [110, 110, 120],
                "a3_road_type_rgb": {"1": [200, 200, 200]},
                "a3_protected_rgb": [220, 80, 80],
                "a3_fallback_rgb": [180, 180, 180],
                "a4_subtype_rgb": {"1": [120, 200, 120]},
                "a4_fallback_rgb": [180, 180, 180],
                "b2_paint_rgb": {"1": [245, 235, 0]},
                "b2_paint_fallback_rgb": [130, 130, 130],
                "c3_type_rgb": {"2": [140, 140, 150]},
                "c3_type_fallback_rgb": [140, 140, 140],
            }
        }
    )

    viz_cfg = _build_viz_cfg(cfg)

    assert viz_cfg.a2_uniform_rgb == (110, 110, 120)
    assert not hasattr(viz_cfg, "default_abstraction_level")
    assert not hasattr(viz_cfg, "group_palette_seed")
