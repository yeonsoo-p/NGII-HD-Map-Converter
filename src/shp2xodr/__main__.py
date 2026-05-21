"""Hydra entry point: ``uv run python -m shp2xodr [dataset=...]``."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from shp2xodr.viz import HdMapViz

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    shp_dir = Path(cfg.dataset.shp_dir)
    log.info("dataset=%s  shp_dir=%s", cfg.dataset.name, shp_dir)
    HdMapViz(shp_dir, junction_merge_dist_m=cfg.segmentation.junction_merge_dist_m).show()


if __name__ == "__main__":
    main()
