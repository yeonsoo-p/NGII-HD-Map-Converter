"""Hydra entry point: ``uv run python -m shp2xodr [dataset=...]``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PySide6.QtWidgets import QApplication

from shp2xodr.shp.gui import HdMapWindow

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    shp_dir = Path(cfg.dataset.shp_dir)
    log.info("dataset=%s  shp_dir=%s", cfg.dataset.name, shp_dir)
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(shp_dir, junction_merge_dist_m=cfg.segmentation.junction_merge_dist_m)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
