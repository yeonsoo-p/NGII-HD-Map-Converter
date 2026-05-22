"""Hydra entry point: ``uv run python -m shp2xodr``.

The window opens empty; pick a section directory via **File → Open SHP
folder…** (Ctrl+O). The Hydra config now only carries segmentation tuning;
the dataset is no longer selected from YAML.
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig
from PySide6.QtWidgets import QApplication

from shp2xodr.shp.gui import HdMapWindow

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(junction_merge_dist_m=cfg.segmentation.junction_merge_dist_m)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
