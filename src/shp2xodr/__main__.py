"""Hydra entry point: ``uv run python -m shp2xodr``.

The window opens empty by default; pick a section directory via
**File → Open SHP folder…** (Ctrl+O). Pass ``shp_dir=/path/to/section`` on
the CLI (or set it in ``conf/config.yaml``) to auto-load on startup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PySide6.QtWidgets import QApplication

from shp2xodr.shp.gui import HdMapWindow

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(
        junction_merge_dist_m=cfg.segmentation.junction_merge_dist_m,
        bidirectional_merge_max_separation_m=cfg.segmentation.bidirectional_merge_max_separation_m,
    )
    window.show()
    if cfg.shp_dir is not None:
        # to_absolute_path resolves against the invocation cwd, not Hydra's
        # per-run output dir, which is what the user means by a relative path.
        window.load_folder(Path(to_absolute_path(str(cfg.shp_dir))).expanduser())
    app.exec()


if __name__ == "__main__":
    main()
