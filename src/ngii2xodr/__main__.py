"""Hydra entry point: ``uv run python -m ngii2xodr``.

The window opens empty by default; select a section directory via
**File → Open NGII folder…** (Ctrl+O). Pass ``ngii_dir=/path/to/section`` on
the CLI (or set it in ``conf/config.yaml``) to auto-load on startup.

Runtime config parsing lives in :mod:`ngii2xodr.config`; this module only wires
Hydra, Qt, and startup folder loading.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PySide6.QtWidgets import QApplication

from ngii2xodr.config import build_runtime_config
from ngii2xodr.gui import HdMapWindow


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HdMapWindow(runtime_cfg=build_runtime_config(cfg))
    window.show()
    if cfg.ngii_dir is not None:
        # to_absolute_path resolves against the invocation cwd, not Hydra's
        # per-run output dir, which is what the user means by a relative path.
        window.load_folder(Path(to_absolute_path(str(cfg.ngii_dir))).expanduser())
    app.exec()


if __name__ == "__main__":
    # Windows stdio defaults to cp1252; UTF-8 makes logging non-ASCII-safe.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
