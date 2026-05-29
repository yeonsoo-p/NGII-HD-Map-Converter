from __future__ import annotations

import logging
import sys
from typing import cast

import hydra
from omegaconf import DictConfig
from PySide6.QtWidgets import QApplication

from ngii.gui import NgiiViewerWindow

logger = logging.getLogger(__name__)


def _configure_terminal_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    logging.getLogger("ngii").setLevel(logging.INFO)


def run_gui(coordinate: str) -> int:
    _configure_terminal_logging()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    window = NgiiViewerWindow(coordinate=coordinate)
    window.show()
    logger.info("Started NGII viewer with coordinate set %s", coordinate)

    if owns_app:
        return cast(QApplication, app).exec()
    return 0


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    coordinate = str(cfg.get("coordinate", "HDMap_UTMK_정표고"))
    raise SystemExit(run_gui(coordinate))


if __name__ == "__main__":
    main()
