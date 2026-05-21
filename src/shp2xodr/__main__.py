"""Hydra entry point: ``uv run python -m shp2xodr [dataset=...]``."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from shp2xodr.viz import A2Viz, RawViz, SegmentsViz

log = logging.getLogger(__name__)

_VIZ_MODES: dict[str, type[A2Viz]] = {
    "raw": RawViz,
    "segments": SegmentsViz,
}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    shp_dir = Path(cfg.dataset.shp_dir)
    log.info("dataset=%s  shp_dir=%s  viz=%s", cfg.dataset.name, shp_dir, cfg.viz)
    viz_cls = _VIZ_MODES.get(cfg.viz)
    if viz_cls is None:
        msg = f"unknown viz mode {cfg.viz!r}; expected one of {sorted(_VIZ_MODES)}"
        raise ValueError(msg)
    viz_cls(shp_dir).show()


if __name__ == "__main__":
    main()
