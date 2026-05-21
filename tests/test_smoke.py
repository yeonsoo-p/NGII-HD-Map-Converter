"""Smoke test — verifies the package is importable and the version is set."""

import shp2xodr


def test_package_imports() -> None:
    assert shp2xodr.__version__ == "0.1.0"
