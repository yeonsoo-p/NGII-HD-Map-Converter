from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ngii2xodr"


def test_shared_app_code_does_not_import_version_packages() -> None:
    checked_roots = (
        SRC / "gui",
        SRC / "ngii" / "segmentation",
        SRC / "ngii" / "viz.py",
        SRC / "ngii" / "app.py",
    )
    offenders = [
        f"{path.relative_to(ROOT)}:{module}"
        for path in _python_files(checked_roots)
        for module in _imported_modules(path)
        if module.startswith(
            (
                "ngii2xodr.ngii.data.v2023",
                "ngii2xodr.ngii.data.v2025",
            )
        )
    ]
    assert offenders == []


def test_layer_modules_do_not_import_sibling_layer_modules() -> None:
    layer_roots = (
        SRC / "ngii" / "data" / "v2023" / "layers",
        SRC / "ngii" / "data" / "v2025" / "layers",
    )
    offenders = [
        f"{path.relative_to(ROOT)}:{module}"
        for path in _python_files(layer_roots)
        if path.name != "__init__.py"
        for module in _imported_modules(path)
        if ".layers." in module
    ]
    assert offenders == []


def _python_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(sorted(path.rglob("*.py")))
    return tuple(files)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)
