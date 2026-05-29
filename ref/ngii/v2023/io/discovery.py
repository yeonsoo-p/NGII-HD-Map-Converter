from __future__ import annotations

from pathlib import Path

from ngii.v2023.dataset import DataSet
from ngii.v2023.section import Section


def discover_dataset(dataset_root: Path, coordinate: str) -> DataSet:
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    dataset = DataSet(path=dataset_root, coordinate=coordinate, sections=[])
    for section_path in sorted(dataset_root.iterdir(), key=lambda path: path.name):
        if not section_path.is_dir() or not section_path.name.startswith("SEC"):
            continue

        coordinate_path = section_path / coordinate
        if not coordinate_path.is_dir():
            continue

        dataset.sections.append(
            Section(
                dataset=dataset,
                name=section_path.name,
                path=section_path,
                coordinate_dir=coordinate_path,
            )
        )

    return dataset
