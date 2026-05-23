from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from numpy.typing import NDArray

from shp2xodr.__main__ import _build_seg_cfg
from shp2xodr.shp.data import A2Data, B2Data
from shp2xodr.shp.gui import _field, _flatten_selected_sections, _raw_section, _SelectedTableRow
from shp2xodr.shp.segmentation import (
    JunctionStage,
    Segmentation,
    SegmentationConfig,
    SegmentationInput,
    SelectedField,
    SelectedSection,
    UTurnStage,
    segmentation_level_labels,
)


def _str_array(values: list[str]) -> NDArray[np.str_]:
    return np.asarray(values, dtype=np.str_)


def _line(points: list[tuple[float, float, float]]) -> NDArray[np.float64]:
    return np.asarray(points, dtype=np.float64)


def _input(
    *,
    a2_ids: list[str],
    a2_link_types: list[str],
    a2_polylines: list[NDArray[np.float64]],
    b2_ids: list[str] | None = None,
    b2_kinds: list[str] | None = None,
    b2_polylines: list[NDArray[np.float64]] | None = None,
    a2_r_link_ids: list[str] | None = None,
    a2_l_link_ids: list[str] | None = None,
    z_tol: float = 2.0,
    proximity_tol: float = 5.0,
) -> SegmentationInput:
    n_links = len(a2_ids)
    return SegmentationInput(
        a2_ids=_str_array(a2_ids),
        a2_link_types=_str_array(a2_link_types),
        a2_r_link_ids=_str_array(a2_r_link_ids or [""] * n_links),
        a2_l_link_ids=_str_array(a2_l_link_ids or [""] * n_links),
        a2_polylines=tuple(a2_polylines),
        b2_ids=_str_array(b2_ids or []),
        b2_kinds=_str_array(b2_kinds or []),
        b2_polylines=tuple(b2_polylines or []),
        cfg=SegmentationConfig(
            z_intersection_tol_m=z_tol,
            junction_proximity_merge_dist_m=proximity_tol,
        ),
    )


def test_uturn_stage_detects_type6_a2_crossing_b2_502_within_z_tolerance() -> None:
    data = _input(
        a2_ids=["a"],
        a2_link_types=["6"],
        a2_polylines=[_line([(0, 0, 0), (10, 0, 0)])],
        b2_ids=["m"],
        b2_kinds=["502"],
        b2_polylines=[_line([(5, -1, 0), (5, 1, 0)])],
    )

    result = UTurnStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0]
    assert len(result.entities) == 1
    assert result.entities[0].link_ids == ("a",)
    assert result.entities[0].selected_fields() == (
        SelectedField.scalar("U-turn", 0),
        SelectedField.list("U-turn marker ID", ("m",)),
    )


def test_uturn_stage_ignores_wrong_type_wrong_marker_and_z_mismatch() -> None:
    data = _input(
        a2_ids=["wrong_type", "wrong_marker", "high_z"],
        a2_link_types=["5", "6", "6"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(0, 10, 0), (10, 10, 0)]),
            _line([(0, 20, 5), (10, 20, 5)]),
        ],
        b2_ids=["m502_a", "m503", "m502_b"],
        b2_kinds=["502", "503", "502"],
        b2_polylines=[
            _line([(5, -1, 0), (5, 1, 0)]),
            _line([(5, 9, 0), (5, 11, 0)]),
            _line([(5, 19, 0), (5, 21, 0)]),
        ],
        z_tol=2.0,
    )

    result = UTurnStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [-1, -1, -1]
    assert result.entities == ()


def test_junction_stage_unions_type1_links_through_lateral_references() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_r_link_ids=["b", ""],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(0, 10, 0), (10, 10, 0)]),
        ],
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 0]
    assert len(result.entities) == 1
    assert result.entities[0].link_ids == ("a", "b")


def test_junction_stage_unions_type1_links_through_same_plane_intersection() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(5, -5, 0), (5, 5, 0)]),
        ],
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 0]
    assert len(result.entities) == 1


def test_junction_stage_does_not_union_over_z_threshold() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(5, -5, 5), (5, 5, 5)]),
        ],
        z_tol=2.0,
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 1]
    assert len(result.entities) == 2


def test_junction_stage_keeps_singleton_type1_link_as_junction() -> None:
    data = _input(
        a2_ids=["a"],
        a2_link_types=["1"],
        a2_polylines=[_line([(0, 0, 0), (10, 0, 0)])],
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0]
    assert len(result.entities) == 1
    assert result.entities[0].selected_fields() == (
        SelectedField.scalar("Junction", 0),
        SelectedField.list("Junction link ID", ("a",)),
    )


def test_junction_stage_proximity_merges_components_within_distance_and_z_tol() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(13, 0, 0), (20, 0, 0)]),
        ],
        proximity_tol=5.0,
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 0]
    assert len(result.entities) == 1


def test_junction_stage_proximity_merge_respects_z_tolerance() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(13, 0, 5), (20, 0, 5)]),
        ],
        z_tol=2.0,
        proximity_tol=5.0,
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 1]
    assert len(result.entities) == 2


def test_junction_stage_proximity_merge_respects_distance() -> None:
    data = _input(
        a2_ids=["a", "b"],
        a2_link_types=["1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(16, 0, 0), (20, 0, 0)]),
        ],
        proximity_tol=5.0,
    )

    result = JunctionStage().run(data, {})

    assert result.entity_id_per_link.tolist() == [0, 1]
    assert len(result.entities) == 2


def test_stage_registry_and_selected_fields_are_incremental_contract() -> None:
    data = _input(
        a2_ids=["uturn", "j1", "j2"],
        a2_link_types=["6", "1", "1"],
        a2_polylines=[
            _line([(0, 0, 0), (10, 0, 0)]),
            _line([(0, 10, 0), (10, 10, 0)]),
            _line([(5, 5, 0), (5, 15, 0)]),
        ],
        b2_ids=["marker"],
        b2_kinds=["502"],
        b2_polylines=[_line([(5, -1, 0), (5, 1, 0)])],
    )
    uturn_result = UTurnStage().run(data, {})
    junction_result = JunctionStage().run(data, {"uturn": uturn_result})
    segmentation = Segmentation(stage_results=(uturn_result, junction_result))

    assert segmentation_level_labels() == ("Raw", "U-turns", "Junctions")
    assert [result.stage_id for result in segmentation.active_results(2)] == [
        "uturn",
        "junction",
    ]
    assert segmentation.selected_fields_for_link(0) == (
        SelectedField.scalar("U-turn", 0),
        SelectedField.list("U-turn marker ID", ("marker",)),
    )
    assert segmentation.selected_fields_for_link(1) == (
        SelectedField.scalar("Junction", 0),
        SelectedField.list("Junction link ID", ("j1", "j2")),
    )


def test_selected_table_sections_and_list_values_flatten_to_multiple_rows() -> None:
    rows = _flatten_selected_sections(
        (
            _raw_section((_field("ID", "a"), _field("LinkType", "1"))),
            SelectedSection(
                "Segmentation",
                (
                    SelectedField.scalar("Junction", 0),
                    SelectedField.list("Junction link ID", ("j1", "j2")),
                    SelectedField.list("U-turn marker ID", ("m1", "m2")),
                ),
            ),
        )
    )

    assert rows == [
        _SelectedTableRow("Raw", "", True),
        _SelectedTableRow("ID", "a"),
        _SelectedTableRow("LinkType", "1"),
        _SelectedTableRow("Segmentation", "", True),
        _SelectedTableRow("Junction", "0"),
        _SelectedTableRow("Junction link ID", "j1"),
        _SelectedTableRow("Junction link ID", "j2"),
        _SelectedTableRow("U-turn marker ID", "m1"),
        _SelectedTableRow("U-turn marker ID", "m2"),
    ]


def test_configured_section_edge_case_links_share_one_junction_entity() -> None:
    ids = ("A2218I541856", "A2218I542034", "A2218I542316", "A2218I541855")
    config_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")
    shp_dir = Path(to_absolute_path(str(cfg.shp_dir))).expanduser()
    a2 = A2Data(shp_dir)
    b2 = B2Data(shp_dir)
    segmentation = Segmentation.from_layers(a2, b2, _build_seg_cfg(cfg))
    junction = segmentation.result("junction")
    id_to_idx = {str(link_id): i for i, link_id in enumerate(a2.ids)}
    junction_ids = {int(junction.entity_id_per_link[id_to_idx[link_id]]) for link_id in ids}

    assert junction_ids == {next(iter(junction_ids))}
    assert next(iter(junction_ids)) >= 0


def test_app_owned_select_naming_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    searched_paths = [
        root / "src" / "shp2xodr" / "__main__.py",
        root / "src" / "shp2xodr" / "shp" / "gui.py",
        root / "src" / "shp2xodr" / "shp" / "viz.py",
        root / "conf" / "config.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in searched_paths)

    assert "selecter" not in text
    assert "_PICK_" not in text
    assert "picker_tol" not in text
