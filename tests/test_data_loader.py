"""Loader-helper unit tests.

The SHP loaders see real NGII data with the occasional ``MultiLineString`` or
``MultiPolygon`` row produced by upstream GIS pipelines (QGIS / ArcGIS) — even
when the row only carries a single part. These tests pin down the unwrap +
reject contract independently of any real SHP file.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely

from shp2xodr.shp.data import _outer_rings_from_gdf, _polylines_from_gdf


def _line_gdf(geom: shapely.geometry.base.BaseGeometry, row_id: str = "L001") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"ID": [row_id]}, geometry=[geom])


def _poly_gdf(geom: shapely.geometry.base.BaseGeometry, row_id: str = "P001") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"ID": [row_id]}, geometry=[geom])


# ---- _polylines_from_gdf -------------------------------------------------------


def test_polylines_pass_through_plain_linestring() -> None:
    line = shapely.LineString([(0.0, 0.0), (1.0, 2.0), (3.0, 4.0)])
    [out] = _polylines_from_gdf(_line_gdf(line))
    np.testing.assert_array_equal(out, np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]))


def test_polylines_unwrap_single_part_multilinestring() -> None:
    inner = shapely.LineString([(0.0, 0.0), (5.0, 5.0)])
    [out] = _polylines_from_gdf(_line_gdf(shapely.MultiLineString([inner])))
    np.testing.assert_array_equal(out, np.array([[0.0, 0.0], [5.0, 5.0]]))


def test_polylines_merge_connected_multipart() -> None:
    """3-part MLS that chains endpoint-to-endpoint merges to one polyline."""
    mls = shapely.MultiLineString(
        [
            shapely.LineString([(0.0, 0.0), (1.0, 0.0)]),
            shapely.LineString([(1.0, 0.0), (2.0, 0.0)]),
            shapely.LineString([(2.0, 0.0), (3.0, 0.0)]),
        ]
    )
    [out] = _polylines_from_gdf(_line_gdf(mls))
    np.testing.assert_array_equal(out, np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))


def test_polylines_merge_connected_multipart_preserves_z() -> None:
    """Same chain in 3-D — Z must survive the merge so OpenDRIVE elevations stay intact."""
    mls = shapely.MultiLineString(
        [
            shapely.LineString([(0.0, 0.0, 10.0), (1.0, 0.0, 11.0)]),
            shapely.LineString([(1.0, 0.0, 11.0), (2.0, 0.0, 12.0)]),
        ]
    )
    [out] = _polylines_from_gdf(_line_gdf(mls))
    np.testing.assert_array_equal(
        out, np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 11.0], [2.0, 0.0, 12.0]])
    )


def test_polylines_merge_sub_cm_drift() -> None:
    """Parts with sub-cm endpoint drift (NGII roundtripping noise) snap-then-merge.

    Pinned by real SEC01_송파대로 data: row A2196I000499 has a 27 mm gap
    between the shared endpoint of its two sub-LineStrings. 3 cm here.
    """
    mls = shapely.MultiLineString(
        [
            shapely.LineString([(0.0, 0.0), (1.0, 0.0)]),
            shapely.LineString([(1.03, 0.0), (2.0, 0.0)]),
        ]
    )
    [out] = _polylines_from_gdf(_line_gdf(mls))
    assert out.shape[0] >= 3, f"expected single merged polyline, got shape {out.shape}"
    np.testing.assert_array_equal(out[0], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(out[-1], np.array([2.0, 0.0]))


def test_polylines_reject_disconnected_multipart() -> None:
    """Sub-lines that don't share endpoints can't be merged → row-id-tagged ValueError."""
    mls = shapely.MultiLineString(
        [shapely.LineString([(0.0, 0.0), (1.0, 0.0)]), shapely.LineString([(5.0, 0.0), (6.0, 0.0)])]
    )
    with pytest.raises(ValueError, match=r"row ID='L042'.*could not be merged"):
        _polylines_from_gdf(_line_gdf(mls, row_id="L042"))


def test_polylines_reject_wrong_type() -> None:
    with pytest.raises(TypeError, match=r"row ID='L007'.*Point"):
        _polylines_from_gdf(_line_gdf(shapely.Point(1.0, 2.0), row_id="L007"))


# ---- _outer_rings_from_gdf -----------------------------------------------------


def test_outer_rings_pass_through_plain_polygon() -> None:
    poly = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    [out] = _outer_rings_from_gdf(_poly_gdf(poly))
    # Exterior ring closes on itself — shapely repeats the first vertex at end.
    np.testing.assert_array_equal(
        out, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    )


def test_outer_rings_unwrap_single_part_multipolygon() -> None:
    inner = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    [out] = _outer_rings_from_gdf(_poly_gdf(shapely.MultiPolygon([inner])))
    np.testing.assert_array_equal(
        out, np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]])
    )


def test_outer_rings_reject_true_multipart() -> None:
    mp = shapely.MultiPolygon(
        [
            shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
            shapely.Polygon([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    )
    with pytest.raises(ValueError, match=r"row ID='P042'.*2 parts"):
        _outer_rings_from_gdf(_poly_gdf(mp, row_id="P042"))


def test_outer_rings_reject_wrong_type() -> None:
    with pytest.raises(TypeError, match=r"row ID='P007'.*LineString"):
        _outer_rings_from_gdf(
            _poly_gdf(shapely.LineString([(0.0, 0.0), (1.0, 1.0)]), row_id="P007")
        )
