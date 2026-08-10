"""Tests for road rasterisation (the prediction targets) and the NLNI land use adapter."""

from __future__ import annotations

import numpy as np
import pytest

from japgo.core import Edge, Node, RoadGraph, load_hierarchy
from japgo.geo import SHIZUOKA, Bounds, Raster
from japgo.pipeline.rasterize import (
    distance_to,
    road_class_raster,
    road_mask,
    road_orientation,
)
from japgo.sources import load_landuse_spec

CRS = SHIZUOKA.crs
BOUNDS = Bounds(0, 0, 200, 200)
RES = 1.0


def _graph(*specs: tuple[str, str, list[tuple[float, float]]]) -> RoadGraph:
    g = RoadGraph(crs=CRS.to_string())
    for eid, road_class, geometry in specs:
        u, v = f"{eid}u", f"{eid}v"
        g.add_node(Node(id=u, x=geometry[0][0], y=geometry[0][1]))
        g.add_node(Node(id=v, x=geometry[-1][0], y=geometry[-1][1]))
        g.add_edge(Edge(id=eid, u=u, v=v, geometry=geometry, road_class=road_class))
    return g


# ---------------------------------------------------------------------------------------------
# Road mask
# ---------------------------------------------------------------------------------------------


def test_road_mask_burns_the_carriageway():
    g = _graph(("e", "residential", [(0, 100), (200, 100)]))
    mask = road_mask(g, BOUNDS, RES, CRS)
    assert mask.data.sum() > 0
    assert set(np.unique(mask.data)) <= {0.0, 1.0}


def test_wider_roads_burn_wider():
    """A model trained on hairlines loses road width, which spec §38 lists as a metric."""
    narrow = road_mask(_graph(("e", "service", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    wide = road_mask(_graph(("e", "expressway", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    assert wide.data.sum() > narrow.data.sum() * 3


def test_hairline_mode_ignores_width():
    g = _graph(("e", "expressway", [(0, 100), (200, 100)]))
    wide = road_mask(g, BOUNDS, RES, CRS, use_width=True)
    thin = road_mask(g, BOUNDS, RES, CRS, use_width=False)
    assert thin.data.sum() < wide.data.sum()


def test_narrow_road_still_registers():
    """all_touched: dropping sub-cell roads would teach the model minor roads do not exist."""
    g = _graph(("e", "path", [(0, 100), (200, 100)]))
    mask = road_mask(g, BOUNDS, 5.0, CRS)  # 5 m cells, ~1.5 m road
    assert mask.data.sum() > 0


def test_empty_graph_yields_an_empty_mask():
    mask = road_mask(RoadGraph(crs=CRS.to_string()), BOUNDS, RES, CRS)
    assert mask.data.sum() == 0
    assert mask.data.shape == (200, 200)


# ---------------------------------------------------------------------------------------------
# Road class
# ---------------------------------------------------------------------------------------------


def test_class_raster_ranks_significance():
    expressway = road_class_raster(
        _graph(("e", "expressway", [(0, 100), (200, 100)])), BOUNDS, RES, CRS
    )
    service = road_class_raster(
        _graph(("e", "service", [(0, 100), (200, 100)])), BOUNDS, RES, CRS
    )
    assert expressway.data.max() > service.data.max()


def test_significant_road_wins_at_a_crossing():
    """Painting in dictionary order would make the raster depend on insertion order."""
    forward = _graph(
        ("minor", "service", [(100, 0), (100, 200)]),
        ("major", "expressway", [(0, 100), (200, 100)]),
    )
    reverse = _graph(
        ("major", "expressway", [(0, 100), (200, 100)]),
        ("minor", "service", [(100, 0), (100, 200)]),
    )
    a = road_class_raster(forward, BOUNDS, RES, CRS)
    b = road_class_raster(reverse, BOUNDS, RES, CRS)
    assert np.array_equal(a.data, b.data)
    assert a.data[100, 100] == pytest.approx(a.data.max())


def test_class_raster_is_zero_off_road():
    r = road_class_raster(_graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    assert r.data[0, 0] == 0.0


# ---------------------------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------------------------


def test_orientation_is_axis_not_direction():
    """A road and its reverse must encode identically: direction is meaningless, axis is not."""
    forward = _graph(("e", "residential", [(0, 100), (200, 100)]))
    backward = _graph(("e", "residential", [(200, 100), (0, 100)]))

    fs, fc = road_orientation(forward, BOUNDS, RES, CRS)
    bs, bc = road_orientation(backward, BOUNDS, RES, CRS)

    assert np.allclose(fs.data, bs.data, atol=1e-5)
    assert np.allclose(fc.data, bc.data, atol=1e-5)


def test_perpendicular_roads_encode_differently():
    ew, _ = road_orientation(_graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    _, ns_cos = road_orientation(
        _graph(("e", "residential", [(100, 0), (100, 200)])), BOUNDS, RES, CRS
    )
    _, ew_cos = road_orientation(
        _graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS
    )
    on_road = ew.data != 0
    assert not np.allclose(ns_cos.data[on_road], ew_cos.data[on_road])


def test_orientation_is_zero_off_road():
    s, c = road_orientation(_graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    assert s.data[0, 0] == 0.0
    assert c.data[0, 0] == 0.0


def test_orientation_of_empty_graph_is_zero():
    s, c = road_orientation(RoadGraph(crs=CRS.to_string()), BOUNDS, RES, CRS)
    assert s.data.sum() == 0
    assert c.data.sum() == 0


# ---------------------------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------------------------


def test_distance_to_road_is_zero_on_the_road():
    pytest.importorskip("scipy")
    mask = road_mask(_graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    d = distance_to(mask)
    assert d.data[mask.data > 0].max() == 0.0


def test_distance_grows_away_from_the_road():
    pytest.importorskip("scipy")
    mask = road_mask(_graph(("e", "residential", [(0, 100), (200, 100)])), BOUNDS, RES, CRS)
    d = distance_to(mask)
    assert d.data[0, 100] > d.data[80, 100] > 0


def test_distance_is_in_metres_not_cells():
    pytest.importorskip("scipy")
    mask = Raster(np.zeros((100, 100), np.float32), Bounds(0, 0, 500, 500), CRS)  # 5 m cells
    mask.data[50, 50] = 1.0
    d = distance_to(mask)
    assert d.data[50, 60] == pytest.approx(50.0)  # 10 cells x 5 m


def test_empty_mask_yields_a_uniform_far_field():
    """The honest encoding of 'nothing of this kind anywhere near'."""
    pytest.importorskip("scipy")
    empty = Raster(np.zeros((50, 50), np.float32), BOUNDS, CRS)
    d = distance_to(empty, max_distance_m=500.0)
    assert np.all(d.data == 500.0)


def test_distance_is_clamped():
    pytest.importorskip("scipy")
    mask = Raster(np.zeros((200, 200), np.float32), BOUNDS, CRS)
    mask.data[0, 0] = 1.0
    assert distance_to(mask, max_distance_m=10.0).data.max() == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------
# Land use config
# ---------------------------------------------------------------------------------------------


def test_every_nlni_code_maps_to_a_declared_class():
    spec = load_landuse_spec()
    for code, entry in spec.nlni_codes.items():
        assert entry["class"] in spec.classes, f"code {code} maps to undeclared class"


def test_every_channel_group_member_is_a_declared_class():
    spec = load_landuse_spec()
    for group, members in spec.channel_groups.items():
        for member in members:
            assert member in spec.classes, f"{group} references undeclared class {member}"


def test_channel_groups_are_disjoint():
    """A class in two groups would double-count coverage."""
    spec = load_landuse_spec()
    seen: set[str] = set()
    for members in spec.channel_groups.values():
        overlap = seen & set(members)
        assert not overlap, f"classes in more than one channel group: {overlap}"
        seen |= set(members)


def test_channel_count_respects_the_memory_budget():
    """Twelve one-hot channels would cost ~12 MB per sample against a 16 GB budget (§20.2)."""
    assert len(load_landuse_spec().channel_names) <= 5


def test_known_codes_resolve():
    spec = load_landuse_spec()
    assert spec.class_for("0700") == ("built_up", True)
    assert spec.class_for("0500") == ("forest", True)
    assert spec.class_for("0100") == ("paddy", True)
    assert spec.class_for("9999") == ("unknown", False)


def test_classes_map_to_their_groups():
    spec = load_landuse_spec()
    assert spec.group_for("built_up") == "landuse_built"
    assert spec.group_for("forest") == "landuse_forest"
    assert spec.group_for("golf") is None  # folds into implicit "other"


def test_road_hierarchy_and_landuse_agree_on_road_representation():
    """Land use has a `road` class and the hierarchy has road classes; they must not be confused.

    Land use `road` is a mesh land-cover category; the road *network* is a graph. Keeping them
    separate is what stops the model being handed its own target as an input.
    """
    assert "road" in load_landuse_spec().classes
    assert "road" not in load_hierarchy().classes
