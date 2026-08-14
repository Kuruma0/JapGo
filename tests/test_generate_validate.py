"""Tests for Phase 4 validation and Phase 5 terrain enforcement.

The Phase 5 tests carry most of the weight. The sensitivity sweep showed the model does not put
roads where terrain says they belong, so this layer is not checking the ML's work — it is doing
the terrain reasoning the ML failed to do, and the switchback test is the proof it works.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from japgo.core import Edge, Node, RoadGraph
from japgo.generate import (
    TerrainSpec,
    ValidationSpec,
    enforce,
    enforce_grade,
    validate,
)
from japgo.geo.tiling import Bounds


def _graph(segments) -> RoadGraph:
    g = RoadGraph(crs="EPSG:6676")
    for i, (a, b) in enumerate(segments):
        for p in (a, b):
            nid = f"n{p[0]:.1f}_{p[1]:.1f}"
            if nid not in g.nodes:
                g.add_node(Node(id=nid, x=float(p[0]), y=float(p[1])))
        g.add_edge(Edge(id=f"e{i}", u=f"n{a[0]:.1f}_{a[1]:.1f}", v=f"n{b[0]:.1f}_{b[1]:.1f}",
                        geometry=[a, b], source_id="model"))
    return g


# ---------------------------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------------------------


def test_a_clean_graph_validates():
    g = _graph([((0, 0), (100, 0)), ((100, 0), (200, 0)), ((100, 0), (100, 100))])
    assert validate(g).valid


def test_duplicate_carriageways_are_an_error_and_are_removed():
    """The same road drawn twice: same endpoints, same bearing."""
    g = _graph([((0, 0), (100, 0))])
    g.add_edge(Edge(id="dup", u="n0.0_0.0", v="n100.0_0.0",
                    geometry=[(0, 0), (100, 0)], source_id="model"))

    assert not validate(g).valid
    cleaned, report = enforce(g)
    assert len(cleaned.edges) == 1 and report.valid


def test_a_skeleton_knot_is_reported_not_silently_collapsed():
    """Nine edges meeting at a point is an extraction artefact, but collapsing it changes the
    topology — that is the repair stage's call with its thresholds, not a validator's."""
    spokes = [((0, 0), (math.cos(i) * 50, math.sin(i) * 50)) for i in range(9)]
    report = validate(_graph(spokes), ValidationSpec(max_junction_degree=6))

    knot = [f for f in report.findings if f.check == "junction degree"]
    assert knot and knot[0].count == 1
    cleaned, _ = enforce(_graph(spokes))
    assert len(cleaned.nodes) == len(_graph(spokes).nodes)     # reported, not removed


def test_a_self_crossing_edge_is_an_error():
    g = RoadGraph(crs="EPSG:6676")
    g.add_node(Node(id="a", x=0.0, y=0.0))
    g.add_node(Node(id="b", x=100.0, y=0.0))
    g.add_edge(Edge(id="bow", u="a", v="b",
                    geometry=[(0, 0), (100, 50), (0, 50), (100, 0)], source_id="model"))
    assert not validate(g).valid


def test_disconnection_is_a_warning_not_an_error():
    """A vehicle cannot cross between components, but two real towns are legitimately separate."""
    g = _graph([((0, 0), (100, 0)), ((900, 900), (1000, 900))])
    report = validate(g)
    assert report.valid                                    # warning only
    assert any(f.check == "disconnected" for f in report.findings)


# ---------------------------------------------------------------------------------------------
# Phase 5 — the stage the sweep result made necessary
# ---------------------------------------------------------------------------------------------


def _hillside(size: int = 240, rise_per_m: float = 0.30) -> np.ndarray:
    """A uniform 30% slope rising north — far beyond the 12% road limit."""
    row = np.arange(size, dtype=np.float32)[:, None] * rise_per_m
    return np.tile(row[::-1], (1, size))


def test_a_road_straight_up_a_hillside_is_rerouted_not_deleted():
    """The whole argument for this stage.

    A real engineer facing a steep slope does not omit the road, they switchback up it. Deleting
    would give the mountain valley no roads at all, which is precisely the failure Kawanehon
    exists to expose.
    """
    bounds = Bounds(0.0, 0.0, 240.0, 240.0)
    g = _graph([((120, 20), (120, 220))])                  # straight up the fall line

    fixed, report = enforce_grade(g, _hillside(), bounds, 1.0, TerrainSpec(max_grade_pct=12.0))

    assert report.violations == 1
    assert report.rerouted == 1 and report.deleted == 0
    assert report.max_grade_after <= 12.0 + 1e-6
    assert report.added_length_m > 0                        # a legal route is necessarily longer


def test_the_reroute_actually_traverses_the_slope():
    """A switchback is what a least-cost path becomes when climbing directly is forbidden --
    nobody programs the zigzag, the constraint produces it."""
    bounds = Bounds(0.0, 0.0, 240.0, 240.0)
    g = _graph([((120, 20), (120, 220))])
    fixed, _ = enforce_grade(g, _hillside(), bounds, 1.0, TerrainSpec(max_grade_pct=12.0))

    route = next(iter(fixed.edges.values())).geometry
    lateral = max(abs(p[0] - 120.0) for p in route)
    assert lateral > 20.0, "the route climbed straight up instead of traversing"
    assert len(route) > 2


def test_a_gentle_road_is_left_exactly_as_it_was():
    bounds = Bounds(0.0, 0.0, 240.0, 240.0)
    flat = np.zeros((240, 240), dtype=np.float32)
    g = _graph([((20, 120), (220, 120))])
    before = next(iter(g.edges.values())).geometry

    fixed, report = enforce_grade(g, flat, bounds, 1.0)
    assert report.violations == 0 and report.rerouted == 0
    assert next(iter(fixed.edges.values())).geometry == before


def test_terrain_that_admits_no_road_deletes_the_edge_and_says_so():
    """A cliff is a cliff. Deleting is right here — and the report must not hide it."""
    bounds = Bounds(0.0, 0.0, 120.0, 120.0)
    cliff = np.zeros((120, 120), dtype=np.float32)
    cliff[:60, :] = 400.0                                   # a sheer 400 m wall across the middle
    g = _graph([((60, 20), (60, 100))])

    fixed, report = enforce_grade(g, cliff, bounds, 1.0, TerrainSpec(max_grade_pct=12.0))
    assert report.deleted == 1 and len(fixed.edges) == 0
    assert report.notes and "no route" in report.notes[0]


def test_grade_enforcement_is_deterministic():
    bounds = Bounds(0.0, 0.0, 240.0, 240.0)
    spec = TerrainSpec()
    a, ra = enforce_grade(_graph([((120, 20), (120, 220))]), _hillside(), bounds, 1.0, spec)
    b, rb = enforce_grade(_graph([((120, 20), (120, 220))]), _hillside(), bounds, 1.0, spec)
    assert [e.geometry for e in a.edges.values()] == [e.geometry for e in b.edges.values()]
    assert ra.describe() == rb.describe()
