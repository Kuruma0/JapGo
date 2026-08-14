"""Tests for Phase 6 geometry and the Phase 7 entry point.

The properties that matter here are the ones a game depends on and a metric cannot see:
smoothing must not undo the alignment grade enforcement just proved legal, junctions must not
drift apart, and the same seed must give the same world.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from japgo.core import Edge, Node, RoadGraph
from japgo.generate import GeometrySpec, build_geometry
from japgo.generate.geometry import _chaikin, _resample, _tangents


def _flat(x: float, y: float) -> float:
    return 0.0


def _staircase() -> RoadGraph:
    """The shape an extractor produces: a diagonal made of 8 m right angles."""
    pts = []
    for i in range(10):
        pts.append((i * 8.0, i * 8.0))
        pts.append(((i + 1) * 8.0, i * 8.0))
    g = RoadGraph(crs="EPSG:6676")
    g.add_node(Node(id="a", x=pts[0][0], y=pts[0][1]))
    g.add_node(Node(id="b", x=pts[-1][0], y=pts[-1][1]))
    g.add_edge(Edge(id="e", u="a", v="b", geometry=pts, source_id="model"))
    return g


def test_smoothing_softens_corners():
    """Road alignments do not have 8 m right angles.

    Measured as the *sharpest* turn at any vertex, not the total. Chaikin conserves total turning
    angle — it spreads the same rotation over more vertices — so a sum would show no change while
    the geometry visibly improves. The sharpest corner is what a driver and a mesh both feel.
    """
    pts = [p for p in _staircase().edges["e"].geometry]

    def sharpest(points):
        worst = 0.0
        for a, b, c in zip(points, points[1:], points[2:], strict=False):
            v1 = math.atan2(b[1] - a[1], b[0] - a[0])
            v2 = math.atan2(c[1] - b[1], c[0] - b[0])
            worst = max(worst, abs((v2 - v1 + math.pi) % (2 * math.pi) - math.pi))
        return worst

    before, after = sharpest(pts), sharpest(_chaikin(pts, 3, 6.0))
    assert before == pytest.approx(math.pi / 2, abs=0.01)   # right angles, as extracted
    assert after < before / 2


def test_smoothing_cannot_wander_off_the_validated_alignment():
    """Grade enforcement proved a specific alignment legal. A smoother free to move vertices
    arbitrarily undoes that proof silently — the road looks better and drives worse."""
    pts = [p for p in _staircase().edges["e"].geometry]
    smoothed = _chaikin(pts, 6, 3.0)

    for p in smoothed:
        assert min(math.dist(p, q) for q in pts) <= 3.0 + 1e-6


def test_endpoints_never_move():
    """They are junctions shared with other roads. A junction that drifts per incident edge
    stops being a junction."""
    pts = [p for p in _staircase().edges["e"].geometry]
    smoothed = _chaikin(pts, 5, 50.0)      # a leash long enough to permit drift
    assert smoothed[0] == pts[0]
    assert smoothed[-1] == pts[-1]


def test_resampling_is_evenly_spaced_and_keeps_the_ends():
    pts = _resample([(0.0, 0.0), (100.0, 0.0)], 10.0)
    gaps = [math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False)]
    assert pts[0] == (0.0, 0.0) and pts[-1] == (100.0, 0.0)
    assert max(gaps) <= 10.0 + 1e-6


def test_tangents_are_unit_length_and_follow_the_road():
    t = _tangents([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    assert all(abs(math.hypot(*v) - 1.0) < 1e-9 for v in t)
    assert t[0] == pytest.approx((1.0, 0.0))


def test_geometry_carries_width_elevation_and_grade():
    g = _staircase()
    g.edges["e"].grade_pct = 7.5
    splines, junctions = build_geometry(g, lambda x, y: 100.0, spec=GeometrySpec())

    s = splines[0]
    assert s.width_m > 0
    assert len(s.points) == len(s.elevations) == len(s.tangents)
    assert all(e > 100.0 for e in s.elevations)       # lifted clear of the terrain mesh
    assert s.grade_pct == 7.5
    assert junctions == []                            # a lone edge has no junction


def test_only_real_junctions_are_emitted():
    g = RoadGraph(crs="EPSG:6676")
    for nid, (x, y) in {"c": (0, 0), "n": (0, 50), "s": (0, -50), "e": (50, 0)}.items():
        g.add_node(Node(id=nid, x=float(x), y=float(y)))
    for i, other in enumerate(("n", "s", "e")):
        g.add_edge(Edge(id=f"e{i}", u="c", v=other,
                        geometry=[(0, 0), g.nodes[other].position], source_id="model"))

    _, junctions = build_geometry(g, _flat)
    assert [j.node_id for j in junctions] == ["c"]     # the three endpoints are not junctions
    assert junctions[0].degree == 3
    assert len(junctions[0].incident) == 3


# ---------------------------------------------------------------------------------------------
# Phase 7 — the entry point
# ---------------------------------------------------------------------------------------------


class _StubModel:
    """Stands in for the frozen model so the pipeline is testable without a checkpoint."""

    def __init__(self, spec, resolution_m=1.0):
        self.spec = spec
        from japgo.generate import ModelCard

        self.card = ModelCard(
            checkpoint="stub", trained_on="stub", channels=spec.names,
            stack_version=spec.stack_version, resolution_m=resolution_m,
            crs="EPSG:6676", registry_hash=None, width=32, threshold=0.5,
        )

    def predict(self, channels, bounds, threshold=None):
        from japgo.generate import RoadPrediction

        rows, cols = channels.shape[1:]
        prob = np.zeros((rows, cols), dtype=np.float32)
        prob[rows // 2 - 2 : rows // 2 + 2, :] = 1.0          # one straight road across
        return RoadPrediction(prob, bounds, "EPSG:6676", self.card.resolution_m,
                              threshold=threshold or 0.5)


def _world(size=200):
    from japgo.pipeline.channels import load_stack_spec

    spec = load_stack_spec()
    channels = np.zeros((spec.depth, size, size), dtype=np.float32)
    channels[spec.index_of("valid")] = 1.0
    return spec, channels


def test_the_entry_point_produces_a_network_and_explains_itself():
    from japgo.generate import GenerationParams, generate_roads
    from japgo.geo.tiling import Bounds

    spec, channels = _world()
    roads = generate_roads(_StubModel(spec), channels, Bounds(0.0, 0.0, 200.0, 200.0),
                           params=GenerationParams(seed=7))

    assert roads.splines and roads.total_length_m > 0
    assert roads.seed == 7
    assert roads.summary()["seed"] == 7
    # Every stage reported, so a disappointing world can be explained rather than guessed at.
    text = roads.diagnostics.describe()
    for stage in ("candidates", "repair", "validation", "terrain"):
        assert stage in text


def test_the_same_seed_and_inputs_give_the_same_world():
    """The property the whole design is arranged around."""
    from japgo.generate import GenerationParams, generate_roads
    from japgo.geo.tiling import Bounds

    spec, channels = _world()
    bounds = Bounds(0.0, 0.0, 200.0, 200.0)
    a = generate_roads(_StubModel(spec), channels, bounds, params=GenerationParams(seed=3))
    b = generate_roads(_StubModel(spec), channels, bounds, params=GenerationParams(seed=3))

    assert [s.points for s in a.splines] == [s.points for s in b.splines]
    assert a.summary() == b.summary()


def test_the_exported_bundle_is_engine_agnostic(tmp_path):
    """Invariant 1: the core emits an interchange format, never engine types. An adapter consumes
    this; it does not reach into the generator."""
    from japgo.generate import GenerationParams, export_bundle, generate_roads
    from japgo.geo.tiling import Bounds

    spec, channels = _world()
    roads = generate_roads(_StubModel(spec), channels, Bounds(0.0, 0.0, 200.0, 200.0),
                           params=GenerationParams(seed=1))
    out = export_bundle(roads, tmp_path / "world")

    geo = json.loads((out / "roads.geojson").read_text())
    assert geo["type"] == "FeatureCollection" and geo["features"]
    coords = geo["features"][0]["geometry"]["coordinates"]
    assert len(coords[0]) == 3, "coordinates must carry elevation"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 1 and "total_length_m" in manifest
    assert (out / "junctions.geojson").is_file()


def test_elevations_are_labelled_relative_unless_a_datum_is_given():
    """The stack's elevation channel is tile-relative by design -- raster_stack.yaml subtracts
    each tile's mean so the model learns slope, not altitude. Exporting it raw puts roads tens of
    metres underground, and an importer cannot tell by looking."""
    from japgo.generate import GenerationParams, generate_roads
    from japgo.geo.tiling import Bounds

    spec, channels = _world()
    channels[spec.index_of("elevation")] = -40.0        # a tile-relative field, as stored
    bounds = Bounds(0.0, 0.0, 200.0, 200.0)

    raw = generate_roads(_StubModel(spec), channels, bounds)
    assert raw.elevation_reference == "tile-relative"
    assert raw.summary()["elevation_reference"] == "tile-relative"
    assert raw.splines[0].elevations[0] < 0

    lifted = generate_roads(_StubModel(spec), channels, bounds,
                            params=GenerationParams(elevation_datum_m=500.0))
    assert lifted.elevation_reference == "absolute"
    assert lifted.splines[0].elevations[0] > 400


def test_grade_is_unaffected_by_the_datum():
    """A difference cancels the offset. Only absolute placement was ever wrong."""
    from japgo.generate import GenerationParams, generate_roads
    from japgo.geo.tiling import Bounds

    spec, channels = _world()
    bounds = Bounds(0.0, 0.0, 200.0, 200.0)
    a = generate_roads(_StubModel(spec), channels, bounds)
    b = generate_roads(_StubModel(spec), channels, bounds,
                       params=GenerationParams(elevation_datum_m=1000.0))
    assert [s.grade_pct for s in a.splines] == [s.grade_pct for s in b.splines]
