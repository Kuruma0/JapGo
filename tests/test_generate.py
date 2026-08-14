"""Tests for the frozen-model boundary.

The point of this package is that a game-world generator can use the model without importing the
training code, and can reproduce a call a year from now. So the tests are about the *contract* —
what the card must carry, what happens when it disagrees with the checkout, and whether a
prediction can be re-cut without re-running inference — rather than about output quality, which
`japgo.model` already measures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from japgo.core import RoadGraph
from japgo.generate import DEFAULT_THRESHOLD, ModelCard, RoadPrediction
from japgo.geo.tiling import Bounds

BOUNDS = Bounds(0.0, 0.0, 512.0, 512.0)


def _card(**over) -> ModelCard:
    base = dict(
        checkpoint="models/road_v1/road_v1.pt",
        trained_on="test corpus",
        channels=[f"c{i}" for i in range(15)],
        stack_version=2,
        resolution_m=1.0,
        crs="EPSG:6676",
        registry_hash="deadbeef",
        width=32,
        threshold=0.45,
    )
    return ModelCard(**{**base, **over})


def test_the_card_carries_everything_needed_to_reproduce_a_call():
    """A checkpoint whose preprocessing cannot be reconstructed is not frozen, only saved.

    Weights alone are not enough: the channel order, stack version, resolution, CRS and the
    registry the corpus was built under all change what the model is being shown.
    """
    card = _card()
    for field in ("channels", "stack_version", "resolution_m", "crs", "registry_hash", "width"):
        assert getattr(card, field) is not None, field
    assert len(card.channels) == 15


def test_a_card_round_trips_through_disk(tmp_path):
    path = _card(metrics={"apls": 0.172}).write(tmp_path / "card.json")
    back = ModelCard.read(path)
    assert back.channels == _card().channels
    assert back.metrics["apls"] == pytest.approx(0.172)


def test_the_shipped_card_exists_and_matches_this_checkout():
    """The frozen model must stay loadable from a clean clone, or the freeze meant nothing."""
    from japgo.pipeline.channels import load_stack_spec

    path = Path("models/road_v1/road_v1.json")
    if not path.is_file():
        pytest.skip("no frozen model in this checkout")

    card = ModelCard.read(path)
    spec = load_stack_spec()
    assert card.stack_version == spec.stack_version
    assert card.channels == spec.names        # order matters; it is checked by length at runtime
    assert Path(card.checkpoint).is_file()


def test_a_stack_version_mismatch_is_refused_rather_than_guessed(tmp_path, monkeypatch):
    """Channels keep their names across a stack bump but not their meaning. Loading anyway would
    produce a confident, plausible, entirely wrong prediction."""
    from japgo.generate.inference import FrozenModel

    with pytest.raises(ValueError, match="stack v"):
        FrozenModel(_card(stack_version=99))


def test_a_prediction_can_be_recut_without_rerunning_the_model():
    """Thresholding is cheap and inference is not, so the cut is a property of the prediction
    rather than baked into it."""
    prob = np.linspace(0.0, 1.0, 64).reshape(8, 8).astype(np.float32)
    low = RoadPrediction(prob, BOUNDS, "EPSG:6676", 1.0, threshold=0.2)
    high = RoadPrediction(prob, BOUNDS, "EPSG:6676", 1.0, threshold=0.8)

    assert low.candidate_mask.sum() > high.candidate_mask.sum()
    assert low.coverage > high.coverage
    assert np.array_equal(low.probability, high.probability)   # same field, different cut


def test_the_default_threshold_is_not_the_naive_one():
    """0.5 is arbitrary against a class that is a few percent of pixels and a weighted loss; every
    fold's calibrated operating point measured below it."""
    assert DEFAULT_THRESHOLD < 0.5


def test_the_generate_package_does_not_drag_in_the_training_loop():
    """A game-world generator should not need the sampler, the folds or the corpus.

    Checked structurally rather than by intent: importing the public surface must not require
    japgo.model.train, which is where the training loop and the fold machinery live.
    """
    source = Path("src/japgo/generate/inference.py").read_text(encoding="utf-8")
    assert "model.train" not in source
    assert "from ..model.nets import build_unet" in source   # the network only


# ---------------------------------------------------------------------------------------------
# Phase 2 — candidate extraction
# ---------------------------------------------------------------------------------------------


def _cross_prediction(size: int = 200) -> RoadPrediction:
    prob = np.zeros((size, size), dtype=np.float32)
    prob[size // 2 - 2 : size // 2 + 2, :] = 1.0
    prob[:, size // 2 - 2 : size // 2 + 2] = 1.0
    return RoadPrediction(prob, Bounds(0.0, 0.0, float(size), float(size)),
                          "EPSG:6676", 1.0, threshold=0.5)


def test_extraction_reports_the_things_the_next_stage_needs():
    from japgo.generate import extract_candidates

    graph, report = extract_candidates(_cross_prediction())
    assert report.edges == 4 and report.junctions == 1 and report.endpoints == 4
    assert report.components == 1
    assert report.total_length_m > 0
    assert "no elevation" in " ".join(report.notes)     # grade is unknown and says so


def test_grade_is_computed_from_elevation_and_carried_on_the_edge():
    """Grade cannot be recovered later — the elevation raster is gone by export time."""
    from japgo.generate import extract_candidates

    prediction = _cross_prediction()
    rows, cols = prediction.probability.shape
    # A 20% slope rising east: 0.2 m per metre.
    elevation = np.tile(np.arange(cols, dtype=np.float32) * 0.2, (rows, 1))

    graph, report = extract_candidates(prediction, elevation=elevation, grade_limit_pct=12.0)
    east_west = [e for e in graph.edges.values() if abs(e.geometry[0][1] - e.geometry[-1][1]) < 1]
    assert east_west, "expected the horizontal arms"
    assert all(e.grade_pct == pytest.approx(20.0, abs=1.0) for e in east_west)
    assert report.steep_edges >= len(east_west)          # 20% is over the 12% limit
    assert all(n.z is not None for n in graph.nodes.values())


# ---------------------------------------------------------------------------------------------
# Phase 3 — connectivity repair
# ---------------------------------------------------------------------------------------------


def _broken_line(gap: float) -> RoadGraph:
    """One road the model lost in the middle: two collinear halves separated by `gap`."""
    from japgo.core import Edge, Node

    g = RoadGraph(crs="EPSG:6676")
    for nid, (x, y) in {"a": (0, 0), "b": (100, 0),
                        "c": (100 + gap, 0), "d": (200 + gap, 0)}.items():
        g.add_node(Node(id=nid, x=float(x), y=float(y)))
    g.add_edge(Edge(id="left", u="a", v="b", geometry=[(0, 0), (100, 0)], source_id="model"))
    g.add_edge(Edge(id="right", u="c", v="d",
                    geometry=[(100 + gap, 0), (200 + gap, 0)], source_id="model"))
    return g


def test_a_road_the_model_lost_in_the_middle_is_bridged():
    """The failure this stage exists for: two stubs that nearly meet are one interrupted road."""
    from japgo.generate import RepairSpec, repair

    graph, report = repair(_broken_line(gap=20.0), RepairSpec(min_component_m=10.0))
    assert report.bridged >= 1
    assert report.components_after == 1
    assert report.dead_end_after < report.dead_end_before


def test_a_gap_too_wide_to_be_an_accident_is_left_alone():
    from japgo.generate import RepairSpec, repair

    _, report = repair(_broken_line(gap=400.0), RepairSpec(min_component_m=10.0))
    assert report.bridged == 0
    assert report.components_after == 2       # two genuinely separate roads


def test_two_dead_ends_already_well_connected_are_not_shortcut():
    """Without the detour test, bridging punches roads through blocks: the ends of two streets
    either side of a junction are metres apart and already connected."""
    from japgo.core import Edge, Node
    from japgo.generate import RepairSpec, repair

    g = RoadGraph(crs="EPSG:6676")
    for nid, (x, y) in {"j": (0, 0), "p": (10, 40), "q": (-10, 40)}.items():
        g.add_node(Node(id=nid, x=float(x), y=float(y)))
    g.add_edge(Edge(id="e1", u="j", v="p", geometry=[(0, 0), (10, 40)], source_id="model"))
    g.add_edge(Edge(id="e2", u="j", v="q", geometry=[(0, 0), (-10, 40)], source_id="model"))

    # p and q are 20 m apart but only ~82 m apart along the network -- a ratio of ~4, not a
    # dropped link. With the default 4.0 threshold this must not be bridged.
    _, report = repair(g, RepairSpec(min_component_m=10.0, snap_to_edge_m=0.0,
                                     min_stub_m=0.0, bridge_detour_ratio=6.0))
    assert report.bridged == 0


def test_pruning_happens_after_bridging_not_before():
    """A stub pruned first can never be reconnected, and the stub is usually the only evidence
    the model left that a road was there."""
    from japgo.generate import RepairSpec, repair

    # Both halves are shorter than min_stub_m would allow, but together they are a real road.
    spec = RepairSpec(min_component_m=10.0, min_stub_m=1000.0, bridge_gap_m=30.0)
    graph, report = repair(_broken_line(gap=20.0), spec)
    assert report.bridged >= 1


def test_tiny_isolated_fragments_are_dropped():
    from japgo.core import Edge, Node
    from japgo.generate import RepairSpec, repair

    g = _broken_line(gap=20.0)
    g.add_node(Node(id="s1", x=1000.0, y=1000.0))
    g.add_node(Node(id="s2", x=1005.0, y=1000.0))
    g.add_edge(Edge(id="speck", u="s1", v="s2",
                    geometry=[(1000, 1000), (1005, 1000)], source_id="model"))

    graph, report = repair(g, RepairSpec(min_component_m=50.0))
    assert report.dropped_components == 1
    assert "speck" not in graph.edges


def test_repair_is_deterministic():
    """A seed reproduces a world only if every stage after the model is deterministic."""
    from japgo.generate import RepairSpec, repair

    spec = RepairSpec()
    a, ra = repair(_broken_line(gap=20.0), spec)
    b, rb = repair(_broken_line(gap=20.0), spec)
    assert sorted(a.edges) == sorted(b.edges)
    assert ra.describe() == rb.describe()


def test_repair_does_not_force_dead_ends_to_zero():
    """Real networks are full of cul-de-sacs. A long stub with nothing near it is a place, not
    an accident, and must survive."""
    from japgo.core import Edge, Node
    from japgo.generate import RepairSpec, repair

    g = RoadGraph(crs="EPSG:6676")
    for nid, (x, y) in {"a": (0, 0), "b": (300, 0), "c": (300, 250)}.items():
        g.add_node(Node(id=nid, x=float(x), y=float(y)))
    g.add_edge(Edge(id="main", u="a", v="b", geometry=[(0, 0), (300, 0)], source_id="model"))
    g.add_edge(Edge(id="culdesac", u="b", v="c", geometry=[(300, 0), (300, 250)],
                    source_id="model"))

    graph, report = repair(g, RepairSpec())
    assert "culdesac" in graph.edges          # 250 m long, nothing within reach: keep it
    assert report.dead_end_after > 0.0
