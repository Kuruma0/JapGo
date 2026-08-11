"""Tests for graph extraction and the topology metrics.

The properties that matter are the ones pixel F1 cannot see. A severed network scores well
per-pixel and badly on APLS, and that gap is the reason §16.2 puts topology first — so the tests
are built around constructing exactly that situation and checking the metrics notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.core import Edge, Node, RoadGraph
from japgo.geo import SHIZUOKA
from japgo.geo.tiling import Bounds
from japgo.model.extract import ExtractionSpec, extract_graph
from japgo.model.topology import apls, compare, topo

CRS = SHIZUOKA.crs.to_string()
BOUNDS = Bounds(0.0, 0.0, 200.0, 200.0)


def _graph(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> RoadGraph:
    g = RoadGraph(crs=CRS)
    for i, (a, b) in enumerate(segments):
        for p in (a, b):
            nid = f"n{p[0]:.1f}_{p[1]:.1f}"
            if nid not in g.nodes:
                g.add_node(Node(id=nid, x=p[0], y=p[1]))
        g.add_edge(
            Edge(id=f"e{i}", u=f"n{a[0]:.1f}_{a[1]:.1f}", v=f"n{b[0]:.1f}_{b[1]:.1f}",
                 geometry=[a, b], source_id="osm")
        )
    return g


def _cross() -> RoadGraph:
    return _graph([
        ((0.0, 100.0), (100.0, 100.0)), ((100.0, 100.0), (200.0, 100.0)),
        ((100.0, 0.0), (100.0, 100.0)), ((100.0, 100.0), (100.0, 200.0)),
    ])


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


def test_a_cross_shaped_mask_becomes_a_four_way_junction():
    prob = np.zeros((200, 200), dtype=np.float32)
    prob[98:102, :] = 1.0
    prob[:, 98:102] = 1.0

    graph = extract_graph(prob, BOUNDS, CRS, resolution=1.0)

    assert len(graph.edges) == 4
    degrees = sorted(graph.degree(n) for n in graph.nodes)
    assert degrees == [1, 1, 1, 1, 4], degrees


def test_extraction_returns_an_empty_graph_rather_than_failing_on_a_blank_raster():
    graph = extract_graph(np.zeros((50, 50), dtype=np.float32), BOUNDS, CRS, resolution=4.0)
    assert len(graph.edges) == 0 and len(graph.nodes) == 0


def test_a_predicted_graph_is_never_mistaken_for_observed_geometry():
    """Invariant 3: source_id on every feature. A model's guess must not look like survey data."""
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[48:52, :] = 1.0
    graph = extract_graph(prob, BOUNDS, CRS, resolution=2.0)

    assert {e.source_id for e in graph.edges.values()} == {"model"}
    assert all(0.0 <= e.confidence <= 1.0 for e in graph.edges.values())


def test_simplification_removes_the_staircase():
    """A skeleton carries a vertex per pixel; unsimplified, every length and sinuosity is wrong."""
    prob = np.zeros((200, 200), dtype=np.float32)
    prob[98:102, :] = 1.0

    detailed = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                             spec=ExtractionSpec(simplify_m=0.0))
    simple = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                           spec=ExtractionSpec(simplify_m=2.0))

    detailed_pts = sum(len(e.geometry) for e in detailed.edges.values())
    simple_pts = sum(len(e.geometry) for e in simple.edges.values())
    assert simple_pts < detailed_pts
    # A straight line needs two points, whatever the pixel grid says.
    assert simple_pts <= 4


def test_a_split_junction_is_merged_back_into_one():
    """A crossroads skeletonises to two Y-junctions a few pixels apart. Left alone that doubles
    intersection density and halves mean degree — both §16.2 measures."""
    near = _graph([
        ((0.0, 100.0), (98.0, 100.0)), ((98.0, 100.0), (102.0, 100.0)),
        ((98.0, 100.0), (98.0, 0.0)), ((102.0, 100.0), (102.0, 200.0)),
    ])
    from japgo.model.extract import _merge_close_nodes

    merged = _merge_close_nodes(near, 6.0)
    assert len(merged.nodes) < len(near.nodes)
    assert max(merged.degree(n) for n in merged.nodes) >= 3


# ---------------------------------------------------------------------------------------------
# APLS and TOPO
# ---------------------------------------------------------------------------------------------


def test_a_graph_scores_perfectly_against_itself():
    cross = _cross()
    assert apls(cross, cross) == pytest.approx(1.0)
    f1, precision, recall = topo(cross, cross)
    assert f1 == pytest.approx(1.0)


def test_apls_falls_when_the_network_is_severed_but_the_pixels_are_nearly_right():
    """The whole reason topology is measured separately.

    Removing one link leaves almost every pixel in place, so a raster score barely moves — but
    routes across the break become impossible, and APLS is where that shows.

    The two directions say different things and the asymmetry is the point. Forward (truth ->
    proposal) is route *recall*: severing the eastern arm isolates one of five nodes, so four of
    ten truth pairs become unroutable and it drops to 0.6. Backward is route *precision*: the
    proposal never claims a route it cannot make, so it stays at 1.0. The reported figure is their
    mean, and a model that deletes the hard parts is caught by the forward term.
    """
    from japgo.model.topology import _apls_one_way

    truth = _cross()
    severed = _cross()
    del severed.edges["e1"]        # drop the eastern arm: one edge of four

    assert apls(truth, truth) == pytest.approx(1.0)

    forward = _apls_one_way(truth, severed, radius=25.0, samples=400, seed=0)
    backward = _apls_one_way(severed, truth, radius=25.0, samples=400, seed=0)
    assert forward == pytest.approx(0.6, abs=0.01)      # four of ten pairs unroutable
    assert backward == pytest.approx(1.0)               # it claims nothing it cannot deliver
    assert apls(truth, severed) == pytest.approx(0.8, abs=0.01)


def test_an_empty_proposal_scores_zero_rather_than_erroring():
    assert apls(_cross(), RoadGraph(crs=CRS)) == 0.0
    assert topo(_cross(), RoadGraph(crs=CRS)) == (0.0, 0.0, 0.0)


def test_a_displaced_network_still_matches_within_the_radius():
    """Registration error should not be scored as a topology failure."""
    truth = _cross()
    shifted = _graph([
        ((5.0, 105.0), (105.0, 105.0)), ((105.0, 105.0), (205.0, 105.0)),
        ((105.0, 5.0), (105.0, 105.0)), ((105.0, 105.0), (105.0, 205.0)),
    ])
    assert apls(truth, shifted, radius=25.0) > 0.9
    assert apls(truth, shifted, radius=2.0) < 0.2      # too tight: measures registration instead


def test_spurious_side_streets_cost_topo_precision():
    truth = _cross()
    noisy = _cross()
    for i, y in enumerate((20.0, 40.0, 60.0, 80.0)):
        noisy.add_node(Node(id=f"x{i}", x=140.0, y=y))
        noisy.add_edge(
            Edge(id=f"junk{i}", u="n100.0_100.0", v=f"x{i}",
                 geometry=[(100.0, 100.0), (140.0, y)], source_id="model")
        )
    clean_f1, clean_p, _ = topo(truth, truth)
    noisy_f1, noisy_p, _ = topo(truth, noisy)
    assert noisy_p < clean_p
    assert noisy_f1 < clean_f1


def test_compare_reports_the_match_counts_that_explain_the_scores():
    result = compare(_cross(), _cross())
    assert result.truth_nodes == result.proposal_nodes == 5
    assert result.matched == 5
    assert "APLS" in result.describe() and "TOPO" in result.describe()


def test_the_metrics_are_reproducible_from_their_seed():
    truth, proposal = _cross(), _cross()
    del proposal.edges["e3"]
    assert apls(truth, proposal, seed=7) == apls(truth, proposal, seed=7)


# ---------------------------------------------------------------------------------------------
# Over-painting defences
# ---------------------------------------------------------------------------------------------


def test_a_pinhole_does_not_become_a_ring():
    """The cheapest fix for junction inflation.

    A hole inside a painted blob survives thinning as a ring, and a ring contributes two junctions
    and two edges that no road produced. An over-painting model makes these by the hundred.
    """
    prob = np.zeros((200, 200), dtype=np.float32)
    prob[95:105, :] = 1.0
    prob[99:101, 100:102] = 0.0            # a 2x2 pinhole in an otherwise solid bar

    leaky = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                          spec=ExtractionSpec(fill_holes_px=0, prune_spur_m=0.0))
    filled = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                           spec=ExtractionSpec(fill_holes_px=64, prune_spur_m=0.0))

    assert len(filled.nodes) < len(leaky.nodes)
    assert len(filled.edges) <= 2          # a bar is one line, however it was punctured


def test_isolated_specks_are_dropped():
    """Specks of probability become unroutable two-node fragments that cost TOPO precision."""
    prob = np.zeros((200, 200), dtype=np.float32)
    prob[98:102, :] = 1.0
    for r, c in ((20, 20), (40, 150), (170, 60)):
        prob[r : r + 3, c : c + 3] = 1.0   # 3x3 specks, well under min_component_px

    kept = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                         spec=ExtractionSpec(min_component_px=0))
    cleaned = extract_graph(prob, BOUNDS, CRS, resolution=1.0,
                            spec=ExtractionSpec(min_component_px=48))
    assert len(cleaned.connected_components()) < len(kept.connected_components())


def test_spur_pruning_is_iterative():
    """Removing one spur exposes the next; a single pass leaves most of a frayed edge in place."""
    from japgo.model.extract import prune_spurs

    g = _graph([
        ((0.0, 100.0), (100.0, 100.0)), ((100.0, 100.0), (200.0, 100.0)),
        ((100.0, 100.0), (100.0, 108.0)),          # spur
        ((100.0, 108.0), (100.0, 114.0)),          # only reachable once the first goes
    ])
    once = prune_spurs(g, 20.0, iterations=1)
    thrice = prune_spurs(g, 20.0, iterations=3)

    assert len(once.edges) == 3          # outermost spur only
    assert len(thrice.edges) == 2        # both, leaving the through-road
    assert all(e.length_m >= 20.0 for e in thrice.edges.values())


def test_pruning_never_removes_a_through_road():
    """Only degree-1 edges are eligible: a spur carries no through traffic by definition."""
    from japgo.model.extract import prune_spurs

    g = _graph([((0.0, 100.0), (5.0, 100.0)), ((5.0, 100.0), (10.0, 100.0))])
    pruned = prune_spurs(g, 100.0, iterations=5)
    # Both edges are short, but the middle node keeps the chain connected until an end is cut.
    assert len(pruned.edges) < len(g.edges)
    assert len(pruned.connected_components()) <= len(g.connected_components())


def test_dice_punishes_painted_area_that_bce_tolerates():
    """The whole reason for adding it: BCE scores pixels independently, so extra painted area is
    only ever a small per-pixel cost. Dice is a ratio, so it responds to the total."""
    torch = pytest.importorskip("torch")
    from japgo.model.nets import masked_dice

    truth = torch.zeros(1, 1, 32, 32)
    truth[..., 15:17, :] = 1.0
    valid = torch.ones(1, 1, 32, 32)

    def logits_for(rows):
        z = torch.full((1, 1, 32, 32), -6.0)
        z[..., rows, :] = 6.0
        return z

    tight = masked_dice(logits_for(slice(15, 17)), truth, valid)
    bloated = masked_dice(logits_for(slice(10, 22)), truth, valid)
    assert tight < bloated                       # painting 6x the width costs more
    assert tight < 0.05                          # near-perfect overlap scores near zero
