"""Tests for the Phase 3 analysis layer.

The statistical assertions here carry more weight than the arithmetic ones. Phase 3's output is a
ranked list of claims about the world, and the two ways it can be wrong without failing loudly are
both tested directly: manufacturing dead ends by clipping a graph at the tile boundary, and
manufacturing confidence by treating spatially autocorrelated tiles as independent samples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from japgo.analysis import (
    ENVIRONMENTAL_FEATURES,
    ROAD_STRUCTURE_METRICS,
    correlate,
    environmental_features,
    road_structure,
    spearman,
)
from japgo.analysis.correlate import MIN_CLUSTERS, NEGLIGIBLE_RHO, TileObservation
from japgo.analysis.features import core_view, coverage
from japgo.analysis.structure import _midpoint, split_by_core
from japgo.core import Building, Edge, Node, RoadGraph
from japgo.core.manifest import SourceRecord, SourceRole
from japgo.geo import SHIZUOKA, Raster, Tile
from japgo.pipeline import TileAssembler, TileInputs

TILE = Tile(zone=8, ix=10, iy=-110)
RES = 1.0
CRS = SHIZUOKA.crs.to_string()


# ---------------------------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------------------------


def test_spearman_is_one_for_a_perfect_monotone_relationship():
    x = np.arange(10, dtype=float)
    assert spearman(x, x * 3 + 1) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_sees_a_nonlinear_relationship_that_pearson_would_understate():
    """The reason the module uses rank correlation at all.

    Road density against slope is expected to fall steeply then flatten. A measure that assumes a
    straight line reports that as a weak relationship; it is a perfect monotone one.
    """
    x = np.arange(1, 11, dtype=float)
    y = 1.0 / x  # strictly decreasing, strongly curved

    centred_x, centred_y = x - x.mean(), y - y.mean()
    pearson = (centred_x * centred_y).sum() / math.sqrt(
        (centred_x**2).sum() * (centred_y**2).sum()
    )

    assert spearman(x, y) == pytest.approx(-1.0)
    assert abs(pearson) < 0.95


def test_spearman_averages_tied_ranks():
    # Ties must not invent an ordering: both orderings of the tied pair give the same answer.
    a = spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0, 40.0]))
    b = spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([10.0, 30.0, 20.0, 40.0]))
    assert a == pytest.approx(b)


def test_spearman_is_undefined_rather_than_zero_for_a_constant_input():
    """A constant column has no ranking. Reporting 0.0 would read as 'measured, no relationship'
    when the truth is 'unmeasurable'."""
    assert math.isnan(spearman(np.arange(5, dtype=float), np.ones(5)))


def test_spearman_drops_pairs_with_a_missing_side():
    x = np.array([1.0, 2.0, 3.0, np.nan])
    y = np.array([1.0, 2.0, 3.0, 100.0])
    assert spearman(x, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# Environmental features
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def bundle(gate):
    """A tile that slopes uniformly: 5% grade rising east, with nine houses on it."""
    bounds = TILE.read
    rows, cols = int(bounds.height / RES), int(bounds.width / RES)
    col = np.arange(cols, dtype=np.float32)[None, :]
    dem = Raster(np.tile(col * 0.05, (rows, 1)).astype(np.float32), bounds, SHIZUOKA.crs)

    cx, cy = TILE.core.centre
    buildings = [
        Building(
            id=f"b{i}{j}",
            source_id="plateau",
            footprint=[
                (cx + i * 60, cy + j * 60),
                (cx + i * 60 + 10, cy + j * 60),
                (cx + i * 60 + 10, cy + j * 60 + 10),
                (cx + i * 60, cy + j * 60 + 10),
                (cx + i * 60, cy + j * 60),
            ],
            height_m=8.0,
            coarse_type="residential",
            fine_type="detached_house",
        )
        for i in range(3)
        for j in range(3)
    ]

    inputs = TileInputs(
        elevation=dem,
        buildings=buildings,
        records=[
            SourceRecord(source_id="virtual_shizuoka", layers=["elevation"]),
            SourceRecord(source_id="plateau", layers=["buildings"]),
        ],
    )
    return TileAssembler(gate, resolution=RES).assemble(TILE, inputs)


def test_core_view_strips_the_halo(bundle):
    full = bundle.channel("slope")
    cropped = core_view(bundle, full)

    expected = int(round(bundle.tile.core.width / RES))
    assert cropped.shape == (expected, expected)
    assert full.shape[0] > cropped.shape[0]


def test_core_view_leaves_a_core_only_tile_alone(gate):
    """A core-only stack is already the core. Cropping it again would discard 56% of the ground
    and report the remainder as the whole tile — no error, just quietly wrong numbers.

    The shape cannot disambiguate this: 1000² is a 1 km core at 1 m or a read extent at 1.512 m.
    The bundle has to carry the answer.
    """
    bounds = TILE.core
    rows, cols = int(bounds.height / RES), int(bounds.width / RES)
    dem = Raster(np.zeros((rows, cols), dtype=np.float32), bounds, SHIZUOKA.crs)

    core_only = TileAssembler(gate, resolution=RES).assemble(
        TILE,
        TileInputs(
            elevation=dem,
            records=[SourceRecord(source_id="virtual_shizuoka", layers=["elevation"])],
        ),
        with_halo=False,
    )

    assert core_only.with_halo is False
    assert core_view(core_only, core_only.channel("slope")).shape == (rows, cols)


def test_the_halo_flag_survives_a_store_round_trip(gate, tmp_path):
    """Because the flag is only useful if it is still there when the study reads the corpus back."""
    from japgo.pipeline.store import read_tile, write_tile

    bounds = TILE.core
    rows, cols = int(bounds.height / RES), int(bounds.width / RES)
    dem = Raster(np.zeros((rows, cols), dtype=np.float32), bounds, SHIZUOKA.crs)
    bundle = TileAssembler(gate, resolution=RES).assemble(
        TILE,
        TileInputs(
            elevation=dem,
            records=[SourceRecord(source_id="virtual_shizuoka", layers=["elevation"])],
        ),
        with_halo=False,
    )

    write_tile(tmp_path, bundle)
    assert read_tile(tmp_path, TILE.id).with_halo is False


def test_features_are_reported_in_real_units_not_normalised_ones(bundle):
    """The stack stores slope divided by 50. A findings table reading 0.1 instead of 5% is not
    a table a geographer can check."""
    features = environmental_features(bundle)
    assert features["slope_median_pct"] == pytest.approx(5.0, abs=0.5)


def test_every_feature_in_the_vocabulary_is_present(bundle):
    features = environmental_features(bundle)
    assert tuple(features) == ENVIRONMENTAL_FEATURES


def test_uniform_slope_scores_high_aspect_alignment(bundle):
    """A single planar hillside faces one way, so the aspect vectors reinforce."""
    assert environmental_features(bundle)["aspect_alignment"] > 0.9


def test_built_fraction_counts_only_the_core(bundle):
    features = environmental_features(bundle)
    # Nine 10 m houses in a 1 km core: small but non-zero, and far below the whole-tile figure
    # would be if the halo were included in the denominator only.
    assert 0.0 < features["built_frac"] < 0.01
    assert features["building_height_mean_m"] == pytest.approx(8.0, abs=0.5)


def test_coverage_is_reported_for_quality_gating(bundle):
    assert coverage(bundle) == pytest.approx(1.0)


def test_voids_are_excluded_rather_than_read_as_zero(bundle):
    """nodata_fill is 0.0 and zero is a plausible slope. If fill leaked into the statistics every
    partly-covered tile would look flatter than it is."""
    valid_index = bundle.spec.index_of("valid")
    slope_index = bundle.spec.index_of("slope")

    # Columns chosen inside the core: a strip in the halo would be cropped away entirely, which
    # is correct behaviour but tests nothing about masking.
    observed = slice(400, 500)
    bundle.stack[valid_index][:, :] = 0.0
    bundle.stack[valid_index][:, observed] = 1.0
    bundle.stack[slope_index][:, observed] = 20.0 / 50.0  # 20%, normalised

    features = environmental_features(bundle)
    assert features["slope_median_pct"] == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------------------------
# Road structure
# ---------------------------------------------------------------------------------------------


def _node_id(point: tuple[float, float]) -> str:
    """Nodes are keyed by position, so two roads meeting at a point share one node.

    Without this the helper builds a graph where every endpoint is its own degree-1 node, and
    every topological metric measures the fixture instead of the network.
    """
    return f"n{point[0]:.3f}_{point[1]:.3f}"


def _line(graph: RoadGraph, name: str, a: tuple[float, float], b: tuple[float, float], **kw):
    for point in (a, b):
        node_id = _node_id(point)
        if node_id not in graph.nodes:
            graph.add_node(Node(id=node_id, x=point[0], y=point[1]))
    graph.add_edge(
        Edge(
            id=name,
            u=_node_id(a),
            v=_node_id(b),
            geometry=kw.pop("geometry", [a, b]),
            road_class=kw.pop("road_class", "residential"),
            source_id="osm",
            **kw,
        )
    )


def test_midpoint_is_by_distance_not_by_vertex_mean():
    """The mistake docs/decision-log.md records making once already. Vertex density varies with
    curvature, so a vertex mean drifts toward the bendy end of a polyline."""
    geometry = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (103.0, 0.0)]

    vertex_mean_x = sum(p[0] for p in geometry) / len(geometry)
    assert _midpoint(geometry)[0] == pytest.approx(51.5)
    assert vertex_mean_x == pytest.approx(21.8)


def test_a_road_continuing_into_the_halo_is_not_counted_as_a_dead_end():
    """The methodological point of the whole module.

    Clipping the graph at the core boundary turns every through-road into a cul-de-sac, and it
    would do so more in sparse mountain tiles than dense plain ones — biasing the exact contrast
    Phase 3 measures. The halo exists so this does not happen (invariant 6).
    """
    cx, cy = TILE.core.centre
    half = TILE.core.width / 2

    graph = RoadGraph(crs=CRS)
    # Runs from the middle of the core out through the boundary into the halo.
    _line(graph, "through", (cx, cy), (cx + half + 200, cy))
    # A genuine cul-de-sac, entirely inside the core.
    _line(graph, "stub", (cx, cy), (cx, cy + 200))

    metrics = road_structure(graph, TILE)
    split = split_by_core(graph, TILE.core)

    # The node out in the halo is not a core node, so it does not vote.
    assert _node_id((cx + half + 200, cy)) not in split.core_node_ids
    # Of the core nodes, only the stub's far end is a true dead end; the shared origin has
    # degree 2 and the boundary crossing keeps its neighbour.
    assert metrics["dead_end_ratio"] == pytest.approx(0.5)


def test_density_uses_the_core_area_not_the_read_area():
    cx, cy = TILE.core.centre
    graph = RoadGraph(crs=CRS)
    _line(graph, "e", (cx - 250, cy), (cx + 250, cy))  # 500 m inside a 1 km core

    metrics = road_structure(graph, TILE)
    assert metrics["road_density_km_per_km2"] == pytest.approx(0.5, abs=1e-6)


def test_a_grid_and_an_organic_network_separate_on_orientation_entropy():
    """The discriminator docs/decision-log.md measured on real data: Atami 0.872, grid town
    0.281. If these ever converge, the measure has stopped working."""
    cx, cy = TILE.core.centre

    grid = RoadGraph(crs=CRS)
    for i in range(4):
        offset = (i - 2) * 100
        _line(grid, f"h{i}", (cx - 300, cy + offset), (cx + 300, cy + offset))
        _line(grid, f"v{i}", (cx + offset, cy - 300), (cx + offset, cy + 300))

    # Enough distinct bearings to fill most of the 36 orientation bins. Eight rays would occupy
    # eight bins and score ~0.58 — high for a grid, but not what "organic" means here.
    organic = RoadGraph(crs=CRS)
    for i in range(24):
        angle = i * math.pi / 24
        _line(
            organic,
            f"r{i}",
            (cx, cy),
            (cx + 300 * math.cos(angle), cy + 300 * math.sin(angle)),
        )

    assert road_structure(grid, TILE)["orientation_entropy"] < 0.4
    assert road_structure(organic, TILE)["orientation_entropy"] > 0.8


def test_sinuosity_separates_a_straight_road_from_a_winding_one():
    cx, cy = TILE.core.centre

    straight = RoadGraph(crs=CRS)
    _line(straight, "s", (cx - 200, cy), (cx + 200, cy))

    winding = RoadGraph(crs=CRS)
    zigzag = [(cx - 200 + i * 40, cy + (40 if i % 2 else -40)) for i in range(11)]
    _line(winding, "w", zigzag[0], zigzag[-1], geometry=zigzag)

    assert road_structure(straight, TILE)["sinuosity_median"] == pytest.approx(1.0, abs=1e-6)
    assert road_structure(winding, TILE)["sinuosity_median"] > 1.4


def test_a_tile_with_no_road_data_is_nan_not_zero():
    """A tile we did not measure and a tile with no roads are different facts. Only one of them
    should be allowed to drag a correlation."""
    metrics = road_structure(None, TILE)
    assert tuple(metrics) == ROAD_STRUCTURE_METRICS
    assert all(math.isnan(v) for v in metrics.values())


# ---------------------------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------------------------


def _observations(n_per_site: int, sites: int, *, relationship: bool, seed: int = 1):
    """Tiles whose road density either does or does not track slope."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(sites):
        for i in range(n_per_site):
            slope = float(rng.uniform(1, 40))
            density = 20.0 - 0.4 * slope if relationship else float(rng.uniform(2, 18))
            out.append(
                TileObservation(
                    tile_id=f"s{s}_t{i}",
                    site=f"site{s}",
                    features={"slope_median_pct": slope},
                    metrics={"road_density_km_per_km2": density},
                )
            )
    return out


PAIR = (("slope_median_pct",), ("road_density_km_per_km2",))


def test_a_real_relationship_is_reported_as_supported():
    study = correlate(
        _observations(12, MIN_CLUSTERS, relationship=True),
        features=PAIR[0],
        metrics=PAIR[1],
        iterations=400,
    )
    (found,) = study.associations
    assert found.rho < -0.9
    assert found.verdict == "supported"
    assert found.ci is not None and found.ci[1] < 0


def test_no_relationship_is_reported_as_a_stated_null_not_an_omission():
    """Phase 3's exit criterion requires the null results. A feature that does not matter has to
    appear in the table saying so."""
    study = correlate(
        _observations(12, MIN_CLUSTERS, relationship=False),
        features=PAIR[0],
        metrics=PAIR[1],
        iterations=400,
    )
    (found,) = study.associations
    assert found.verdict == "null"
    assert found.ci is not None and found.ci[0] < 0 < found.ci[1]
    assert "nothing there (1)" in study.report()


def test_a_wide_interval_is_inconclusive_rather_than_null():
    """The distinction the report exists to preserve.

    A strong point estimate whose interval spans zero has established nothing — but it has not
    established *absence* either. Filing it under 'null' would let 'we need more sites' read as
    'this feature does not matter', which is the opposite conclusion.
    """
    # Sites that disagree on the *sign*. Pooled, they look like little; resampled, a draw of
    # three site-0s says +1 and a draw of three site-1s says -1. The honest answer is that three
    # sites cannot settle it — which is a different statement from "slope does not matter".
    slopes = [float(i * 5 + 1) for i in range(8)]
    directions = {0: +1.0, 1: -1.0, 2: -1.0}

    observations = [
        TileObservation(
            tile_id=f"s{s}_t{i}",
            site=f"site{s}",
            features={"slope_median_pct": slope},
            metrics={"road_density_km_per_km2": 12.0 + direction * 0.4 * slope},
        )
        for s, direction in directions.items()
        for i, slope in enumerate(slopes)
    ]

    study = correlate(observations, features=PAIR[0], metrics=PAIR[1], iterations=600)
    (found,) = study.associations

    assert found.ci is not None
    spans_zero = found.ci[0] < 0 < found.ci[1]
    wide = max(abs(found.ci[0]), abs(found.ci[1])) > NEGLIGIBLE_RHO
    assert spans_zero and wide
    assert found.verdict == "inconclusive"
    assert "NOT a null result" in study.report()


def test_too_few_sites_yields_insufficient_rather_than_false_confidence():
    """Twelve tiles from one site are not twelve independent observations — they share a
    settlement and a 256 m halo. Invariant 4's logic, applied to inference."""
    study = correlate(
        _observations(12, 1, relationship=True),
        features=PAIR[0],
        metrics=PAIR[1],
        iterations=400,
    )
    (found,) = study.associations
    assert found.verdict == "insufficient"
    assert found.ci is None
    assert found.rho < -0.9  # the estimate still exists; only the confidence does not


def test_the_cluster_bootstrap_is_wider_than_a_naive_one_would_be():
    """Resampling sites rather than tiles is the entire honesty mechanism. If it ever produced
    the tighter interval, it would have stopped doing its job."""
    observations = _observations(10, MIN_CLUSTERS, relationship=True, seed=7)
    study = correlate(observations, features=PAIR[0], metrics=PAIR[1], iterations=600)
    (found,) = study.associations
    assert found.ci is not None

    x = np.array([o.features["slope_median_pct"] for o in observations])
    y = np.array([o.metrics["road_density_km_per_km2"] for o in observations])
    rng = np.random.default_rng(0)
    naive = [
        spearman(x[idx], y[idx])
        for idx in (rng.integers(0, x.size, size=x.size) for _ in range(600))
    ]
    naive_width = float(np.percentile(naive, 97.5) - np.percentile(naive, 2.5))

    assert (found.ci[1] - found.ci[0]) >= naive_width


def test_a_study_is_reproducible_from_its_seed():
    """Invariant 8: an experiment that cannot be re-run from its config is a failed experiment."""
    observations = _observations(10, MIN_CLUSTERS, relationship=True)
    a = correlate(observations, features=PAIR[0], metrics=PAIR[1], iterations=300, seed=42)
    b = correlate(observations, features=PAIR[0], metrics=PAIR[1], iterations=300, seed=42)
    assert a.associations[0].ci == b.associations[0].ci


def test_ranking_puts_the_strongest_association_first():
    observations = []
    for s in range(MIN_CLUSTERS):
        for i in range(10):
            slope = float(i * 4 + s)
            observations.append(
                TileObservation(
                    tile_id=f"s{s}_t{i}",
                    site=f"site{s}",
                    features={"slope_median_pct": slope, "landuse_water_frac": 0.1},
                    metrics={"road_density_km_per_km2": 20.0 - 0.4 * slope},
                )
            )

    study = correlate(
        observations,
        features=("landuse_water_frac", "slope_median_pct"),
        metrics=("road_density_km_per_km2",),
        iterations=200,
    )
    assert study.ranked()[0].feature == "slope_median_pct"


def test_an_empty_corpus_produces_an_empty_study_rather_than_an_error():
    study = correlate([])
    assert study.tiles == 0
    assert study.associations == []
    assert "0 tiles" in study.report()


# ---------------------------------------------------------------------------------------------
# Reading a corpus off disk
# ---------------------------------------------------------------------------------------------

COARSE = 8.0
"""Neighbourhood resolution — enough to exercise the store without 1512² arrays per tile."""


def _write_corpus(gate, root, tiles, *, with_roads=True):
    from japgo.pipeline.store import write_tile

    for tile in tiles:
        bounds = tile.read
        rows, cols = int(bounds.height / COARSE), int(bounds.width / COARSE)
        col = np.arange(cols, dtype=np.float32)[None, :]
        dem = Raster(np.tile(col * 0.4, (rows, 1)).astype(np.float32), bounds, SHIZUOKA.crs)

        records = [SourceRecord(source_id="virtual_shizuoka", layers=["elevation"])]
        graph = None
        if with_roads:
            cx, cy = tile.core.centre
            graph = RoadGraph(crs=CRS)
            _line(graph, f"{tile.id}_e", (cx - 300, cy), (cx + 300, cy))
            records.append(
                SourceRecord(source_id="osm", role=SourceRole.TARGET, layers=["roads"])
            )

        bundle = TileAssembler(gate, resolution=COARSE).assemble(
            tile, TileInputs(elevation=dem, roads=graph, records=records)
        )
        write_tile(root, bundle)
    return root


def test_observe_reads_a_corpus_into_paired_observations(gate, tmp_path):
    from japgo.analysis.study import observe

    tiles = [Tile(zone=8, ix=10 + i, iy=-110) for i in range(3)]
    _write_corpus(gate, tmp_path, tiles)

    observations, skipped = observe(tmp_path)

    assert len(observations) == 3
    assert skipped == []
    assert tuple(observations[0].features) == ENVIRONMENTAL_FEATURES
    assert tuple(observations[0].metrics) == ROAD_STRUCTURE_METRICS
    assert observations[0].metrics["road_density_km_per_km2"] == pytest.approx(0.6, abs=0.01)


def test_observe_reports_what_it_dropped_rather_than_dropping_it_silently(gate, tmp_path):
    """An analysis that quietly discards half its corpus looks identical to one that only ever
    had half a corpus."""
    from japgo.analysis.study import observe

    _write_corpus(gate, tmp_path, [Tile(zone=8, ix=10, iy=-110)], with_roads=False)

    observations, skipped = observe(tmp_path)
    assert observations == []
    assert len(skipped) == 1 and "no road graph" in skipped[0]


def test_sites_come_from_the_split_definition(gate, tmp_path):
    """Grouping must agree with the project's existing authority on which tiles belong together
    (invariant 4). Deriving it a second way here would eventually disagree."""
    from japgo.analysis.study import UNASSIGNED_SITE, observe
    from japgo.pipeline.splits import Site, Split, make_split

    tiles = [Tile(zone=8, ix=10 + i, iy=-110) for i in range(2)]
    _write_corpus(gate, tmp_path, tiles)

    sites = {
        "izu_coast": Site(
            name="izu_coast", archetype="coastal", tiles=frozenset({tiles[0].id})
        )
    }
    split = make_split(sites, {"izu_coast": Split.TRAIN})

    observations, _ = observe(tmp_path, split=split)
    by_tile = {o.tile_id: o.site for o in observations}

    assert by_tile[tiles[0].id] == "izu_coast"
    assert by_tile[tiles[1].id] == UNASSIGNED_SITE
