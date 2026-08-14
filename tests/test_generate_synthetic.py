"""Tests for synthetic terrain, whole-world inference, and the blind experiment's measurements.

The experiment's conclusion is negative, which makes the machinery producing it worth more
scrutiny than usual: a bug that silently zeroed a channel or misplaced a window would produce
exactly the same empty result and the same confident write-up. So what is tested here is mostly
"is the model being fed what we claim it is being fed".
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.generate.synthetic import (
    ARCHETYPES,
    _fbm,
    params_for,
    synthesise,
)
from japgo.generate.world import TERRAIN_CHANNELS, WindowPlan, terrain_stack
from japgo.geo.raster import Raster
from japgo.geo.tiling import Bounds
from japgo.pipeline.channels import load_stack_spec

SMALL = 512.0        # a 512 m world keeps these tests to about a second


def _world(archetype: str, seed: int = 1):
    return synthesise(params_for(archetype, seed, size_m=SMALL))


def test_the_same_seed_gives_the_same_terrain():
    """The experiment's reproducibility claim rests entirely on this."""
    a, b = _world("mountain_valley", 7), _world("mountain_valley", 7)
    assert np.array_equal(a.elevation, b.elevation)
    assert a.params.to_dict() == b.params.to_dict()


def test_different_seeds_give_different_terrain():
    """Three seeds per archetype are only a replicate if they actually differ."""
    a, b = _world("basin", 1), _world("basin", 2)
    assert not np.array_equal(a.elevation, b.elevation)


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_every_archetype_produces_finite_terrain_with_relief(archetype):
    world = _world(archetype)
    assert np.isfinite(world.elevation).all(), "a NaN here poisons every downstream channel"
    assert np.ptp(world.elevation) > 1.0, "flat output would make the archetype meaningless"
    assert world.elevation.dtype == np.float32
    assert world.notes, "a world that cannot describe its own landforms cannot be reported"


def test_valleys_are_lower_than_the_ground_around_them():
    """The archetype's whole claim. Checked as a distribution rather than at a point, because
    where the valley runs depends on the seed."""
    world = _world("mountain_valley", 3)
    z = world.elevation
    assert np.percentile(z, 5) < np.percentile(z, 95) - 50


def test_the_basin_floor_is_flatter_than_its_rim():
    from japgo.geo.terrain import slope

    world = _world("basin", 1)
    grade = slope(world.raster(), as_percent=True).data
    n = grade.shape[0]
    middle = grade[n // 3 : 2 * n // 3, n // 3 : 2 * n // 3]
    assert middle.mean() < grade.mean()


def test_the_coast_has_ground_below_sea_level_and_ground_well_above_it():
    world = _world("coastal", 1)
    assert world.elevation.min() < -1.0
    assert world.elevation.max() > 50.0


def test_noise_is_bounded():
    """fBm outside [-1, 1] would silently rescale every landform that multiplies by it."""
    rng = np.random.default_rng(0)
    field = _fbm(rng, (64, 64), params_for("plain", 1))
    assert -1.0001 <= field.min() and field.max() <= 1.0001


# ---------------------------------------------------------------------------------------------
# the stack handed to the model
# ---------------------------------------------------------------------------------------------


def test_only_the_terrain_channels_carry_signal():
    """The experiment's central constraint, enforced rather than trusted.

    If a building or land-use channel were accidentally populated, the model would be receiving
    settlement information and the whole question would be a different one.
    """
    spec = load_stack_spec()
    world = _world("plain", 1)
    stack = terrain_stack(world.raster(), spec)

    assert stack.shape == (spec.depth, *world.elevation.shape)
    for i, channel in enumerate(spec.channels):
        if channel.name in TERRAIN_CHANNELS:
            continue
        expected = 1.0 if channel.name == "valid" else spec.nodata_fill
        assert np.allclose(stack[i], expected), f"{channel.name} should be constant {expected}"

    assert np.abs(stack[spec.index_of("elevation")].mean()) < 1e-3, "elevation is tile-relative"
    assert stack[spec.index_of("slope")].max() > 0.0
    assert np.isfinite(stack).all()


def test_the_stack_reproduces_a_real_tile_s_stored_channels():
    """Guards the comparison the controls depend on.

    The experiment measures a synthetic world against a real tile run through the same code. If
    ``terrain_stack`` did not reproduce what the assembler stored, that comparison would be
    between two different definitions of slope.
    """
    from japgo.pipeline.assemble import terrain_planes

    spec = load_stack_spec()
    world = _world("coastal", 2)
    raster = world.raster()

    stack = terrain_stack(raster, spec)
    planes = terrain_planes(raster)
    for name in TERRAIN_CHANNELS:
        channel = next(c for c in spec.channels if c.name == name)
        expected = np.nan_to_num(
            channel.apply_normalisation(np.asarray(planes[name], np.float32)),
            nan=spec.nodata_fill,
        )
        assert np.allclose(stack[spec.index_of(name)], expected)


# ---------------------------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------------------------


def test_windows_tile_the_world_exactly_once():
    """Overlapping cores would double-predict; a gap would leave a stripe of zeros that looks
    exactly like a model refusing to answer."""
    plan = WindowPlan(window_px=1000, halo_px=256)
    covered = np.zeros((2500, 2500), np.uint8)
    for r0, r1, c0, c1 in plan.windows(2500, 2500):
        covered[r0:r1, c0:c1] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_the_read_window_is_always_the_training_read_size():
    """1000 m core inside a 256 m halo is 1512 px — the read extent of a corpus tile. Ragged edge
    windows are what made a 4 km world take 285 s instead of 8."""
    plan = WindowPlan()
    elevation = np.arange(2000 * 2000, dtype=np.float32).reshape(2000, 2000)
    shapes = {
        plan.read_window(elevation, r0, c0)[0].shape
        for r0, _, c0, _ in plan.windows(2000, 2000)
    }
    assert shapes == {(plan.read_px, plan.read_px)} == {(1512, 1512)}


def test_edge_windows_are_mirrored_not_zero_filled():
    """A wall of zero elevation at the world edge is a cliff, and a cliff is a feature the model
    would answer."""
    plan = WindowPlan(window_px=64, halo_px=16)
    elevation = np.full((128, 128), 500.0, np.float32)
    block, _, _ = plan.read_window(elevation, 0, 0)
    assert block.shape == (96, 96)
    assert (block == 500.0).all()


def test_predict_world_stitches_windows_into_the_right_places():
    """A stub model that answers with its own position proves the reassembly, without a GPU."""
    from japgo.generate.inference import RoadPrediction

    plan = WindowPlan(window_px=32, halo_px=8)
    spec = load_stack_spec()

    card = type("C", (), {"resolution_m": 1.0, "crs": "EPSG:6676", "threshold": 0.5,
                          "channels": spec.names, "stack_version": spec.stack_version})()

    class _Positional:
        def predict(self, stack, bounds, *, threshold=None, crs=None):
            # Encode the window's own left edge, so a misplaced write is visible in the output.
            value = np.full(stack.shape[1:], bounds.minx / 1000.0, np.float32)
            return RoadPrediction(value, bounds, "EPSG:6676", 1.0, threshold=0.5)

    from japgo.generate.world import predict_world

    model = _Positional()
    model.card, model.spec = card, spec
    out = predict_world(model, np.zeros((64, 64), np.float32), Bounds(0, 0, 64, 64), plan=plan)
    # Left half came from the window starting at x=-8, right half from the one at x=24.
    assert out.probability[0, 0] == pytest.approx(-0.008)
    assert out.probability[0, 40] == pytest.approx(0.024)


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------


def test_stage_metrics_reports_an_empty_stage_without_crashing():
    """Nine of the twelve worlds produced no roads at all. A metrics function that raises on an
    empty graph would have hidden the experiment's actual result."""
    from japgo.core import RoadGraph
    from japgo.generate.blind import stage_metrics

    world = _world("plain", 1)
    m = stage_metrics(
        "RAW ML", RoadGraph(crs="EPSG:6676"), world.tile,
        elevation=world.elevation, bounds=world.bounds, resolution_m=1.0, grade_limit_pct=12.0,
    )
    assert m["edges"] == 0 and m["components"] == 0 and m["grade_violations"] == 0
    assert m["max_grade_pct"] is None, "no roads means no grade, not a grade of zero"


def test_stage_metrics_does_not_modify_the_graph_it_measures():
    """Each stage is measured and then handed on. Annotating in place would let one stage's
    measurement change the next stage's input."""
    from japgo.core import Edge, Node, RoadGraph
    from japgo.generate.blind import stage_metrics

    world = _world("plain", 1)
    graph = RoadGraph(crs="EPSG:6676")
    graph.add_node(Node(id="a", x=20.0, y=20.0))
    graph.add_node(Node(id="b", x=200.0, y=200.0))
    graph.add_edge(Edge(id="e", u="a", v="b", geometry=[(20.0, 20.0), (200.0, 200.0)],
                        source_id="model"))

    stage_metrics("RAW ML", graph, world.tile, elevation=world.elevation, bounds=world.bounds,
                  resolution_m=1.0, grade_limit_pct=12.0)
    assert graph.edges["e"].grade_pct is None
    assert graph.nodes["a"].z is None


def test_terrain_response_orders_a_field_that_depends_on_slope():
    """The diagnostic that distinguishes 'faint but ordered' from 'nothing'. Verified against a
    probability field constructed to prefer flat ground."""
    from japgo.geo.terrain import slope
    from japgo.generate.blind import terrain_response

    world = _world("mountain_valley", 1)
    grade = slope(world.raster(), as_percent=True).data
    probability = np.exp(-grade / 20.0).astype(np.float32)

    response = terrain_response(probability, world.elevation, world, 1.0)["slope_pct"]
    means = [q["mean_probability"] for q in response["quintiles"]]
    assert response["monotonic"] and means[0] > means[-1]
    assert response["spread"] > 1.5
