"""Tests for the Phase 5 sensitivity sweep.

The sweep is the project's thesis test, so the properties that matter are the ones that stop it
flattering itself: expectations fixed before the run, a null control that must not move, and a
dead band so noise is not read as response.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.model.sweep import DEFAULT_SWEEP, RESPONSES, Perturbation, SweepResult


def _result(before: dict, after: dict, expect: dict, name: str = "test") -> SweepResult:
    return SweepResult(name, "slope", 2.0, before, after, expect, tiles=3)


def test_the_sweep_declares_its_expectations_before_it_runs():
    """A direction read off after the fact is not a prediction. These come from site-selection.md
    and §1.3, both written before the project had a model."""
    for p in DEFAULT_SWEEP:
        assert set(p.expect) <= set(RESPONSES)
        assert set(p.expect.values()) <= {"up", "down", "flat"}
        assert p.expect, f"{p.name} predicts nothing"


def test_a_null_perturbation_is_included_and_predicts_no_movement():
    """Without it, inference nondeterminism reads as environmental response."""
    null = next(p for p in DEFAULT_SWEEP if p.name == "null")
    assert null.factor == 1.0
    assert set(null.expect.values()) == {"flat"}


def test_steepening_and_flattening_predict_opposite_directions():
    """If they agreed, the sweep could be satisfied by a model that ignores the channel."""
    steep = next(p for p in DEFAULT_SWEEP if p.name == "steepen")
    flat = next(p for p in DEFAULT_SWEEP if p.name == "flatten")
    for response in RESPONSES:
        assert steep.expect[response] != flat.expect[response], response


def test_direction_has_a_dead_band_so_noise_is_not_a_response():
    r = _result(
        {"road_density_km_per_km2": 10.0}, {"road_density_km_per_km2": 10.2},
        {"road_density_km_per_km2": "up"},
    )
    assert r.direction("road_density_km_per_km2") == "flat"   # 2% is inside the band
    bigger = _result(
        {"road_density_km_per_km2": 10.0}, {"road_density_km_per_km2": 13.0},
        {"road_density_km_per_km2": "up"},
    )
    assert bigger.direction("road_density_km_per_km2") == "up"


def test_a_response_the_metric_could_not_measure_is_not_scored():
    """NaN means "not measurable here", and counting it either way would be an invention."""
    r = _result(
        {"road_density_km_per_km2": float("nan")}, {"road_density_km_per_km2": 5.0},
        {"road_density_km_per_km2": "up"},
    )
    assert r.direction("road_density_km_per_km2") == "n/a"
    assert np.isnan(r.score)


def test_the_score_counts_only_agreement_with_the_declared_direction():
    r = _result(
        {"a": 10.0, "b": 10.0}, {"a": 20.0, "b": 20.0},
        {"a": "up", "b": "down"},
    )
    assert r.agrees("a") and not r.agrees("b")
    assert r.score == pytest.approx(0.5)


def test_a_perturbation_changes_one_channel_and_leaves_the_rest_alone():
    """One at a time, or the response cannot be attributed."""
    stack = np.ones((4, 8, 8), dtype=np.float32)
    altered = Perturbation("x", "slope", 3.0, {}).apply(stack, 1)

    assert np.allclose(altered[1], 3.0)
    for other in (0, 2, 3):
        assert np.allclose(altered[other], 1.0)
    assert np.allclose(stack, 1.0)        # the original is untouched


def test_a_perturbation_cannot_drive_a_channel_negative():
    stack = np.full((2, 4, 4), 0.5, dtype=np.float32)
    zeroed = Perturbation("unbuild", "landuse_built", 0.0, {}).apply(stack, 0)
    assert np.allclose(zeroed[0], 0.0)
    assert zeroed.min() >= 0.0
