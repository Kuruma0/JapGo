"""Tests for the generation-relevant measure.

APLS asks whether a specific real network was reproduced; these ask whether the output has the
right character for its environment. The second question is the one a generator is judged on, and
the ordering test is its sharpest form — a model can be wrong everywhere and still be useful if it
keeps the archetypes apart, and right on average yet useless if it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.model.plausibility import (
    PLAUSIBILITY_METRICS,
    MetricAgreement,
    SitePlausibility,
    ordering_preserved,
)


def _site(name: str, **values) -> SitePlausibility:
    s = SitePlausibility(site=name, tiles=5)
    for metric in PLAUSIBILITY_METRICS:
        real, predicted = values.get(metric, (1.0, 1.0))
        s.agreements.append(MetricAgreement(metric=metric, real=real, predicted=predicted))
    return s


def test_a_factor_of_two_is_the_bar_and_it_is_deliberately_loose():
    """Nothing calibrates 'close enough to look right' yet; a tight threshold would imply a
    precision the project has not earned."""
    assert MetricAgreement("m", 10.0, 6.0).plausible          # x0.6
    assert MetricAgreement("m", 10.0, 19.0).plausible         # x1.9
    assert not MetricAgreement("m", 10.0, 4.0).plausible      # x0.4
    assert not MetricAgreement("m", 10.0, 21.0).plausible     # x2.1


def test_a_missing_real_value_is_not_scored_either_way():
    a = MetricAgreement("m", float("nan"), 5.0)
    assert np.isnan(a.ratio) and not a.plausible


def test_ordering_is_kept_when_predictions_are_uniformly_scaled():
    """The distinction that matters for a generator.

    A model whose densities are all half of reality needs a scale factor. One whose densities are
    right on average but unordered between archetypes needs a different model. Ranks separate them.
    """
    sites = [
        _site("plain",  road_density_km_per_km2=(20.0, 10.0)),
        _site("coast",  road_density_km_per_km2=(10.0, 5.0)),
        _site("valley", road_density_km_per_km2=(4.0, 2.0)),
    ]
    agrees, real_order, pred_order = ordering_preserved(sites, "road_density_km_per_km2")
    assert agrees
    assert real_order == pred_order == ["valley", "coast", "plain"]


def test_ordering_fails_when_the_output_is_generic():
    """Right on average, indistinguishable between environments — the damaging failure."""
    sites = [
        _site("plain",  road_density_km_per_km2=(20.0, 8.0)),
        _site("coast",  road_density_km_per_km2=(10.0, 9.0)),
        _site("valley", road_density_km_per_km2=(4.0, 8.5)),
    ]
    agrees, real_order, pred_order = ordering_preserved(sites, "road_density_km_per_km2")
    assert not agrees
    assert real_order == ["valley", "coast", "plain"]
    assert pred_order != real_order


def test_the_site_score_counts_only_measurable_axes():
    s = _site("x", road_density_km_per_km2=(10.0, 10.0), sinuosity_median=(float("nan"), 1.2))
    assert s.score == pytest.approx(1.0)      # the NaN axis is skipped, not failed
