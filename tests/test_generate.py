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
