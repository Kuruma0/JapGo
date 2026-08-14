"""Tests for the Phase 4 baseline.

Torch is in the ``ml`` extra, so the network itself is exercised only where it is installed. What
is always tested is the part that decides whether a Phase 4 number means anything: how folds are
formed, that no training tile touches a held-out one, and that the priors are computed the way the
comparison claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.analysis import spearman  # noqa: F401  (import guard: analysis must stay torch-free)
from japgo.model import assert_no_overlap, configured_fold, leave_one_site_out, score
from japgo.model.baseline import best_threshold, built_proximity_prior, constant_prior
from japgo.model.dataset import Fold
from japgo.pipeline.splits import Site, Split, make_split

SITES = {
    "hamamatsu_plain": Site(
        name="hamamatsu_plain", archetype="suburban_plain",
        tiles=frozenset({"z08_x-00067_y-00144", "z08_x-00066_y-00144"}),
    ),
    "izu_coast": Site(
        name="izu_coast", archetype="coastal_constrained",
        tiles=frozenset({"z08_x000053_y-00108", "z08_x000054_y-00108"}),
    ),
    "kawanehon_valley": Site(
        name="kawanehon_valley", archetype="mountain_valley",
        tiles=frozenset({"z08_x-00035_y-00102", "z08_x-00036_y-00102"}),
    ),
}
ASSIGNMENT = {
    "hamamatsu_plain": Split.TRAIN,
    "izu_coast": Split.VAL,
    "kawanehon_valley": Split.TEST,
}


# ---------------------------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------------------------


def test_leave_one_site_out_holds_out_a_whole_archetype_each_time():
    """§16.1 requires held-out sets to cover unseen archetypes, not merely unseen cities."""
    folds = leave_one_site_out(make_split(SITES, ASSIGNMENT))

    assert len(folds) == 3
    for fold in folds:
        assert fold.held_out not in fold.train_sites
        assert set(fold.train_tiles).isdisjoint(fold.eval_tiles)
        assert len(fold.train_sites) == 2
    assert {f.held_out for f in folds} == set(SITES)


def test_every_loso_fold_trains_on_both_flat_and_steep_ground():
    """The reason the scheme exists.

    The configured split trains on the Hamamatsu plain alone, where slope barely varies. A model
    cannot learn a response to terrain from data holding terrain nearly constant, so the first
    Phase 4 objective would be unanswerable rather than answered.
    """
    steep = {"izu_coast", "kawanehon_valley"}
    for fold in leave_one_site_out(make_split(SITES, ASSIGNMENT)):
        assert steep & set(fold.train_sites), f"{fold.name} trains on flat ground only"


def test_the_configured_fold_matches_the_split_file():
    fold = configured_fold(make_split(SITES, ASSIGNMENT))
    assert fold.held_out == "kawanehon_valley"
    assert fold.train_sites == ("hamamatsu_plain",)
    assert set(fold.eval_tiles) == SITES["kawanehon_valley"].tiles


def test_adjacent_tiles_across_a_fold_boundary_are_rejected():
    """Adjacent tiles share halo pixels, so a training tile touching an evaluation tile leaks
    input straight across the split — invariant 4, checked rather than assumed."""
    touching = Fold(
        name="bad", train_sites=("a",), held_out="b",
        train_tiles=["z08_x000010_y-00010"],
        eval_tiles=["z08_x000011_y-00010"],          # immediately east
    )
    with pytest.raises(ValueError, match="within 1 tile"):
        assert_no_overlap(touching)


def test_a_tile_on_both_sides_is_rejected():
    both = Fold(
        name="bad", train_sites=("a",), held_out="b",
        train_tiles=["z08_x000010_y-00010"], eval_tiles=["z08_x000010_y-00010"],
    )
    with pytest.raises(ValueError, match="both sides"):
        assert_no_overlap(both)


def test_the_real_sites_are_far_enough_apart():
    for fold in leave_one_site_out(make_split(SITES, ASSIGNMENT)):
        assert_no_overlap(fold)


# ---------------------------------------------------------------------------------------------
# Priors and scoring
# ---------------------------------------------------------------------------------------------


def test_void_pixels_do_not_count_toward_a_score():
    """A tile that is 40% sea would otherwise flatter every model for predicting nothing there."""
    truth = np.zeros((10, 10)); truth[0, :] = 1
    prob = np.zeros((10, 10)); prob[0, :] = 1
    valid = np.zeros((10, 10)); valid[0:2, :] = 1

    full = score(prob, truth, threshold=0.5)
    masked = score(prob, truth, valid=valid, threshold=0.5)
    assert full.f1 == pytest.approx(1.0) and masked.f1 == pytest.approx(1.0)
    assert masked.positives == 10          # only the observed road pixels counted


def test_a_model_predicting_nothing_scores_zero_not_high():
    """On a target that is a few percent of pixels, accuracy would reward exactly this."""
    truth = np.zeros((20, 20)); truth[5, :] = 1
    assert score(np.zeros((20, 20)), truth, threshold=0.5).f1 == 0.0


def test_the_threshold_sweep_is_applied_to_every_side():
    """A comparison won by tuning one side's cutoff is not a comparison."""
    truth = np.zeros((10, 10)); truth[0, :] = 1
    prob = np.full((10, 10), 0.2); prob[0, :] = 0.4

    assert score(prob, truth, threshold=0.5).f1 == 0.0        # everything below the default
    assert best_threshold(prob, truth).f1 == pytest.approx(1.0)


def test_the_constant_prior_is_the_score_for_learning_nothing():
    truth = np.zeros((10, 10)); truth[0, :] = 1
    prior = constant_prior(truth.shape, 0.1)
    assert prior.shape == truth.shape and np.allclose(prior, 0.1)
    # It is not zero: at a low enough threshold it predicts everything and recovers all roads.
    assert best_threshold(prior, truth).recall == pytest.approx(1.0)


def test_built_proximity_puts_probability_where_the_buildings_are():
    """The harder floor: a U-Net that cannot beat 'roads are where the town is' has learned only
    that."""
    stack = np.zeros((3, 40, 40), dtype=np.float32)
    stack[1, 20, 20] = 1.0                      # one building, channel index 1
    prior = built_proximity_prior(stack, built_index=1, radius=5)

    assert prior[20, 20] > 0
    assert prior[0, 0] == 0
    assert prior[20, 20] > prior[20, 30]        # falls off with distance


def test_built_proximity_is_flat_in_window_size():
    """Summed-area, not a sliding window: AGENTS.md names the O(n.w^2) version as a trap."""
    rng = np.random.default_rng(0)
    stack = np.zeros((3, 200, 200), dtype=np.float32)
    stack[1] = (rng.random((200, 200)) > 0.98).astype(np.float32)

    small = built_proximity_prior(stack, built_index=1, radius=2)
    large = built_proximity_prior(stack, built_index=1, radius=40)
    assert small.shape == large.shape == (200, 200)

    # It is a local *density*, so widening the window spreads probability over more pixels without
    # inflating the total — the mean stays put while the covered fraction grows.
    assert (large > 0).mean() > (small > 0).mean()
    assert large.mean() == pytest.approx(small.mean(), rel=0.05)


def test_the_reference_configuration_is_the_one_that_measured_best():
    """v2 — width target, Dice, positive weight capped at 5, no distance weighting.

    Five configurations were measured end to end on the same folds. v2 scored mean APLS 0.092
    against 0.031-0.037 for the three centreline variants that followed it, so the defaults are
    pinned here rather than left as whatever the last experiment used. Every alternative remains
    reachable through the CLI; none of them is the default.
    """
    from japgo.model.baseline import ROAD_CENTRELINE_TARGET, ROAD_TARGET
    from japgo.model.train import RunConfig

    c = RunConfig(root="x", fold="y", train_tiles=[], eval_tiles=[])
    assert ROAD_TARGET == "road_mask"                 # width, not the hairline
    assert c.dice_weight == 1.0                       # Dice earned its place in v2
    assert c.max_positive_weight == 5.0               # raising it cost precision without APLS
    assert c.distance_tolerance_px == 0.0             # a thin-target device, off here

    # The alternative stays selectable, because the comparison is worth being able to re-run.
    assert ROAD_CENTRELINE_TARGET == "road_centreline"


def test_epoch_batches_keep_patches_grouped_by_tile():
    """The loader was costing four fifths of training time.

    A plain permutation over every patch means consecutive reads come from different tiles, and
    each miss re-decompresses a 183 MB zarr: measured 221 ms against 8.7 ms for a resident tile,
    25x. Grouping keeps batches mixing several tiles while all of them stay in memory.
    """
    from japgo.model.dataset import Patch, epoch_batches

    patches = [Patch(f"tile{t}", r, 0) for t in range(20) for r in range(16)]
    batches = epoch_batches(patches, batch=8, resident_tiles=4, seed=0, epoch=0)

    assert batches and all(len(b) == 8 for b in batches)
    # No batch may span more tiles than can be resident at once.
    for b in batches:
        assert len({p.tile_id for p in b}) <= 4
    # And a batch should usually mix tiles rather than come from one: gradients need the variety.
    assert max(len({p.tile_id for p in b}) for b in batches) > 1


def test_epoch_batches_reshuffle_between_epochs_but_are_seed_stable():
    from japgo.model.dataset import Patch, epoch_batches

    patches = [Patch(f"tile{t}", r, 0) for t in range(8) for r in range(16)]
    first = epoch_batches(patches, batch=8, resident_tiles=4, seed=0, epoch=0)
    again = epoch_batches(patches, batch=8, resident_tiles=4, seed=0, epoch=0)
    later = epoch_batches(patches, batch=8, resident_tiles=4, seed=0, epoch=1)

    ids = lambda bs: [[p.tile_id + str(p.row) for p in b] for b in bs]  # noqa: E731
    assert ids(first) == ids(again)      # invariant 8: re-runnable from the seed
    assert ids(first) != ids(later)      # but a different epoch is a different order


def test_every_patch_is_reachable_within_an_epoch():
    """Dropping patches silently would shrink the training set without anyone noticing."""
    from japgo.model.dataset import Patch, epoch_batches

    patches = [Patch(f"tile{t}", r, 0) for t in range(6) for r in range(16)]
    seen = {
        (p.tile_id, p.row)
        for b in epoch_batches(patches, batch=8, resident_tiles=3, seed=0, epoch=0)
        for p in b
    }
    assert len(seen) == len(patches)     # 96 patches, 16 per tile, divides evenly into batches
