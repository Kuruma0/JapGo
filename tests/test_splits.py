"""Tests for geographic splits.

Invariant 4 says splits are geographic, never random. These tests are what make that claim
checkable — the point of the module is that a leaking split fails loudly rather than producing
flattering numbers.
"""

from __future__ import annotations

import pytest

from japgo.geo import Tile
from japgo.pipeline import (
    MIN_BUFFER_TILES,
    Site,
    Split,
    SplitDefinition,
    SplitError,
    assert_valid,
    load_sites,
    make_split,
    validate_split,
)


def _tiles(ix_range: range, iy: int = 0) -> frozenset[str]:
    return frozenset(Tile(zone=8, ix=ix, iy=iy).id for ix in ix_range)


def _site(name: str, ix_range: range, archetype: str = "suburban_plain") -> Site:
    return Site(name=name, archetype=archetype, tiles=_tiles(ix_range))


# ---------------------------------------------------------------------------------------------
# Assignment is per site, never per tile
# ---------------------------------------------------------------------------------------------


def test_split_is_assigned_by_site():
    sites = {
        "a": _site("a", range(0, 5)),
        "b": _site("b", range(20, 25), archetype="mountain_valley"),
    }
    definition = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})
    assert set(definition.tiles_in(Split.TRAIN)) == set(sites["a"].tiles)
    assert set(definition.tiles_in(Split.TEST)) == set(sites["b"].tiles)


def test_unassigned_site_is_refused():
    """An unassigned site would silently disappear from the dataset."""
    sites = {"a": _site("a", range(0, 3)), "b": _site("b", range(20, 23))}
    with pytest.raises(SplitError, match="no split assignment"):
        make_split(sites, {"a": Split.TRAIN})


def test_unknown_site_in_assignment_is_refused():
    with pytest.raises(SplitError, match="unknown sites"):
        make_split({"a": _site("a", range(0, 3))}, {"a": Split.TRAIN, "ghost": Split.TEST})


# ---------------------------------------------------------------------------------------------
# The buffer zone
# ---------------------------------------------------------------------------------------------


def test_zero_buffer_is_refused():
    """Adjacent tiles share halo pixels, so a zero buffer leaks input data."""
    sites = {"a": _site("a", range(0, 3)), "b": _site("b", range(20, 23))}
    with pytest.raises(SplitError, match="below the minimum"):
        make_split(sites, {"a": Split.TRAIN, "b": Split.TEST}, buffer_tiles=0)


def test_adjacent_sites_are_buffered_apart():
    """Touching regions must lose their boundary tiles, not share them."""
    sites = {
        "a": _site("a", range(0, 5)),
        "b": _site("b", range(5, 10), archetype="coastal_constrained"),
    }
    definition = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})

    buffered = set(definition.tiles_in(Split.BUFFER))
    assert Tile(zone=8, ix=4, iy=0).id in buffered
    assert Tile(zone=8, ix=5, iy=0).id in buffered
    assert Tile(zone=8, ix=0, iy=0).id in set(definition.tiles_in(Split.TRAIN))


def test_distant_sites_lose_nothing():
    sites = {
        "a": _site("a", range(0, 5)),
        "b": _site("b", range(50, 55), archetype="mountain_valley"),
    }
    definition = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})
    assert definition.tiles_in(Split.BUFFER) == []


def test_larger_buffer_discards_more():
    sites = {
        "a": _site("a", range(0, 10)),
        "b": _site("b", range(12, 20), archetype="mountain_valley"),
    }
    small = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST}, buffer_tiles=1)
    large = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST}, buffer_tiles=3)
    assert len(large.tiles_in(Split.BUFFER)) > len(small.tiles_in(Split.BUFFER))


def test_buffer_is_chebyshev_so_diagonals_count():
    """Diagonal neighbours share halo pixels too."""
    sites = {
        "a": Site("a", "suburban_plain", frozenset({Tile(zone=8, ix=0, iy=0).id})),
        "b": Site("b", "mountain_valley", frozenset({Tile(zone=8, ix=1, iy=1).id})),
    }
    definition = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})
    assert len(definition.tiles_in(Split.BUFFER)) == 2


# ---------------------------------------------------------------------------------------------
# Validation — the part that turns the invariant into a property
# ---------------------------------------------------------------------------------------------


def test_a_generated_split_validates_clean():
    sites = {
        "a": _site("a", range(0, 5)),
        "b": _site("b", range(20, 25), archetype="mountain_valley"),
    }
    assert validate_split(make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})) == []


def test_hand_built_leaking_split_is_caught():
    """The failure this module exists to prevent, constructed deliberately."""
    leaking = SplitDefinition(
        assignment={
            Tile(zone=8, ix=0, iy=0).id: Split.TRAIN,
            Tile(zone=8, ix=1, iy=0).id: Split.TEST,  # adjacent
        },
        sites={},
        buffer_tiles=MIN_BUFFER_TILES,
    )
    problems = validate_split(leaking)
    assert problems
    assert "share halo pixels" in problems[0]

    with pytest.raises(SplitError, match="leak geography"):
        assert_valid(leaking)


def test_random_tile_split_is_caught():
    """The specific mistake invariant 4 names: splitting tiles rather than regions."""
    tiles = [Tile(zone=8, ix=ix, iy=0).id for ix in range(10)]
    random_ish = SplitDefinition(
        assignment={t: (Split.TRAIN if i % 2 == 0 else Split.TEST) for i, t in enumerate(tiles)},
        sites={},
        buffer_tiles=MIN_BUFFER_TILES,
    )
    assert validate_split(random_ish)


def test_site_spanning_two_splits_is_caught():
    site = _site("spanning", range(0, 4))
    definition = SplitDefinition(
        assignment={
            Tile(zone=8, ix=0, iy=0).id: Split.TRAIN,
            Tile(zone=8, ix=30, iy=0).id: Split.TEST,
        },
        sites={"spanning": Site("spanning", "suburban_plain", frozenset(
            {Tile(zone=8, ix=0, iy=0).id, Tile(zone=8, ix=30, iy=0).id}
        ))},
        buffer_tiles=MIN_BUFFER_TILES,
    )
    del site
    problems = validate_split(definition)
    assert any("spans" in p for p in problems)


# ---------------------------------------------------------------------------------------------
# Archetype coverage
# ---------------------------------------------------------------------------------------------


def test_archetypes_are_tracked_per_split():
    sites = {
        "plain": _site("plain", range(0, 3), archetype="suburban_plain"),
        "valley": _site("valley", range(20, 23), archetype="mountain_valley"),
    }
    definition = make_split(sites, {"plain": Split.TRAIN, "valley": Split.TEST})
    assert definition.archetypes_in(Split.TRAIN) == {"suburban_plain"}
    assert definition.archetypes_in(Split.TEST) == {"mountain_valley"}


def test_held_out_archetype_differs_from_training():
    """Testing on the same archetype only proves transfer between similar places."""
    sites = {
        "plain": _site("plain", range(0, 3), archetype="suburban_plain"),
        "valley": _site("valley", range(20, 23), archetype="mountain_valley"),
    }
    definition = make_split(sites, {"plain": Split.TRAIN, "valley": Split.TEST})
    assert not (definition.archetypes_in(Split.TEST) & definition.archetypes_in(Split.TRAIN))


# ---------------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------------


def test_split_roundtrips(tmp_path):
    sites = {
        "a": _site("a", range(0, 4)),
        "b": _site("b", range(20, 24), archetype="mountain_valley"),
    }
    definition = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST})
    path = definition.write(tmp_path / "split.json")

    back = SplitDefinition.read(path)
    assert back.assignment == definition.assignment
    assert back.buffer_tiles == definition.buffer_tiles
    assert back.archetypes_in(Split.TEST) == {"mountain_valley"}


def test_counts_report_every_category(tmp_path):
    sites = {
        "a": _site("a", range(0, 5)),
        "b": _site("b", range(5, 10), archetype="mountain_valley"),
    }
    counts = make_split(sites, {"a": Split.TRAIN, "b": Split.TEST}).counts
    assert counts["buffer"] > 0
    assert sum(counts.values()) == 10


# ---------------------------------------------------------------------------------------------
# The configured MVP sites
# ---------------------------------------------------------------------------------------------


def test_configured_sites_have_distinct_archetypes():
    """Three sites with the same archetype could not demonstrate environmental response."""
    sites = load_sites()
    archetypes = [s.archetype for s in sites.sites.values()]
    assert len(set(archetypes)) == len(archetypes)


def test_every_configured_site_has_a_default_split():
    sites = load_sites()
    assert set(sites.default_split) == set(sites.sites)


def test_the_mountain_valley_is_held_out():
    """The archetype that most distinguishes the project; testing on the easy case proves nothing."""
    sites = load_sites()
    assert sites.default_split["kawanehon_valley"] == "test"


def test_configured_bboxes_are_well_formed():
    for name, spec in load_sites().sites.items():
        minlon, minlat, maxlon, maxlat = spec.bbox
        assert minlon < maxlon, name
        assert minlat < maxlat, name
        assert 122 < minlon < 154, f"{name} is not in Japan"
        assert 24 < minlat < 46, f"{name} is not in Japan"
