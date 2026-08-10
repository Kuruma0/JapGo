"""Tests for the provenance gate.

These encode the licensing analysis from ``docs/phase0-research.md`` §6 as executable assertions.
If one of these fails, either a licence changed or someone weakened a rule that was protecting the
project's ability to ship.
"""

from __future__ import annotations

import pytest

from japgo.provenance import (
    OutputRole,
    ProvenanceViolation,
    ShareAlike,
    SourceGate,
    UnregisteredSourceError,
    UsageTier,
    check_registry,
    check_share_alike,
    errors,
    find_registry,
    load_registry,
    registry_hash,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry(find_registry())


@pytest.fixture(scope="module")
def gate(registry):
    return SourceGate(registry)


# ---------------------------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------------------------


def test_registry_passes_its_own_checks(registry):
    assert errors(check_registry(registry)) == []


def test_registry_hash_is_stable():
    assert registry_hash() == registry_hash()


def test_every_ingestible_source_declares_an_output_role(registry):
    """Omitting output_role would silently bypass the export gate."""
    missing = [s.id for s in registry.ingestible if s.output_role is None]
    assert missing == []


# ---------------------------------------------------------------------------------------------
# Invariant 2: no source without a registry entry
# ---------------------------------------------------------------------------------------------


def test_unregistered_source_is_refused(gate):
    with pytest.raises(UnregisteredSourceError, match="no registry entry"):
        gate.assert_ingestible("google_maps_scrape")


def test_error_message_lists_known_sources(gate):
    with pytest.raises(UnregisteredSourceError, match="plateau"):
        gate.assert_ingestible("nope")


# ---------------------------------------------------------------------------------------------
# GSI decision (§6.3) — avoid bulk use
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", ["gsi_dem", "gsi_tiles"])
def test_gsi_is_quarantined(registry, gate, source_id):
    assert registry.require(source_id).usage_tier is UsageTier.QUARANTINED
    with pytest.raises(ProvenanceViolation, match="quarantined"):
        gate.assert_ingestible(source_id)


def test_virtual_shizuoka_replaces_gsi_at_finer_resolution(registry, gate):
    vs = gate.assert_ingestible("virtual_shizuoka")
    assert vs.usage_tier is UsageTier.PUBLIC
    assert vs.output_role is OutputRole.REDISTRIBUTABLE_CORE
    # CC BY 4.0 elected from the dual licence keeps terrain out of share-alike entirely.
    assert vs.share_alike is ShareAlike.NONE
    assert "0.5 m" in vs.resolution


# ---------------------------------------------------------------------------------------------
# §6.1c — OSM geometry must never reach shipped output
# ---------------------------------------------------------------------------------------------


def test_osm_is_trainable_but_not_shippable(registry, gate):
    osm = gate.assert_ingestible("osm")  # training is fine
    assert osm.output_role is OutputRole.TRAINING_ONLY
    assert not osm.may_ship_geometry


def test_export_refuses_osm_geometry_as_attribution_only(gate):
    with pytest.raises(ProvenanceViolation, match="training_only"):
        gate.assert_exportable(["plateau", "osm"], redistribution_class="attribution-only")


def test_export_allows_osm_when_declared_share_alike(gate):
    """The refusal is about mislabelling, not about OSM itself."""
    gate.assert_exportable(["plateau", "osm"], redistribution_class="share-alike")


def test_redistributable_core_export_is_permitted(gate):
    findings = gate.assert_exportable(
        ["plateau", "virtual_shizuoka", "nlni_landuse", "aw3d30"],
        redistribution_class="attribution-only",
    )
    assert errors(findings) == []


def test_overture_is_treated_exactly_like_osm(registry):
    """Adopting Overture buys ergonomics, not licence relief."""
    overture = registry.require("overture")
    assert overture.output_role is OutputRole.TRAINING_ONLY
    assert overture.share_alike is ShareAlike.ODBL


# ---------------------------------------------------------------------------------------------
# §6.4 — incompatible copyleft families
# ---------------------------------------------------------------------------------------------


def test_odbl_and_cc_by_sa_cannot_be_combined(registry):
    found = check_share_alike(registry, ["osm", "mapillary"])
    assert any("incompatible copyleft" in f.message for f in found)


def test_redistributable_core_is_free_of_share_alike(registry):
    families = {s.share_alike for s in registry.redistributable_core}
    assert families == {ShareAlike.NONE}


@pytest.mark.parametrize("source_id", ["mapillary", "kartaview", "meijo_gnss_imu"])
def test_street_level_and_research_only_sources_are_blocked(gate, source_id):
    with pytest.raises(ProvenanceViolation):
        gate.assert_ingestible(source_id)


# ---------------------------------------------------------------------------------------------
# Commercial intent
# ---------------------------------------------------------------------------------------------


def test_commercial_intent_is_set(registry):
    assert registry.policy.commercial_intent is True


def test_no_ingestible_source_has_commercial_restrictions(registry):
    restricted = [s.id for s in registry.ingestible if not s.is_commercially_clear]
    assert restricted == []


# ---------------------------------------------------------------------------------------------
# The AW3D30 version trap
# ---------------------------------------------------------------------------------------------


def test_aw3d30_is_pinned_to_v31(registry):
    assert registry.require("aw3d30").version_pin == "3.1"


def test_ingesting_aw3d30_latest_is_refused(gate):
    """v4.1 excludes Japan; fetching 'latest' silently yields no data."""
    with pytest.raises(ProvenanceViolation, match="pinned to version 3.1"):
        gate.assert_version("aw3d30", "4.1")


def test_ingesting_aw3d30_at_pinned_version_is_allowed(gate):
    gate.assert_version("aw3d30", "3.1")


# ---------------------------------------------------------------------------------------------
# Attribution assembly
# ---------------------------------------------------------------------------------------------


def test_attribution_is_assembled_for_every_contributing_source(gate):
    lines = gate.attribution_for(["plateau", "virtual_shizuoka", "aw3d30"])
    assert len(lines) == 3
    assert any("PLATEAU" in line for line in lines)
    assert any("VIRTUAL SHIZUOKA" in line for line in lines)
    assert any("JAXA" in line for line in lines)


def test_attribution_deduplicates(gate):
    assert len(gate.attribution_for(["plateau", "plateau"])) == 1
