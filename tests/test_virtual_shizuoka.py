"""Tests for the VIRTUAL SHIZUOKA terrain adapter.

The LAS fixture is synthesised rather than downloaded: real tiles average ~300 MB and reach 5.6 GB.
It carries both ground (ASPRS 2) and vegetation (ASPRS 5) returns at known elevations, so the
bare-earth filter can be checked by its effect rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.geo import SHIZUOKA, Bounds, slope
from japgo.provenance import ProvenanceViolation
from japgo.sources import VirtualShizuokaAdapter
from japgo.sources.virtual_shizuoka import ASPRS_GROUND, NATIVE_RESOLUTION_M

GROUND_Z = 100.0
CANOPY_Z = 115.0  # 15 m of trees on top of flat ground

# A 20 m x 20 m patch somewhere plausible in Zone 8.
X0, Y0, SIZE = 10_000.0, -110_000.0, 20.0


@pytest.fixture
def las_path(tmp_path):
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(X0, X0 + SIZE, n)
    y = rng.uniform(Y0, Y0 + SIZE, n)

    # Half the returns are canopy sitting 15 m above flat ground.
    classification = np.where(rng.random(n) < 0.5, ASPRS_GROUND, 5).astype(np.uint8)
    z = np.where(classification == ASPRS_GROUND, GROUND_Z, CANOPY_Z)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [X0, Y0, 0.0]
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = classification

    path = tmp_path / "09LD1234.las"
    las.write(path)
    return path


@pytest.fixture
def dsm_only_path(tmp_path):
    """A file with no ground-classified returns at all — an unclassified delivery."""
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(1)
    n = 2000
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [X0, Y0, 0.0]
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = rng.uniform(X0, X0 + SIZE, n)
    las.y = rng.uniform(Y0, Y0 + SIZE, n)
    las.z = np.full(n, CANOPY_Z)
    las.classification = np.full(n, 1, dtype=np.uint8)  # unclassified

    path = tmp_path / "unclassified.las"
    las.write(path)
    return path


@pytest.fixture
def adapter(gate):
    return VirtualShizuokaAdapter(gate)


# ---------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------


def test_source_is_permitted_and_in_the_redistributable_core(adapter, registry):
    src = adapter.open()
    assert src.usage_tier.value == "public"
    assert src.may_ship_geometry


def test_gate_cannot_be_bypassed_by_subclassing(gate):
    class Sneaky(VirtualShizuokaAdapter):
        source_id = "gsi_dem"

    with pytest.raises(ProvenanceViolation, match="quarantined"):
        Sneaky(gate).open()


def test_defaults_to_the_published_crs(adapter):
    assert adapter.target_crs.to_epsg() == SHIZUOKA.epsg


# ---------------------------------------------------------------------------------------------
# Bare earth — the failure that would be silent
# ---------------------------------------------------------------------------------------------


def test_ground_filter_yields_terrain_not_canopy(adapter, las_path):
    dem = adapter.read(las_path).layers["elevation"][0]
    assert np.nanmean(dem.data) == pytest.approx(GROUND_Z, abs=0.01)


def test_without_the_filter_the_surface_includes_canopy(adapter, las_path):
    dsm = adapter.read(las_path, ground_only=False).layers["elevation"][0]
    assert np.nanmean(dsm.data) > GROUND_Z + 5


def test_missing_ground_class_is_reported_loudly(adapter, dsm_only_path):
    """Returning a DSM where a DTM was asked for must never pass quietly.

    §12 assigns grade compliance to the deterministic half of the system, where it is expected to
    be exact. An unnoticed DSM would make an exact constraint enforce the wrong thing.
    """
    result = adapter.read(dsm_only_path, ground_only=True)
    assert any("SURFACE model" in w for w in result.warnings)
    assert any("canopy" in w for w in result.warnings)


def test_record_states_which_filter_was_applied(adapter, las_path):
    assert "bare-earth" in adapter.read(las_path).record.note
    assert "all returns" in adapter.read(las_path, ground_only=False).record.note


# ---------------------------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------------------------


def test_grids_at_native_resolution(adapter, las_path):
    dem = adapter.read(las_path).layers["elevation"][0]
    assert dem.resolution == pytest.approx(NATIVE_RESOLUTION_M)


def test_respects_a_requested_window(adapter, las_path):
    """Files reach 5.6 GB, so a windowed read is the normal case, not an optimisation."""
    window = Bounds(X0, Y0, X0 + 10, Y0 + 10)
    dem = adapter.read(las_path, bounds=window).layers["elevation"][0]
    assert dem.bounds.as_tuple() == pytest.approx(window.as_tuple())
    assert dem.data.shape == (20, 20)  # 10 m at 0.5 m


def test_flat_ground_yields_near_zero_slope(adapter, las_path):
    dem = adapter.read(las_path).layers["elevation"][0]
    assert np.nanmax(slope(dem, as_percent=True).data) < 1.0


def test_window_with_no_overlap_raises_rather_than_returning_an_empty_tile(adapter, las_path):
    """Zero overlap is a caller error — wrong extent, zone or file — not a data gap.

    An all-NaN raster would propagate a valid-looking but empty tile downstream.
    """
    with pytest.raises(ValueError, match="no points fall inside") as exc:
        adapter.read(las_path, bounds=Bounds(0, 0, 10, 10))

    # The message must say where the data actually is, or the caller cannot act on it.
    assert "The file spans" in str(exc.value)
    assert "same CRS" in str(exc.value)


def test_coverage_gap_is_reported(adapter, las_path):
    """Sparse cells are an honest gap, not something to silently interpolate over."""
    result = adapter.read(las_path, resolution=0.05)  # far finer than the point spacing
    assert result.layers["elevation"][0].coverage < 1.0
    assert any("no return" in w for w in result.warnings)


# ---------------------------------------------------------------------------------------------
# Working tier (research doc §8)
# ---------------------------------------------------------------------------------------------


def test_working_tier_downsamples_to_one_metre(adapter, las_path):
    result = adapter.read_to_working_tier(las_path)
    assert result.layers["elevation"][0].resolution == pytest.approx(1.0)


def test_working_tier_retains_the_native_resolution_separately(adapter, las_path):
    """0.5 m is kept for validation and high-detail export, but not fed to the model."""
    result = adapter.read_to_working_tier(las_path)
    assert result.layers["elevation_native"][0].resolution == pytest.approx(0.5)
    assert result.record.layers == ["elevation", "elevation_native"]
    assert "working tier" in result.record.note
