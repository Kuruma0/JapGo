"""Tests for the tile assembler and store.

The assertions that matter most here are the ones about the manifest: it must be *computed* from
what actually went into a tile, because a hand-maintained provenance list is a licensing failure
waiting to happen.
"""

from __future__ import annotations

import numpy as np
import pytest

from japgo.core import Building
from japgo.core.manifest import SourceRecord
from japgo.geo import SHIZUOKA, Bounds, Raster, Tile
from japgo.pipeline import (
    TileAssembler,
    TileInputs,
    channel_summary,
    list_tiles,
    load_stack_spec,
    read_tile,
    write_index,
    write_tile,
)
from japgo.pipeline.rasterize import building_height, building_mask
from japgo.provenance import ProvenanceViolation

TILE = Tile(zone=8, ix=10, iy=-110)
RES = 1.0


@pytest.fixture
def spec():
    return load_stack_spec()


@pytest.fixture
def assembler(gate, spec):
    return TileAssembler(gate, spec=spec, resolution=RES)


@pytest.fixture
def dem():
    """A DEM over the tile's read extent, sloping 10% eastward."""
    b = TILE.read
    rows = int(b.height / RES)
    cols = int(b.width / RES)
    col = np.arange(cols, dtype=np.float32)[None, :]
    return Raster(np.tile(col * 0.1, (rows, 1)).astype(np.float32), b, SHIZUOKA.crs)


def _house(x: float, y: float, size: float = 8.0, **kw) -> Building:
    ring = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    return Building(
        id=kw.pop("id", f"b-{x:.0f}-{y:.0f}"),
        source_id=kw.pop("source_id", "plateau"),
        footprint=ring,
        **kw,
    )


@pytest.fixture
def buildings():
    cx, cy = TILE.core.centre
    return [
        _house(cx, cy, height_m=6.0, coarse_type="residential", fine_type="detached_house"),
        _house(cx + 20, cy, height_m=24.0, coarse_type="commercial", fine_type="office"),
        _house(cx + 40, cy, height_m=12.0, coarse_type="industrial", fine_type="warehouse"),
    ]


@pytest.fixture
def inputs(dem, buildings):
    return TileInputs(
        elevation=dem,
        buildings=buildings,
        records=[
            SourceRecord(source_id="virtual_shizuoka", layers=["elevation"]),
            SourceRecord(source_id="plateau", layers=["buildings"]),
        ],
    )


# ---------------------------------------------------------------------------------------------
# Stack spec
# ---------------------------------------------------------------------------------------------


def test_stack_depth_is_derivable_from_config_alone(spec):
    assert spec.depth == len(spec.channels)
    assert spec.depth > 0


def test_required_sources_are_derived_from_channel_declarations(spec):
    assert set(spec.required_sources) == {"virtual_shizuoka", "plateau"}


def test_valid_channel_has_no_source(spec):
    assert spec.get("valid").source is None


def test_aspect_is_split_into_sin_and_cos(spec):
    """Raw degrees would put 359 and 1 far apart numerically though they are adjacent."""
    assert "aspect_sin" in spec.names
    assert "aspect_cos" in spec.names
    assert "aspect" not in spec.names


def test_scale_normalisation_divides(spec):
    slope = spec.get("slope")
    assert slope.apply_normalisation(np.array([25.0])) == pytest.approx([0.5])


def test_tile_relative_normalisation_removes_the_mean(spec):
    out = spec.get("elevation").apply_normalisation(np.array([100.0, 110.0, 120.0]))
    assert out.mean() == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------------------------


def test_building_mask_burns_footprints(buildings):
    b = TILE.core
    mask = building_mask(buildings, b, RES, SHIZUOKA.crs)
    assert mask.data.sum() > 0
    assert set(np.unique(mask.data)) <= {0.0, 1.0}


def test_building_height_burns_metres(buildings):
    heights = building_height(buildings, TILE.core, RES, SHIZUOKA.crs)
    assert heights.data.max() == pytest.approx(24.0)


def test_overlapping_buildings_resolve_to_the_taller():
    """Otherwise the height field depends on file ordering, which §44 forbids."""
    short = _house(0, 0, 10, height_m=5.0, id="short")
    tall = _house(2, 2, 10, height_m=30.0, id="tall")
    bounds = Bounds(0, 0, 20, 20)

    a = building_height([short, tall], bounds, RES, SHIZUOKA.crs)
    b = building_height([tall, short], bounds, RES, SHIZUOKA.crs)
    assert np.array_equal(a.data, b.data)
    assert a.data.max() == pytest.approx(30.0)


def test_building_without_height_still_appears_in_the_mask():
    """The footprint is real even when the height is unknown."""
    b = _house(0, 0, 10, id="no-height")
    bounds = Bounds(0, 0, 20, 20)
    assert building_mask([b], bounds, RES, SHIZUOKA.crs).data.sum() > 0
    assert building_height([b], bounds, RES, SHIZUOKA.crs).data.max() == 0.0


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


def test_stack_shape_matches_spec_and_read_extent(assembler, inputs, spec):
    bundle = assembler.assemble(TILE, inputs)
    expected = TILE.read
    assert bundle.shape == (
        spec.depth,
        int(expected.height / RES),
        int(expected.width / RES),
    )


def test_halo_is_included_by_default(assembler, inputs):
    """The halo must exist from the first ingest; adding it later invalidates every sample."""
    with_halo = assembler.assemble(TILE, inputs).shape
    core_only = assembler.assemble(TILE, inputs, with_halo=False).shape
    assert with_halo[1] > core_only[1]
    assert core_only[1] == int(TILE.core.height / RES)


def test_stack_is_finite_everywhere(assembler, inputs):
    """NaN would poison convolutions; the `valid` channel carries the gap information instead."""
    stack = assembler.assemble(TILE, inputs).stack
    assert np.isfinite(stack).all()


def test_slope_channel_reflects_the_terrain(assembler, inputs, spec):
    """A 10% eastward slope, normalised by 50, should land near 0.2."""
    bundle = assembler.assemble(TILE, inputs)
    interior = bundle.channel("slope")[5:-5, 5:-5]
    assert interior.mean() == pytest.approx(0.2, abs=0.02)


def test_flat_cells_encode_no_aspect_direction(gate, spec):
    """sin/cos of a NaN aspect would poison the stack; a zero-length vector is the honest encoding."""
    b = TILE.read
    flat = Raster(
        np.zeros((int(b.height / RES), int(b.width / RES)), np.float32), b, SHIZUOKA.crs
    )
    bundle = TileAssembler(gate, spec=spec, resolution=RES).assemble(
        TILE, TileInputs(elevation=flat, records=[SourceRecord(source_id="virtual_shizuoka")])
    )
    assert np.allclose(bundle.channel("aspect_sin"), 0.0)
    assert np.allclose(bundle.channel("aspect_cos"), 0.0)


def test_building_channels_are_populated(assembler, inputs):
    bundle = assembler.assemble(TILE, inputs)
    assert bundle.channel("building_mask").sum() > 0
    assert bundle.channel("building_coarse_residential").sum() > 0
    assert bundle.channel("building_coarse_commercial").sum() > 0
    assert bundle.channel("building_coarse_industrial").sum() > 0


def test_empty_rural_tile_is_legitimate(assembler, dem):
    """Forest, water and farmland tiles have no buildings. Rejecting them would reject exactly
    the rural sites the Kawanehon case depends on."""
    bundle = assembler.assemble(
        TILE,
        TileInputs(elevation=dem, records=[SourceRecord(source_id="virtual_shizuoka")]),
    )
    assert bundle.channel("building_mask").sum() == 0
    assert bundle.warnings == []


def test_missing_elevation_is_refused(assembler):
    with pytest.raises(ValueError, match="cannot determine the tile CRS"):
        assembler.assemble(TILE, TileInputs())


def test_channel_summary_covers_every_channel(assembler, inputs, spec):
    assert len(channel_summary(assembler.assemble(TILE, inputs))) == spec.depth


# ---------------------------------------------------------------------------------------------
# Manifest — computed, not declared
# ---------------------------------------------------------------------------------------------


def test_manifest_records_contributing_sources(assembler, inputs):
    manifest = assembler.assemble(TILE, inputs).manifest
    assert set(manifest.source_ids) == {"virtual_shizuoka", "plateau"}


def test_manifest_pins_registry_hash_and_preprocessing_version(assembler, inputs):
    manifest = assembler.assemble(TILE, inputs).manifest
    assert manifest.registry_hash
    assert manifest.preprocessing_version


def test_core_only_tile_is_attribution_only(assembler, inputs, gate):
    """PLATEAU + VIRTUAL SHIZUOKA ship under attribution alone."""
    bundle = assembler.assemble(TILE, inputs)
    assert bundle.redistribution_class(gate) == "attribution-only"


def test_osm_contribution_downgrades_the_tile_to_share_alike(assembler, inputs, gate):
    """The whole point of computing this: it cannot drift from what actually went in."""
    inputs.records.append(SourceRecord(source_id="osm", layers=["roads"]))
    bundle = assembler.assemble(TILE, inputs)
    assert bundle.redistribution_class(gate) == "share-alike"


def test_quarantined_source_is_refused_at_assembly(assembler, inputs):
    """A tile legitimate when built can become illegitimate if a licence is reclassified."""
    inputs.records.append(SourceRecord(source_id="gsi_dem", layers=["elevation"]))
    with pytest.raises(ProvenanceViolation, match="quarantined"):
        assembler.assemble(TILE, inputs)


def test_attribution_is_assembled_from_the_manifest(assembler, inputs, gate):
    lines = assembler.assemble(TILE, inputs).attribution(gate)
    assert any("PLATEAU" in line for line in lines)
    assert any("VIRTUAL SHIZUOKA" in line for line in lines)


def test_duplicate_source_record_is_refused(assembler, inputs):
    inputs.records.append(SourceRecord(source_id="plateau", layers=["roads"]))
    with pytest.raises(ValueError, match="already recorded"):
        assembler.assemble(TILE, inputs)


# ---------------------------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------------------------


def test_roundtrip_preserves_the_stack(assembler, inputs, tmp_path):
    bundle = assembler.assemble(TILE, inputs)
    write_tile(tmp_path, bundle)
    back = read_tile(tmp_path, TILE.id)
    assert np.array_equal(back.stack, bundle.stack)


def test_roundtrip_preserves_the_manifest(assembler, inputs, tmp_path):
    bundle = assembler.assemble(TILE, inputs)
    write_tile(tmp_path, bundle)
    back = read_tile(tmp_path, TILE.id)
    assert set(back.manifest.source_ids) == set(bundle.manifest.source_ids)
    assert back.manifest.registry_hash == bundle.manifest.registry_hash


def test_roundtrip_preserves_buildings(assembler, inputs, tmp_path):
    bundle = assembler.assemble(TILE, inputs)
    write_tile(tmp_path, bundle)
    back = read_tile(tmp_path, TILE.id)
    assert len(back.buildings) == len(bundle.buildings)
    assert {b.fine_type for b in back.buildings} == {b.fine_type for b in bundle.buildings}
    assert back.buildings[0].footprint[0] == pytest.approx(bundle.buildings[0].footprint[0])


def test_reading_a_stack_without_a_manifest_is_refused(assembler, inputs, tmp_path):
    """A stack without provenance is an orphan, not a tile."""
    bundle = assembler.assemble(TILE, inputs)
    out = write_tile(tmp_path, bundle)
    (out / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="not a tile"):
        read_tile(tmp_path, TILE.id)


def test_channel_drift_is_detected_on_read(assembler, inputs, tmp_path, spec):
    """Reinterpreting an old stack against a new spec would silently mislabel every channel."""
    bundle = assembler.assemble(TILE, inputs)
    write_tile(tmp_path, bundle)

    drifted = spec.model_copy(update={"channels": spec.channels[:-1]})
    with pytest.raises(ValueError, match="do not match the current stack spec"):
        read_tile(tmp_path, TILE.id, spec=drifted)


def test_list_and_index_tiles(assembler, inputs, tmp_path):
    write_tile(tmp_path, assembler.assemble(TILE, inputs))
    assert list_tiles(tmp_path) == [TILE.id]
    index = write_index(tmp_path, list_tiles(tmp_path))
    assert TILE.id in index.read_text(encoding="utf-8")
