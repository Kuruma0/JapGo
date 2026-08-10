"""Tests for CRS handling and the tile grid."""

from __future__ import annotations

import math

import pytest

from japgo.geo import (
    CORE_SIZE_M,
    HALO_M,
    MICRO,
    SCALE_TIERS,
    SHIZUOKA,
    Bounds,
    Tile,
    TileGrid,
    assert_metric,
    from_wgs84,
    parse_tile_id,
    tier_bounds,
    to_wgs84,
    zone,
)

# Shizuoka city centre, roughly.
SHIZUOKA_LON, SHIZUOKA_LAT = 138.383, 34.976


# ---------------------------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------------------------


def test_shizuoka_is_zone_8():
    assert SHIZUOKA.zone == 8
    assert SHIZUOKA.epsg == 6676


def test_zone_numbers_map_to_consecutive_epsg_codes():
    assert zone(1).epsg == 6669
    assert zone(19).epsg == 6687


@pytest.mark.parametrize("bad", [0, 20, -1])
def test_invalid_zone_is_rejected(bad):
    with pytest.raises(ValueError, match="1-19"):
        zone(bad)


def test_geographic_crs_is_refused_where_metric_is_required():
    """Computing setbacks and areas in degrees is the most destructive silent error available."""
    with pytest.raises(ValueError, match="geographic CRS"):
        assert_metric(4326)


def test_plane_rectangular_is_accepted_as_metric():
    assert_metric(SHIZUOKA.crs)


def test_roundtrip_projection_is_stable():
    x, y = from_wgs84(SHIZUOKA_LON, SHIZUOKA_LAT, SHIZUOKA.crs)
    lon, lat = to_wgs84(x, y, SHIZUOKA.crs)
    assert lon == pytest.approx(SHIZUOKA_LON, abs=1e-7)
    assert lat == pytest.approx(SHIZUOKA_LAT, abs=1e-7)


def test_projected_coordinates_stay_within_the_zone_s_low_distortion_band():
    """Zone 8's origin is 36°N, 138°30'E — north of Shizuoka, so northings run negative here.

    The property worth asserting is that the site sits well inside the ~130 km band where plane
    rectangular distortion stays negligible, not that it is close to the false origin.
    """
    x, y = from_wgs84(SHIZUOKA_LON, SHIZUOKA_LAT, SHIZUOKA.crs)
    assert abs(x) < 130_000
    assert -130_000 < y < 0  # south of the origin latitude


def test_one_degree_of_latitude_is_about_111_km():
    x0, y0 = from_wgs84(SHIZUOKA_LON, 35.0, SHIZUOKA.crs)
    x1, y1 = from_wgs84(SHIZUOKA_LON, 36.0, SHIZUOKA.crs)
    assert math.dist((x0, y0), (x1, y1)) == pytest.approx(110_900, rel=0.01)


# ---------------------------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------------------------


def test_tile_id_roundtrips():
    tile = Tile(zone=8, ix=123, iy=-45)
    assert parse_tile_id(tile.id) == tile


def test_tile_id_is_stable_and_readable():
    assert Tile(zone=8, ix=123, iy=45).id == "z08_x000123_y000045"


def test_malformed_tile_id_is_rejected():
    with pytest.raises(ValueError, match="malformed tile id"):
        parse_tile_id("not-a-tile")


def test_core_is_one_kilometre():
    core = Tile(zone=8, ix=0, iy=0).core
    assert core.width == CORE_SIZE_M
    assert core.height == CORE_SIZE_M


def test_halo_extends_the_read_extent_symmetrically():
    tile = Tile(zone=8, ix=0, iy=0)
    assert tile.read.width == CORE_SIZE_M + 2 * HALO_M
    assert tile.read.minx == -HALO_M


def test_adjacent_tiles_share_halo_but_not_core():
    """The reason random tile splits are invalid, expressed as a test."""
    a = Tile(zone=8, ix=0, iy=0)
    b = Tile(zone=8, ix=1, iy=0)
    assert not a.core.intersects(b.core)
    assert a.read.intersects(b.read)


def test_cores_tile_the_plane_without_gaps():
    grid = TileGrid(SHIZUOKA)
    area = Bounds(0, 0, 3000, 2000)
    tiles = list(grid.tiles_covering(area))
    assert len(tiles) == 6
    covered = sum(t.core.width * t.core.height for t in tiles)
    assert covered == pytest.approx(area.width * area.height)


def test_tile_at_is_consistent_with_tiles_covering():
    grid = TileGrid(SHIZUOKA)
    x, y = from_wgs84(SHIZUOKA_LON, SHIZUOKA_LAT, SHIZUOKA.crs)
    tile = grid.tile_at(x, y)
    assert tile.core.contains_point(x, y)
    assert tile in list(grid.tiles_covering(tile.core))


def test_neighbours_are_the_eight_adjacent_tiles():
    grid = TileGrid(SHIZUOKA)
    tile = Tile(zone=8, ix=5, iy=5)
    neighbours = grid.neighbours(tile)
    assert len(neighbours) == 8
    assert tile not in neighbours
    assert all(abs(n.ix - tile.ix) <= 1 and abs(n.iy - tile.iy) <= 1 for n in neighbours)


def test_negative_coordinates_floor_correctly():
    """Plane rectangular coordinates go negative south and west of the origin."""
    grid = TileGrid(SHIZUOKA)
    assert grid.tile_at(-1.0, -1.0) == Tile(zone=8, ix=-1, iy=-1)
    assert grid.tile_at(-1000.0, -1000.0) == Tile(zone=8, ix=-1, iy=-1)
    assert grid.tile_at(-1001.0, -1001.0) == Tile(zone=8, ix=-2, iy=-2)


def test_raster_shape_at_one_metre():
    tile = Tile(zone=8, ix=0, iy=0)
    assert tile.raster_shape(1.0) == (1000, 1000)
    assert tile.raster_shape(1.0, with_halo=True) == (1512, 1512)


# ---------------------------------------------------------------------------------------------
# Scale tiers
# ---------------------------------------------------------------------------------------------


def test_scale_tiers_increase_in_footprint_and_coarsen():
    resolutions = [t.resolution_m for t in SCALE_TIERS]
    footprints = [t.footprint_m for t in SCALE_TIERS]
    assert resolutions == sorted(resolutions)
    assert footprints == sorted(footprints)


def test_tier_bounds_are_centred_on_the_tile():
    tile = Tile(zone=8, ix=10, iy=10)
    for tier in SCALE_TIERS:
        b = tier_bounds(tile, tier)
        assert b.centre == pytest.approx(tile.core.centre)
        assert b.width == pytest.approx(tier.footprint_m)


def test_micro_tier_matches_the_read_extent():
    tile = Tile(zone=8, ix=0, iy=0)
    assert tier_bounds(tile, MICRO).as_tuple() == pytest.approx(tile.read.as_tuple())
