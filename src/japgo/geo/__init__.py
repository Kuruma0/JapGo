"""Geometry, projection and tiling primitives."""

from .crs import SHIZUOKA, WGS84, PlaneZone, assert_metric, from_wgs84, to_wgs84, zone
from .raster import NODATA, Raster
from .terrain import aspect, exceeds_grade, hillshade, roughness, slope
from .tiling import (
    CORE_SIZE_M,
    HALO_M,
    MICRO,
    NEIGHBOURHOOD,
    REGIONAL,
    SCALE_TIERS,
    Bounds,
    ScaleTier,
    Tile,
    TileGrid,
    parse_tile_id,
    tier_bounds,
)

__all__ = [
    "CORE_SIZE_M",
    "HALO_M",
    "MICRO",
    "NEIGHBOURHOOD",
    "NODATA",
    "REGIONAL",
    "SCALE_TIERS",
    "SHIZUOKA",
    "WGS84",
    "Bounds",
    "PlaneZone",
    "Raster",
    "ScaleTier",
    "Tile",
    "TileGrid",
    "aspect",
    "assert_metric",
    "exceeds_grade",
    "from_wgs84",
    "hillshade",
    "parse_tile_id",
    "roughness",
    "slope",
    "tier_bounds",
    "to_wgs84",
    "zone",
]
