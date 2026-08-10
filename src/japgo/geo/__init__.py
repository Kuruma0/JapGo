"""Geometry, projection and tiling primitives."""

from .crs import SHIZUOKA, WGS84, PlaneZone, assert_metric, from_wgs84, to_wgs84, zone
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
    "REGIONAL",
    "SCALE_TIERS",
    "SHIZUOKA",
    "WGS84",
    "Bounds",
    "PlaneZone",
    "ScaleTier",
    "Tile",
    "TileGrid",
    "assert_metric",
    "from_wgs84",
    "parse_tile_id",
    "tier_bounds",
    "to_wgs84",
    "zone",
]
