"""The tile grid.

A fixed-extent geographic tile is the unit of everything: dataset sample, model input, cache
entry, and export chunk.

Two properties are load-bearing and must not be changed casually once data exists:

* **1 km core.** The predicted extent.
* **256 m halo.** Read but never predicted into. The halo is what stops roads dead-ending at tile
  seams. Retrofitting it invalidates every cached sample, which is why it exists from day one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from pyproj import CRS

from .crs import PlaneZone, assert_metric

CORE_SIZE_M = 1000.0
HALO_M = 256.0


@dataclass(frozen=True, slots=True)
class Bounds:
    """Axis-aligned metric bounds, ``min`` inclusive / ``max`` exclusive."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def centre(self) -> tuple[float, float]:
        return (self.minx + self.width / 2, self.miny + self.height / 2)

    def buffered(self, distance: float) -> Bounds:
        return Bounds(
            self.minx - distance,
            self.miny - distance,
            self.maxx + distance,
            self.maxy + distance,
        )

    def intersects(self, other: Bounds) -> bool:
        return not (
            self.maxx <= other.minx
            or other.maxx <= self.minx
            or self.maxy <= other.miny
            or other.maxy <= self.miny
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.minx <= x < self.maxx and self.miny <= y < self.maxy

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


@dataclass(frozen=True, slots=True)
class Tile:
    """One tile on the grid.

    Tile indices are absolute in the zone's projected coordinate space, so a tile id identifies a
    fixed patch of ground independent of which region happened to be requested.
    """

    zone: int
    ix: int
    iy: int
    core_size_m: float = CORE_SIZE_M
    halo_m: float = HALO_M

    @property
    def id(self) -> str:
        """Stable identifier, e.g. ``z08_x000123_y-00045``."""
        return f"z{self.zone:02d}_x{self.ix:06d}_y{self.iy:06d}"

    @property
    def core(self) -> Bounds:
        return Bounds(
            self.ix * self.core_size_m,
            self.iy * self.core_size_m,
            (self.ix + 1) * self.core_size_m,
            (self.iy + 1) * self.core_size_m,
        )

    @property
    def read(self) -> Bounds:
        """Core plus halo — the extent to read inputs over."""
        return self.core.buffered(self.halo_m)

    def raster_shape(self, resolution_m: float, *, with_halo: bool = False) -> tuple[int, int]:
        """Pixel shape (rows, cols) at a given resolution."""
        b = self.read if with_halo else self.core
        return (
            int(round(b.height / resolution_m)),
            int(round(b.width / resolution_m)),
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.id


def parse_tile_id(tile_id: str, *, core_size_m: float = CORE_SIZE_M, halo_m: float = HALO_M) -> Tile:
    """Inverse of :attr:`Tile.id`."""
    try:
        z, x, y = tile_id.split("_")
        return Tile(
            zone=int(z[1:]),
            ix=int(x[1:]),
            iy=int(y[1:]),
            core_size_m=core_size_m,
            halo_m=halo_m,
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"malformed tile id {tile_id!r}; expected like 'z08_x000123_y000045'") from exc


class TileGrid:
    """A tile grid anchored to a projected zone."""

    def __init__(
        self,
        zone: PlaneZone,
        *,
        core_size_m: float = CORE_SIZE_M,
        halo_m: float = HALO_M,
    ) -> None:
        self.zone = zone
        self.crs: CRS = assert_metric(zone.crs)
        self.core_size_m = float(core_size_m)
        self.halo_m = float(halo_m)

    def tile_at(self, x: float, y: float) -> Tile:
        """The tile whose *core* contains a projected coordinate."""
        return Tile(
            zone=self.zone.zone,
            ix=math.floor(x / self.core_size_m),
            iy=math.floor(y / self.core_size_m),
            core_size_m=self.core_size_m,
            halo_m=self.halo_m,
        )

    def tiles_covering(self, bounds: Bounds) -> Iterator[Tile]:
        """Every tile whose core intersects ``bounds``, in row-major order."""
        x0 = math.floor(bounds.minx / self.core_size_m)
        x1 = math.ceil(bounds.maxx / self.core_size_m)
        y0 = math.floor(bounds.miny / self.core_size_m)
        y1 = math.ceil(bounds.maxy / self.core_size_m)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                yield Tile(
                    zone=self.zone.zone,
                    ix=ix,
                    iy=iy,
                    core_size_m=self.core_size_m,
                    halo_m=self.halo_m,
                )

    def neighbours(self, tile: Tile) -> list[Tile]:
        """The eight adjacent tiles. Used by the seam-merge pass."""
        return [
            Tile(tile.zone, tile.ix + dx, tile.iy + dy, self.core_size_m, self.halo_m)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        ]

    def count_covering(self, bounds: Bounds) -> int:
        x0 = math.floor(bounds.minx / self.core_size_m)
        x1 = math.ceil(bounds.maxx / self.core_size_m)
        y0 = math.floor(bounds.miny / self.core_size_m)
        y1 = math.ceil(bounds.maxy / self.core_size_m)
        return max(0, x1 - x0) * max(0, y1 - y0)


# --------------------------------------------------------------------------------------------
# Multi-scale tiers (research doc §8)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScaleTier:
    """One resolution tier of the hierarchical context.

    The same tile is materialised at three resolutions so the model reads local and regional
    context together, without a separate regional pipeline.
    """

    name: str
    resolution_m: float
    footprint_m: float
    """Extent read around the tile centre at this tier."""


MICRO = ScaleTier("micro", resolution_m=1.0, footprint_m=CORE_SIZE_M + 2 * HALO_M)
NEIGHBOURHOOD = ScaleTier("neighbourhood", resolution_m=8.0, footprint_m=8_000.0)
REGIONAL = ScaleTier("regional", resolution_m=64.0, footprint_m=64_000.0)

SCALE_TIERS = (MICRO, NEIGHBOURHOOD, REGIONAL)


def tier_bounds(tile: Tile, tier: ScaleTier) -> Bounds:
    """Extent to read for a tile at a given tier, centred on the tile core."""
    cx, cy = tile.core.centre
    half = tier.footprint_m / 2
    return Bounds(cx - half, cy - half, cx + half, cy + half)
