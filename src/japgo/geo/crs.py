"""Coordinate reference systems.

All computation happens in a metric CRS. Every morphological measure in the specification is in
metres, so a geographic CRS would silently corrupt every one of them.

Japan Plane Rectangular CS on JGD2011 is the working CRS (EPSG:6669-6687, one per zone). WGS84
appears only at I/O boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)

# JGD2011 / Japan Plane Rectangular CS zones I..XIX are EPSG:6669..6687 in order.
_ZONE_I_EPSG = 6669
MIN_ZONE = 1
MAX_ZONE = 19


@dataclass(frozen=True)
class PlaneZone:
    """One Japan Plane Rectangular zone."""

    zone: int
    epsg: int
    note: str = ""

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)


def zone(number: int) -> PlaneZone:
    """Return the plane rectangular zone by its number (1-19)."""
    if not MIN_ZONE <= number <= MAX_ZONE:
        raise ValueError(f"Japan Plane Rectangular zone must be {MIN_ZONE}-{MAX_ZONE}, got {number}")
    return PlaneZone(zone=number, epsg=_ZONE_I_EPSG + number - 1)


# The MVP region. VIRTUAL SHIZUOKA is published in this zone already, so the largest and most
# awkward input needs no reprojection.
SHIZUOKA = zone(8)


@lru_cache(maxsize=64)
def transformer(src: CRS | str | int, dst: CRS | str | int) -> Transformer:
    """Cached transformer. ``always_xy`` so coordinates are consistently (x, y) / (lon, lat)."""
    return Transformer.from_crs(_as_crs(src), _as_crs(dst), always_xy=True)


def _as_crs(value: CRS | str | int) -> CRS:
    if isinstance(value, CRS):
        return value
    if isinstance(value, int):
        return CRS.from_epsg(value)
    return CRS.from_user_input(value)


def to_wgs84(x: float, y: float, src: CRS | str | int) -> tuple[float, float]:
    """Project a metric coordinate to (lon, lat)."""
    return transformer(src, WGS84).transform(x, y)


def from_wgs84(lon: float, lat: float, dst: CRS | str | int) -> tuple[float, float]:
    """Project (lon, lat) into a metric CRS."""
    return transformer(WGS84, dst).transform(lon, lat)


def assert_metric(crs: CRS | str | int) -> CRS:
    """Refuse a geographic CRS where a metric one is required.

    Cheap guard against the most destructive silent error available here: computing areas,
    distances and setbacks in degrees.
    """
    resolved = _as_crs(crs)
    if resolved.is_geographic:
        raise ValueError(
            f"{resolved.name!r} is a geographic CRS; metric units are required. "
            "Use a Japan Plane Rectangular zone (japgo.geo.crs.zone) for computation."
        )
    axis_units = {ax.unit_name for ax in resolved.axis_info}
    if not axis_units <= {"metre", "meter"}:
        raise ValueError(f"{resolved.name!r} axis units are {axis_units}, expected metres")
    return resolved
