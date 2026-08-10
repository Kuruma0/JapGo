"""Terrain derivatives.

Slope, aspect and roughness from a DEM. These are the channels through which terrain reaches the
model, and — via the maximum-grade constraint — the channels through which the deterministic half
of the system enforces buildability (research doc §13).

Correctness matters more here than almost anywhere else in the pipeline, for a reason worth
stating: §12 assigns grade compliance to the procedural side, where it is expected to be *exact*.
A slope field that is quietly wrong makes an exact constraint enforce the wrong thing.

Slope and aspect use **Horn's method** — the 3x3 weighted difference used by GDAL and ArcGIS —
rather than a simple central difference, because it is markedly less noisy on real LiDAR terrain.
"""

from __future__ import annotations

import numpy as np

from .raster import Raster


def _horn_gradients(dem: Raster) -> tuple[np.ndarray, np.ndarray]:
    """Return (dz/dx, dz/dy) in metres per metre, using Horn's 3x3 operator.

    ``dz/dy`` is expressed in **map** orientation (positive northward), which requires flipping
    the sign of the row-wise difference because row 0 is the north edge.
    """
    z = np.pad(dem.data.astype(np.float64), 1, mode="edge")

    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, _, f = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dem.res_x)
    # Rows increase southward, so a positive row-difference is a *decrease* in northing.
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dem.res_y)
    dz_dy = -dz_dy

    return dz_dx, dz_dy


def slope(dem: Raster, *, as_percent: bool = False) -> Raster:
    """Slope magnitude, in degrees by default.

    ``as_percent=True`` returns grade (rise/run x 100), which is the unit road engineering uses and
    the unit the maximum-grade constraint is expressed in.
    """
    dz_dx, dz_dy = _horn_gradients(dem)
    magnitude = np.hypot(dz_dx, dz_dy)
    values = magnitude * 100.0 if as_percent else np.degrees(np.arctan(magnitude))
    return Raster(_restore_nodata(values, dem), dem.bounds, dem.crs)


def aspect(dem: Raster) -> Raster:
    """Downslope direction in compass degrees (0 = north, 90 = east).

    Flat cells — where the gradient vanishes — are returned as NaN rather than 0, because 0 is a
    real direction and conflating "north-facing" with "flat" corrupts any circular statistic
    computed downstream.
    """
    dz_dx, dz_dy = _horn_gradients(dem)

    # Downslope direction is the negative gradient. atan2(east, north) gives compass bearing.
    bearing = np.degrees(np.arctan2(-dz_dx, -dz_dy))
    bearing = np.mod(bearing, 360.0)

    flat = np.isclose(dz_dx, 0.0) & np.isclose(dz_dy, 0.0)
    bearing = np.where(flat, np.nan, bearing)

    return Raster(_restore_nodata(bearing, dem), dem.bounds, dem.crs)


def roughness(dem: Raster, window: int = 3) -> Raster:
    """Terrain roughness as the local standard deviation of elevation.

    A scale-aware measure of how broken the ground is — the signal that separates a valley floor
    from the slope above it more cleanly than slope alone.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    pad = window // 2
    z = np.pad(dem.data.astype(np.float64), pad, mode="edge")

    windows = np.lib.stride_tricks.sliding_window_view(z, (window, window))
    with np.errstate(invalid="ignore"):
        values = np.nanstd(windows, axis=(-2, -1))

    return Raster(_restore_nodata(values, dem), dem.bounds, dem.crs)


def hillshade(dem: Raster, *, azimuth: float = 315.0, altitude: float = 45.0) -> Raster:
    """Shaded relief, 0-255. For the Phase 2 visualiser.

    Cheap and disproportionately useful: a hillshade makes a wrong sign convention or a misaligned
    tile obvious to a human in one glance, which is exactly what the Phase 2 gate is for.
    """
    dz_dx, dz_dy = _horn_gradients(dem)

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect_rad = np.arctan2(-dz_dx, -dz_dy)

    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)

    shaded = np.sin(alt) * np.cos(slope_rad) + np.cos(alt) * np.sin(slope_rad) * np.cos(
        az - aspect_rad
    )
    values = np.clip(shaded * 255.0, 0, 255)
    return Raster(_restore_nodata(values, dem), dem.bounds, dem.crs)


def exceeds_grade(dem: Raster, max_grade_percent: float) -> np.ndarray:
    """Boolean mask of cells too steep for a road at the given maximum grade.

    The deterministic constraint from research doc §13. Japanese national highway design generally
    caps grade around 8-12% depending on class and design speed; mountain roads use switchbacks
    precisely to stay inside that envelope, which is the behaviour the Kawanehon site should
    exhibit and the model should reproduce.
    """
    grade = slope(dem, as_percent=True).data
    return np.nan_to_num(grade, nan=0.0) > max_grade_percent


def _restore_nodata(values: np.ndarray, dem: Raster) -> np.ndarray:
    """Propagate the DEM's NaN mask into a derived array.

    ``np.pad(mode="edge")`` would otherwise invent plausible-looking values at the margins of a
    void, which is worse than an honest gap.
    """
    out = values.astype(np.float32)
    out[np.isnan(dem.data)] = np.nan
    return out
