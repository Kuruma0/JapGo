"""VIRTUAL SHIZUOKA adapter — terrain, and the imagery that closed risk R1d.

Shizuoka Prefecture publishes prefecture-wide aerial LiDAR and mobile mapping under a
**CC BY 4.0 / ODbL dual licence**, from which this project elects CC BY 4.0 (registry:
``virtual_shizuoka``). That election keeps terrain out of the share-alike analysis entirely.

The package ships several products. Two matter here:

* a **pre-gridded 0.5 m elevation model**, and
* the **raw LAS point clouds**, both original and ground-filtered.

Prefer the gridded product: it is authoritative, already in JGD2011 Plane Rectangular Zone 8, and
avoids re-deriving what the publisher already derived. Fall back to gridding from LAS when a tile
is not covered by the grid, or when bare-earth control is needed that the grid does not expose.

**Bare earth is not optional.** A DSM includes tree canopy and rooftops; slope computed over it is
slope over vegetation, not ground. Since §12 assigns grade compliance to the deterministic half of
the system where it must be exact, ingesting a DSM as if it were a DTM would make an exact
constraint enforce the wrong thing — silently. This adapter therefore filters to ASPRS class 2
(ground) by default and records what it did.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..geo.crs import assert_metric
from ..geo.raster import Raster
from ..geo.tiling import Bounds
from .base import ReadResult, SourceAdapter

log = logging.getLogger(__name__)

#: ASPRS standard classification codes present in VIRTUAL SHIZUOKA LAS files.
ASPRS_GROUND = 2
ASPRS_WATER = 9

NATIVE_RESOLUTION_M = 0.5
"""Published grid spacing. The working tier is 1 m — see research doc §8."""


class VirtualShizuokaAdapter(SourceAdapter):
    """Reads VIRTUAL SHIZUOKA LAS point clouds into a gridded DEM."""

    source_id = "virtual_shizuoka"
    provides = ("elevation", "point_cloud", "aerial_rgb", "contours", "water_polygons")

    def __init__(self, gate, *, target_crs=None) -> None:
        super().__init__(gate)
        # Published in JGD2011 Plane Rectangular Zone 8, which is also the MVP working CRS, so the
        # largest and most awkward input needs no reprojection.
        from ..geo.crs import SHIZUOKA

        self.target_crs = assert_metric(target_crs or SHIZUOKA.crs)
        self._cache_key: tuple[str, bool] | None = None
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None = None

    # -----------------------------------------------------------------------------------------

    def load_points(
        self, path: Path, *, ground_only: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Parse a LAS/LAZ file into filtered (x, y, z) arrays, cached by path.

        A tile-by-tile build otherwise re-parses the same ~300 MB file once per overlapping tile.
        The cache holds one file, which is the right size for a row-major tile walk: consecutive
        tiles hit the same file, and moving to the next file evicts the previous one rather than
        accumulating hundreds of megabytes.
        """
        key = (str(path), ground_only)
        if self._cache_key == key and self._cache is not None:
            return self._cache

        import laspy

        with laspy.open(path) as reader:
            points = reader.read()

        x = np.asarray(points.x, dtype=np.float64)
        y = np.asarray(points.y, dtype=np.float64)
        z = np.asarray(points.z, dtype=np.float64)
        classification = np.asarray(getattr(points, "classification", []), dtype=np.uint8)

        warnings: list[str] = []
        if ground_only:
            if classification.size == x.size and np.any(classification == ASPRS_GROUND):
                keep = classification == ASPRS_GROUND
                x, y, z = x[keep], y[keep], z[keep]
            else:
                warnings.append(
                    f"{Path(path).name}: ground_only requested but the file carries no ASPRS "
                    "class 2 returns; the result is a SURFACE model (DSM), not bare earth. Slope "
                    "and grade derived from it will include canopy and rooftops."
                )

        self._cache_key = key
        self._cache = (x, y, z, warnings)
        return self._cache

    def read(
        self,
        path: Path,
        *,
        bounds: Bounds | None = None,
        resolution: float = NATIVE_RESOLUTION_M,
        ground_only: bool = True,
        **kwargs,
    ) -> ReadResult:
        """Grid a LAS/LAZ file into an elevation :class:`~japgo.geo.raster.Raster`.

        Parameters
        ----------
        bounds:
            Restrict to this extent. Files average ~300 MB and reach 5.6 GB, so a windowed read is
            the normal case rather than an optimisation.
        ground_only:
            Filter to ASPRS class 2. Leave this true unless you specifically want a surface model.
        """
        self.open()  # provenance gate

        path = Path(path)
        x, y, z, load_warnings = self.load_points(path, ground_only=ground_only)
        warnings = list(load_warnings)
        total = x.size

        if x.size == 0:
            raise ValueError(f"{path}: no points remain after filtering")

        extent = bounds or Bounds(
            float(np.floor(x.min())),
            float(np.floor(y.min())),
            float(np.ceil(x.max())),
            float(np.ceil(y.max())),
        )

        dem = _grid_points(x, y, z, extent, resolution, self.target_crs)

        if dem.coverage == 0.0:
            # No overlap at all is a caller error — wrong extent, wrong zone, or wrong file — not
            # a data gap. Returning an all-NaN raster would propagate a valid-looking empty tile.
            raise ValueError(
                f"{path.name}: no points fall inside {extent}. The file spans "
                f"x=[{x.min():.0f}, {x.max():.0f}] y=[{y.min():.0f}, {y.max():.0f}]. "
                "Check the requested bounds and that both are in the same CRS."
            )

        if dem.coverage < 1.0:
            warnings.append(
                f"{path.name}: {(1 - dem.coverage) * 100:.1f}% of cells have no return "
                "(occlusion, water absorption); call Raster.fill_gaps() or accept the voids"
            )

        for w in warnings:
            log.warning(w)

        return ReadResult(
            layers={"elevation": [dem]},
            record=self.make_record(
                layers=["elevation"],
                note=(
                    f"{path.name}; {'bare-earth (ASPRS 2)' if ground_only else 'all returns'}; "
                    f"{resolution} m grid; {x.size}/{total} points used"
                ),
            ),
            warnings=warnings,
        )

    # -----------------------------------------------------------------------------------------

    def read_to_working_tier(self, path: Path, **kwargs) -> ReadResult:
        """Read at native 0.5 m and downsample to the 1 m working tier.

        Research doc §8: the model consumes terrain at a common 1 m resolution so that it cannot
        learn to depend on 0.5 m detail that exists nowhere outside this prefecture. The native
        data is retained separately for validation and high-detail export.
        """
        result = self.read(path, resolution=NATIVE_RESOLUTION_M, **kwargs)
        dem: Raster = result.layers["elevation"][0]
        result.layers["elevation"] = [dem.downsample(2)]
        result.layers["elevation_native"] = [dem]
        if result.record is not None:
            result.record = result.record.model_copy(
                update={
                    "layers": ["elevation", "elevation_native"],
                    "note": f"{result.record.note}; downsampled 0.5 m -> 1 m working tier",
                }
            )
        return result


def _grid_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: Bounds,
    resolution: float,
    crs,
) -> Raster:
    """Bin points to a grid, taking the mean elevation per cell.

    Mean rather than minimum: with ground-classified returns the spread within a 0.5 m cell is
    measurement noise, and the minimum would systematically bias terrain downward.
    """
    rows = int(round(bounds.height / resolution))
    cols = int(round(bounds.width / resolution))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"bounds {bounds} at {resolution} m yields an empty raster")

    # Cheap bounding-box filter first. A production LAS file covers far more ground than one tile,
    # so computing row/col indices over every point in the file and discarding most of them is the
    # dominant cost of a build. Two comparisons per axis are much cheaper than two divisions.
    near = (x >= bounds.minx) & (x < bounds.maxx) & (y >= bounds.miny) & (y < bounds.maxy)
    if not near.all():
        x, y, z = x[near], y[near], z[near]

    if x.size == 0:
        return Raster(np.full((rows, cols), np.nan, np.float32), bounds, crs)

    col = np.floor((x - bounds.minx) / resolution).astype(np.int64)
    row = np.floor((bounds.maxy - y) / resolution).astype(np.int64)

    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    if not inside.all():
        col, row, z = col[inside], row[inside], z[inside]

    flat = row * cols + col
    counts = np.bincount(flat, minlength=rows * cols)
    sums = np.bincount(flat, weights=z, minlength=rows * cols)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    return Raster(mean.reshape(rows, cols).astype(np.float32), bounds, crs)
