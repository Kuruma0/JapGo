"""A minimal raster grid.

Deliberately thin. Rasterio owns file I/O; this type owns the in-memory contract between the
source adapters, the terrain derivatives and the tile assembler — an array plus enough georeference
to know where each cell is, in a metric CRS.

Row 0 is the **north** edge, matching the GeoTIFF convention and rasterio's default transform, so
that ``y`` decreases as the row index increases. Getting this backwards flips terrain aspect by 180
degrees and is invisible until a hillshade looks wrong.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from pyproj import CRS

from .crs import assert_metric
from .tiling import Bounds

NODATA = np.float32(np.nan)


@dataclass(frozen=True)
class Raster:
    """A georeferenced 2-D array in a metric CRS.

    ``data`` is indexed ``[row, col]``. ``bounds`` describes the outer edge of the cell block, not
    cell centres.
    """

    data: np.ndarray
    bounds: Bounds
    crs: CRS

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"raster data must be 2-D, got shape {self.data.shape}")
        assert_metric(self.crs)

    # -- geometry ---------------------------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def res_x(self) -> float:
        return self.bounds.width / self.width

    @property
    def res_y(self) -> float:
        return self.bounds.height / self.height

    @property
    def resolution(self) -> float:
        """Cell size, for the square-cell case this project always uses."""
        if not np.isclose(self.res_x, self.res_y):
            raise ValueError(f"non-square cells: {self.res_x} x {self.res_y}")
        return self.res_x

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.bounds.minx + (col + 0.5) * self.res_x,
            self.bounds.maxy - (row + 0.5) * self.res_y,
        )

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        """Row/col containing a projected coordinate. May be out of range; caller checks."""
        col = int(np.floor((x - self.bounds.minx) / self.res_x))
        row = int(np.floor((self.bounds.maxy - y) / self.res_y))
        return row, col

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def empty(cls, bounds: Bounds, resolution: float, crs: CRS, *, fill: float = np.nan) -> Raster:
        rows = int(round(bounds.height / resolution))
        cols = int(round(bounds.width / resolution))
        if rows <= 0 or cols <= 0:
            raise ValueError(f"bounds {bounds} at {resolution} m yields an empty raster")
        return cls(np.full((rows, cols), fill, dtype=np.float32), bounds, crs)

    # -- operations -------------------------------------------------------------------------

    @property
    def valid_mask(self) -> np.ndarray:
        return ~np.isnan(self.data)

    @property
    def coverage(self) -> float:
        """Fraction of cells with data. A low value on a terrain tile means gaps to fill."""
        return float(self.valid_mask.mean())

    def downsample(self, factor: int) -> Raster:
        """Block-mean downsample by an integer factor, ignoring NaN.

        Used to bring VIRTUAL SHIZUOKA's 0.5 m terrain to the 1 m working tier (research doc §8),
        so the model does not learn to depend on detail unavailable outside Shizuoka.
        """
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if factor == 1:
            return self

        h = self.height - self.height % factor
        w = self.width - self.width % factor
        if h == 0 or w == 0:
            raise ValueError(f"raster {self.height}x{self.width} too small to downsample by {factor}")

        block = self.data[:h, :w].reshape(h // factor, factor, w // factor, factor)
        # An all-NaN block is a legitimate void, not an anomaly. numpy raises "Mean of empty slice"
        # through the warnings module, which errstate does not intercept.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced = np.nanmean(block, axis=(1, 3)).astype(np.float32)

        return Raster(
            reduced,
            Bounds(
                self.bounds.minx,
                self.bounds.maxy - h * self.res_y,
                self.bounds.minx + w * self.res_x,
                self.bounds.maxy,
            ),
            self.crs,
        )

    def fill_gaps(self, max_iterations: int = 8) -> Raster:
        """Fill NaN cells by iterative mean of valid 4-neighbours.

        Cheap and adequate for the small voids left by LiDAR occlusion — building shadows, water
        surfaces that absorb the return. Not a substitute for a proper interpolator on large gaps,
        which is why :attr:`coverage` is reported rather than silently repaired.
        """
        out = self.data.copy()
        for _ in range(max_iterations):
            holes = np.isnan(out)
            if not holes.any():
                break
            padded = np.pad(out, 1, constant_values=np.nan)
            stack = np.stack(
                [
                    padded[:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, :-2],
                    padded[1:-1, 2:],
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                neighbour_mean = np.nanmean(stack, axis=0)
            out = np.where(holes & ~np.isnan(neighbour_mean), neighbour_mean, out)
        return Raster(out.astype(np.float32), self.bounds, self.crs)

    def clip_to(self, bounds: Bounds) -> Raster:
        """Crop to a sub-extent, snapping outward to whole cells."""
        r0, c0 = self.index_of(bounds.minx, bounds.maxy)
        r1, c1 = self.index_of(bounds.maxx, bounds.miny)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.height, r1 + 1), min(self.width, c1 + 1)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("clip bounds do not intersect the raster")

        return Raster(
            self.data[r0:r1, c0:c1].copy(),
            Bounds(
                self.bounds.minx + c0 * self.res_x,
                self.bounds.maxy - r1 * self.res_y,
                self.bounds.minx + c1 * self.res_x,
                self.bounds.maxy - r0 * self.res_y,
            ),
            self.crs,
        )
