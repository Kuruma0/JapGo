"""Vector to raster conversion.

Buildings arrive as footprint polygons and must reach the model as raster channels. Rasterio's
``features.rasterize`` does the work; this module owns the conventions — extent, transform sign,
and what "height" means for overlapping footprints.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import Polygon, mapping

from ..core.buildings import Building
from ..geo.raster import Raster
from ..geo.tiling import Bounds


def transform_for(bounds: Bounds, resolution: float) -> Affine:
    """Affine transform for a north-up raster over ``bounds``.

    The negative y-scale is what makes row 0 the north edge, matching
    :class:`~japgo.geo.raster.Raster`.
    """
    return Affine(resolution, 0.0, bounds.minx, 0.0, -resolution, bounds.maxy)


def _shape(bounds: Bounds, resolution: float) -> tuple[int, int]:
    return int(round(bounds.height / resolution)), int(round(bounds.width / resolution))


def _polygons(buildings: Iterable[Building]) -> list[tuple[Building, Polygon]]:
    out = []
    for b in buildings:
        if len(b.footprint) < 4:
            continue
        poly = Polygon(b.footprint)
        if not poly.is_valid:
            poly = poly.buffer(0)  # standard self-intersection repair
        if poly.is_empty or poly.area <= 0:
            continue
        out.append((b, poly))
    return out


def building_mask(
    buildings: Iterable[Building],
    bounds: Bounds,
    resolution: float,
    crs,
) -> Raster:
    """Binary coverage mask: 1 where a building footprint falls."""
    shape = _shape(bounds, resolution)
    pairs = _polygons(buildings)

    if not pairs:
        return Raster(np.zeros(shape, np.float32), bounds, crs)

    burned = features.rasterize(
        ((mapping(poly), 1) for _, poly in pairs),
        out_shape=shape,
        transform=transform_for(bounds, resolution),
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return Raster(burned.astype(np.float32), bounds, crs)


def building_height(
    buildings: Iterable[Building],
    bounds: Bounds,
    resolution: float,
    crs,
) -> Raster:
    """Height surface in metres, 0 outside footprints.

    Buildings are burned **tallest last** so that where footprints overlap — which happens at
    LOD1 with adjoining structures — the taller one wins. Painting in arbitrary order would make
    the height field depend on file ordering, which is exactly the kind of irreproducibility §44
    forbids.

    Buildings with neither a measured height nor a storey count are burned at 0 rather than
    dropped: their footprint is real even when their height is unknown, and ``building_mask``
    still records them.
    """
    shape = _shape(bounds, resolution)
    pairs = _polygons(buildings)

    shapes = []
    for b, poly in pairs:
        height = b.estimated_height_m
        if height is None:
            continue
        shapes.append((mapping(poly), float(height)))

    if not shapes:
        return Raster(np.zeros(shape, np.float32), bounds, crs)

    shapes.sort(key=lambda pair: pair[1])  # tallest burned last

    burned = features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform_for(bounds, resolution),
        fill=0.0,
        dtype="float32",
        all_touched=False,
        merge_alg=features.MergeAlg.replace,
    )
    return Raster(burned.astype(np.float32), bounds, crs)


def building_class_mask(
    buildings: Iterable[Building],
    coarse_type: str,
    bounds: Bounds,
    resolution: float,
    crs,
) -> Raster:
    """Binary mask for one coarse building type.

    One channel per coarse type rather than a single categorical channel: a convolution over
    category *indices* would imply that residential(1) is nearer commercial(2) than industrial(3),
    which is meaningless.
    """
    selected = [b for b in buildings if b.coarse_type == coarse_type]
    return building_mask(selected, bounds, resolution, crs)


def density(
    values: Sequence[tuple[float, float]],
    bounds: Bounds,
    resolution: float,
    crs,
    *,
    radius_m: float,
) -> Raster:
    """Point density per square kilometre, box-filtered at ``radius_m``.

    Used for population and, later, POI density.
    """
    rows, cols = _shape(bounds, resolution)
    counts = np.zeros((rows, cols), np.float32)

    for x, y in values:
        col = int(np.floor((x - bounds.minx) / resolution))
        row = int(np.floor((bounds.maxy - y) / resolution))
        if 0 <= row < rows and 0 <= col < cols:
            counts[row, col] += 1

    window = max(1, int(round(2 * radius_m / resolution)))
    if window > 1:
        kernel = np.ones((window, window), np.float32)
        padded = np.pad(counts, window // 2, mode="constant")
        summed = np.zeros_like(counts)
        for dr in range(window):
            for dc in range(window):
                summed += padded[dr : dr + rows, dc : dc + cols]
        counts = summed
        del kernel

    cell_area_km2 = (resolution * resolution) / 1e6
    area = cell_area_km2 * (window * window)
    return Raster(counts / max(area, 1e-9), bounds, crs)
