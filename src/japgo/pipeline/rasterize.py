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
from shapely.geometry import LineString, Polygon, mapping

from ..core.buildings import Building
from ..core.roads import RoadGraph
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


# ------------------------------------------------------------------------------------------------
# Roads — the prediction targets
# ------------------------------------------------------------------------------------------------


def _road_shapes(graph: RoadGraph, hierarchy, *, use_width: bool):
    """Road centrelines buffered to their carriageway width, ordered least-significant first.

    Ordering matters: where an expressway crosses a service road, the burned value should be the
    expressway's. Painting in dictionary order would make the raster depend on insertion order.
    """
    edges = sorted(
        graph.edges.values(),
        key=lambda e: -hierarchy.spec(e.road_class).rank,  # highest rank number burned first
    )
    for edge in edges:
        if len(edge.geometry) < 2:
            continue
        line = LineString(edge.geometry)
        if use_width:
            width = edge.width_m or hierarchy.spec(edge.road_class).typical_width_m
            geom = line.buffer(max(width, 0.5) / 2.0, cap_style=2)
        else:
            geom = line
        if not geom.is_empty:
            yield edge, geom


def road_mask(
    graph: RoadGraph,
    bounds: Bounds,
    resolution: float,
    crs,
    *,
    use_width: bool = True,
    hierarchy=None,
) -> Raster:
    """Binary road coverage — the primary prediction target.

    Burned at carriageway width rather than as hairlines, so that a 4 m service road and an 18 m
    expressway are distinguishable in the target itself. A model trained on hairlines learns
    centrelines and loses road width entirely, which spec §38 lists as a validation metric.

    ``all_touched=True``: a road narrower than one cell must still register. Dropping it would
    teach the model that minor roads do not exist, which is the opposite of what the dense local
    networks in Japanese cities require.
    """
    from ..core.roads import load_hierarchy

    hierarchy = hierarchy or load_hierarchy()
    shape = _shape(bounds, resolution)
    shapes = [(mapping(geom), 1) for _, geom in _road_shapes(graph, hierarchy, use_width=use_width)]

    if not shapes:
        return Raster(np.zeros(shape, np.float32), bounds, crs)

    burned = features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform_for(bounds, resolution),
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return Raster(burned.astype(np.float32), bounds, crs)


def road_class_raster(
    graph: RoadGraph,
    bounds: Bounds,
    resolution: float,
    crs,
    *,
    hierarchy=None,
) -> Raster:
    """Road hierarchy as an ordinal surface: 0 outside roads, higher where more significant.

    Ordinal rather than one-hot because road class genuinely *is* ordered — an expressway
    outranks an arterial outranks a lane — unlike building type, which is not.
    """
    from ..core.roads import load_hierarchy

    hierarchy = hierarchy or load_hierarchy()
    shape = _shape(bounds, resolution)

    max_rank = max(spec.rank for name, spec in hierarchy.classes.items() if name != "unknown")
    shapes = []
    for edge, geom in _road_shapes(graph, hierarchy, use_width=True):
        rank = hierarchy.spec(edge.road_class).rank
        if rank > max_rank:
            continue
        shapes.append((mapping(geom), float(max_rank - rank + 1)))

    if not shapes:
        return Raster(np.zeros(shape, np.float32), bounds, crs)

    burned = features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform_for(bounds, resolution),
        fill=0.0,
        dtype="float32",
        all_touched=True,
        merge_alg=features.MergeAlg.replace,
    )
    return Raster(burned.astype(np.float32), bounds, crs)


def road_orientation(
    graph: RoadGraph,
    bounds: Bounds,
    resolution: float,
    crs,
    *,
    hierarchy=None,
) -> tuple[Raster, Raster]:
    """Road bearing as (sin, cos) fields over the road mask.

    Predicting orientation alongside occupancy measurably improves topology in the road-extraction
    literature — it is what lets a model keep a road straight through an occlusion instead of
    breaking it. Bearings are doubled before the sin/cos so that a road and its reverse map to the
    same value: direction is meaningless here, axis is not.
    """
    from ..core.roads import load_hierarchy

    hierarchy = hierarchy or load_hierarchy()
    shape = _shape(bounds, resolution)
    transform = transform_for(bounds, resolution)

    sin_shapes, cos_shapes = [], []
    for edge, geom in _road_shapes(graph, hierarchy, use_width=True):
        angle = np.radians(2.0 * edge.bearing_deg())
        sin_shapes.append((mapping(geom), float(np.sin(angle))))
        cos_shapes.append((mapping(geom), float(np.cos(angle))))

    if not sin_shapes:
        zeros = np.zeros(shape, np.float32)
        return Raster(zeros.copy(), bounds, crs), Raster(zeros.copy(), bounds, crs)

    def _burn(shapes):
        return features.rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
            all_touched=True,
            merge_alg=features.MergeAlg.replace,
        ).astype(np.float32)

    return Raster(_burn(sin_shapes), bounds, crs), Raster(_burn(cos_shapes), bounds, crs)


def distance_to(mask: Raster, *, max_distance_m: float | None = None) -> Raster:
    """Euclidean distance in metres from every cell to the nearest set cell of ``mask``.

    Used for distance-to-road, distance-to-water and distance-to-rail. Where the mask is empty the
    field is uniformly ``max_distance_m`` (or the tile diagonal), which is the honest encoding of
    "nothing of this kind anywhere near".
    """
    from scipy import ndimage

    occupied = mask.data > 0
    if not occupied.any():
        far = max_distance_m if max_distance_m is not None else float(
            np.hypot(mask.bounds.width, mask.bounds.height)
        )
        return Raster(np.full(mask.data.shape, far, np.float32), mask.bounds, mask.crs)

    distance = ndimage.distance_transform_edt(~occupied, sampling=mask.resolution)
    distance = np.asarray(distance, dtype=np.float32)
    if max_distance_m is not None:
        distance = np.minimum(distance, max_distance_m)
    return Raster(distance, mask.bounds, mask.crs)


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
