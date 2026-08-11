"""Road probability raster -> centreline -> :class:`~japgo.core.roads.RoadGraph`.

This is the procedural half of the §13 hand-off. The interface between the two halves is
deliberately a probability field, not a finished graph — ML proposes where roads want to be, and
deterministic code decides what network that actually is (invariant 5).

Four steps, each with a way of going wrong that is worth naming:

1. **Threshold.** Turns a probability into a footprint. Too low and the network fuses into blobs;
   too high and it fragments. Chosen by the caller, because the right value depends on the model's
   calibration and Phase 4's baseline over-paints.
2. **Skeletonise.** Reduces the footprint to a one-pixel centreline. Taken from scikit-image
   rather than hand-rolled: a stray 8-connected diagonal in a home-grown thinning becomes a
   spurious junction, and junction count is a headline TOPO term.
3. **Trace.** Walks the skeleton into nodes and polylines. Junctions are pixels with three or more
   skeleton neighbours, endpoints have one; everything between is an edge.
4. **Simplify.** Douglas–Peucker on each polyline, then merge nodes closer than a tolerance.
   Skeleton pixels are a staircase, and an unsimplified graph has a vertex every metre — which
   inflates every length measure and makes sinuosity meaningless.

The output carries ``source_id="model"`` and a per-edge confidence, so a predicted network is
never mistaken for observed geometry downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.roads import Edge, Node, NodeKind, RoadGraph
from ..geo.tiling import Bounds

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class ExtractionSpec:
    """Knobs, kept in one place so an extraction is reproducible from a record of them."""

    threshold: float = 0.5
    min_branch_px: int = 12
    """Spurs shorter than this are pruned. Skeletonisation frays at blob edges, and a 3 px hair is
    a dead end in every topology metric that counts them."""

    simplify_m: float = 2.0
    merge_nodes_m: float = 6.0
    """Junctions within this distance collapse. A crossroads skeletonises to two Y-junctions a few
    pixels apart, which doubles the intersection count if left alone."""


def extract_graph(
    probability: np.ndarray,
    bounds: Bounds,
    crs,
    *,
    spec: ExtractionSpec | None = None,
    resolution: float | None = None,
    tile_id: str | None = None,
) -> RoadGraph:
    """Vectorise a probability raster into a road graph in the raster's CRS."""
    spec = spec or ExtractionSpec()
    rows, cols = probability.shape
    resolution = resolution or (bounds.width / cols)

    mask = np.asarray(probability) >= spec.threshold
    if not mask.any():
        return RoadGraph(crs=str(crs), tile_id=tile_id)

    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    paths = _trace(skeleton, min_branch_px=spec.min_branch_px)

    graph = RoadGraph(crs=str(crs), tile_id=tile_id)
    positions: dict[tuple[int, int], str] = {}

    def node_for(pixel: tuple[int, int]) -> str:
        if pixel not in positions:
            x, y = _to_world(pixel, bounds, resolution)
            nid = f"n{len(positions)}"
            positions[pixel] = nid
            graph.add_node(Node(id=nid, x=x, y=y, kind=NodeKind.INTERSECTION, source_id="model"))
        return positions[pixel]

    for index, path in enumerate(paths):
        simplified = _simplify([_to_world(p, bounds, resolution) for p in path], spec.simplify_m)
        if len(simplified) < 2:
            continue
        u, v = node_for(path[0]), node_for(path[-1])
        if u == v:
            continue  # a loop closing on one pixel carries no length worth keeping
        graph.add_edge(
            Edge(
                id=f"e{index}",
                u=u,
                v=v,
                geometry=simplified,
                road_class="unknown",
                source_id="model",
                confidence=float(np.mean([probability[p] for p in path])),
            )
        )

    return _merge_close_nodes(graph, spec.merge_nodes_m)


# -- tracing ------------------------------------------------------------------------------------


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    """Skeleton neighbours per pixel, by shifting rather than looping."""
    padded = np.pad(skeleton.astype(np.uint8), 1)
    total = np.zeros(skeleton.shape, dtype=np.uint8)
    for dr, dc in NEIGHBOURS:
        total += padded[1 + dr : 1 + dr + skeleton.shape[0], 1 + dc : 1 + dc + skeleton.shape[1]]
    return total * skeleton


def _trace(skeleton: np.ndarray, *, min_branch_px: int) -> list[list[tuple[int, int]]]:
    """Walk the skeleton into polylines between junctions and endpoints."""
    counts = _neighbour_count(skeleton)
    nodes = {tuple(p) for p in np.argwhere((counts == 1) | (counts >= 3))}
    occupied = {tuple(p) for p in np.argwhere(skeleton)}

    paths: list[list[tuple[int, int]]] = []
    walked: set[frozenset] = set()

    def neighbours(p):
        return [
            (p[0] + dr, p[1] + dc)
            for dr, dc in NEIGHBOURS
            if (p[0] + dr, p[1] + dc) in occupied
        ]

    for start in nodes:
        for first in neighbours(start):
            if frozenset((start, first)) in walked:
                continue
            path = [start, first]
            walked.add(frozenset((start, first)))
            current, previous = first, start
            while current not in nodes:
                nxt = [n for n in neighbours(current) if n != previous]
                if not nxt:
                    break
                previous, current = current, nxt[0]
                walked.add(frozenset((previous, current)))
                path.append(current)
            # A spur is a path ending at a degree-1 pixel that is not a real cul-de-sac, just
            # fraying where the mask was fat. Length is the only signal available to tell them
            # apart, so it is a threshold rather than a rule.
            if len(path) < min_branch_px and counts[current] == 1 and counts[start] >= 3:
                continue
            paths.append(path)

    # Rings touch no junction at all and would otherwise vanish.
    seen = {p for path in paths for p in path}
    for pixel in sorted(occupied - seen):
        if counts[pixel] != 2:
            continue
        ring, current, previous = [pixel], pixel, None
        while True:
            nxt = [n for n in neighbours(current) if n != previous and n not in ring]
            if not nxt:
                break
            previous, current = current, nxt[0]
            ring.append(current)
        if len(ring) >= min_branch_px:
            paths.append(ring + [ring[0]])
            seen.update(ring)

    return paths


# -- geometry -----------------------------------------------------------------------------------


def _to_world(pixel: tuple[int, int], bounds: Bounds, resolution: float) -> tuple[float, float]:
    """Pixel centre, in the raster's CRS. Row zero is the *north* edge."""
    row, col = pixel
    return (bounds.minx + (col + 0.5) * resolution, bounds.maxy - (row + 0.5) * resolution)


def _simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Douglas–Peucker. Skeleton staircases carry a vertex per pixel otherwise."""
    if len(points) < 3 or tolerance <= 0:
        return points

    start, end = points[0], points[-1]
    worst, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _point_line_distance(points[i], start, end)
        if d > worst:
            worst, index = d, i

    if worst <= tolerance:
        return [start, end]
    left = _simplify(points[: index + 1], tolerance)
    right = _simplify(points[index:], tolerance)
    return left[:-1] + right


def _point_line_distance(p, a, b) -> float:
    if a == b:
        return math.dist(p, a)
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def _merge_close_nodes(graph: RoadGraph, tolerance: float) -> RoadGraph:
    """Collapse junctions within ``tolerance``, keeping edges between the survivors.

    A four-way crossing skeletonises into two Y-junctions a few pixels apart. Left alone that
    doubles intersection density and halves the average degree — both §16.2 measures.
    """
    if tolerance <= 0 or not graph.nodes:
        return graph

    ids = list(graph.nodes)
    representative = {nid: nid for nid in ids}
    for i, a in enumerate(ids):
        if representative[a] != a:
            continue
        for b in ids[i + 1 :]:
            if representative[b] != b:
                continue
            if math.dist(graph.nodes[a].position, graph.nodes[b].position) <= tolerance:
                representative[b] = a

    merged = RoadGraph(crs=graph.crs, tile_id=graph.tile_id, lod_level=graph.lod_level)
    for nid, node in graph.nodes.items():
        if representative[nid] == nid:
            merged.add_node(node)

    for eid, edge in graph.edges.items():
        u, v = representative[edge.u], representative[edge.v]
        if u == v:
            continue
        merged.add_edge(edge.model_copy(update={"id": eid, "u": u, "v": v}))
    return merged
