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


def _drop_small(fn, mask, size: int, legacy_kwarg: str):
    """Call a scikit-image size filter across the 0.26 parameter rename.

    Both filters moved to ``max_size`` and, importantly, changed the comparison from "smaller
    than" to "smaller than **or equal to**". Passing the old threshold to the new API would remove
    one pixel more than intended, so the bound is adjusted rather than the call merely renamed.
    """
    import inspect

    if "max_size" in inspect.signature(fn).parameters:
        return fn(mask, max_size=size - 1)
    return fn(mask, **{legacy_kwarg: size})


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

    fill_holes_px: int = 64
    """Pinholes smaller than this are filled before skeletonising.

    The single cheapest fix for junction inflation. A hole anywhere inside a painted blob survives
    thinning as a **ring**, and a ring contributes two junctions and two edges that no road
    produced. A model that over-paints produces these by the hundred.
    """

    min_component_px: int = 48
    """Connected components smaller than this are dropped: isolated specks of probability become
    isolated two-node fragments, which cost TOPO precision and contribute nothing routable."""

    prune_spur_m: float = 20.0
    prune_iterations: int = 3
    """Dead-end edges shorter than ``prune_spur_m`` are removed, repeatedly.

    Pixel-level pruning only catches first-order spurs: removing one exposes the next, and a frayed
    blob edge is several deep. Pruning on the graph instead is both simpler and iterative — three
    passes settle it in practice.
    """


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

    from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

    # Clean the mask before thinning, not the graph afterwards. A pinhole becomes a ring and a
    # speck becomes a fragment; both are far cheaper to remove here than to recognise later.
    if spec.fill_holes_px > 0:
        mask = _drop_small(remove_small_holes, mask, spec.fill_holes_px, "area_threshold")
    if spec.min_component_px > 0:
        mask = _drop_small(remove_small_objects, mask, spec.min_component_px, "min_size")
    if not mask.any():
        return RoadGraph(crs=str(crs), tile_id=tile_id)

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

    graph = _merge_close_nodes(graph, spec.merge_nodes_m)
    return prune_spurs(graph, spec.prune_spur_m, iterations=spec.prune_iterations)


def prune_spurs(graph: RoadGraph, min_length_m: float, *, iterations: int = 3) -> RoadGraph:
    """Drop short dead-end edges, repeatedly.

    Iterative because removing a spur exposes the one behind it: a frayed blob edge is several
    layers deep, and a single pass leaves most of the damage in place. Only degree-1 edges are
    touched, so nothing routable is ever removed — a spur carries no through-traffic by definition.
    """
    if min_length_m <= 0:
        return graph

    for _ in range(max(iterations, 0)):
        degree = {nid: graph.degree(nid) for nid in graph.nodes}
        doomed = {
            eid
            for eid, edge in graph.edges.items()
            if edge.length_m < min_length_m and (degree[edge.u] == 1 or degree[edge.v] == 1)
        }
        if not doomed:
            break

        kept = {eid: e for eid, e in graph.edges.items() if eid not in doomed}
        used = {n for e in kept.values() for n in (e.u, e.v)}
        graph = RoadGraph(
            crs=graph.crs,
            tile_id=graph.tile_id,
            lod_level=graph.lod_level,
            nodes={nid: n for nid, n in graph.nodes.items() if nid in used},
            edges=kept,
        )
    return graph


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
