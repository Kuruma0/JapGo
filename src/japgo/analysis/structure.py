"""Road structural responses: a tile's road graph reduced to scalars.

The *right* half of the Phase 3 study. These are the measures the environment is supposed to move,
and they are the same family the §16.2 metric suite uses to judge a model — deliberately, so that
Phase 3's findings and Phase 4's evaluation speak one language.

**The halo does real work here**, in a way it does not for the raster features. Clipping a graph
to the core manufactures dead ends at the boundary: every road that continues into the next tile
terminates at the cut and is counted as a cul-de-sac. On a 1 km core with a typical network that
is not a rounding error — it can dominate the statistic, and it would do so *systematically more*
in the sparse mountain tiles than the dense plain ones, which is precisely the contrast the study
is trying to measure.

So the two families are computed differently, and the difference is the point:

* **Extent measures** (density, length) sum edges lying in the core, over the core's area.
* **Topological measures** (degree, dead ends, intersections) count only nodes inside the core but
  compute their degree against the **full** graph, halo included. A node at the core edge keeps
  the neighbour that lies outside, so it is a dead end only if it genuinely is one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.roads import RoadGraph
from ..geo.tiling import Bounds, Tile

ROAD_STRUCTURE_METRICS: tuple[str, ...] = (
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "dead_end_ratio",
    "orientation_entropy",
    "sinuosity_median",
    "sinuosity_p90",
    "segment_length_median_m",
    "component_count",
)
"""The response vocabulary, fixed and ordered — see :data:`..features.ENVIRONMENTAL_FEATURES`."""


@dataclass(frozen=True)
class CoreGraph:
    """A graph split into what is in the core and what the halo still contributes."""

    full: RoadGraph
    core_edge_ids: frozenset[str]
    core_node_ids: frozenset[str]

    @property
    def core_length_m(self) -> float:
        return sum(self.full.edges[e].length_m for e in self.core_edge_ids)


def _midpoint(geometry: list[tuple[float, float]]) -> tuple[float, float]:
    """The point half way along the polyline by distance.

    Not the mean of the vertices: vertex density varies roughly tenfold between a curve and a
    straight, so a vertex mean is pulled toward the bendy end. docs/decision-log.md records this
    same mistake being made once already, in the road target alignment check.
    """
    if len(geometry) == 1:
        return geometry[0]

    spans = [math.dist(a, b) for a, b in zip(geometry, geometry[1:], strict=False)]
    half = sum(spans) / 2.0
    travelled = 0.0
    for (a, b), span in zip(zip(geometry, geometry[1:], strict=False), spans, strict=True):
        if travelled + span >= half:
            if span <= 0:
                return a
            t = (half - travelled) / span
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        travelled += span
    return geometry[-1]


def split_by_core(graph: RoadGraph, core: Bounds) -> CoreGraph:
    """Identify which edges and nodes belong to the core, keeping the full graph intact."""
    core_edges = frozenset(
        eid for eid, e in graph.edges.items() if core.contains_point(*_midpoint(e.geometry))
    )
    core_nodes = frozenset(
        nid for nid, n in graph.nodes.items() if core.contains_point(n.x, n.y)
    )
    return CoreGraph(full=graph, core_edge_ids=core_edges, core_node_ids=core_nodes)


def road_structure(graph: RoadGraph | None, tile: Tile) -> dict[str, float]:
    """Reduce one tile's road network to the :data:`ROAD_STRUCTURE_METRICS` vector.

    ``None`` — a tile with no road data at all — yields all-NaN rather than all-zero. A tile we
    did not measure and a tile with genuinely no roads are different facts, and only one of them
    should be allowed to influence a correlation.
    """
    if graph is None:
        return dict.fromkeys(ROAD_STRUCTURE_METRICS, float("nan"))

    core = tile.core
    area_km2 = (core.width * core.height) / 1e6
    split = split_by_core(graph, core)

    metrics: dict[str, float] = {
        "road_density_km_per_km2": (split.core_length_m / 1000.0) / area_km2 if area_km2 else 0.0,
    }

    # Degree against the full graph; membership by the core. See the module docstring.
    degrees = {nid: graph.degree(nid) for nid in split.core_node_ids}
    intersections = sum(1 for d in degrees.values() if d >= 3)
    metrics["intersection_density_per_km2"] = intersections / area_km2 if area_km2 else 0.0
    metrics["dead_end_ratio"] = (
        sum(1 for d in degrees.values() if d == 1) / len(degrees) if degrees else float("nan")
    )

    core_only = _subgraph(graph, split.core_edge_ids)
    metrics["orientation_entropy"] = (
        core_only.orientation_entropy() if core_only.edges else float("nan")
    )

    sinuosities = np.array(
        [graph.edges[e].sinuosity for e in split.core_edge_ids], dtype=np.float64
    )
    lengths = np.array([graph.edges[e].length_m for e in split.core_edge_ids], dtype=np.float64)
    metrics["sinuosity_median"] = (
        float(np.median(sinuosities)) if sinuosities.size else float("nan")
    )
    metrics["sinuosity_p90"] = (
        float(np.percentile(sinuosities, 90)) if sinuosities.size else float("nan")
    )
    metrics["segment_length_median_m"] = float(np.median(lengths)) if lengths.size else float("nan")

    # Components of the core-clipped graph: an island here is an island in the prediction too.
    metrics["component_count"] = (
        float(len([c for c in core_only.connected_components() if c])) if core_only.nodes else 0.0
    )

    return {name: metrics[name] for name in ROAD_STRUCTURE_METRICS}


def _subgraph(graph: RoadGraph, edge_ids: frozenset[str]) -> RoadGraph:
    edges = {eid: graph.edges[eid] for eid in edge_ids}
    used = {n for e in edges.values() for n in (e.u, e.v)}
    return RoadGraph(
        crs=graph.crs,
        tile_id=graph.tile_id,
        lod_level=graph.lod_level,
        nodes={nid: n for nid, n in graph.nodes.items() if nid in used},
        edges=edges,
    )
