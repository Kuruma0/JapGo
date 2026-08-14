"""Phase 2 — turn a probability field into a road graph the procedural layer can reason about.

The vectorisation itself is :func:`japgo.model.extract.extract_graph`, which already exists and is
tested; this module is what makes its output *usable downstream*. A bare graph of nodes and
polylines is not enough to validate against terrain — the repair and validation stages need to ask
how steep an edge is, how long, which way it runs, and whether it connects to anything. So those
are computed once, here, and carried on the graph.

Grade is the one that matters most and the one that cannot be recovered later: the elevation
raster is available now and will not be by the time a graph reaches an engine exporter. Sampling
it per edge at extraction time is the difference between a network that can be checked against the
12% limit and one that merely looks plausible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.roads import NodeKind, RoadGraph
from ..model.extract import ExtractionSpec, extract_graph
from .inference import RoadPrediction


@dataclass
class CandidateReport:
    """What extraction produced, in the terms the next stage cares about."""

    nodes: int = 0
    edges: int = 0
    junctions: int = 0
    endpoints: int = 0
    components: int = 0
    total_length_m: float = 0.0
    dead_end_ratio: float = 0.0
    steep_edges: int = 0
    """Edges exceeding the grade limit — a count, not a judgement. Phase 5 decides what to do."""
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.edges} edges / {self.nodes} nodes  "
            f"({self.junctions} junctions, {self.endpoints} endpoints, "
            f"{self.components} components)  "
            f"{self.total_length_m / 1000:.2f} km  "
            f"dead ends {self.dead_end_ratio:.0%}  steep {self.steep_edges}"
        )


def _sample(raster: np.ndarray, bounds, resolution: float, x: float, y: float) -> float:
    """Nearest-cell lookup, clamped. Row zero is the north edge."""
    rows, cols = raster.shape
    col = int(np.clip((x - bounds.minx) / resolution, 0, cols - 1))
    row = int(np.clip((bounds.maxy - y) / resolution, 0, rows - 1))
    return float(raster[row, col])


def annotate_terrain(
    graph: RoadGraph,
    elevation: np.ndarray,
    bounds,
    resolution: float,
    *,
    grade_limit_pct: float = 12.0,
) -> int:
    """Attach elevation to nodes and grade to edges. Returns the count over ``grade_limit_pct``.

    Grade is measured end to end rather than along the polyline. A road that climbs and descends
    back to its starting height has an end-to-end grade of zero and is not flat, but the segment
    lengths here are already short — simplification caps them — and the alternative is a
    per-vertex maximum that a single noisy DEM cell can dominate. The 0.5 m LiDAR is detailed
    enough that spikes are real terrain *and* real noise, and telling them apart is not this
    stage's job.
    """
    for node in graph.nodes.values():
        node.z = _sample(elevation, bounds, resolution, node.x, node.y)

    steep = 0
    for edge in graph.edges.values():
        run = edge.length_m
        if run <= 0:
            continue
        rise = abs(
            _sample(elevation, bounds, resolution, *edge.geometry[-1])
            - _sample(elevation, bounds, resolution, *edge.geometry[0])
        )
        edge.grade_pct = 100.0 * rise / run
        if edge.grade_pct > grade_limit_pct:
            steep += 1
    return steep


def classify_nodes(graph: RoadGraph) -> tuple[int, int]:
    """Mark each node an intersection or an endpoint. Returns ``(junctions, endpoints)``."""
    junctions = endpoints = 0
    for nid, node in graph.nodes.items():
        degree = graph.degree(nid)
        if degree == 1:
            node.kind = NodeKind.ENDPOINT
            endpoints += 1
        else:
            node.kind = NodeKind.INTERSECTION
            junctions += degree >= 3
    return junctions, endpoints


def extract_candidates(
    prediction: RoadPrediction,
    *,
    elevation: np.ndarray | None = None,
    spec: ExtractionSpec | None = None,
    grade_limit_pct: float = 12.0,
) -> tuple[RoadGraph, CandidateReport]:
    """Probability field in, annotated road graph out.

    The graph is a *candidate* network: fragmented, over-junctioned and unvalidated. Making it
    coherent is the repair stage's job, and separating the two keeps each testable — extraction
    should be judged on whether it faithfully vectorises the raster, not on whether the raster was
    any good.
    """
    spec = spec or ExtractionSpec(threshold=prediction.threshold)
    graph = extract_graph(
        prediction.probability,
        prediction.bounds,
        prediction.crs,
        spec=spec,
        resolution=prediction.resolution_m,
    )

    report = CandidateReport()
    if elevation is not None:
        report.steep_edges = annotate_terrain(
            graph, elevation, prediction.bounds, prediction.resolution_m,
            grade_limit_pct=grade_limit_pct,
        )
    elif graph.edges:
        report.notes.append("no elevation supplied; grade is unknown and cannot be validated")

    report.junctions, report.endpoints = classify_nodes(graph)
    report.nodes = len(graph.nodes)
    report.edges = len(graph.edges)
    report.components = len(graph.connected_components())
    report.total_length_m = graph.total_length_m
    report.dead_end_ratio = graph.dead_end_ratio
    return graph, report


def bearing_of(edge) -> float:
    """Edge bearing folded to [0, 180). A road has no inherent direction."""
    (x0, y0), (x1, y1) = edge.geometry[0], edge.geometry[-1]
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 180.0
