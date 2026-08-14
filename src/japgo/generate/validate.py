"""Phase 4 — is this a valid transportation graph, or merely a tidy picture?

Everything up to here has been about recovering roads. This asks whether what was recovered obeys
the rules a road network obeys: junctions of plausible degree, edges long enough to be roads,
intersections not stacked on top of each other, no duplicate carriageways, no edge crossing itself.

The checks report rather than delete. A validator that silently repairs is a validator whose
findings can never be counted, and the count is the point — "37 junctions above degree 6" is a
diagnosis, while a graph that quietly lost them is just a different graph. :func:`enforce` applies
the subset that is safe to fix automatically, and says what it did.

Limits come from ``config/road_hierarchy.yaml`` where the project already states them, rather than
from numbers invented here. Competing definitions of the same constraint is how two parts of a
system come to disagree about what is legal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from ..core.roads import RoadGraph


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    count: int
    detail: str

    def __str__(self) -> str:
        mark = "ERROR " if self.severity is Severity.ERROR else "warn  "
        return f"{mark} {self.check:<26} {self.count:>5}  {self.detail}"


@dataclass(frozen=True)
class ValidationSpec:
    max_junction_degree: int = 6
    """Above this a 'junction' is an extraction artefact. Real crossroads are degree 4; a six-way
    exists but a nine-way is a skeleton knot."""

    min_edge_m: float = 8.0
    min_junction_spacing_m: float = 15.0
    """Two junctions closer than this are one junction the raster split in two."""

    duplicate_bearing_deg: float = 12.0
    """Parallel edges between the same pair, within this bearing, are the same road twice."""


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    nodes: int = 0
    edges: int = 0

    @property
    def errors(self) -> int:
        return sum(f.count for f in self.findings if f.severity is Severity.ERROR)

    @property
    def valid(self) -> bool:
        return self.errors == 0

    def describe(self) -> str:
        head = f"{self.edges} edges / {self.nodes} nodes — " + (
            "valid" if self.valid else f"{self.errors} error(s)"
        )
        return "\n".join([head, *(str(f) for f in self.findings)])


def _bearing(edge) -> float:
    (x0, y0), (x1, y1) = edge.geometry[0], edge.geometry[-1]
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 180.0


def _segments_cross(a1, a2, b1, b2) -> bool:
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = side(b1, b2, a1), side(b1, b2, a2)
    d3, d4 = side(a1, a2, b1), side(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def validate(graph: RoadGraph, spec: ValidationSpec | None = None) -> ValidationReport:
    """Check the graph against the constraints a road network must satisfy."""
    spec = spec or ValidationSpec()
    report = ValidationReport(nodes=len(graph.nodes), edges=len(graph.edges))

    over = [n for n in graph.nodes if graph.degree(n) > spec.max_junction_degree]
    if over:
        report.findings.append(Finding(
            "junction degree", Severity.WARNING, len(over),
            f"above degree {spec.max_junction_degree}; likely skeleton knots, not crossroads",
        ))

    short = [e for e in graph.edges.values() if e.length_m < spec.min_edge_m]
    if short:
        report.findings.append(Finding(
            "short edges", Severity.WARNING, len(short),
            f"under {spec.min_edge_m:g} m; too short to be a road segment",
        ))

    junctions = [n for n in graph.nodes if graph.degree(n) >= 3]
    close = 0
    for i, a in enumerate(junctions):
        for b in junctions[i + 1 :]:
            if math.dist(graph.nodes[a].position, graph.nodes[b].position) < spec.min_junction_spacing_m:
                close += 1
    if close:
        report.findings.append(Finding(
            "junction spacing", Severity.WARNING, close,
            f"pairs under {spec.min_junction_spacing_m:g} m apart; one junction split in two",
        ))

    seen: dict[tuple[str, str], list] = {}
    duplicates = 0
    for edge in graph.edges.values():
        key = tuple(sorted((edge.u, edge.v)))
        for other in seen.setdefault(key, []):
            if abs(_bearing(edge) - _bearing(other)) < spec.duplicate_bearing_deg:
                duplicates += 1
                break
        seen[key].append(edge)
    if duplicates:
        report.findings.append(Finding(
            "duplicate roads", Severity.ERROR, duplicates,
            "same node pair, near-identical bearing — one carriageway drawn twice",
        ))

    self_crossing = 0
    for edge in graph.edges.values():
        pts = edge.geometry
        for i in range(len(pts) - 1):
            for j in range(i + 2, len(pts) - 1):
                if _segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                    self_crossing += 1
                    break
            else:
                continue
            break
    if self_crossing:
        report.findings.append(Finding(
            "self-intersections", Severity.ERROR, self_crossing,
            "an edge crossing its own geometry is not a road",
        ))

    components = graph.connected_components()
    if len(components) > 1:
        report.findings.append(Finding(
            "disconnected", Severity.WARNING, len(components),
            "components; a vehicle cannot cross between them",
        ))

    isolated = graph.isolated_nodes
    if isolated:
        report.findings.append(Finding(
            "isolated nodes", Severity.ERROR, len(isolated),
            "nodes with no edge at all",
        ))

    return report


def enforce(
    graph: RoadGraph, spec: ValidationSpec | None = None
) -> tuple[RoadGraph, ValidationReport]:
    """Apply the fixes that are unambiguous, and re-validate.

    Only duplicates and isolated nodes are removed automatically: both are cases where the graph
    says the same thing twice or says nothing at all, so deleting cannot lose information. Short
    edges and tight junctions are *reported* — collapsing them changes the network's topology, and
    that is a decision for the repair stage with its thresholds, not a validator's to take
    quietly.
    """
    spec = spec or ValidationSpec()

    seen: dict[tuple[str, str], list] = {}
    keep = {}
    for eid, edge in graph.edges.items():
        key = tuple(sorted((edge.u, edge.v)))
        if any(
            abs(_bearing(edge) - _bearing(other)) < spec.duplicate_bearing_deg
            for other in seen.setdefault(key, [])
        ):
            continue
        seen[key].append(edge)
        keep[eid] = edge

    used = {n for e in keep.values() for n in (e.u, e.v)}
    cleaned = RoadGraph(
        crs=graph.crs, tile_id=graph.tile_id, lod_level=graph.lod_level,
        nodes={nid: n for nid, n in graph.nodes.items() if nid in used},
        edges=keep,
    )
    return cleaned, validate(cleaned, spec)
