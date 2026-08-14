"""Phase 6 — turn a validated graph into road geometry an engine can build.

A graph says where roads go. Geometry says what they *are*: a centreline smooth enough to drive,
a width, a height above the ground, and tangents so a mesh generator or spline component can take
it without guessing.

Smoothing is the delicate part, because the two obvious approaches are both wrong here. Leaving
the polyline alone ships the extractor's staircase — road alignments do not have 8 m corners.
Fitting an unconstrained spline lets the curve leave the terrain and cut corners the grade
enforcement just finished proving were illegal. So the centreline is smoothed by **Chaikin
corner-cutting under a displacement cap**: each round softens corners, and no vertex is allowed to
move further from its original position than ``max_shift_m``. The road stops looking extracted
without wandering off the alignment that was validated.

Endpoints never move. They are junctions, shared with other edges, and a junction that drifts by
half a metre per incident road stops being a junction at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.roads import RoadGraph, load_hierarchy

Point = tuple[float, float]


@dataclass(frozen=True)
class GeometrySpec:
    smoothing_rounds: int = 3
    max_shift_m: float = 6.0
    """How far smoothing may move a vertex from where validation put it.

    The cap is what keeps this stage honest. Grade enforcement proved a specific alignment was
    legal; a smoother free to move vertices arbitrarily can undo that proof silently, and the
    result looks better and drives worse.
    """

    resample_m: float = 10.0
    """Spacing for the emitted centreline. Even spacing is what spline importers expect, and it
    makes tangent estimates stable."""

    default_width_m: float = 5.0
    camber_lift_m: float = 0.15
    """How far the road surface sits above sampled ground, so it does not z-fight the terrain
    mesh. Cosmetic, but the alternative is a visible defect in every engine."""


@dataclass
class RoadSpline:
    """One road, ready for an engine."""

    edge_id: str
    points: list[Point]
    elevations: list[float]
    tangents: list[Point]
    width_m: float
    road_class: str
    grade_pct: float | None = None

    @property
    def length_m(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:], strict=False))


@dataclass
class JunctionPoint:
    node_id: str
    position: Point
    elevation: float
    degree: int
    incident: list[str] = field(default_factory=list)


def _chaikin(points: list[Point], rounds: int, max_shift: float) -> list[Point]:
    """Corner-cutting with a leash, endpoints pinned."""
    if len(points) < 3 or rounds <= 0:
        return list(points)

    original = list(points)
    current = list(points)
    for _ in range(rounds):
        out: list[Point] = [current[0]]
        for a, b in zip(current, current[1:], strict=False):
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        out.append(current[-1])
        current = out

    # Pull anything that wandered too far back toward the alignment that was validated.
    leashed: list[Point] = []
    for p in current:
        nearest = min(original, key=lambda q: math.dist(p, q))
        d = math.dist(p, nearest)
        if d > max_shift and d > 0:
            f = max_shift / d
            p = (nearest[0] + (p[0] - nearest[0]) * f, nearest[1] + (p[1] - nearest[1]) * f)
        leashed.append(p)

    leashed[0], leashed[-1] = original[0], original[-1]
    return leashed


def _resample(points: list[Point], spacing: float) -> list[Point]:
    """Even spacing along the polyline, endpoints preserved."""
    if len(points) < 2 or spacing <= 0:
        return list(points)

    spans = [math.dist(a, b) for a, b in zip(points, points[1:], strict=False)]
    total = sum(spans)
    if total <= spacing:
        return [points[0], points[-1]]

    out = [points[0]]
    target, travelled, i = spacing, 0.0, 0
    while target < total and i < len(spans):
        if travelled + spans[i] < target:
            travelled += spans[i]
            i += 1
            continue
        f = (target - travelled) / spans[i] if spans[i] else 0.0
        a, b = points[i], points[i + 1]
        out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        target += spacing
    out.append(points[-1])
    return out


def _tangents(points: list[Point]) -> list[Point]:
    """Unit tangents by central difference, one per point."""
    out: list[Point] = []
    for i, _ in enumerate(points):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, len(points) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        out.append((dx / n, dy / n))
    return out


def build_geometry(
    graph: RoadGraph,
    sample_elevation,
    *,
    spec: GeometrySpec | None = None,
    hierarchy=None,
) -> tuple[list[RoadSpline], list[JunctionPoint]]:
    """Emit one spline per edge and one junction record per node.

    ``sample_elevation`` is any ``(x, y) -> float``; the caller owns how terrain is looked up, so
    this works equally against a real DEM raster and against a synthetic heightfield the game
    generated. That is the whole point of taking a callable rather than an array.
    """
    spec = spec or GeometrySpec()
    hierarchy = hierarchy or load_hierarchy()

    splines: list[RoadSpline] = []
    for eid, edge in sorted(graph.edges.items()):
        smoothed = _chaikin(list(edge.geometry), spec.smoothing_rounds, spec.max_shift_m)
        points = _resample(smoothed, spec.resample_m)
        try:
            width = edge.width_m or hierarchy.spec(edge.road_class).typical_width_m
        except (KeyError, AttributeError):
            width = spec.default_width_m

        splines.append(RoadSpline(
            edge_id=eid,
            points=points,
            elevations=[sample_elevation(x, y) + spec.camber_lift_m for x, y in points],
            tangents=_tangents(points),
            width_m=float(width),
            road_class=edge.road_class,
            grade_pct=edge.grade_pct,
        ))

    junctions = [
        JunctionPoint(
            node_id=nid,
            position=node.position,
            elevation=sample_elevation(*node.position),
            degree=graph.degree(nid),
            incident=sorted(e for e, x in graph.edges.items() if nid in (x.u, x.v)),
        )
        for nid, node in sorted(graph.nodes.items())
        if graph.degree(nid) >= 3
    ]
    return splines, junctions
