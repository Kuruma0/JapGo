"""Phase 5 — terrain constraints, enforced by *rerouting* rather than deletion.

The sensitivity sweep settled what this stage has to be. Across three folds and three perturbation
designs, the model responds to the slope channel intensively and still does not put roads where
terrain says they should go: change the values and the output tracks the size of the change rather
than its direction. So the ML cannot be relied on for terrain reasoning, and this layer cannot
merely *check* its work — it has to do the reasoning.

Deleting a road that exceeds the grade limit is the wrong repair. A real engineer facing a steep
hillside does not omit the road, they switchback up it, and a generator that deletes instead
produces exactly the failure mode the Kawanehon site exists to expose: a mountain valley with no
roads in it. So a violating edge is rerouted by least-cost path across the terrain, where cost
rises steeply with gradient and steps beyond the limit are forbidden outright. Deletion is the
fallback for when no route exists at all.

The search runs on a downsampled grid — 1 m terrain is far finer than a road alignment needs, and
a 1512² lattice is 2.3 M nodes. At 8 m it is 36 k, which Dijkstra crosses in well under a second
and which still resolves a switchback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.roads import Edge, RoadGraph

def _neighbourhood(reach: int = 3) -> list[tuple[int, int]]:
    """Grid moves out to ``reach`` cells, coprime offsets only.

    Eight neighbours is not enough, and the reason is geometric rather than a tuning detail. On a
    uniform slope of gradient G, the gentlest ascending move an 8-connected grid offers is the
    diagonal, at G/sqrt(2) — so a 30% hillside cannot be climbed below 21%, and a 12% limit makes
    *every* upward move illegal. The search then reports no route and the road gets deleted, which
    is exactly the wrong answer on ground a switchback could cross.

    A real switchback climbs at a shallow angle to the contour. Representing that needs moves like
    (3, 1), which on the same 30% slope grades at 9.5% and is legal. Coprime offsets only, because
    (2, 2) is just (1, 1) twice and would double-count the same direction.
    """
    from math import gcd

    return [
        (dr, dc)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if (dr or dc) and gcd(abs(dr), abs(dc)) == 1
    ]


NEIGHBOURS = _neighbourhood()


@dataclass(frozen=True)
class TerrainSpec:
    max_grade_pct: float = 12.0
    """From ``config/road_hierarchy.yaml`` — the limit for local streets and the classes just
    above them. Not invented here."""

    search_resolution_m: float = 8.0
    """Grid spacing for the reroute search. Fine enough for a switchback, coarse enough to run."""

    grade_penalty: float = 8.0
    """How much a steep step costs relative to a flat one. High enough that the search prefers a
    long contour to a short climb, which is what a switchback *is*."""

    max_cut_fill_m: float = 4.0
    """How far the ground may depart from a straight segment before that move is disallowed.

    A road is built on terrain, not over it. Without this the search takes long moves that hop
    ridges and bridge gullies for free."""

    max_detour_ratio: float = 4.0
    """Abandon a reroute longer than this multiple of the direct line. Past it the 'road' is a
    scenic tour, and deleting the edge is more honest than inventing one."""


@dataclass
class TerrainReport:
    violations: int = 0
    rerouted: int = 0
    deleted: int = 0
    max_grade_before: float = 0.0
    max_grade_after: float = 0.0
    added_length_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.violations} over limit -> {self.rerouted} rerouted, {self.deleted} deleted; "
            f"max grade {self.max_grade_before:.0f}% -> {self.max_grade_after:.0f}%; "
            f"+{self.added_length_m / 1000:.2f} km of alignment"
        )


class TerrainGrid:
    """A downsampled elevation lattice with a grade-aware least-cost search over it."""

    def __init__(self, elevation: np.ndarray, bounds, resolution_m: float, spec: TerrainSpec):
        self.bounds = bounds
        self.spec = spec
        step = max(int(round(spec.search_resolution_m / resolution_m)), 1)
        self.step_m = resolution_m * step
        self.z = elevation[::step, ::step].astype(np.float64)
        self.rows, self.cols = self.z.shape

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip((x - self.bounds.minx) / self.step_m, 0, self.cols - 1))
        row = int(np.clip((self.bounds.maxy - y) / self.step_m, 0, self.rows - 1))
        return row, col

    def world_of(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            self.bounds.minx + (col + 0.5) * self.step_m,
            self.bounds.maxy - (row + 0.5) * self.step_m,
        )

    def _follows_ground(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """Whether the ground under a straight move stays close to the line joining its ends.

        A road spanning several cells is built on the terrain, not floating over it. Sampling the
        cells between and comparing against linear interpolation is what stops the search hopping
        a ridge or bridging a gully it has not paid for.
        """
        span = max(abs(r1 - r0), abs(c1 - c0))
        za, zb = self.z[r0, c0], self.z[r1, c1]
        for i in range(1, span):
            f = i / span
            r = int(round(r0 + (r1 - r0) * f))
            c = int(round(c0 + (c1 - c0) * f))
            if abs(self.z[r, c] - (za + (zb - za) * f)) > self.spec.max_cut_fill_m:
                return False
        return True

    def route(self, start: tuple[float, float], end: tuple[float, float]) -> list | None:
        """Least-cost path between two world points, or ``None`` if terrain forbids every route.

        Cost is horizontal distance scaled by a gradient penalty, and any step steeper than the
        limit is simply not an edge. That combination is what produces switchbacks without anyone
        programming a switchback: climbing directly is illegal, so the cheapest legal path
        traverses the slope and doubles back.
        """
        import heapq

        source, target = self.cell_of(*start), self.cell_of(*end)
        if source == target:
            return None

        limit = self.spec.max_grade_pct / 100.0
        best = {source: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(0.0, source)]
        seen = set()

        while queue:
            cost, cell = heapq.heappop(queue)
            if cell in seen:
                continue
            seen.add(cell)
            if cell == target:
                break

            row, col = cell
            for dr, dc in NEIGHBOURS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                run = self.step_m * math.hypot(dr, dc)
                rise = abs(self.z[nr, nc] - self.z[row, col])
                grade = rise / run if run else 0.0
                if grade > limit:
                    continue  # not an edge: the terrain forbids it
                # A long move must not jump a ridge or a gully between its ends. That is a
                # cut-and-fill question, not a grade one: the segment's own gradient is uniform
                # by construction, so what matters is how far the ground departs from it.
                #
                # An earlier version compared the midpoint's rise against half the run and
                # rejected almost everything — integer division puts the midpoint of a (1, 3)
                # move a *whole* row up, so a uniform slope read as twice its real gradient and
                # no long move was ever legal. The search then found no route and deleted roads
                # that a switchback could have carried.
                if max(abs(dr), abs(dc)) > 1 and not self._follows_ground(row, col, nr, nc):
                    continue

                step_cost = run * (1.0 + self.spec.grade_penalty * grade)
                nxt = (nr, nc)
                if step_cost + cost < best.get(nxt, math.inf):
                    best[nxt] = step_cost + cost
                    came[nxt] = cell
                    heapq.heappush(queue, (step_cost + cost, nxt))

        if target not in came and target != source:
            return None

        path, cell = [], target
        while cell != source:
            path.append(self.world_of(cell))
            cell = came[cell]
        path.append(self.world_of(source))
        path.reverse()
        return path


def _polyline_length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))


def _max_grade(points, grid: TerrainGrid) -> float:
    worst = 0.0
    for a, b in zip(points, points[1:], strict=False):
        run = math.dist(a, b)
        if run <= 0:
            continue
        za = grid.z[grid.cell_of(*a)]
        zb = grid.z[grid.cell_of(*b)]
        worst = max(worst, 100.0 * abs(zb - za) / run)
    return worst


def enforce_grade(
    graph: RoadGraph,
    elevation: np.ndarray,
    bounds,
    resolution_m: float,
    spec: TerrainSpec | None = None,
) -> tuple[RoadGraph, TerrainReport]:
    """Reroute every edge that exceeds the grade limit; delete only where no route exists."""
    spec = spec or TerrainSpec()
    grid = TerrainGrid(elevation, bounds, resolution_m, spec)
    report = TerrainReport()

    kept: dict[str, Edge] = {}
    for eid, edge in graph.edges.items():
        grade = _max_grade(edge.geometry, grid)
        report.max_grade_before = max(report.max_grade_before, grade)

        if grade <= spec.max_grade_pct:
            kept[eid] = edge
            continue

        report.violations += 1
        route = grid.route(edge.geometry[0], edge.geometry[-1])
        direct = max(math.dist(edge.geometry[0], edge.geometry[-1]), 1.0)

        if route is None or _polyline_length(route) > spec.max_detour_ratio * direct:
            report.deleted += 1
            continue

        report.added_length_m += _polyline_length(route) - edge.length_m
        report.rerouted += 1
        kept[eid] = edge.model_copy(update={
            "geometry": route,
            "grade_pct": _max_grade(route, grid),
            "attributes": {**edge.attributes, "rerouted": "grade"},
        })

    used = {n for e in kept.values() for n in (e.u, e.v)}
    out = RoadGraph(
        crs=graph.crs, tile_id=graph.tile_id, lod_level=graph.lod_level,
        nodes={nid: n for nid, n in graph.nodes.items() if nid in used},
        edges=kept,
    )
    report.max_grade_after = max((_max_grade(e.geometry, grid) for e in kept.values()), default=0.0)

    if report.deleted:
        report.notes.append(
            f"{report.deleted} edge(s) had no route within {spec.max_grade_pct:g}% grade and a "
            f"{spec.max_detour_ratio:g}x detour budget — terrain there does not admit a road"
        )
    return out, report
