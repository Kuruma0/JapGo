"""Phase 3 — connectivity repair. Deterministic, configurable, and the reason the ML output is usable.

This is where invariant 5 earns its keep. The frozen model predicts a dead-end ratio of 0.6–0.9
against a real 0.04–0.25 — a network of stubs that nearly meet. Almost every one of those stubs is
a road the model found and then lost for a few metres, and no amount of retraining has fixed it;
five loss configurations tried and the ratio never moved. It is a repair problem.

**The objective is not zero dead ends.** Real networks are full of them: cul-de-sacs, service
spurs, roads that stop at a field. Forcing the ratio down would trade one implausibility for
another, so the aim is to make the *distribution and spatial role* of dead ends plausible — bridge
the ones that are accidents, keep the ones that are places.

Four passes, in an order that matters:

1. **Drop tiny components.** Isolated fragments carry no route and inflate every count.
2. **Bridge near-misses.** Two dead ends a few metres apart, with no short path already between
   them, are one road the model dropped. This is the pass that does the work.
3. **Snap dead ends onto passing edges.** A stub that ends beside another road is a T-junction the
   raster lost.
4. **Prune what is left.** Only after bridging, because a stub pruned in step 1 can never be
   reconnected in step 2 — and the stub was usually the evidence of a real road.

Every pass is deterministic and every threshold is on :class:`RepairSpec`, so a given graph and
spec always produce the same network. That is what lets a seed reproduce a world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.roads import Edge, RoadGraph


@dataclass(frozen=True)
class RepairSpec:
    """Thresholds, all in metres. Defaults tuned against the 191-tile corpus's real networks."""

    min_component_m: float = 60.0
    """Total length below which an isolated component is noise rather than a road."""

    bridge_gap_m: float = 45.0
    """How far apart two dead ends may be and still be treated as one interrupted road.

    Generous, because the failure being repaired is a model that loses roads for short stretches,
    and a gap it cannot close leaves a false cul-de-sac. Bounded by the detour test below rather
    than by distance alone.
    """

    bridge_detour_ratio: float = 4.0
    """Only bridge when the existing route between two ends is at least this many times the gap.

    Without it, bridging punches shortcuts through blocks: two dead ends either side of a street
    are metres apart and already well connected, and joining them invents a road that is not
    there. Requiring the current path to be a long way round is what distinguishes "the model
    dropped this link" from "these are two different streets".
    """

    snap_to_edge_m: float = 25.0
    """How far a dead end may be from a passing edge and still be snapped onto it as a junction."""

    min_stub_m: float = 30.0
    """Dead-end edges shorter than this are pruned — *after* bridging has had its chance."""

    prune_iterations: int = 3


@dataclass
class RepairReport:
    """Before and after, per pass, so a change in the network can be attributed."""

    components_before: int = 0
    components_after: int = 0
    dead_end_before: float = 0.0
    dead_end_after: float = 0.0
    dropped_components: int = 0
    bridged: int = 0
    snapped: int = 0
    pruned: int = 0
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"components {self.components_before} -> {self.components_after}   "
            f"dead ends {self.dead_end_before:.0%} -> {self.dead_end_after:.0%}   "
            f"(dropped {self.dropped_components}, bridged {self.bridged}, "
            f"snapped {self.snapped}, pruned {self.pruned})"
        )


def _rebuild(graph: RoadGraph, edges: dict[str, Edge]) -> RoadGraph:
    used = {n for e in edges.values() for n in (e.u, e.v)}
    return RoadGraph(
        crs=graph.crs, tile_id=graph.tile_id, lod_level=graph.lod_level,
        nodes={nid: n for nid, n in graph.nodes.items() if nid in used},
        edges=edges,
    )


def drop_small_components(graph: RoadGraph, spec: RepairSpec) -> tuple[RoadGraph, int]:
    """Remove components whose total length is below ``min_component_m``."""
    if not graph.edges:
        return graph, 0

    keep: dict[str, Edge] = {}
    dropped = 0
    for component in graph.connected_components():
        edges = {
            eid: e for eid, e in graph.edges.items() if e.u in component and e.v in component
        }
        if sum(e.length_m for e in edges.values()) >= spec.min_component_m:
            keep.update(edges)
        else:
            dropped += 1
    return _rebuild(graph, keep), dropped


def bridge_dead_ends(graph: RoadGraph, spec: RepairSpec) -> tuple[RoadGraph, int]:
    """Join dead-end pairs that are close in space but far apart along the network."""
    import networkx as nx

    ends = [n for n in graph.nodes if graph.degree(n) == 1]
    if len(ends) < 2:
        return graph, 0

    g = nx.Graph()
    for eid, e in graph.edges.items():
        g.add_edge(e.u, e.v, weight=e.length_m)

    positions = {n: graph.nodes[n].position for n in ends}
    candidates = []
    for i, a in enumerate(ends):
        for b in ends[i + 1 :]:
            gap = math.dist(positions[a], positions[b])
            if gap <= spec.bridge_gap_m:
                candidates.append((gap, a, b))
    candidates.sort()

    added = 0
    joined: set[str] = set()
    for gap, a, b in candidates:
        if a in joined or b in joined or a == b:
            continue
        # A pair already well connected is two streets, not one broken road.
        try:
            existing = nx.shortest_path_length(g, a, b, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            existing = math.inf
        if existing < spec.bridge_detour_ratio * max(gap, 1.0):
            continue

        eid = f"bridge{added}"
        graph.add_edge(
            Edge(id=eid, u=a, v=b, geometry=[positions[a], positions[b]],
                 road_class="unknown", source_id="model", confidence=0.5)
        )
        g.add_edge(a, b, weight=gap)
        joined.update({a, b})
        added += 1
    return graph, added


def snap_dead_ends_to_edges(graph: RoadGraph, spec: RepairSpec) -> tuple[RoadGraph, int]:
    """Attach a dead end to a nearby edge it runs alongside, forming a T-junction.

    Only to an edge it is not already attached to, and only via the edge's nearest vertex — a
    proper split would insert a node mid-polyline, which is correct and is Phase 4's business.
    Here the concern is connectivity, and joining to the nearest existing vertex achieves it
    without rewriting geometry.
    """
    ends = [n for n in graph.nodes if graph.degree(n) == 1]
    if not ends:
        return graph, 0

    added = 0
    for end in ends:
        if graph.degree(end) != 1:
            continue  # an earlier snap in this pass already connected it
        ex, ey = graph.nodes[end].position
        attached = graph.neighbours(end) | {end}

        best = None
        for eid, edge in graph.edges.items():
            if edge.u in attached and edge.v in attached:
                continue
            for target in (edge.u, edge.v):
                if target in attached:
                    continue
                d = math.dist((ex, ey), graph.nodes[target].position)
                if d <= spec.snap_to_edge_m and (best is None or d < best[0]):
                    best = (d, target)

        if best is not None:
            _, target = best
            graph.add_edge(
                Edge(id=f"snap{added}", u=end, v=target,
                     geometry=[(ex, ey), graph.nodes[target].position],
                     road_class="unknown", source_id="model", confidence=0.5)
            )
            added += 1
    return graph, added


def prune_stubs(graph: RoadGraph, spec: RepairSpec) -> tuple[RoadGraph, int]:
    """Remove short dead-end edges that bridging could not save.

    Deliberately last. A stub removed before bridging is a road that can never be reconnected, and
    the stub is usually the only evidence the model left that a road was there at all.
    """
    removed = 0
    for _ in range(max(spec.prune_iterations, 0)):
        degree = {n: graph.degree(n) for n in graph.nodes}
        doomed = {
            eid for eid, e in graph.edges.items()
            if e.length_m < spec.min_stub_m and (degree[e.u] == 1 or degree[e.v] == 1)
        }
        if not doomed:
            break
        removed += len(doomed)
        graph = _rebuild(graph, {k: v for k, v in graph.edges.items() if k not in doomed})
    return graph, removed


def repair(graph: RoadGraph, spec: RepairSpec | None = None) -> tuple[RoadGraph, RepairReport]:
    """Run the four passes in order and report what each did."""
    spec = spec or RepairSpec()
    report = RepairReport(
        components_before=len(graph.connected_components()),
        dead_end_before=graph.dead_end_ratio,
    )

    graph, report.dropped_components = drop_small_components(graph, spec)
    graph, report.bridged = bridge_dead_ends(graph, spec)
    graph, report.snapped = snap_dead_ends_to_edges(graph, spec)
    graph, report.pruned = prune_stubs(graph, spec)

    report.components_after = len(graph.connected_components())
    report.dead_end_after = graph.dead_end_ratio
    if report.dead_end_after > 0.5:
        report.notes.append(
            f"dead-end ratio still {report.dead_end_after:.0%}; real networks run 0.04-0.25, so "
            "the prediction is likely fragmented beyond what bridging at "
            f"{spec.bridge_gap_m:.0f} m can close"
        )
    return graph, report


def largest_component(graph: RoadGraph) -> RoadGraph:
    """The biggest connected subgraph — a last resort for a caller that needs one network."""
    components = graph.connected_components()
    if len(components) <= 1:
        return graph
    biggest = max(components, key=len)
    return _rebuild(
        graph, {eid: e for eid, e in graph.edges.items() if e.u in biggest and e.v in biggest}
    )
