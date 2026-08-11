"""APLS and TOPO — the topology metrics Phase 4's exit criterion names.

Pixel F1 answers "did it paint road in the right places". Neither of these does, and that is the
point: a network can score well per-pixel and still be useless, because a single missing metre at
a junction disconnects two halves of a town. §16.2 puts topology first for that reason.

**APLS** (Average Path Length Similarity) compares *routes*. Sample node pairs in the ground-truth
graph, find the nearest proposal node to each, and compare shortest-path length between the pairs.
A road that exists but is severed shows up here and nowhere else. Missing paths score zero rather
than being skipped — skipping them would reward a model for deleting the hard parts.

**TOPO** compares *local reachability*. Around sampled seed points, take everything reachable
within a radius in each graph and score the overlap as precision/recall/F1. It is more forgiving
of geometric offset than APLS and more sensitive to spurious side-streets.

Both are computed symmetrically where the definition calls for it, and both use networkx for the
shortest paths — §3 puts graph algorithms on the do-not-reinvent list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.roads import RoadGraph

MATCH_RADIUS_M = 25.0
"""How far a proposal node may be from a truth node and still count as the same place.

The standard APLS figure. Wider forgives real positional error; narrower turns a 30 m offset into
a total miss, which measures registration rather than topology.
"""


@dataclass(frozen=True)
class TopologyScore:
    apls: float
    topo_f1: float
    topo_precision: float
    topo_recall: float
    truth_nodes: int
    proposal_nodes: int
    matched: int

    def describe(self) -> str:
        return (
            f"APLS {self.apls:.3f}  TOPO F1 {self.topo_f1:.3f} "
            f"(P {self.topo_precision:.3f} R {self.topo_recall:.3f})  "
            f"matched {self.matched}/{self.truth_nodes}"
        )


def _to_networkx(graph: RoadGraph):
    """A weighted undirected graph, weights in metres."""
    import networkx as nx

    g = nx.Graph()
    for nid, node in graph.nodes.items():
        g.add_node(nid, pos=node.position)
    for edge in graph.edges.values():
        length = edge.length_m
        # Keep the shorter of any parallel pair: routes take the shorter one anyway.
        if g.has_edge(edge.u, edge.v) and g[edge.u][edge.v]["weight"] <= length:
            continue
        g.add_edge(edge.u, edge.v, weight=length)
    return g


def _match_nodes(
    truth: RoadGraph, proposal: RoadGraph, *, radius: float
) -> dict[str, str]:
    """Nearest proposal node to each truth node, within ``radius``.

    Greedy by distance so that one proposal node cannot stand in for several truth nodes — which
    would let a single blob claim a whole junction cluster.
    """
    if not truth.nodes or not proposal.nodes:
        return {}

    truth_ids = list(truth.nodes)
    prop_ids = list(proposal.nodes)
    tp = np.array([truth.nodes[i].position for i in truth_ids])
    pp = np.array([proposal.nodes[i].position for i in prop_ids])

    distances = np.hypot(
        tp[:, 0][:, None] - pp[None, :, 0], tp[:, 1][:, None] - pp[None, :, 1]
    )
    pairs = [
        (distances[i, j], i, j)
        for i, j in zip(*np.where(distances <= radius), strict=True)
    ]
    pairs.sort()

    matched: dict[str, str] = {}
    used: set[str] = set()
    for _, i, j in pairs:
        t, p = truth_ids[i], prop_ids[j]
        if t in matched or p in used:
            continue
        matched[t] = p
        used.add(p)
    return matched


def apls(
    truth: RoadGraph,
    proposal: RoadGraph,
    *,
    radius: float = MATCH_RADIUS_M,
    samples: int = 400,
    seed: int = 0,
) -> float:
    """Average path length similarity, symmetric, in [0, 1].

    A truth pair whose route has no counterpart contributes zero. That is the whole value of the
    metric: it is where "the raster looked right but the network is severed" becomes a number.
    """
    forward = _apls_one_way(truth, proposal, radius=radius, samples=samples, seed=seed)
    backward = _apls_one_way(proposal, truth, radius=radius, samples=samples, seed=seed)
    if math.isnan(forward) and math.isnan(backward):
        return 0.0
    values = [v for v in (forward, backward) if not math.isnan(v)]
    return float(np.mean(values))


def _apls_one_way(
    source: RoadGraph, target: RoadGraph, *, radius: float, samples: int, seed: int
) -> float:
    import networkx as nx

    if len(source.nodes) < 2:
        return float("nan")

    matched = _match_nodes(source, target, radius=radius)
    gs, gt = _to_networkx(source), _to_networkx(target)

    rng = np.random.default_rng(seed)
    ids = list(source.nodes)
    pairs = set()
    # Sample pairs rather than enumerate: APLS is defined over all pairs, but that is O(n^2)
    # shortest paths and a tile can hold thousands of nodes. A fixed seed keeps it reproducible.
    limit = min(samples, len(ids) * (len(ids) - 1) // 2)
    guard = 0
    while len(pairs) < limit and guard < limit * 20:
        a, b = rng.choice(len(ids), size=2, replace=False)
        pairs.add((min(a, b), max(a, b)))
        guard += 1

    scores = []
    for ai, bi in pairs:
        a, b = ids[ai], ids[bi]
        try:
            source_length = nx.shortest_path_length(gs, a, b, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue  # unconnected in the source: no claim to make about the target

        if a not in matched or b not in matched:
            scores.append(0.0)
            continue
        try:
            target_length = nx.shortest_path_length(gt, matched[a], matched[b], weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            scores.append(0.0)
            continue

        if source_length <= 0:
            continue
        scores.append(max(0.0, 1.0 - abs(source_length - target_length) / source_length))

    return float(np.mean(scores)) if scores else float("nan")


def topo(
    truth: RoadGraph,
    proposal: RoadGraph,
    *,
    radius: float = MATCH_RADIUS_M,
    hop_m: float = 300.0,
    seeds: int = 60,
    seed: int = 0,
) -> tuple[float, float, float]:
    """TOPO precision, recall and F1 over local reachable sets.

    At each seed, everything reachable within ``hop_m`` in the truth graph is what *should* be
    found, and the same ball in the proposal is what *was* found. Comparing balls rather than
    whole graphs is what makes it local: one severed corner does not zero the whole tile.
    """
    import networkx as nx

    if not truth.nodes or not proposal.nodes:
        return 0.0, 0.0, 0.0

    matched = _match_nodes(truth, proposal, radius=radius)
    reverse = {p: t for t, p in matched.items()}
    gs, gt = _to_networkx(truth), _to_networkx(proposal)

    rng = np.random.default_rng(seed)
    candidates = [t for t in truth.nodes if t in matched]
    if not candidates:
        return 0.0, 0.0, 0.0
    picks = rng.choice(len(candidates), size=min(seeds, len(candidates)), replace=False)

    tp = fp = fn = 0
    for index in picks:
        origin = candidates[int(index)]
        truth_ball = set(nx.single_source_dijkstra_path_length(gs, origin, cutoff=hop_m, weight="weight"))
        prop_ball = set(
            nx.single_source_dijkstra_path_length(gt, matched[origin], cutoff=hop_m, weight="weight")
        )
        # Project the proposal ball back into truth identities so the sets are comparable.
        projected = {reverse[p] for p in prop_ball if p in reverse}

        tp += len(truth_ball & projected)
        fn += len(truth_ball - projected)
        fp += len(prop_ball) - len(truth_ball & projected)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def compare(
    truth: RoadGraph,
    proposal: RoadGraph,
    *,
    radius: float = MATCH_RADIUS_M,
    seed: int = 0,
) -> TopologyScore:
    """Both metrics at once, with the match counts that explain them."""
    matched = _match_nodes(truth, proposal, radius=radius)
    f1, precision, recall = topo(truth, proposal, radius=radius, seed=seed)
    return TopologyScore(
        apls=apls(truth, proposal, radius=radius, seed=seed),
        topo_f1=f1,
        topo_precision=precision,
        topo_recall=recall,
        truth_nodes=len(truth.nodes),
        proposal_nodes=len(proposal.nodes),
        matched=len(matched),
    )
