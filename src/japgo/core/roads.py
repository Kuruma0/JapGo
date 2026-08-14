"""The road network graph — the project's primary output representation.

Follows research doc §11. Four decisions there are load-bearing and are enforced here rather than
left to convention:

* **Undirected topology, directed attributes.** ``oneway`` is an edge attribute, not a topology
  decision. Everything downstream — hierarchy assignment, tessellation, export — is simpler when
  the graph itself is undirected.
* **``confidence`` on every edge.** Without it the Phase 2 visualiser cannot show what the model is
  unsure about, and showing exactly that is why Phase 2 exists.
* **LOD is a property of the graph, not the export.** The five levels of spec §23 are produced by
  filtering on ``road_class``, so one model produces all five and they cannot disagree.
* **Elevation is carried, not baked.** Bridges and tunnels are edge attributes with a structure
  type; what geometry that becomes is the exporter's decision.

Nothing here references an engine type, and nothing here describes appearance.
"""

from __future__ import annotations

import math
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_HIERARCHY_PATH = Path("config/road_hierarchy.yaml")


class NodeKind(StrEnum):
    INTERSECTION = "intersection"
    JUNCTION = "junction"
    ENDPOINT = "endpoint"
    BRIDGE_HEAD = "bridge_head"
    TUNNEL_MOUTH = "tunnel_mouth"


class Control(StrEnum):
    NONE = "none"
    SIGNAL = "signal"
    STOP = "stop"
    ROUNDABOUT = "roundabout"


class Structure(StrEnum):
    AT_GRADE = "at_grade"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    EMBANKMENT = "embankment"
    CUT = "cut"


class RoadClassSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    rank: int
    min_lod: int
    typical_lanes: int = 2
    typical_width_m: float = 5.0
    max_grade_percent: float = 12.0
    note: str | None = None


class RoadHierarchy(BaseModel):
    """The versioned road classification."""

    model_config = ConfigDict(frozen=True)

    hierarchy_version: int
    classes: dict[str, RoadClassSpec]
    osm_highway: dict[str, str]
    lod_levels: dict[int, str] = Field(default_factory=dict)

    def spec(self, road_class: str) -> RoadClassSpec:
        return self.classes.get(road_class) or self.classes["unknown"]

    def from_osm(self, highway: str) -> tuple[str, bool]:
        """Map an OSM ``highway=*`` value to a road class. Returns ``(class, was_mapped)``."""
        value = self.osm_highway.get(str(highway).strip().lower())
        return (value, True) if value else ("unknown", False)

    def classes_at_lod(self, lod: int) -> set[str]:
        return {name for name, spec in self.classes.items() if spec.min_lod <= lod}

    @property
    def vehicular_classes(self) -> set[str]:
        """Everything a car can use. ``path`` is excluded from road-network metrics by default."""
        return {name for name in self.classes if name != "path"}


def _find_hierarchy(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_HIERARCHY_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find {DEFAULT_HIERARCHY_PATH} walking up from {here}")


@lru_cache(maxsize=4)
def load_hierarchy(path: Path | None = None) -> RoadHierarchy:
    path = path or _find_hierarchy()
    return RoadHierarchy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------------


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    x: float
    y: float
    z: float | None = None

    kind: NodeKind = NodeKind.INTERSECTION
    control: Control = Control.NONE
    source_id: str = "unknown"

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    u: str
    v: str
    geometry: list[tuple[float, float]]
    """Polyline in the graph's metric CRS, from ``u`` to ``v``."""

    road_class: str = "unknown"
    width_m: float | None = None
    lane_count: int | None = None
    oneway: bool = False

    grade_pct: float | None = None
    """End-to-end gradient, percent. Specified in research doc §11 and unimplemented until the
    generation module needed it: grade cannot be recovered downstream because the elevation raster
    is gone by the time a graph reaches an exporter, so it is sampled at extraction and carried."""
    structure: Structure = Structure.AT_GRADE
    surface: str | None = None

    source_id: str = "unknown"
    """Registry source id. Present on every feature (invariant 3)."""

    confidence: float = 1.0
    """1.0 for observed geometry; a model's estimate otherwise. Mandatory — see module docstring."""

    attributes: dict[str, str] = Field(default_factory=dict)

    @property
    def length_m(self) -> float:
        return sum(
            math.dist(a, b) for a, b in zip(self.geometry, self.geometry[1:], strict=False)
        )

    @property
    def straight_length_m(self) -> float:
        if len(self.geometry) < 2:
            return 0.0
        return math.dist(self.geometry[0], self.geometry[-1])

    @property
    def sinuosity(self) -> float:
        """Path length over straight-line distance. 1.0 is straight; mountain roads run higher.

        One of the cleanest discriminators between the Hamamatsu plain and the Kawanehon valley,
        and therefore one of the first things the sensitivity sweep should move.
        """
        straight = self.straight_length_m
        return self.length_m / straight if straight > 1e-9 else 1.0

    def bearing_deg(self) -> float:
        """Compass bearing of the straight line from ``u`` to ``v``."""
        if len(self.geometry) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.geometry[0], self.geometry[-1]
        return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


class RoadGraph(BaseModel):
    """A road network over one tile or region."""

    model_config = ConfigDict(extra="forbid")

    crs: str
    tile_id: str | None = None
    lod_level: int | None = None
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)

    # -- construction -------------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        for endpoint in (edge.u, edge.v):
            if endpoint not in self.nodes:
                raise KeyError(f"edge {edge.id!r} references unknown node {endpoint!r}")
        self.edges[edge.id] = edge

    # -- topology -----------------------------------------------------------------------------

    def degree(self, node_id: str) -> int:
        return sum(1 for e in self.edges.values() if node_id in (e.u, e.v))

    def neighbours(self, node_id: str) -> set[str]:
        out = set()
        for e in self.edges.values():
            if e.u == node_id:
                out.add(e.v)
            elif e.v == node_id:
                out.add(e.u)
        return out

    @property
    def dead_ends(self) -> list[str]:
        return [n for n in self.nodes if self.degree(n) == 1]

    @property
    def isolated_nodes(self) -> list[str]:
        return [n for n in self.nodes if self.degree(n) == 0]

    def connected_components(self) -> list[set[str]]:
        """Node sets of each connected component.

        A generated network that scores well locally but splits into disconnected islands is the
        failure mode risk R7 names, and this is how it is detected.
        """
        seen: set[str] = set()
        components = []
        for start in self.nodes:
            if start in seen:
                continue
            stack, component = [start], set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(self.neighbours(current) - component)
            seen |= component
            components.append(component)
        return components

    # -- metrics (research doc §16.2) -----------------------------------------------------------

    @property
    def total_length_m(self) -> float:
        return sum(e.length_m for e in self.edges.values())

    def road_density_km_per_km2(self, area_km2: float) -> float:
        return (self.total_length_m / 1000.0) / area_km2 if area_km2 > 0 else 0.0

    def intersection_density_per_km2(self, area_km2: float) -> float:
        intersections = sum(1 for n in self.nodes if self.degree(n) >= 3)
        return intersections / area_km2 if area_km2 > 0 else 0.0

    @property
    def dead_end_ratio(self) -> float:
        if not self.nodes:
            return 0.0
        return len(self.dead_ends) / len(self.nodes)

    def degree_histogram(self) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for n in self.nodes:
            d = self.degree(n)
            histogram[d] = histogram.get(d, 0) + 1
        return histogram

    def orientation_entropy(self, bins: int = 36) -> float:
        """Shannon entropy of edge bearings, normalised to [0, 1].

        A good grid-versus-organic discriminator: a perfect grid concentrates bearings into a few
        bins and scores near 0, while terrain-following networks spread across many and score near
        1. Bearings are folded modulo 180 degrees because a road has no inherent direction.
        """
        if not self.edges:
            return 0.0

        counts = [0.0] * bins
        width = 180.0 / bins
        for e in self.edges.values():
            index = int((e.bearing_deg() % 180.0) / width) % bins
            counts[index] += e.length_m  # weight by length: a long road matters more

        total = sum(counts)
        if total <= 0:
            return 0.0

        entropy = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
        # A single occupied bin sums to exactly zero, which negation renders as -0.0. Harmless
        # arithmetically, but it reads as a bug in any report.
        return max(0.0, entropy / math.log(bins))

    # -- level of detail ------------------------------------------------------------------------

    def at_lod(self, lod: int, hierarchy: RoadHierarchy | None = None) -> RoadGraph:
        """A filtered copy containing only classes present at ``lod``.

        Nodes left isolated by the filter are dropped, so the result is a valid graph rather than
        a graph with debris.
        """
        hierarchy = hierarchy or load_hierarchy()
        keep = hierarchy.classes_at_lod(lod)

        edges = {eid: e for eid, e in self.edges.items() if e.road_class in keep}
        used = {n for e in edges.values() for n in (e.u, e.v)}

        return RoadGraph(
            crs=self.crs,
            tile_id=self.tile_id,
            lod_level=lod,
            nodes={nid: n for nid, n in self.nodes.items() if nid in used},
            edges=edges,
        )

    @property
    def source_ids(self) -> set[str]:
        return {e.source_id for e in self.edges.values()}

    def __len__(self) -> int:
        return len(self.edges)
