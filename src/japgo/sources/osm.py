"""OpenStreetMap adapter — **training only**.

OSM is the richest available source of road-network labels and the natural training target for
Phase 4. It is also the one source whose geometry must never reach shipped output.

The distinction, from research doc §6.1/§6.1c:

* Training on OSM is fine. Model *predictions* are not implicated by ODbL.
* A reconstruction *transformed from* OSM geometry is a Derivative Database, and publicly using it
  triggers share-alike — which conflicts with the project's commercial intent.

So this adapter exists, is useful, and is fenced. Every feature it produces carries
``source_id="osm"``, and the registry marks that source ``output_role: training_only``, which makes
the export gate refuse any attribution-only artifact touching it. :func:`assert_training_only_use`
below is the belt to that braces: it lets a caller state the intent explicitly at the point of use
rather than discovering the problem at export time.

Reads ``.osm`` XML. PBF is the realistic format for a prefecture-sized extract and needs
``pyosmium``; the parser here is factored so a PBF backend slots in behind the same node/way
callbacks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from ..core.roads import Edge, Node, NodeKind, RoadGraph, RoadHierarchy, load_hierarchy
from ..geo.crs import assert_metric, from_wgs84
from ..geo.tiling import Bounds
from ..provenance import ProvenanceViolation
from .base import ReadResult, SourceAdapter

log = logging.getLogger(__name__)

#: Ways carrying these keys are roads for our purposes.
HIGHWAY_KEY = "highway"


def assert_training_only_use(purpose: str) -> None:
    """Guard for OSM-derived geometry at the point of use.

    ``purpose`` must be one of ``training``, ``evaluation`` or ``gap_filling``. Anything else —
    notably ``export`` or ``reconstruction`` — raises, so the mistake surfaces where it is made
    rather than at the end of a pipeline run.
    """
    permitted = {"training", "evaluation", "gap_filling", "analysis"}
    if purpose not in permitted:
        raise ProvenanceViolation(
            f"OSM-derived geometry may not be used for {purpose!r}. Permitted uses are "
            f"{sorted(permitted)}. Shipped reconstruction geometry must come from the "
            "redistributable core (PLATEAU, NLNI, VIRTUAL SHIZUOKA, AW3D30) — see research doc "
            "§6.1c. Training on OSM is fine; transforming its geometry into a product is not."
        )


class OsmAdapter(SourceAdapter):
    """Reads OSM XML into a :class:`~japgo.core.roads.RoadGraph`."""

    source_id = "osm"
    provides = ("roads", "buildings", "water", "rail", "landuse")

    def __init__(self, gate, *, target_crs, hierarchy: RoadHierarchy | None = None) -> None:
        super().__init__(gate)
        self.target_crs = assert_metric(target_crs)
        self.hierarchy = hierarchy or load_hierarchy()

    # -----------------------------------------------------------------------------------------

    def read(
        self,
        path: Path,
        *,
        bounds: Bounds | None = None,
        purpose: str = "training",
        keep_paths: bool = False,
        **kwargs,
    ) -> ReadResult:
        """Read roads from an ``.osm`` XML extract.

        Parameters
        ----------
        purpose:
            Declared use. Validated by :func:`assert_training_only_use`.
        keep_paths:
            Include footways, cycleways and steps. Off by default: they are not part of the road
            network and including them distorts density and connectivity metrics.
        """
        assert_training_only_use(purpose)
        self.open()  # provenance gate

        path = Path(path)
        root = ET.parse(path).getroot()

        coords = self._read_nodes(root)
        graph = RoadGraph(crs=self.target_crs.to_string())
        warnings: list[str] = []
        unmapped: dict[str, int] = {}
        skipped_paths = 0

        # An OSM way is a cartographic object, not a topological one: one way can run through a
        # dozen junctions. A node referenced by two or more ways IS a junction, and splitting on
        # that is exact — no coordinate comparison, no rounding tolerance.
        accepted: list[tuple[str, list[str], str, dict[str, str]]] = []
        ref_uses: dict[str, int] = {}

        for way_id, refs, tags in self._read_ways(root):
            highway = tags.get(HIGHWAY_KEY)
            if not highway:
                continue

            road_class, mapped = self.hierarchy.from_osm(highway)
            if not mapped:
                unmapped[highway] = unmapped.get(highway, 0) + 1

            if road_class == "path" and not keep_paths:
                skipped_paths += 1
                continue

            usable = [r for r in refs if r in coords]
            if len(usable) < 2:
                continue

            if bounds is not None and not self._intersects(
                [coords[r] for r in usable], bounds
            ):
                continue

            accepted.append((way_id, usable, road_class, tags))
            for ref in usable:
                ref_uses[ref] = ref_uses.get(ref, 0) + 1

        junctions = {ref for ref, uses in ref_uses.items() if uses >= 2}

        for way_id, refs, road_class, tags in accepted:
            for index, span in enumerate(self._split_refs(refs, junctions)):
                self._add_way(
                    graph,
                    way_id if index == 0 and len(refs) == len(span) else f"{way_id}#{index}",
                    span,
                    [coords[r] for r in span],
                    road_class,
                    tags,
                )

        if unmapped:
            for value, count in sorted(unmapped.items(), key=lambda kv: -kv[1]):
                warnings.append(
                    f"OSM highway={value!r} ({count} way(s)) is not in the road hierarchy config; "
                    "add it to config/road_hierarchy.yaml"
                )
        if skipped_paths:
            warnings.append(
                f"{skipped_paths} non-vehicular way(s) excluded; pass keep_paths=True to include"
            )

        for w in warnings:
            log.warning("%s: %s", path.name, w)

        return ReadResult(
            layers={"roads": [graph]},
            record=self.make_record(
                layers=["roads"],
                note=f"{path.name}; purpose={purpose}; {len(graph.edges)} edges; TRAINING ONLY",
            ),
            warnings=warnings,
        )

    # -----------------------------------------------------------------------------------------

    def _read_nodes(self, root: ET.Element) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for node in root.iter("node"):
            node_id = node.get("id")
            lat, lon = node.get("lat"), node.get("lon")
            if node_id and lat and lon:
                out[node_id] = from_wgs84(float(lon), float(lat), self.target_crs)
        return out

    def _read_ways(self, root: ET.Element) -> Iterator[tuple[str, list[str], dict[str, str]]]:
        for way in root.iter("way"):
            way_id = way.get("id")
            if not way_id:
                continue
            refs = [nd.get("ref") for nd in way.iter("nd")]
            tags = {
                t.get("k"): t.get("v")
                for t in way.iter("tag")
                if t.get("k") is not None and t.get("v") is not None
            }
            yield way_id, [r for r in refs if r], tags  # type: ignore[misc]

    def _intersects(self, geometry: list[tuple[float, float]], bounds: Bounds) -> bool:
        xs = [p[0] for p in geometry]
        ys = [p[1] for p in geometry]
        return not (
            max(xs) < bounds.minx
            or min(xs) > bounds.maxx
            or max(ys) < bounds.miny
            or min(ys) > bounds.maxy
        )

    @staticmethod
    def _split_refs(refs: list[str], junctions: set[str]) -> list[list[str]]:
        """Split a way's node refs into spans that break at junctions.

        Endpoints always terminate a span; interior junction refs both end one span and begin the
        next, so the shared node is present in both and the graph stays connected.
        """
        spans: list[list[str]] = [[refs[0]]]
        for ref in refs[1:]:
            spans[-1].append(ref)
            if ref in junctions and ref is not refs[-1]:
                spans.append([ref])
        return [s for s in spans if len(s) >= 2]

    def _add_way(
        self,
        graph: RoadGraph,
        way_id: str,
        refs: list[str],
        geometry: list[tuple[float, float]],
        road_class: str,
        tags: dict[str, str],
    ) -> None:
        """Add one span as an edge between its endpoint nodes."""
        spec = self.hierarchy.spec(road_class)

        u_id, v_id = f"n{refs[0]}", f"n{refs[-1]}"
        for node_id, position in ((u_id, geometry[0]), (v_id, geometry[-1])):
            if node_id not in graph.nodes:
                graph.add_node(
                    Node(
                        id=node_id,
                        x=position[0],
                        y=position[1],
                        kind=NodeKind.INTERSECTION,
                        source_id=self.source_id,
                    )
                )

        structure = self._structure(tags)

        graph.add_edge(
            Edge(
                id=f"w{way_id}",
                u=u_id,
                v=v_id,
                geometry=geometry,
                road_class=road_class,
                lane_count=_int(tags.get("lanes")) or spec.typical_lanes,
                width_m=_float(tags.get("width")) or spec.typical_width_m,
                oneway=tags.get("oneway") in {"yes", "true", "1", "-1"},
                structure=structure,
                surface=tags.get("surface"),
                source_id=self.source_id,
                confidence=1.0,  # observed, not inferred
                attributes={k: v for k, v in tags.items() if k in {"name", "ref", "bridge", "tunnel"}},
            )
        )

    def _structure(self, tags: dict[str, str]):
        from ..core.roads import Structure

        if tags.get("bridge") not in (None, "no"):
            return Structure.BRIDGE
        if tags.get("tunnel") not in (None, "no"):
            return Structure.TUNNEL
        if tags.get("embankment") not in (None, "no"):
            return Structure.EMBANKMENT
        if tags.get("cutting") not in (None, "no"):
            return Structure.CUT
        return Structure.AT_GRADE


def split_at_intersections(graph: RoadGraph) -> RoadGraph:
    """Split edges wherever their geometry passes through another edge's endpoint node.

    OSM ways are cartographic objects, not topological ones: a single way can run through a dozen
    junctions. Left unsplit, intersection density is undercounted and every connectivity metric is
    wrong — so this runs before any metric is trusted.
    """
    node_at: dict[tuple[float, float], str] = {
        (round(n.x, 3), round(n.y, 3)): nid for nid, n in graph.nodes.items()
    }

    out = RoadGraph(crs=graph.crs, tile_id=graph.tile_id, lod_level=graph.lod_level)
    for nid, node in graph.nodes.items():
        out.add_node(node.model_copy())
        del nid

    for edge in graph.edges.values():
        segments: list[list[tuple[float, float]]] = [[edge.geometry[0]]]
        for point in edge.geometry[1:]:
            segments[-1].append(point)
            key = (round(point[0], 3), round(point[1], 3))
            if key in node_at and point is not edge.geometry[-1]:
                segments.append([point])

        if len(segments) == 1:
            out.add_edge(edge.model_copy())
            continue

        for index, geometry in enumerate(segments):
            if len(geometry) < 2:
                continue
            u_id = _node_for(out, geometry[0], node_at, edge.source_id)
            v_id = _node_for(out, geometry[-1], node_at, edge.source_id)
            out.add_edge(
                edge.model_copy(
                    update={
                        "id": f"{edge.id}#{index}",
                        "u": u_id,
                        "v": v_id,
                        "geometry": geometry,
                    }
                )
            )

    return out


def _node_for(
    graph: RoadGraph,
    point: tuple[float, float],
    node_at: dict[tuple[float, float], str],
    source_id: str,
) -> str:
    key = (round(point[0], 3), round(point[1], 3))
    if key in node_at and node_at[key] in graph.nodes:
        return node_at[key]

    node_id = f"s{key[0]:.0f}_{key[1]:.0f}"
    if node_id not in graph.nodes:
        graph.add_node(
            Node(id=node_id, x=point[0], y=point[1], kind=NodeKind.JUNCTION, source_id=source_id)
        )
    node_at[key] = node_id
    return node_id


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None
