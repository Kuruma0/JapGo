"""Tests for the road graph and the OSM adapter.

The licensing fence around OSM is tested as hard as the geometry: it is the one place where a
quiet mistake is a licensing failure rather than a wrong number.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from japgo.core import Edge, Node, NodeKind, RoadGraph, Structure, load_hierarchy
from japgo.geo import SHIZUOKA
from japgo.provenance import ProvenanceViolation
from japgo.sources import OsmAdapter, assert_training_only_use, split_at_intersections

CRS = SHIZUOKA.crs.to_string()


def _graph(*edges: tuple[str, str, str, list[tuple[float, float]]]) -> RoadGraph:
    g = RoadGraph(crs=CRS)
    for _, u, v, geometry in edges:
        for node_id, position in ((u, geometry[0]), (v, geometry[-1])):
            if node_id not in g.nodes:
                g.add_node(Node(id=node_id, x=position[0], y=position[1]))
    for eid, u, v, geometry in edges:
        g.add_edge(Edge(id=eid, u=u, v=v, geometry=geometry, road_class="residential"))
    return g


# ---------------------------------------------------------------------------------------------
# Hierarchy config
# ---------------------------------------------------------------------------------------------


def test_every_osm_mapping_targets_a_declared_class():
    h = load_hierarchy()
    for osm_value, road_class in h.osm_highway.items():
        assert road_class in h.classes, f"{osm_value} maps to undeclared class {road_class}"


def test_lod_levels_are_nested():
    """Level n must contain everything level n-1 does, or the ladder is not a ladder."""
    h = load_hierarchy()
    for lod in range(1, 5):
        assert h.classes_at_lod(lod - 1) <= h.classes_at_lod(lod)


def test_lod_zero_is_major_highways_only():
    assert load_hierarchy().classes_at_lod(0) == {"expressway", "trunk"}


def test_paths_are_excluded_from_vehicular_classes():
    assert "path" not in load_hierarchy().vehicular_classes


def test_grade_limits_loosen_down_the_hierarchy():
    """An expressway cannot climb what a forestry track can."""
    h = load_hierarchy()
    assert h.spec("expressway").max_grade_percent < h.spec("residential").max_grade_percent
    assert h.spec("residential").max_grade_percent < h.spec("track").max_grade_percent


# ---------------------------------------------------------------------------------------------
# Edge geometry
# ---------------------------------------------------------------------------------------------


def test_edge_length_follows_the_polyline():
    e = Edge(id="e", u="a", v="b", geometry=[(0, 0), (3, 0), (3, 4)])
    assert e.length_m == pytest.approx(7.0)
    assert e.straight_length_m == pytest.approx(5.0)


def test_sinuosity_is_one_for_a_straight_edge():
    e = Edge(id="e", u="a", v="b", geometry=[(0, 0), (10, 0)])
    assert e.sinuosity == pytest.approx(1.0)


def test_sinuosity_rises_with_switchbacks():
    """The discriminator between the Hamamatsu plain and the Kawanehon valley."""
    straight = Edge(id="s", u="a", v="b", geometry=[(0, 0), (100, 0)])
    winding = Edge(
        id="w", u="a", v="b",
        geometry=[(0, 0), (25, 30), (50, -30), (75, 30), (100, 0)],
    )
    assert winding.sinuosity > straight.sinuosity
    assert winding.sinuosity > 1.5


@pytest.mark.parametrize(
    ("end", "expected"),
    [((0, 10), 0.0), ((10, 0), 90.0), ((0, -10), 180.0), ((-10, 0), 270.0)],
)
def test_bearing_is_a_compass_direction(end, expected):
    e = Edge(id="e", u="a", v="b", geometry=[(0, 0), end])
    assert e.bearing_deg() == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------------------------


def test_edge_referencing_an_unknown_node_is_refused():
    g = RoadGraph(crs=CRS)
    g.add_node(Node(id="a", x=0, y=0))
    with pytest.raises(KeyError, match="unknown node"):
        g.add_edge(Edge(id="e", u="a", v="ghost", geometry=[(0, 0), (1, 1)]))


def test_degree_and_dead_ends():
    g = _graph(
        ("e1", "a", "b", [(0, 0), (10, 0)]),
        ("e2", "b", "c", [(10, 0), (20, 0)]),
        ("e3", "b", "d", [(10, 0), (10, 10)]),
    )
    assert g.degree("b") == 3
    assert set(g.dead_ends) == {"a", "c", "d"}
    assert g.dead_end_ratio == pytest.approx(0.75)


def test_connected_components_detect_islands():
    """A network that scores well locally but splits into islands is risk R7."""
    g = _graph(
        ("e1", "a", "b", [(0, 0), (10, 0)]),
        ("e2", "c", "d", [(100, 100), (110, 100)]),
    )
    components = g.connected_components()
    assert len(components) == 2
    assert {frozenset(c) for c in components} == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_degree_histogram():
    g = _graph(
        ("e1", "a", "b", [(0, 0), (10, 0)]),
        ("e2", "b", "c", [(10, 0), (20, 0)]),
    )
    assert g.degree_histogram() == {1: 2, 2: 1}


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def test_road_and_intersection_density():
    g = _graph(
        ("e1", "a", "b", [(0, 0), (1000, 0)]),
        ("e2", "b", "c", [(1000, 0), (2000, 0)]),
        ("e3", "b", "d", [(1000, 0), (1000, 1000)]),
    )
    assert g.total_length_m == pytest.approx(3000.0)
    assert g.road_density_km_per_km2(1.0) == pytest.approx(3.0)
    assert g.intersection_density_per_km2(1.0) == pytest.approx(1.0)


def test_orientation_entropy_is_low_for_a_grid():
    """A grid concentrates bearings into two bins; terrain-following networks spread out."""
    edges = []
    for i in range(4):
        edges.append((f"h{i}", f"h{i}a", f"h{i}b", [(0, i * 100), (300, i * 100)]))
        edges.append((f"v{i}", f"v{i}a", f"v{i}b", [(i * 100, 0), (i * 100, 300)]))
    grid = _graph(*edges)

    organic = _graph(
        *[
            (f"o{i}", f"o{i}a", f"o{i}b",
             [(0, 0), (100 * math.cos(i * 0.4), 100 * math.sin(i * 0.4))])
            for i in range(16)
        ]
    )
    assert grid.orientation_entropy() < organic.orientation_entropy()
    assert grid.orientation_entropy() < 0.3


def test_orientation_entropy_is_empty_safe():
    assert RoadGraph(crs=CRS).orientation_entropy() == 0.0


def test_orientation_entropy_never_reports_negative_zero():
    """A single occupied bin sums to exactly zero; negation would render it -0.0 in reports."""
    parallel = _graph(
        ("e1", "a", "b", [(0, 0), (0, 100)]),
        ("e2", "c", "d", [(50, 0), (50, 100)]),
    )
    entropy = parallel.orientation_entropy()
    assert entropy == 0.0
    assert math.copysign(1.0, entropy) > 0


# ---------------------------------------------------------------------------------------------
# Level of detail
# ---------------------------------------------------------------------------------------------


def _mixed_graph() -> RoadGraph:
    g = RoadGraph(crs=CRS)
    for i, road_class in enumerate(["expressway", "primary", "residential", "service"]):
        u, v = f"u{i}", f"v{i}"
        g.add_node(Node(id=u, x=i * 100.0, y=0.0))
        g.add_node(Node(id=v, x=i * 100.0, y=100.0))
        g.add_edge(
            Edge(id=f"e{i}", u=u, v=v, geometry=[(i * 100.0, 0.0), (i * 100.0, 100.0)],
                 road_class=road_class)
        )
    return g


def test_lod_filtering_is_monotonic():
    g = _mixed_graph()
    counts = [len(g.at_lod(lod)) for lod in range(5)]
    assert counts == sorted(counts)
    assert counts[0] == 1  # expressway only
    assert counts[4] == 4


def test_lod_filtering_drops_orphaned_nodes():
    """A filtered graph must be a valid graph, not a graph with debris."""
    filtered = _mixed_graph().at_lod(0)
    assert filtered.isolated_nodes == []
    assert len(filtered.nodes) == 2


def test_lod_level_is_recorded():
    assert _mixed_graph().at_lod(2).lod_level == 2


# ---------------------------------------------------------------------------------------------
# The OSM licensing fence
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("purpose", ["training", "evaluation", "gap_filling", "analysis"])
def test_permitted_purposes_are_allowed(purpose):
    assert_training_only_use(purpose)


@pytest.mark.parametrize("purpose", ["export", "reconstruction", "shipping", "product"])
def test_forbidden_purposes_are_refused(purpose):
    with pytest.raises(ProvenanceViolation, match="redistributable core"):
        assert_training_only_use(purpose)


def test_osm_source_is_marked_training_only(registry):
    assert registry.require("osm").output_role.value == "training_only"
    assert not registry.require("osm").may_ship_geometry


def test_reading_osm_for_export_is_refused_before_any_io(gate, tmp_path):
    """The guard fires at the point of use, not at the end of a pipeline run."""
    adapter = OsmAdapter(gate, target_crs=SHIZUOKA.crs)
    missing = tmp_path / "does-not-exist.osm"
    with pytest.raises(ProvenanceViolation, match="may not be used for 'export'"):
        adapter.read(missing, purpose="export")


# ---------------------------------------------------------------------------------------------
# OSM parsing
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def osm_path(tmp_path):
    """A tiny extract: two crossing roads, a bridge, and a footway."""
    content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <osm version="0.6">
          <node id="1" lat="34.9760" lon="138.3830"/>
          <node id="2" lat="34.9760" lon="138.3850"/>
          <node id="3" lat="34.9760" lon="138.3870"/>
          <node id="4" lat="34.9740" lon="138.3850"/>
          <node id="5" lat="34.9780" lon="138.3850"/>
          <way id="100">
            <nd ref="1"/><nd ref="2"/><nd ref="3"/>
            <tag k="highway" v="secondary"/>
            <tag k="lanes" v="4"/>
            <tag k="name" v="Test Avenue"/>
          </way>
          <way id="200">
            <nd ref="4"/><nd ref="2"/><nd ref="5"/>
            <tag k="highway" v="residential"/>
            <tag k="oneway" v="yes"/>
          </way>
          <way id="300">
            <nd ref="1"/><nd ref="4"/>
            <tag k="highway" v="trunk"/>
            <tag k="bridge" v="yes"/>
          </way>
          <way id="400">
            <nd ref="3"/><nd ref="5"/>
            <tag k="highway" v="footway"/>
          </way>
        </osm>
        """)
    path = tmp_path / "extract.osm"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def osm_adapter(gate):
    return OsmAdapter(gate, target_crs=SHIZUOKA.crs)


def test_reads_roads_split_at_junctions(osm_adapter, osm_path):
    """Ways 100 and 200 both pass through node 2, so both split there. Way 300 stays whole.

    Splitting happens at ingest because an unsplit graph is wrong for every topology metric, and
    a graph that is wrong by default invites metrics computed on it by accident.
    """
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    assert set(graph.edges) == {"w100#0", "w100#1", "w200#0", "w200#1", "w300"}


def test_footways_are_excluded_by_default(osm_adapter, osm_path):
    """Including them distorts density and connectivity metrics."""
    result = osm_adapter.read(osm_path)
    assert "w400" not in result.layers["roads"][0].edges
    assert any("non-vehicular" in w for w in result.warnings)


def test_footways_can_be_kept_explicitly(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path, keep_paths=True).layers["roads"][0]
    assert "w400" in graph.edges


def test_tags_map_to_attributes(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    assert graph.edges["w100#0"].road_class == "secondary"
    assert graph.edges["w100#0"].lane_count == 4
    assert graph.edges["w100#0"].attributes["name"] == "Test Avenue"
    assert graph.edges["w200#0"].oneway is True
    assert graph.edges["w300"].structure is Structure.BRIDGE


def test_split_spans_inherit_the_parent_way_attributes(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    for span in ("w100#0", "w100#1"):
        assert graph.edges[span].road_class == "secondary"
        assert graph.edges[span].lane_count == 4


def test_width_falls_back_to_the_class_default(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    expected = load_hierarchy().spec("residential").typical_width_m
    assert graph.edges["w200#0"].width_m == pytest.approx(expected)


def test_geometry_is_projected_to_metres(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    # Way 100 spans ~0.004 degrees of longitude at 35N: roughly 360 m across both spans.
    total = graph.edges["w100#0"].length_m + graph.edges["w100#1"].length_m
    assert 200 < total < 500


def test_split_preserves_the_full_way_length(osm_adapter, osm_path):
    """Splitting is a topology operation; it must not move or lose geometry."""
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    a, b = graph.edges["w100#0"], graph.edges["w100#1"]
    assert a.geometry[-1] == b.geometry[0]  # shared junction node, present in both
    assert a.v == b.u


def test_every_feature_carries_its_source_id(osm_adapter, osm_path):
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    assert graph.source_ids == {"osm"}
    assert all(n.source_id == "osm" for n in graph.nodes.values())


def test_record_states_training_only(osm_adapter, osm_path):
    assert "TRAINING ONLY" in osm_adapter.read(osm_path).record.note


# ---------------------------------------------------------------------------------------------
# Topology construction
# ---------------------------------------------------------------------------------------------


def test_junction_is_detected_as_an_intersection(osm_adapter, osm_path):
    """Node 2 carries four edge ends; unsplit it would have looked like a mid-way vertex."""
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    assert graph.degree("n2") == 4
    assert graph.intersection_density_per_km2(1.0) > 0


def test_ingest_is_already_topologically_split(osm_adapter, osm_path):
    """The geometric pass should find nothing left to do on an OSM graph."""
    graph = osm_adapter.read(osm_path).layers["roads"][0]
    again = split_at_intersections(graph)
    assert len(again.edges) == len(graph.edges)
    assert again.total_length_m == pytest.approx(graph.total_length_m, rel=1e-6)


def test_geometric_split_handles_graphs_without_node_refs():
    """`split_at_intersections` exists for non-OSM graphs, where ref counting is unavailable.

    A way passing through another edge's endpoint must be broken there.
    """
    g = RoadGraph(crs=CRS)
    g.add_node(Node(id="a", x=0, y=0))
    g.add_node(Node(id="b", x=200, y=0))
    g.add_node(Node(id="mid", x=100, y=0))
    g.add_node(Node(id="spur", x=100, y=100))
    g.add_edge(
        Edge(id="through", u="a", v="b", geometry=[(0, 0), (100, 0), (200, 0)],
             road_class="residential", source_id="plateau")
    )
    g.add_edge(
        Edge(id="spur", u="mid", v="spur", geometry=[(100, 0), (100, 100)],
             road_class="residential", source_id="plateau")
    )

    split = split_at_intersections(g)
    assert len(split.edges) == 3
    assert split.total_length_m == pytest.approx(g.total_length_m, rel=1e-6)
    assert split.degree("mid") == 3
