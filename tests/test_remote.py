"""Tests for the remote ingest path.

Everything here runs offline. The network calls are the part that cannot be tested in CI, so what
is tested instead is everything that decides *whether and what* to fetch — mesh selection, the
training-only guard, caching, retry, and clipping. Those are where the failures are silent: a
wrong mesh filter produces a corpus that builds cleanly with every building channel zero, which
looks exactly like an area with no buildings.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from japgo.geo import SHIZUOKA, Tile
from japgo.geo.tiling import Bounds
from japgo.provenance import ProvenanceViolation
from japgo.sources.jismesh import MeshCodeError, code_in, covering, decode
from japgo.sources.overpass import OverpassClient, OverpassError

TILE = Tile(zone=8, ix=53, iy=-108)
CRS = SHIZUOKA.crs


# ---------------------------------------------------------------------------------------------
# JIS mesh codes
# ---------------------------------------------------------------------------------------------


def test_primary_meshes_match_their_known_squares():
    """5339 is the Tokyo primary mesh and 5239 the one south of it, covering Atami. If these
    drift, every PLATEAU member selection silently drifts with them."""
    tokyo = decode("5339")
    assert (tokyo.min_lon, tokyo.max_lon) == (139.0, 140.0)
    assert tokyo.min_lat == pytest.approx(35.3333, abs=1e-4)
    assert tokyo.intersects(139.76, 35.68, 139.77, 35.69)

    assert decode("5239").intersects(139.076, 35.096, 139.08, 35.10)


def test_mesh_squares_shrink_by_level():
    for code, lon_size in (("5339", 1.0), ("533945", 0.125), ("53394567", 0.0125)):
        square = decode(code)
        assert square.max_lon - square.min_lon == pytest.approx(lon_size)


def test_third_mesh_sits_inside_its_parents():
    third, second, first = decode("52394036"), decode("523940"), decode("5239")
    for parent in (second, first):
        assert parent.min_lon <= third.min_lon and third.max_lon <= parent.max_lon
        assert parent.min_lat <= third.min_lat and third.max_lat <= parent.max_lat


def test_touching_edges_do_not_count_as_overlap():
    square = decode("52394036")
    assert not square.intersects(square.max_lon, square.min_lat, square.max_lon + 0.01, square.max_lat)


@pytest.mark.parametrize("bad", ["", "533", "53394", "5339456", "abcd", "533998"])
def test_malformed_codes_are_rejected(bad):
    with pytest.raises(MeshCodeError):
        decode(bad)


def test_an_epsg_code_in_a_filename_is_not_mistaken_for_a_mesh():
    """PLATEAU names carry an EPSG code too. 6697 decodes to longitude 197, off the globe."""
    with pytest.raises(MeshCodeError):
        decode("6697")
    assert code_in("52394036_bldg_6697_op.gml") == "52394036"


def test_an_undecodable_name_is_kept_rather_than_dropped():
    """The safe direction. A spurious download costs bandwidth; a spurious exclusion costs data,
    and the loss is invisible."""
    assert code_in("codelists/Building_usage.xml") is None
    names = ["codelists/Building_usage.xml", "52394036_bldg_6697_op.gml"]
    assert covering(names, 139.0, 35.0, 139.2, 35.1) == names


def test_covering_excludes_squares_outside_the_box():
    # 52394036 is lon 139.075..139.0875; ask for somewhere else entirely.
    assert covering(["52394036_bldg_6697_op.gml"], 140.0, 36.0, 140.1, 36.1) == []


# ---------------------------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------------------------


def test_a_non_training_purpose_is_refused_before_any_request(gate, tmp_path, monkeypatch):
    """Invariant 3b at the point of the mistake. OSM geometry must never reach shipped output,
    and the check has to fire before the fetch, not after."""
    client = OverpassClient(gate, cache_dir=tmp_path)
    monkeypatch.setattr(
        client, "_request", lambda q: pytest.fail("a request was made despite a bad purpose")
    )
    with pytest.raises(ProvenanceViolation):
        client.fetch(TILE.core, CRS, purpose="export")


def test_the_query_carries_the_bbox_converted_to_wgs84(gate, tmp_path):
    query = OverpassClient(gate, cache_dir=tmp_path).query_for(TILE.core, CRS)
    assert '["highway"]' in query
    # The Atami tile: lat ~35.025..35.034, lon ~139.081..139.092.
    import re

    match = re.search(r"\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)", query)
    assert match, query
    south, west, north, east = (float(v) for v in match.groups())
    assert 35.02 < south < north < 35.04
    assert 139.07 < west < east < 139.10


def test_a_cached_response_makes_no_request(gate, tmp_path, monkeypatch):
    """Overpass is a free shared service. A rebuild must not re-ask it for what it already gave."""
    client = OverpassClient(gate, cache_dir=tmp_path)
    calls = []

    def fake_request(query):
        calls.append(query)
        return b'<?xml version="1.0"?><osm version="0.6"></osm>'

    monkeypatch.setattr(client, "_request", fake_request)

    first = client.fetch(TILE.core, CRS, key="region")
    second = client.fetch(TILE.core, CRS, key="region")

    assert len(calls) == 1
    assert first.from_cache is False and second.from_cache is True
    assert second.path == first.path and second.bytes_fetched == 0


def test_a_response_that_is_not_osm_is_an_error_not_a_cached_file(gate, tmp_path, monkeypatch):
    """An HTML error page written to the cache would be read as an empty road network on every
    later run, and the corpus would quietly have no roads."""
    client = OverpassClient(gate, cache_dir=tmp_path)
    monkeypatch.setattr(client, "_request", lambda q: b"<html>rate limited</html>")

    with pytest.raises(OverpassError):
        client.fetch(TILE.core, CRS, key="region")
    assert list(tmp_path.glob("*.osm")) == []


def test_rate_limiting_is_retried_and_other_errors_are_not(gate, tmp_path, monkeypatch):
    client = OverpassClient(gate, cache_dir=tmp_path, retries=3, backoff_s=0)
    monkeypatch.setattr("time.sleep", lambda s: None)

    attempts = []

    def flaky(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(client.endpoint, 429, "Too Many Requests", {}, None)
        raise urllib.error.HTTPError(client.endpoint, 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", flaky)

    with pytest.raises(OverpassError, match="HTTP 400"):
        client.fetch(TILE.core, CRS, key="region")
    assert len(attempts) == 3  # two retried 429s, then a 400 that is raised immediately


# ---------------------------------------------------------------------------------------------
# RemoteSources
# ---------------------------------------------------------------------------------------------


def test_roads_are_clipped_to_the_tile_not_handed_over_whole(gate, tmp_path):
    """The staged-files path clips per tile; the remote path must agree.

    Handing every tile the region's graph writes a region-sized roads.json into each directory and
    makes ``bundle.roads`` mean something different depending on which provider built the tile.
    """
    from japgo.core import Edge, Node, RoadGraph
    from japgo.pipeline.remote import RemoteSources

    graph = RoadGraph(crs=CRS.to_string())
    cx, cy = TILE.core.centre
    graph.add_node(Node(id="near_u", x=cx - 100, y=cy))
    graph.add_node(Node(id="near_v", x=cx + 100, y=cy))
    graph.add_node(Node(id="far_u", x=cx + 50_000, y=cy))
    graph.add_node(Node(id="far_v", x=cx + 50_100, y=cy))
    graph.add_edge(
        Edge(id="near", u="near_u", v="near_v", geometry=[(cx - 100, cy), (cx + 100, cy)],
             source_id="osm")
    )
    graph.add_edge(
        Edge(id="far", u="far_u", v="far_v",
             geometry=[(cx + 50_000, cy), (cx + 50_100, cy)], source_id="osm")
    )

    source = RemoteSources(gate, CRS, cache_dir=tmp_path)
    source._roads = graph
    source._prepared = True
    source._terrain = _StubTerrain()

    inputs = source.inputs_for(TILE)
    assert set(inputs.roads.edges) == {"near"}


def test_a_tile_with_no_terrain_is_skipped_rather_than_written_empty(gate, tmp_path):
    from japgo.pipeline.remote import RemoteSources

    source = RemoteSources(gate, CRS, cache_dir=tmp_path)
    source._prepared = True
    source._terrain = _StubTerrain(meshes=[])

    assert source.inputs_for(TILE) is None


def test_the_terrain_fetcher_is_built_once_for_the_whole_run(gate, tmp_path):
    """Its mesh index is a network fetch. Rebuilding it per tile re-fetches that index for every
    tile in the site."""
    from japgo.pipeline.remote import RemoteSources

    source = RemoteSources(gate, CRS, cache_dir=tmp_path)
    source._prepared = True
    source._terrain = stub = _StubTerrain()

    for ix in (52, 53, 54):
        source.inputs_for(Tile(zone=8, ix=ix, iy=-108))

    assert source._terrain is stub
    assert stub.calls == 3


def test_prepare_needs_tiles_to_know_what_to_fetch(gate, tmp_path):
    from japgo.pipeline.remote import RemoteSources

    with pytest.raises(ValueError, match="at least one tile"):
        RemoteSources(gate, CRS, cache_dir=tmp_path).prepare([])


class _StubTerrain:
    """Stands in for TerrainFetcher so the per-tile logic is testable without the network."""

    def __init__(self, meshes: list[str] | None = None) -> None:
        self.meshes = ["52394036"] if meshes is None else meshes
        self.calls = 0

    def dem_for(self, bounds: Bounds, crs, *, resolution=1.0, key=None, progress=False):
        import numpy as np

        from japgo.geo.raster import Raster

        self.calls += 1
        rows = int(bounds.height / resolution)
        cols = int(bounds.width / resolution)
        return Raster(np.zeros((rows, cols), dtype=np.float32), bounds, crs), list(self.meshes)


# ---------------------------------------------------------------------------------------------
# NLNI land use
# ---------------------------------------------------------------------------------------------


def test_primary_meshes_cover_the_mvp_sites():
    """Land use ships one file per primary mesh, so a site extent has to resolve to codes."""
    from japgo.sources.jismesh import primary_meshes_for

    # Atami, Kawanehon, Hamamatsu respectively.
    assert primary_meshes_for(139.07, 35.02, 139.11, 35.04) == ["5239"]
    assert primary_meshes_for(138.08, 35.08, 138.12, 35.12) == ["5238"]
    assert primary_meshes_for(137.73, 34.70, 137.77, 34.74) == ["5237"]


def test_a_box_spanning_two_primary_meshes_returns_both():
    from japgo.sources.jismesh import primary_meshes_for

    assert primary_meshes_for(138.95, 35.02, 139.05, 35.04) == ["5238", "5239"]


def test_the_landuse_url_pins_the_vintage_and_the_datum():
    """The registry pins FY2016 because it is the only vintage the datalist marks open data. If
    the URL did not carry the vintage, the registry and the bytes could drift apart silently.

    ``-jgd`` not ``-tky``: the same data ships in JGD2000 and the old Tokyo datum, which differ by
    a few hundred metres — more than a tile's halo.
    """
    from japgo.pipeline.remote import LANDUSE_STEM, LANDUSE_URL

    url = LANDUSE_URL.format(stem=LANDUSE_STEM, mesh="5239")
    assert "L03-b-16" in url and url.endswith("L03-b-16_5239-jgd_GML.zip")
    assert "-tky" not in url


def test_an_unpublished_primary_mesh_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A primary mesh is ~90 km across; one clipping a site can be all sea and unpublished."""
    import urllib.error

    from japgo.pipeline.remote import _fetch_landuse_mesh

    def missing(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    assert _fetch_landuse_mesh("9999", tmp_path) is None


def test_landuse_uses_the_mesh_that_contains_the_tile(gate, tmp_path, monkeypatch):
    """A tile sits in exactly one primary mesh. Reading the wrong 50 MB file would rasterise land
    use for somewhere else entirely, and every cell would come back zero."""
    from japgo.pipeline.remote import RemoteSources
    from japgo.sources.jismesh import decode

    source = RemoteSources(gate, CRS, cache_dir=tmp_path)
    source._prepared = True
    source._terrain = _StubTerrain()
    source._landuse = [
        (decode("5237"), tmp_path / "wrong.shp"),   # Hamamatsu, far west
        (decode("5239"), tmp_path / "right.shp"),   # Atami
    ]

    used = []

    class _Adapter:
        def __init__(self, gate, *, target_crs):
            pass

        def read(self, path, **kwargs):
            used.append(path)
            from japgo.sources.base import ReadResult

            return ReadResult(layers={}, record=None, warnings=[])

    monkeypatch.setattr("japgo.sources.NlniLanduseAdapter", _Adapter)
    source.inputs_for(TILE)

    assert used == [tmp_path / "right.shp"]


def test_the_road_class_is_not_an_input_channel():
    """NLNI code 0901 道路 must not reach landuse_built.

    landuse_built is a model input; road_mask is what the model predicts. Grouping road corridors
    into an input leaks the target — and leaks it precisely where the wide arterials are, which
    are the easiest and most valuable roads to get right.
    """
    from japgo.sources.nlni import load_landuse_spec

    spec = load_landuse_spec()
    assert spec.class_for("0901") == ("road", True)      # still mapped, so it is not "unmapped"
    for channel, members in spec.channel_groups.items():
        assert "road" not in members, f"{channel} contains the road class"
    assert spec.group_for("road") is None                # falls through to "other"
    # Rail is environmental context, not the target, and stays.
    assert spec.group_for("railway") == "landuse_built"
