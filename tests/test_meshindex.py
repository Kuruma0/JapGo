"""Tests for VIRTUAL SHIZUOKA mesh discovery.

The MVT reader is hand-rolled — a protobuf dependency for two field types was not worth it — so it
gets tested against bytes built by an independent encoder here. Decoding with the same assumptions
used to encode would prove nothing, so the encoder is written from the MVT spec rather than by
inverting the decoder.
"""

from __future__ import annotations

import http.server
import threading

import numpy as np
import pytest

from japgo.geo.crs import zone
from japgo.geo.tiling import Bounds
from japgo.provenance import ProvenanceViolation
from japgo.sources.meshindex import GRID_INDEX, Mesh, MeshIndex, _packed, _rings, _varint


# ---------------------------------------------------------------------------------------------
# Independent MVT encoder, per the spec
# ---------------------------------------------------------------------------------------------


def _enc_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _enc_varint((field << 3) | wire)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _enc_varint(len(payload)) + payload


def _varint_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _enc_varint(value)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _square_geometry(x0: int, y0: int, size: int) -> bytes:
    """MoveTo(x0,y0), LineTo x3, ClosePath — a closed square in tile coordinates."""
    commands = [
        (1 << 3) | 1, _zigzag(x0), _zigzag(y0),
        (3 << 3) | 2,
        _zigzag(size), _zigzag(0),
        _zigzag(0), _zigzag(size),
        _zigzag(-size), _zigzag(0),
        (1 << 3) | 7,
    ]
    return b"".join(_enc_varint(c) for c in commands)


def _build_tile(entries: list[tuple[str, str, int, int, int]]) -> bytes:
    """entries: (mesh_no, url, x0, y0, size) in tile coordinates."""
    keys = ["MESH_NO", "URL"]
    values: list[str] = []
    features = b""

    for mesh_no, url, x0, y0, size in entries:
        values += [mesh_no, url]
        tags = [0, len(values) - 2, 1, len(values) - 1]
        feature = (
            _bytes_field(2, b"".join(_enc_varint(t) for t in tags))
            + _varint_field(3, 3)  # POLYGON
            + _bytes_field(4, _square_geometry(x0, y0, size))
        )
        features += _bytes_field(2, feature)

    layer = _bytes_field(1, b"Grid") + features
    for key in keys:
        layer += _bytes_field(3, key.encode())
    for value in values:
        layer += _bytes_field(4, _bytes_field(1, value.encode()))
    layer += _varint_field(5, 4096) + _varint_field(15, 2)

    return _bytes_field(3, layer)


# ---------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 4096, 2**21])
def test_varint_roundtrip(value):
    decoded, index = _varint(_enc_varint(value), 0)
    assert decoded == value
    assert index == len(_enc_varint(value))


def test_packed_reads_a_sequence():
    assert _packed(b"".join(_enc_varint(v) for v in [1, 300, 7])) == [1, 300, 7]


def test_rings_decodes_a_closed_square():
    rings = _rings(_packed(_square_geometry(10, 20, 100)))
    assert len(rings) == 1
    assert rings[0][0] == (10, 20)
    assert (110, 20) in rings[0]
    assert (110, 120) in rings[0]


def test_rings_handles_zigzag_negatives():
    """Coordinates move backwards as often as forwards; a sign error would skew every extent."""
    ring = _rings(_packed(_square_geometry(500, 500, -100)))[0]
    assert (400, 500) in ring


def test_rings_of_empty_geometry_is_empty():
    assert _rings([]) == []


# ---------------------------------------------------------------------------------------------
# Index served over HTTP
# ---------------------------------------------------------------------------------------------

# A square covering roughly the whole z14 tile that contains Atami.
ATAMI_LON, ATAMI_LAT = 139.084, 35.032


@pytest.fixture(scope="module")
def index_server():
    payload = _build_tile(
        [
            ("08NF5373", "https://example.invalid/08NF5373.zip", 0, 0, 4095),
            ("08NF9999", "https://example.invalid/08NF9999.zip", 4000, 4000, 90),
        ]
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.pbf"
    server.shutdown()


def _bounds_around(lon: float, lat: float, metres: float = 300.0) -> Bounds:
    from japgo.geo.crs import from_wgs84

    x, y = from_wgs84(lon, lat, zone(8).crs)
    return Bounds(x - metres, y - metres, x + metres, y + metres)


def test_finds_the_mesh_covering_an_area(gate, index_server):
    index = MeshIndex(gate, template=index_server)
    meshes = index.meshes_for(_bounds_around(ATAMI_LON, ATAMI_LAT), zone(8).crs)
    assert "08NF5373" in {m.mesh_no for m in meshes}


def test_returns_the_download_url(gate, index_server):
    index = MeshIndex(gate, template=index_server)
    mesh = next(
        m for m in index.meshes_for(_bounds_around(ATAMI_LON, ATAMI_LAT), zone(8).crs)
        if m.mesh_no == "08NF5373"
    )
    assert mesh.url.endswith("08NF5373.zip")


def test_meshes_are_deduplicated_across_index_tiles(gate, index_server):
    """A large area spans several index tiles, and the same mesh can appear in more than one."""
    index = MeshIndex(gate, template=index_server)
    meshes = index.meshes_for(_bounds_around(ATAMI_LON, ATAMI_LAT, 3000), zone(8).crs)
    assert len({m.mesh_no for m in meshes}) == len(meshes)


def test_unavailable_index_tile_is_not_fatal(gate):
    """Off-coverage areas simply have no index tile; that is normal, not an error."""
    index = MeshIndex(gate, template="http://127.0.0.1:9/{z}/{x}/{y}.pbf")
    assert index.meshes_for(_bounds_around(ATAMI_LON, ATAMI_LAT), zone(8).crs) == []


def test_discovery_is_gated_by_provenance(gate, index_server):
    index = MeshIndex(gate, template=index_server)
    index.source_id = "gsi_dem"
    with pytest.raises(ProvenanceViolation, match="quarantined"):
        index.meshes_for(_bounds_around(ATAMI_LON, ATAMI_LAT), zone(8).crs)


# ---------------------------------------------------------------------------------------------
# Mesh geometry
# ---------------------------------------------------------------------------------------------


def test_intersection_test():
    mesh = Mesh("m", "u", 139.0, 35.0, 139.1, 35.1)
    assert mesh.intersects_lonlat(139.05, 35.05, 139.06, 35.06)
    assert not mesh.intersects_lonlat(140.0, 36.0, 140.1, 36.1)


def test_published_index_endpoint_is_the_grid_product():
    """The Grid product is the publisher-derived 0.5 m DEM, which the project prefers."""
    assert "LPGRD" in GRID_INDEX



# ---------------------------------------------------------------------------------------------
# TerrainFetcher — in-memory gridding and the raster cache
# ---------------------------------------------------------------------------------------------


def _grid_zip(x0: float, y0: float, size: int = 20, spacing: float = 0.5) -> bytes:
    """A Grid-product ZIP: one .txt of 'x y z' posts, as the prefecture publishes."""
    import io
    import zipfile as zf

    lines = []
    for i in range(size):
        for j in range(size):
            x = x0 + i * spacing
            y = y0 + j * spacing
            lines.append(f"{x:.3f} {y:.3f} {100.0 + i:.3f}")
    buffer = io.BytesIO()
    with zf.ZipFile(buffer, "w", zf.ZIP_DEFLATED) as archive:
        archive.writestr("MESH_DEM.txt", "\r\n".join(lines) + "\r\n")
    return buffer.getvalue()


@pytest.fixture
def mesh_server(tmp_path):
    from japgo.geo.crs import from_wgs84

    x0, y0 = from_wgs84(ATAMI_LON, ATAMI_LAT, zone(8).crs)
    payload = _grid_zip(x0, y0)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mesh.zip", (x0, y0)
    server.shutdown()


def test_mesh_is_parsed_without_touching_disk(gate, mesh_server, tmp_path):
    """The XYZ text is ~7x its zipped size; writing it out is the storage problem."""
    from japgo.sources.meshindex import Mesh, MeshIndex

    url, (x0, y0) = mesh_server
    index = MeshIndex(gate)
    x, y, z = index.read_mesh(Mesh("m", url, 0, 0, 0, 0))

    assert x.size == 400
    assert z.min() == pytest.approx(100.0)
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_terrain_fetcher_grids_into_a_raster(gate, mesh_server, monkeypatch, tmp_path):
    from japgo.geo.tiling import Bounds
    from japgo.sources.meshindex import Mesh, TerrainFetcher

    url, (x0, y0) = mesh_server
    fetcher = TerrainFetcher(gate, cache_dir=tmp_path)
    monkeypatch.setattr(
        fetcher.index, "meshes_for", lambda *a, **k: [Mesh("m", url, 0, 0, 0, 0)]
    )

    bounds = Bounds(x0, y0, x0 + 10, y0 + 10)
    dem, meshes = fetcher.dem_for(bounds, zone(8).crs, resolution=1.0, key="t")
    assert meshes == ["m"]
    assert dem.data.shape == (10, 10)
    assert dem.coverage == 1.0


def test_cache_hit_avoids_refetching(gate, mesh_server, monkeypatch, tmp_path):
    from japgo.geo.tiling import Bounds
    from japgo.sources.meshindex import Mesh, TerrainFetcher

    url, (x0, y0) = mesh_server
    fetcher = TerrainFetcher(gate, cache_dir=tmp_path)
    calls = []

    def _meshes(*a, **k):
        calls.append(1)
        return [Mesh("m", url, 0, 0, 0, 0)]

    monkeypatch.setattr(fetcher.index, "meshes_for", _meshes)
    bounds = Bounds(x0, y0, x0 + 10, y0 + 10)

    first, _ = fetcher.dem_for(bounds, zone(8).crs, resolution=1.0, key="t")
    second, _ = fetcher.dem_for(bounds, zone(8).crs, resolution=1.0, key="t")

    assert len(calls) == 1  # the second call never reached the index
    assert np.array_equal(np.nan_to_num(first.data), np.nan_to_num(second.data))


def test_cached_raster_is_far_smaller_than_the_text(gate, mesh_server, monkeypatch, tmp_path):
    from japgo.geo.tiling import Bounds
    from japgo.sources.meshindex import Mesh, TerrainFetcher

    url, (x0, y0) = mesh_server
    fetcher = TerrainFetcher(gate, cache_dir=tmp_path)
    monkeypatch.setattr(fetcher.index, "meshes_for", lambda *a, **k: [Mesh("m", url, 0, 0, 0, 0)])
    fetcher.dem_for(Bounds(x0, y0, x0 + 10, y0 + 10), zone(8).crs, resolution=1.0, key="t")

    cached = sum(p.stat().st_size for p in tmp_path.glob("*.npz"))
    text = 400 * len("53200.250 -107399.750 298.517\r\n")  # what the old route wrote
    assert cached < text


def test_no_cache_dir_still_works(gate, mesh_server, monkeypatch):
    from japgo.geo.tiling import Bounds
    from japgo.sources.meshindex import Mesh, TerrainFetcher

    url, (x0, y0) = mesh_server
    fetcher = TerrainFetcher(gate)
    monkeypatch.setattr(fetcher.index, "meshes_for", lambda *a, **k: [Mesh("m", url, 0, 0, 0, 0)])
    dem, _ = fetcher.dem_for(Bounds(x0, y0, x0 + 10, y0 + 10), zone(8).crs, resolution=1.0)
    assert dem.coverage > 0
