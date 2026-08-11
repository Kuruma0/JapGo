"""Tests for the Phase 2 visualiser, and for the alignment property it exists to reveal.

The alignment assertions here matter more than the rendering ones. Risk R8 is that vector/raster
misalignment corrupts training without producing an error — the visualiser lets a human catch it,
but a test catches it in CI, every run, without anyone having to look.
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np
import pytest

from japgo.core import Building, Edge, Node, RoadGraph
from japgo.core.manifest import SourceRecord, SourceRole
from japgo.geo import SHIZUOKA, Bounds, Raster, Tile
from japgo.pipeline import TileAssembler, TileInputs
from japgo.pipeline.rasterize import building_mask
from japgo.viz import colourise, png_bytes, render_tile, shade, summarise
from japgo.viz.report import CHANNEL_STYLE

TILE = Tile(zone=8, ix=10, iy=-110)
RES = 1.0


# ---------------------------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------------------------


def test_png_has_a_valid_signature_and_chunks():
    data = png_bytes(np.zeros((4, 6, 3), np.uint8))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in data[:20]
    # The trailing chunk is [length:4][IEND:4][crc:4], so the tag sits at -8:-4.
    assert data[-8:-4] == b"IEND"
    assert data[-12:-8] == b"\x00\x00\x00\x00"  # IEND carries no payload


def test_png_header_records_the_right_dimensions():
    data = png_bytes(np.zeros((7, 11, 4), np.uint8))
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (11, 7)


def test_png_pixels_survive_a_roundtrip():
    """Decode the IDAT back out and compare, rather than trusting the encoder by inspection."""
    rgb = np.zeros((3, 3, 3), np.uint8)
    rgb[1, 1] = (200, 100, 50)
    data = png_bytes(rgb)

    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8 : start - 4])[0]
    raw = zlib.decompress(data[start : start + length])

    # Each row is a filter byte followed by width*3 samples.
    decoded = np.frombuffer(raw, np.uint8).reshape(3, 1 + 3 * 3)[:, 1:].reshape(3, 3, 3)
    assert np.array_equal(decoded, rgb)


def test_png_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="uint8"):
        png_bytes(np.zeros((2, 2, 3), np.float32))


# ---------------------------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------------------------


def test_nodata_is_transparent_not_coloured():
    """Painting nodata as a colour is how a void starts looking like a measurement."""
    data = np.array([[1.0, np.nan], [3.0, 4.0]])
    rgba = colourise(data)
    assert rgba[0, 1, 3] == 0
    assert rgba[0, 0, 3] == 255


def test_mask_excludes_cells():
    data = np.ones((2, 2))
    mask = np.array([[True, False], [True, True]])
    assert colourise(data, mask=mask)[0, 1, 3] == 0


def test_colour_varies_across_the_range():
    rgba = colourise(np.array([[0.0, 0.5, 1.0]]))
    assert not np.array_equal(rgba[0, 0, :3], rgba[0, 2, :3])


def test_constant_field_does_not_divide_by_zero():
    rgba = colourise(np.full((3, 3), 5.0))
    assert np.isfinite(rgba).all()
    assert (rgba[..., 3] == 255).all()


def test_shade_produces_opaque_grey():
    rgba = shade(np.full((2, 2), 128.0))
    assert (rgba[..., 3] == 255).all()
    assert rgba[0, 0, 0] == rgba[0, 0, 1] == rgba[0, 0, 2]


def test_every_stack_channel_has_a_display_style():
    """An unstyled channel still renders, but silently loses its label and colour intent."""
    from japgo.pipeline import load_stack_spec

    spec = load_stack_spec()
    for name in spec.names + spec.target_names:
        assert name in CHANNEL_STYLE, f"{name} has no entry in CHANNEL_STYLE"


# ---------------------------------------------------------------------------------------------
# Fixtures for alignment and rendering
# ---------------------------------------------------------------------------------------------


def _house(x: float, y: float, size: float = 10.0) -> Building:
    return Building(
        id=f"b{x:.0f}-{y:.0f}",
        source_id="plateau",
        footprint=[(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)],
        height_m=8.0,
        coarse_type="residential",
        fine_type="detached_house",
    )


@pytest.fixture
def bundle(gate):
    bounds = TILE.read
    rows = int(bounds.height / RES)
    cols = int(bounds.width / RES)
    col = np.arange(cols, dtype=np.float32)[None, :]
    dem = Raster(np.tile(col * 0.05, (rows, 1)).astype(np.float32), bounds, SHIZUOKA.crs)

    cx, cy = TILE.core.centre
    buildings = [_house(cx + i * 60, cy + j * 60) for i in range(3) for j in range(3)]

    graph = RoadGraph(crs=SHIZUOKA.crs.to_string())
    graph.add_node(Node(id="a", x=cx - 400, y=cy))
    graph.add_node(Node(id="b", x=cx + 400, y=cy))
    graph.add_edge(
        Edge(id="e", u="a", v="b", geometry=[(cx - 400, cy), (cx + 400, cy)],
             road_class="secondary", source_id="osm")
    )

    inputs = TileInputs(
        elevation=dem,
        buildings=buildings,
        roads=graph,
        records=[
            SourceRecord(source_id="virtual_shizuoka", layers=["elevation"]),
            SourceRecord(source_id="plateau", layers=["buildings"]),
            SourceRecord(source_id="osm", role=SourceRole.TARGET, layers=["roads"]),
        ],
    )
    return TileAssembler(gate, resolution=RES).assemble(TILE, inputs)


# ---------------------------------------------------------------------------------------------
# Alignment — risk R8, made checkable
# ---------------------------------------------------------------------------------------------


def test_building_raster_and_vector_centroids_agree(bundle):
    """A systematic offset between a footprint and its mask is exactly the silent corruption
    Phase 2 exists to catch. Sub-pixel disagreement is discretisation; more is a bug."""
    bounds = bundle.tile.read
    mask = bundle.channel("building_mask")
    rows, cols = np.nonzero(mask > 0)
    assert rows.size > 0

    raster_x = bounds.minx + (cols.mean() + 0.5) * RES
    raster_y = bounds.maxy - (rows.mean() + 0.5) * RES

    total = sum(b.area_m2 for b in bundle.buildings)
    vector_x = sum(
        b.area_m2 * sum(p[0] for p in b.footprint[:-1]) / (len(b.footprint) - 1)
        for b in bundle.buildings
    ) / total
    vector_y = sum(
        b.area_m2 * sum(p[1] for p in b.footprint[:-1]) / (len(b.footprint) - 1)
        for b in bundle.buildings
    ) / total

    assert abs(raster_x - vector_x) < RES
    assert abs(raster_y - vector_y) < RES


def test_burned_building_area_matches_vector_area(bundle):
    """Area conservation catches a scale or resolution error that a centroid check would miss."""
    burned = float((bundle.channel("building_mask") > 0).sum()) * RES * RES
    vector = sum(b.area_m2 for b in bundle.buildings)
    assert burned == pytest.approx(vector, rel=0.05)


def test_road_target_lies_under_the_road_vector(bundle):
    """The target raster must sit on the graph it was burned from, not beside it."""
    bounds = bundle.tile.read
    target = bundle.target("road_mask")
    rows, cols = np.nonzero(target > 0)
    assert rows.size > 0

    ys = bounds.maxy - (rows + 0.5) * RES
    _, road_y = bundle.tile.core.centre
    assert abs(ys.mean() - road_y) < 2 * RES


def test_road_target_stays_within_the_carriageway_plus_one_cell(bundle):
    """Burned road cells must lie within half a carriageway of the centreline, plus one
    half-cell-diagonal.

    The tolerance is not slack, it is arithmetic: ``all_touched=True`` burns any cell the buffered
    polygon clips at all, so a cell centre can sit up to half a cell diagonal outside the
    carriageway edge. Measured on the real Atami tile the worst case was 3.44 m against a 2.75 m
    half-width — i.e. 2.75 + 0.69, matching the bound exactly. Anything beyond it is a genuine
    misalignment rather than discretisation.
    """
    from shapely.geometry import LineString, Point

    bounds = bundle.tile.read
    target = bundle.target("road_mask") > 0
    assert target.any()

    from japgo.core import load_hierarchy

    hierarchy = load_hierarchy()
    lines = [LineString(e.geometry) for e in bundle.roads.edges.values() if len(e.geometry) > 1]

    # Must mirror the rasteriser's own rule, or the test is checking a different road.
    half_width = max(
        (e.width_m or hierarchy.spec(e.road_class).typical_width_m)
        for e in bundle.roads.edges.values()
    ) / 2.0
    tolerance = half_width + RES * math.sqrt(2) / 2

    rows, cols = np.nonzero(target)
    step = max(1, len(rows) // 300)  # sample; the full mask is tens of thousands of cells
    for row, col in zip(rows[::step], cols[::step], strict=False):
        x = bounds.minx + (col + 0.5) * RES
        y = bounds.maxy - (row + 0.5) * RES
        assert min(line.distance(Point(x, y)) for line in lines) <= tolerance


def test_misalignment_would_be_detected(gate, bundle):
    """A guard on the guard: shift the vectors and confirm the check fails.

    A test that only ever passes proves nothing about its own sensitivity.
    """
    bounds = bundle.tile.read
    shifted = [
        Building(
            id=b.id,
            source_id=b.source_id,
            footprint=[(x + 50.0, y) for x, y in b.footprint],
            height_m=b.height_m,
        )
        for b in bundle.buildings
    ]
    mask = building_mask(shifted, bounds, RES, SHIZUOKA.crs)
    rows, cols = np.nonzero(mask.data > 0)
    shifted_x = bounds.minx + (cols.mean() + 0.5) * RES

    original = bundle.channel("building_mask")
    orows, ocols = np.nonzero(original > 0)
    original_x = bounds.minx + (ocols.mean() + 0.5) * RES

    assert abs(shifted_x - original_x) == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------------------------


def test_report_is_self_contained(bundle, gate):
    """No server, no CDN. It must open from the filesystem on a headless training box."""
    html = render_tile(bundle, gate, decimate=8)
    assert "http://" not in html
    assert "https://" not in html
    assert "data:image/png;base64," in html


def test_report_includes_every_channel_and_target(bundle, gate):
    html = render_tile(bundle, gate, decimate=8)
    for name in bundle.spec.names:
        assert f'data-layer="{name}"' in html
    for name in bundle.spec.target_names:
        assert f'data-layer="target_{name}"' in html


def test_report_draws_the_core_halo_boundary(bundle, gate):
    """The halo is invisible in the data; a human needs to see features crossing it."""
    html = render_tile(bundle, gate, decimate=8)
    assert 'id="v-core"' in html
    assert "halo" in html


def test_report_draws_vectors_over_the_raster(bundle, gate):
    html = render_tile(bundle, gate, decimate=8)
    assert 'id="v-buildings"' in html
    assert 'id="v-roads"' in html
    assert "<svg" in html


def test_report_states_both_redistribution_classes(bundle, gate):
    """Training on OSM is fine; shipping its geometry is not. The page must say both."""
    html = render_tile(bundle, gate, decimate=8)
    assert "inputs: attribution-only" in html
    assert "bundle: share-alike" in html


def test_report_carries_the_attribution_block(bundle, gate):
    html = render_tile(bundle, gate, decimate=8)
    assert "PLATEAU" in html
    assert "VIRTUAL SHIZUOKA" in html


def test_report_escapes_untrusted_source_text(gate, bundle):
    """Source ids and layer names come from external files and reach the page unmediated.

    A PLATEAU package or an OSM extract is not hostile, but it is not ours either, and a report
    that mangles on an unexpected ampersand is a report nobody trusts.
    """
    bundle.manifest.sources.append(
        SourceRecord(source_id="nlni_landuse", layers=['<img src=x onerror="boom()">'])
    )
    html = render_tile(bundle, gate, decimate=8)
    assert 'onerror="boom()"' not in html
    assert "&lt;img" in html


def test_summarise_reports_the_gate_relevant_facts(bundle, gate):
    summary = summarise(bundle, gate)
    assert summary["tile_id"] == TILE.id
    assert summary["buildings"] == 9
    assert summary["edges"] == 1
    assert summary["inputs"] == "attribution-only"


def test_decimation_shrinks_the_page(bundle, gate):
    assert len(render_tile(bundle, gate, decimate=8)) < len(
        render_tile(bundle, gate, decimate=4)
    )
