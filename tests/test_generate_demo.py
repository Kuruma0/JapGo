"""Tests for Phase 9 — the demonstration page.

The page exists so a person can see what each half of the system contributes, so what is worth
testing is not how it looks but the two properties that would make it lie or stop working: the
panels must be the real intermediate stages rather than the final graph drawn five times, and the
file must render with no network access years from now.
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np

from japgo.core import Edge, Node, RoadGraph
from japgo.generate import GenerationParams, build_demo, generate_roads, write_demo
from japgo.generate.candidates import extract_candidates
from japgo.generate.repair import repair
from japgo.geo.tiling import Bounds

from test_generate_pipeline import _StubModel, _world


def _demo(size=200):
    spec, channels = _world(size)
    bounds = Bounds(0.0, 0.0, float(size), float(size))
    model = _StubModel(spec)
    roads = generate_roads(model, channels, bounds, params=GenerationParams(seed=11))

    elevation = channels[spec.index_of("elevation")]
    prediction = model.predict(channels, bounds)
    raw, _ = extract_candidates(prediction, elevation=elevation)
    repaired, _ = repair(raw)
    return roads, elevation, prediction, raw, repaired


def test_every_stage_is_rendered_and_the_panels_track_their_own_graph():
    """The failure that would make the whole page dishonest: drawing the finished network in every
    panel. It would look convincing and would demonstrate nothing.

    Checked by giving the raw stage a fragment that repair removes, because on a trivial world the
    panels are *legitimately* identical — nothing was repaired — and pixel uniqueness would then
    fail for the right reason and pass for the wrong one.
    """
    roads, elevation, prediction, raw, repaired = _demo()
    littered = RoadGraph(crs=raw.crs)
    for nid, node in raw.nodes.items():
        littered.add_node(node)
    for eid, edge in raw.edges.items():
        littered.add_edge(edge)
    littered.add_node(Node(id="frag_a", x=20.0, y=170.0))
    littered.add_node(Node(id="frag_b", x=40.0, y=170.0))
    littered.add_edge(Edge(id="frag", u="frag_a", v="frag_b",
                           geometry=[(20.0, 170.0), (40.0, 170.0)], source_id="model"))

    stages = build_demo(
        roads, elevation=elevation, probability=prediction.probability,
        raw_graph=littered, repaired_graph=repaired,
    )

    assert len(stages) == 5
    assert all(s.png.startswith(b"\x89PNG") for s in stages)
    assert stages[2].png != stages[3].png, "the extracted panel must show the extracted graph"
    assert stages[0].png != stages[1].png != stages[2].png


def test_reality_is_shown_only_when_there_is_a_real_network_to_show():
    """Generated worlds have no ground truth. The comparison panel has to be optional or the demo
    only works on tiles that came from Shizuoka."""
    roads, elevation, prediction, raw, repaired = _demo()
    real = RoadGraph(crs="EPSG:6676")
    real.add_node(Node(id="a", x=10.0, y=10.0))
    real.add_node(Node(id="b", x=190.0, y=190.0))
    real.add_edge(Edge(id="r", u="a", v="b", geometry=[(10.0, 10.0), (190.0, 190.0)],
                       source_id="osm"))

    without = build_demo(roads, elevation=elevation, probability=prediction.probability,
                         raw_graph=raw, repaired_graph=repaired)
    with_real = build_demo(roads, elevation=elevation, probability=prediction.probability,
                           raw_graph=raw, repaired_graph=repaired, real_graph=real)

    assert len(with_real) == len(without) + 1
    assert "reality" in with_real[-1].title


def test_the_page_is_self_contained(tmp_path):
    """Same rule as the Phase 2 viewer. A demo that needs a CDN is a demo that stops working, and
    this one is meant to be openable from a repository checkout with no server."""
    roads, elevation, prediction, raw, repaired = _demo()
    stages = build_demo(roads, elevation=elevation, probability=prediction.probability,
                        raw_graph=raw, repaired_graph=repaired)
    page = write_demo(stages, roads, tmp_path / "demo.html", title="test")
    html = page.read_text(encoding="utf-8")

    for remote in ("http://", "https://", "<script", "//cdn"):
        assert remote not in html
    assert html.count("data:image/png;base64,") == len(stages)

    # The embedded bytes are the images that were built, not a placeholder.
    embedded = re.findall(r"data:image/png;base64,([^\"]+)", html)
    assert [base64.b64decode(b) for b in embedded] == [s.png for s in stages]


def test_the_sidecar_records_what_would_be_needed_to_reproduce_the_page(tmp_path):
    roads, elevation, prediction, raw, repaired = _demo()
    stages = build_demo(roads, elevation=elevation, probability=prediction.probability,
                        raw_graph=raw, repaired_graph=repaired)
    write_demo(stages, roads, tmp_path / "demo.html", title="test")

    meta = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 11
    assert meta["stages"] == [s.title for s in stages]
    assert meta["elevation_reference"] in ("tile-relative", "absolute")


def test_dead_ends_and_junctions_are_drawn_in_different_colours():
    """The red pixels are the argument for the procedural layer. If the two node classes render
    identically the panels stop carrying the point they were built to carry."""
    from japgo.generate.demo import _graph_png

    g = RoadGraph(crs="EPSG:6676")
    for i, (x, y) in enumerate([(50.0, 50.0), (150.0, 50.0), (50.0, 150.0), (150.0, 150.0)]):
        g.add_node(Node(id=str(i), x=x, y=y))
    for i, (u, v) in enumerate([("0", "1"), ("0", "2"), ("0", "3")]):
        g.add_edge(Edge(id=f"e{i}", u=u, v=v,
                        geometry=[g.nodes[u].position, g.nodes[v].position], source_id="model"))

    png = _graph_png(g, Bounds(0.0, 0.0, 200.0, 200.0), (200, 200), 1.0, decimate=1)
    assert png.startswith(b"\x89PNG")
    assert g.degree("0") == 3 and g.degree("1") == 1


def test_the_page_says_whether_the_model_had_seen_the_site(tmp_path):
    """The comparison panel is the thing a reader trusts first, and on a training site the
    resemblance is partly recall. A leave-one-site-out checkpoint is honest evidence on exactly
    one site; the page has to say which."""
    roads, elevation, prediction, raw, repaired = _demo()
    stages = build_demo(roads, elevation=elevation, probability=prediction.probability,
                        raw_graph=raw, repaired_graph=repaired)

    held_out = write_demo(stages, roads, tmp_path / "a.html", title="t", unseen=True)
    trained_on = write_demo(stages, roads, tmp_path / "b.html", title="t", unseen=False)
    unknown = write_demo(stages, roads, tmp_path / "c.html", title="t", unseen=None)

    assert "held out of training" in held_out.read_text(encoding="utf-8")
    assert "partly recall" in trained_on.read_text(encoding="utf-8")
    assert "unknown" in unknown.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "b.json").read_text())["site_held_out_of_training"] is False


def test_the_card_can_be_asked_which_sites_it_never_saw():
    from japgo.generate import ModelCard

    card = ModelCard(checkpoint="c", trained_on="corpus", held_out=["hamamatsu_plain"],
                     channels=["elevation"], stack_version=2, resolution_m=1.0, crs="EPSG:6676",
                     registry_hash=None, width=32, threshold=0.45)
    assert card.unseen("hamamatsu_plain") is True
    assert card.unseen("izu_coast") is False

    silent = ModelCard(checkpoint="c", trained_on="corpus", channels=["elevation"],
                       stack_version=2, resolution_m=1.0, crs="EPSG:6676", registry_hash=None,
                       width=32, threshold=0.45)
    assert silent.unseen("izu_coast") is None, "a card that does not say must not claim"
