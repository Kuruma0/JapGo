"""Phase 9 — a reproducible demonstration of what each half of the system contributes.

Every metric so far collapses a road network to a number. The directive is right that the decisive
check is different: *look at it*. A network can score adequately and still read as a noisy
segmentation rather than as roads belonging to a place, and no APLS value will tell you which one
you have.

So a demo renders the whole transformation over one tile — terrain, the model's raw probability,
the graph that was extracted from it, the graph after repair, and the final geometry — onto a
single self-contained page, with the numbers for each stage beside the picture. Laid out that way
the question "what did the ML contribute and what did the procedural layer contribute" stops being
an argument and becomes something you can see.

Self-contained by the same rule as the Phase 2 viewer: hand-rolled PNG, inline data URIs, no
plotting dependency and no network fetch. A demo that needs a CDN is a demo that stops working.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..core.roads import RoadGraph
from ..geo.tiling import Bounds
from ..viz.image import colourise, png_bytes, shade
from .pipeline import GeneratedRoads


@dataclass
class DemoStage:
    """One panel: an image, a caption, and the numbers that go with it."""

    title: str
    png: bytes
    caption: str
    stats: str = ""


def _uri(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _decimate(rgba: np.ndarray, factor: int) -> np.ndarray:
    return rgba[::factor, ::factor] if factor > 1 else rgba


def _draw_graph(
    canvas: np.ndarray,
    graph: RoadGraph,
    bounds: Bounds,
    resolution: float,
    *,
    road: tuple[int, int, int] = (235, 235, 235),
    width: int = 1,
) -> np.ndarray:
    """Rasterise a graph's polylines onto an RGBA canvas — no plotting library, same as Phase 2."""
    rows, cols = canvas.shape[:2]

    def plot(x: float, y: float, rgb: tuple[int, int, int], size: int = 1) -> None:
        c = int((x - bounds.minx) / resolution)
        r = int((bounds.maxy - y) / resolution)
        r0, r1 = max(r - size, 0), min(r + size + 1, rows)
        c0, c1 = max(c - size, 0), min(c + size + 1, cols)
        if r1 > r0 and c1 > c0:
            canvas[r0:r1, c0:c1, :3] = rgb

    for edge in graph.edges.values():
        for a, b in zip(edge.geometry, edge.geometry[1:], strict=False):
            steps = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / resolution), 1)
            for i in range(steps + 1):
                f = i / steps
                plot(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), road, size=width)

    for nid, node in graph.nodes.items():
        degree = graph.degree(nid)
        if degree == 1:
            plot(node.x, node.y, (235, 90, 90), size=width + 1)      # dead end
        elif degree >= 3:
            plot(node.x, node.y, (90, 200, 235), size=width + 1)     # junction

    return canvas


def _graph_png(
    graph: RoadGraph, bounds: Bounds, shape: tuple[int, int], resolution: float, *, decimate: int
) -> bytes:
    """A graph on black."""
    canvas = np.zeros((*shape, 4), np.uint8)
    canvas[..., 3] = 255
    return png_bytes(_decimate(_draw_graph(canvas, graph, bounds, resolution), decimate))


def overlay_png(
    relief: np.ndarray,
    graph: RoadGraph,
    bounds: Bounds,
    resolution: float,
    *,
    decimate: int = 2,
) -> bytes:
    """The final network drawn over the terrain that produced it.

    The panel that answers the question the other five cannot: not "is this a network" but "does
    this network belong to *this* ground". A road following a valley and a road crossing it look
    identical side by side and unmistakable superimposed.
    """
    canvas = shade(relief).copy()
    canvas[..., :3] = (canvas[..., :3] * 0.62).astype(np.uint8)     # dim, so roads read on top
    drawn = _draw_graph(canvas, graph, bounds, resolution, road=(255, 196, 60), width=1)
    return png_bytes(_decimate(drawn, decimate))


def build_demo(
    roads: GeneratedRoads,
    *,
    elevation: np.ndarray,
    probability: np.ndarray,
    raw_graph: RoadGraph,
    repaired_graph: RoadGraph,
    real_graph: RoadGraph | None = None,
    resolution_m: float = 1.0,
    decimate: int = 2,
) -> list[DemoStage]:
    """Assemble the panels, in the order the transformation happens."""
    from ..geo.raster import Raster
    from ..geo.terrain import hillshade

    shape = elevation.shape
    # hillshade works on a Raster because it needs the pixel size to get gradients right.
    relief = hillshade(Raster(elevation.astype("float32"), roads.bounds, roads.crs)).data
    bounds = roads.bounds
    d = roads.diagnostics

    stages = [
        DemoStage(
            "1. terrain",
            png_bytes(_decimate(shade(relief), decimate)),
            "The input. Everything downstream is conditioned on this.",
            f"{elevation.max() - elevation.min():.0f} m of relief across the tile",
        ),
        DemoStage(
            "2. ML proposal",
            png_bytes(_decimate(colourise(probability, cmap="magma", vmin=0.0, vmax=1.0), decimate)),
            "Road probability from the frozen model. A proposal, not a network.",
            f"{len(roads.splines)} of these candidates survive to the final network",
        ),
        DemoStage(
            "3. extracted graph",
            _graph_png(raw_graph, bounds, shape, resolution_m, decimate=decimate),
            "Vectorised as-is. Red is a dead end, blue a junction — the red tells the story.",
            d.candidates.describe(),
        ),
        DemoStage(
            "4. after repair",
            _graph_png(repaired_graph, bounds, shape, resolution_m, decimate=decimate),
            "Fragments bridged, stubs pruned, noise dropped. Deterministic.",
            d.repair.describe(),
        ),
        DemoStage(
            "5. final roads",
            _graph_png(roads.graph, bounds, shape, resolution_m, decimate=decimate),
            "Grade-legal and smoothed. Steep alignments rerouted, not deleted.",
            d.terrain.describe(),
        ),
        DemoStage(
            "6. roads on terrain",
            overlay_png(relief, roads.graph, bounds, resolution_m, decimate=decimate),
            "The same network over the ground it was generated from.",
            "Does it follow the terrain, or merely sit on it?",
        ),
    ]
    if real_graph is not None:
        stages.append(DemoStage(
            "for comparison: reality",
            _graph_png(real_graph, bounds, shape, resolution_m, decimate=decimate),
            "The actual road network. Not a target the generator should match — a reference "
            "for whether the output has the right character.",
            f"{len(real_graph.edges)} edges, dead ends {real_graph.dead_end_ratio:.0%}",
        ))
    return stages


_CSS = """
body{background:#101216;color:#dfe3ea;font:14px/1.55 system-ui,sans-serif;margin:0;padding:28px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8b93a1;margin-bottom:24px}
.grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
figure{margin:0;background:#171a20;border:1px solid #232833;border-radius:8px;overflow:hidden}
img{width:100%;display:block;background:#000;image-rendering:pixelated}
figcaption{padding:10px 12px}
.t{font-weight:600;margin-bottom:3px} .c{color:#9aa3b2;font-size:13px}
.s{font-size:12px;font-family:ui-monospace,monospace;margin-top:7px;color:#7f8899;
   word-break:break-word}
.prov{border-left:3px solid;padding:8px 12px;margin:0 0 22px;border-radius:0 5px 5px 0;
      font-size:13px}
.prov.ok{border-color:#4f9d69;background:#16211a;color:#a8cdb5}
.prov.warn{border-color:#c08a3e;background:#231d13;color:#dcbd8a}
.key{margin-top:22px;color:#8b93a1;font-size:13px}
.key b{color:#eb5a5a}.key i{color:#5ac8eb;font-style:normal}
"""


#: What to say about a tile the checkpoint trained on. Shown in place of nothing, because the
#: resemblance in the comparison panel is the thing a reader will trust first and it is partly
#: recall on a training site.
_SEEN = ("this site was in the checkpoint's training set — the resemblance to reality below is "
         "partly recall, not generalisation")
_UNSEEN = "this site was held out of training — the model had never seen it"
_UNKNOWN = ("the model card does not name its held-out sites, so whether this site was seen in "
            "training is unknown")


def provenance_note(unseen: bool | None) -> str:
    """One sentence on how much the comparison panel is worth."""
    return _UNSEEN if unseen else (_UNKNOWN if unseen is None else _SEEN)


def write_demo(
    stages: list[DemoStage], roads: GeneratedRoads, path: Path, *, title: str,
    unseen: bool | None = None,
) -> Path:
    """Write the panels to one self-contained HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = "\n".join(
        f'<figure><img alt="{s.title}" src="{_uri(s.png)}">'
        f'<figcaption><div class="t">{s.title}</div>'
        f'<div class="c">{s.caption}</div>'
        f'<div class="s">{s.stats}</div></figcaption></figure>'
        for s in stages
    )
    summary = roads.summary()
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style>"
        f"<h1>{title}</h1>"
        f"<div class='sub'>seed {summary['seed']} &middot; {summary['roads']} roads &middot; "
        f"{summary['junctions']} junctions &middot; {summary['total_length_m'] / 1000:.2f} km "
        f"&middot; {summary['components']} components &middot; "
        f"dead ends {summary['dead_end_ratio']:.0%} &middot; "
        f"elevations {summary['elevation_reference']}</div>"
        f"<div class='prov {'ok' if unseen else 'warn'}'>{provenance_note(unseen)}</div>"
        f"<div class='grid'>{panels}</div>"
        "<div class='key'><b>red</b> = dead end &nbsp; <i>blue</i> = junction. "
        "Panels 3 to 5 are the same tile at the same scale, so the change between them is "
        "entirely the procedural layer's doing.</div>",
        encoding="utf-8",
    )
    (path.parent / f"{path.stem}.json").write_text(
        json.dumps({**summary, "site_held_out_of_training": unseen,
                    "stages": [s.title for s in stages]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
