"""Tile inspection report — the Phase 2 gate.

Phase 2 exists to answer one question before any modelling begins: **is every layer spatially
aligned?** Vector/raster misalignment corrupts training without producing a single error, and no
amount of unit testing catches it, because each layer is individually correct.

So this renders one self-contained HTML page per tile:

* raster channels as toggleable, adjustable-opacity layers over a hillshade base;
* buildings and the road graph as **SVG on top**, not burned into the raster — vectors stay crisp
  at any zoom, and a half-pixel offset between a footprint and its raster mask is visible rather
  than smeared away by resampling;
* the **core/halo boundary drawn explicitly**, because the halo is invisible in the data and a
  human needs to see that features continue across it;
* the manifest, so provenance is on the same page as the pixels it describes.

No server, no plotting library, no external requests. It opens from the filesystem on any machine
that can run the pipeline.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geo import terrain as terrain_ops
from ..geo.raster import Raster
from ..geo.tiling import Bounds
from ..pipeline.assemble import TileBundle
from ..provenance import SourceGate
from .image import colourise, downsample_rgba, png_bytes, shade

#: Per-channel display hints. Anything unlisted falls back to viridis over its own range.
CHANNEL_STYLE: dict[str, dict] = {
    "elevation": {"cmap": "diverging", "label": "Elevation (tile-relative)"},
    "slope": {"cmap": "magma", "label": "Slope"},
    "aspect_sin": {"cmap": "diverging", "label": "Aspect (sin)"},
    "aspect_cos": {"cmap": "diverging", "label": "Aspect (cos)"},
    "roughness": {"cmap": "magma", "label": "Roughness"},
    "building_mask": {"cmap": "grey", "label": "Building mask"},
    "building_height": {"cmap": "viridis", "label": "Building height"},
    "building_coarse_residential": {"cmap": "grey", "label": "Buildings: residential"},
    "building_coarse_commercial": {"cmap": "grey", "label": "Buildings: commercial"},
    "building_coarse_industrial": {"cmap": "grey", "label": "Buildings: industrial"},
    "landuse_built": {"cmap": "grey", "label": "Land use: built"},
    "landuse_agricultural": {"cmap": "grey", "label": "Land use: agricultural"},
    "landuse_forest": {"cmap": "grey", "label": "Land use: forest"},
    "landuse_water": {"cmap": "grey", "label": "Land use: water"},
    "valid": {"cmap": "grey", "label": "Valid mask"},
    "road_mask": {"cmap": "magma", "label": "TARGET road mask (width)"},
    "road_centreline": {"cmap": "magma", "label": "TARGET road centreline"},
    "road_class": {"cmap": "magma", "label": "TARGET road class"},
    "road_orientation_sin": {"cmap": "diverging", "label": "TARGET orientation (sin)"},
    "road_orientation_cos": {"cmap": "diverging", "label": "TARGET orientation (cos)"},
}


@dataclass
class Layer:
    name: str
    label: str
    data_uri: str
    default_on: bool = False
    group: str = "input"


def _data_uri(rgba: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgba)).decode("ascii")


def _world_to_pixel(bounds: Bounds, resolution: float):
    def convert(x: float, y: float) -> tuple[float, float]:
        return ((x - bounds.minx) / resolution, (bounds.maxy - y) / resolution)

    return convert


def _svg_overlay(
    bundle: TileBundle,
    bounds: Bounds,
    resolution: float,
    width: int,
    height: int,
) -> str:
    """Buildings, roads, and the core/halo boundary as SVG."""
    convert = _world_to_pixel(bounds, resolution)
    parts: list[str] = []

    # --- buildings ---------------------------------------------------------------------------
    building_paths = []
    for building in bundle.buildings:
        points = " ".join(f"{convert(x, y)[0]:.1f},{convert(x, y)[1]:.1f}" for x, y in building.footprint)
        if points:
            building_paths.append(f'<polygon points="{points}"/>')
    if building_paths:
        parts.append(
            '<g id="v-buildings" class="vec" fill="rgba(255,120,0,0.35)" '
            'stroke="#ff7800" stroke-width="1.2">' + "".join(building_paths) + "</g>"
        )

    # --- roads -------------------------------------------------------------------------------
    if bundle.roads is not None:
        road_paths = []
        for edge in bundle.roads.edges.values():
            points = " ".join(
                f"{convert(x, y)[0]:.1f},{convert(x, y)[1]:.1f}" for x, y in edge.geometry
            )
            if points:
                road_paths.append(f'<polyline points="{points}"/>')
        if road_paths:
            parts.append(
                '<g id="v-roads" class="vec" fill="none" stroke="#00e5ff" stroke-width="2" '
                'stroke-linecap="round">' + "".join(road_paths) + "</g>"
            )

        node_dots = []
        for node in bundle.roads.nodes.values():
            px, py = convert(node.x, node.y)
            degree = bundle.roads.degree(node.id)
            colour = "#ff2d6f" if degree == 1 else "#ffffff"
            node_dots.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{colour}"/>')
        if node_dots:
            parts.append('<g id="v-nodes" class="vec">' + "".join(node_dots) + "</g>")

    # --- core / halo boundary ------------------------------------------------------------------
    core = bundle.tile.core
    x0, y0 = convert(core.minx, core.maxy)
    x1, y1 = convert(core.maxx, core.miny)
    parts.append(
        f'<g id="v-core" class="vec">'
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
        f'fill="none" stroke="#ffe600" stroke-width="2" stroke-dasharray="8 6"/>'
        f'<text x="{x0 + 8:.1f}" y="{y0 + 22:.1f}" fill="#ffe600" font-size="16" '
        f'font-family="monospace">core 1 km — outside is halo, read but never predicted</text>'
        f"</g>"
    )

    return (
        f'<svg id="overlay" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">{"".join(parts)}</svg>'
    )


def render_tile(
    bundle: TileBundle,
    gate: SourceGate,
    *,
    decimate: int = 2,
) -> str:
    """Render one tile bundle to a self-contained HTML document."""
    bounds = bundle.tile.read
    rows, cols = bundle.stack.shape[1], bundle.stack.shape[2]
    resolution = bounds.width / cols

    valid = bundle.channel("valid") > 0 if "valid" in bundle.spec.names else None

    # --- base: hillshade from the elevation channel -------------------------------------------
    elevation = bundle.channel("elevation")
    dem = Raster(elevation.astype(np.float32), bounds, _crs_of(bundle))
    base = shade(terrain_ops.hillshade(dem).data)
    base_uri = _data_uri(downsample_rgba(base, decimate))

    # --- raster layers -------------------------------------------------------------------------
    layers: list[Layer] = []
    for name in bundle.spec.names:
        style = CHANNEL_STYLE.get(name, {})
        rgba = colourise(
            bundle.channel(name),
            cmap=style.get("cmap", "viridis"),
            mask=valid if name != "valid" else None,
        )
        layers.append(
            Layer(
                name=name,
                label=style.get("label", name),
                data_uri=_data_uri(downsample_rgba(rgba, decimate)),
                default_on=(name == "slope"),
                group="input",
            )
        )

    if bundle.has_targets:
        for name in bundle.spec.target_names:
            style = CHANNEL_STYLE.get(name, {})
            rgba = colourise(bundle.target(name), cmap=style.get("cmap", "magma"))
            layers.append(
                Layer(
                    name=f"target_{name}",
                    label=style.get("label", name),
                    data_uri=_data_uri(downsample_rgba(rgba, decimate)),
                    default_on=(name == "road_centreline"),
                    group="target",
                )
            )

    overlay = _svg_overlay(bundle, bounds, resolution, cols, rows)
    return _document(bundle, gate, base_uri, layers, overlay, cols, rows)


def _crs_of(bundle: TileBundle):
    from pyproj import CRS

    return CRS.from_user_input(bundle.manifest.crs)


def _document(
    bundle: TileBundle,
    gate: SourceGate,
    base_uri: str,
    layers: list[Layer],
    overlay: str,
    width: int,
    height: int,
) -> str:
    klass = bundle.redistribution_class(gate)
    input_klass = bundle.input_redistribution_class(gate)

    def rows(group: str) -> str:
        out = []
        for layer in layers:
            if layer.group != group:
                continue
            checked = " checked" if layer.default_on else ""
            out.append(
                f'<label class="layer"><input type="checkbox" data-layer="{layer.name}"{checked}>'
                f'<span>{html.escape(layer.label)}</span>'
                f'<input type="range" min="0" max="100" value="75" data-opacity="{layer.name}">'
                f"</label>"
            )
        return "".join(out)

    images = "".join(
        f'<img class="layer-img" id="img-{layer.name}" src="{layer.data_uri}" '
        f'style="opacity:{0.75 if layer.default_on else 0};'
        f'{"" if layer.default_on else "display:none;"}">'
        for layer in layers
    )

    sources = "".join(
        f"<tr><td>{html.escape(record.source_id)}</td>"
        f"<td>{record.role.value}</td>"
        f"<td>{html.escape(', '.join(record.layers))}</td></tr>"
        for record in bundle.manifest.sources
    )

    stats = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{lo:.3f}</td><td>{mean:.3f}</td><td>{hi:.3f}</td></tr>"
        for name, lo, mean, hi in _summary(bundle)
    )

    attribution = "<br>".join(html.escape(line) for line in bundle.attribution(gate))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(bundle.tile.id)} — JapGo tile inspection</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#c9d1d9;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:12px 20px; border-bottom:1px solid #30363d; display:flex;
            gap:20px; align-items:baseline; flex-wrap:wrap; }}
  h1 {{ font-size:16px; margin:0; font-family:ui-monospace,monospace; }}
  .pill {{ padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }}
  .ok {{ background:#1a7f37; color:#fff; }}
  .warn {{ background:#9e6a03; color:#fff; }}
  main {{ display:flex; gap:20px; padding:20px; align-items:flex-start; flex-wrap:wrap; }}
  #stage {{ position:relative; flex:1 1 620px; min-width:340px; max-width:900px;
            aspect-ratio:1; background:#010409; border:1px solid #30363d; overflow:hidden; }}
  #stage img, #stage svg {{ position:absolute; inset:0; width:100%; height:100%; }}
  #stage img {{ image-rendering:pixelated; }}
  aside {{ flex:0 0 320px; }}
  .panel {{ border:1px solid #30363d; border-radius:8px; margin-bottom:16px; }}
  .panel h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em;
               margin:0; padding:10px 14px; border-bottom:1px solid #30363d; color:#8b949e; }}
  .panel .body {{ padding:10px 14px; }}
  label.layer {{ display:grid; grid-template-columns:auto 1fr 90px; gap:8px;
                 align-items:center; padding:3px 0; font-size:13px; }}
  label.layer input[type=range] {{ width:90px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px;
           font-family:ui-monospace,monospace; }}
  td, th {{ padding:3px 6px; border-bottom:1px solid #21262d; text-align:left; }}
  .note {{ font-size:12px; color:#8b949e; padding:10px 14px; }}
  .vec {{ pointer-events:none; }}
</style></head>
<body>
<header>
  <h1>{html.escape(bundle.tile.id)}</h1>
  <span>{width}×{height} px @ {bundle.tile.core_size_m:.0f} m core + {bundle.tile.halo_m:.0f} m halo</span>
  <span>coverage {bundle.coverage:.1%}</span>
  <span class="pill {"ok" if input_klass == "attribution-only" else "warn"}">
    inputs: {input_klass}</span>
  <span class="pill {"ok" if klass == "attribution-only" else "warn"}">bundle: {klass}</span>
</header>
<main>
  <div id="stage">
    <img src="{base_uri}" style="opacity:1">
    {images}
    {overlay}
  </div>
  <aside>
    <div class="panel"><h2>Alignment overlay</h2><div class="body">
      <label class="layer"><input type="checkbox" data-vec="v-buildings" checked>
        <span style="color:#ff7800">Building footprints</span><span></span></label>
      <label class="layer"><input type="checkbox" data-vec="v-roads" checked>
        <span style="color:#00e5ff">Road graph</span><span></span></label>
      <label class="layer"><input type="checkbox" data-vec="v-nodes" checked>
        <span>Nodes (pink = dead end)</span><span></span></label>
      <label class="layer"><input type="checkbox" data-vec="v-core" checked>
        <span style="color:#ffe600">Core / halo boundary</span><span></span></label>
    </div>
    <p class="note">Vectors are drawn over the rasters, not burned into them. If a footprint
    outline does not sit on its raster mask, the layers are misaligned — that is what this page
    is for.</p></div>

    <div class="panel"><h2>Input channels</h2><div class="body">{rows("input")}</div></div>
    {'<div class="panel"><h2>Targets</h2><div class="body">' + rows("target") + "</div></div>"
     if bundle.has_targets else ""}

    <div class="panel"><h2>Provenance</h2><div class="body">
      <table><tr><th>source</th><th>role</th><th>layers</th></tr>{sources}</table>
      <p class="note" style="padding:8px 0 0">registry {html.escape(bundle.manifest.registry_hash or '-')}
      · preproc v{html.escape(bundle.manifest.preprocessing_version or '-')}</p>
      <p class="note" style="padding:8px 0 0">{attribution}</p>
    </div></div>

    <div class="panel"><h2>Channel statistics</h2><div class="body">
      <table><tr><th>channel</th><th>min</th><th>mean</th><th>max</th></tr>{stats}</table>
    </div></div>
  </aside>
</main>
<script>
document.querySelectorAll('input[data-layer]').forEach(box => {{
  box.addEventListener('change', () => {{
    const img = document.getElementById('img-' + box.dataset.layer);
    img.style.display = box.checked ? 'block' : 'none';
  }});
}});
document.querySelectorAll('input[data-opacity]').forEach(slider => {{
  slider.addEventListener('input', () => {{
    document.getElementById('img-' + slider.dataset.opacity).style.opacity = slider.value / 100;
  }});
}});
document.querySelectorAll('input[data-vec]').forEach(box => {{
  box.addEventListener('change', () => {{
    const g = document.getElementById(box.dataset.vec);
    if (g) g.style.display = box.checked ? 'inline' : 'none';
  }});
}});
</script>
</body></html>
"""


def _summary(bundle: TileBundle):
    from ..pipeline.assemble import channel_summary

    return channel_summary(bundle)


def write_report(bundle: TileBundle, gate: SourceGate, path: Path, *, decimate: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_tile(bundle, gate, decimate=decimate), encoding="utf-8")
    return path


def write_index_page(paths: list[Path], out: Path, entries: list[dict]) -> Path:
    """A contact sheet linking every tile report, so a site is reviewable in one pass."""
    out = Path(out)
    cards = "".join(
        f'<a class="card" href="{html.escape(p.name)}">'
        f'<strong>{html.escape(e["tile_id"])}</strong>'
        f'<span>coverage {e["coverage"]:.0%}</span>'
        f'<span>{e["buildings"]} buildings · {e["edges"]} road edges</span>'
        f'<span class="k {"ok" if e["inputs"] == "attribution-only" else "warn"}">'
        f'inputs: {e["inputs"]}</span></a>'
        for p, e in zip(paths, entries, strict=False)
    )
    out.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>JapGo tile set</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#c9d1d9;
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px}}
h1{{font-size:18px}} .grid{{display:grid;gap:12px;
grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}}
.card{{display:flex;flex-direction:column;gap:4px;padding:12px 14px;border:1px solid #30363d;
border-radius:8px;text-decoration:none;color:inherit;background:#161b22}}
.card:hover{{border-color:#58a6ff}} .card strong{{font-family:ui-monospace,monospace}}
.card span{{font-size:12px;color:#8b949e}} .k{{margin-top:4px;font-weight:600}}
.ok{{color:#3fb950}} .warn{{color:#d29922}}
</style></head><body><h1>JapGo — tile set ({len(entries)} tiles)</h1>
<p style="color:#8b949e;font-size:13px">Phase 2 gate: confirm every layer is spatially aligned
before any modelling begins.</p>
<div class="grid">{cards}</div></body></html>
""",
        encoding="utf-8",
    )
    return out


def summarise(bundle: TileBundle, gate: SourceGate) -> dict:
    return {
        "tile_id": bundle.tile.id,
        "coverage": bundle.coverage,
        "buildings": len(bundle.buildings),
        "edges": len(bundle.roads.edges) if bundle.roads else 0,
        "inputs": bundle.input_redistribution_class(gate),
    }
