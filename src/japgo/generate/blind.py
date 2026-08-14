"""The blind generation experiment: what the frozen model infers from terrain it has never seen.

Every evaluation of this model so far has been a *reconstruction* test — predict the roads that
are actually there, score against OSM. Reconstruction and generation are different questions, and
a model can be adequate at the first while having learned nothing that serves the second. A game
does not hand the system a real place. It hands it a heightfield that has never existed.

So: four synthetic archetypes, three seeds each, terrain channels only, one configuration for all
twelve worlds, and no ground truth anywhere. APLS, TOPO and pixel F1 are not merely omitted here,
they are undefined — there is no correct network for a world that does not exist, and inventing
one to score against would be inventing the answer.

What replaces them is a set of controls. A blind world that produces nothing is uninformative on
its own: it could mean the model cannot generalise, or that the ten channels this experiment must
withhold were carrying the signal, or that synthetic ground is unlike real ground in some way that
has nothing to do with roads. Each of those is separable by running the *same* treatment on a real
tile whose full-channel answer is known, so :func:`run_controls` does exactly that before the
worlds are generated. The controls are what make the result mean something.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.structure import road_structure
from ..core.roads import RoadGraph
from ..geo.raster import Raster
from ..geo.tiling import Bounds, Tile
from .candidates import annotate_terrain, extract_candidates
from .demo import DemoStage, _graph_png, _uri, overlay_png
from .geometry import build_geometry
from .inference import FrozenModel, RoadPrediction
from .pipeline import GeneratedRoads, GenerationDiagnostics, GenerationParams, export_bundle
from .repair import repair
from .synthetic import ARCHETYPES, SyntheticWorld, params_for, synthesise
from .terrain import enforce_grade
from .validate import enforce
from .world import WindowPlan, predict_world, terrain_stack

STAGES = ("RAW ML", "REPAIRED", "FINAL")


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------


def stage_metrics(
    label: str,
    graph: RoadGraph,
    tile: Tile,
    *,
    elevation: np.ndarray,
    bounds: Bounds,
    resolution_m: float,
    grade_limit_pct: float,
) -> dict:
    """Every measure the experiment asks for, at one stage of the pipeline.

    Grade is recomputed from the terrain at each stage rather than carried forward. Repair and
    rerouting both create edges, and an edge with no grade recorded would silently drop out of the
    violation count — making each stage look better than the last for the wrong reason.
    """
    # On a copy: annotate_terrain writes grades into the graph it is given, and a stage's own
    # numbers must not modify the graph a later stage will be measured against.
    measured = graph.model_copy(deep=True)
    annotate_terrain(measured, elevation, bounds, resolution_m, grade_limit_pct=grade_limit_pct)
    structure = road_structure(measured, tile)

    lengths = [e.length_m for e in measured.edges.values()]
    grades = [e.grade_pct for e in measured.edges.values() if e.grade_pct is not None]
    degrees = [measured.degree(n) for n in measured.nodes]
    histogram: dict[str, int] = {}
    for d in degrees:
        key = str(d) if d < 5 else "5+"
        histogram[key] = histogram.get(key, 0) + 1

    return {
        "stage": label,
        "edges": len(measured.edges),
        "nodes": len(measured.nodes),
        "junctions": sum(1 for d in degrees if d >= 3),
        "dead_ends": sum(1 for d in degrees if d == 1),
        "components": len(measured.connected_components()),
        "total_length_m": round(sum(lengths), 1),
        "mean_edge_length_m": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "dead_end_ratio": round(measured.dead_end_ratio, 4) if degrees else None,
        "degree_histogram": dict(sorted(histogram.items())),
        "road_density_km_per_km2": _round(structure.get("road_density_km_per_km2")),
        "intersection_density_per_km2": _round(structure.get("intersection_density_per_km2")),
        "orientation_entropy": _round(structure.get("orientation_entropy")),
        "sinuosity_median": _round(structure.get("sinuosity_median")),
        "max_grade_pct": round(max(grades), 1) if grades else None,
        "mean_grade_pct": round(float(np.mean(grades)), 1) if grades else None,
        "grade_violations": sum(1 for g in grades if g > grade_limit_pct),
    }


def _round(value, digits: int = 3):
    if value is None:
        return None
    return None if isinstance(value, float) and math.isnan(value) else round(float(value), digits)


@dataclass
class WorldResult:
    """One world, every stage of it."""

    name: str
    archetype: str
    seed: int
    params: dict
    terrain_notes: list[str]
    elevation_range_m: tuple[float, float]
    probability: dict
    metrics: list[dict]
    diagnostics: str
    terrain_response: dict = field(default_factory=dict)
    reroutes: int = 0
    deleted_no_route: int = 0
    observations: list[str] = field(default_factory=list)

    def stage(self, label: str) -> dict:
        return next(m for m in self.metrics if m["stage"] == label)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# running a world
# ---------------------------------------------------------------------------------------------


@dataclass
class WorldRun:
    """The result plus the arrays needed to draw it. Not serialised — 16 M cells per field."""

    result: WorldResult
    world: SyntheticWorld
    prediction: RoadPrediction
    raw: RoadGraph
    repaired: RoadGraph
    roads: GeneratedRoads


def run_world(
    model: FrozenModel,
    world: SyntheticWorld,
    *,
    params: GenerationParams | None = None,
    plan: WindowPlan | None = None,
) -> WorldRun:
    """Terrain in, every stage out.

    Deliberately re-implements :func:`generate_roads` stage by stage rather than calling it: the
    experiment needs the intermediate graphs, and a pipeline that only returns its endpoint cannot
    show what the procedural layer changed.
    """
    params = params or GenerationParams()
    elevation = world.elevation
    bounds = world.bounds
    resolution = model.card.resolution_m
    limit = params.terrain.max_grade_pct

    prediction = predict_world(model, elevation, bounds, plan=plan, threshold=params.threshold)

    raw, candidates = extract_candidates(
        prediction, elevation=elevation, grade_limit_pct=limit
    )
    # Deep copies between stages: repair and enforce mutate what they are handed, and the point
    # of the experiment is to be able to show the graph as it was before each pass.
    repaired, repair_report = repair(raw.model_copy(deep=True), params.repair)
    validated, validation = enforce(repaired.model_copy(deep=True), params.validation)
    final, terrain_report = enforce_grade(
        validated.model_copy(deep=True), elevation, bounds, resolution, params.terrain
    )

    # The world's own datum: synthetic elevations are absolute already, so nothing is added.
    splines, junctions = build_geometry(
        final, _world_sampler(elevation, bounds, resolution), spec=params.geometry
    )
    roads = GeneratedRoads(
        graph=final, splines=splines, junctions=junctions, bounds=bounds,
        crs=prediction.crs, seed=params.seed,
        diagnostics=GenerationDiagnostics(
            candidates=candidates, repair=repair_report,
            validation=validation, terrain=terrain_report,
        ),
        elevation_reference="absolute",
    )

    tile = world.tile
    metrics = [
        stage_metrics(label, graph, tile, elevation=elevation, bounds=bounds,
                      resolution_m=resolution, grade_limit_pct=limit)
        for label, graph in (("RAW ML", raw), ("REPAIRED", repaired), ("FINAL", final))
    ]

    p = prediction.probability
    result = WorldResult(
        terrain_response=terrain_response(p, elevation, world, resolution),
        name=world.name,
        archetype=world.params.archetype,
        seed=world.params.seed,
        params=world.params.to_dict(),
        terrain_notes=world.notes,
        elevation_range_m=(round(float(elevation.min()), 1), round(float(elevation.max()), 1)),
        probability={
            "threshold": prediction.threshold,
            "coverage_at_threshold": round(float(prediction.coverage), 8),
            "max": round(float(p.max()), 4),
            "mean": round(float(p.mean()), 6),
            "p99": round(float(np.percentile(p, 99)), 6),
            "p99_9": round(float(np.percentile(p, 99.9)), 6),
            # A uniform diagnostic, not a per-world tuning knob: the same cuts are reported for
            # every world so a faint field can be distinguished from an empty one.
            "coverage_at": {
                str(t): round(float((p >= t).mean()), 8) for t in (0.05, 0.1, 0.2, 0.45)
            },
        },
        metrics=metrics,
        diagnostics=roads.diagnostics.describe(),
        reroutes=terrain_report.rerouted,
        deleted_no_route=terrain_report.deleted,
    )
    return WorldRun(result, world, prediction, raw, repaired, roads)


def terrain_response(
    probability: np.ndarray, elevation: np.ndarray, world: SyntheticWorld, resolution_m: float
) -> dict:
    """Whether the probability field varies with the terrain, and in which direction.

    Asked quantitatively because the alternative is squinting at a mostly black image. Mean
    probability per slope quintile answers the question a picture cannot: a field that is faint
    but *ordered* by terrain is a model reading the ground weakly, and a field that is flat across
    every quintile is a model not reading it at all. The two look identical rendered.
    """
    from ..geo.terrain import slope as slope_op

    grade = slope_op(world.raster(), as_percent=True).data
    flat = probability.reshape(-1)
    out: dict[str, dict] = {}

    for name, field_ in (("slope_pct", grade), ("elevation_m", elevation)):
        values = field_.reshape(-1)
        edges = np.nanpercentile(values, [0, 20, 40, 60, 80, 100])
        bins: list[dict] = []
        for lo, hi in zip(edges, edges[1:], strict=False):
            mask = (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
            bins.append({
                "from": round(float(lo), 2), "to": round(float(hi), 2),
                "mean_probability": round(float(flat[mask].mean()), 7) if mask.any() else None,
            })
        means = [b["mean_probability"] or 0.0 for b in bins]
        out[name] = {
            "quintiles": bins,
            "spread": round(max(means) / max(min(means), 1e-9), 2) if means else None,
            "monotonic": bool(
                all(a <= b for a, b in zip(means, means[1:], strict=False))
                or all(a >= b for a, b in zip(means, means[1:], strict=False))
            ),
        }
    return out


def _world_sampler(elevation: np.ndarray, bounds: Bounds, resolution_m: float):
    rows, cols = elevation.shape

    def sample(x: float, y: float) -> float:
        col = int(np.clip((x - bounds.minx) / resolution_m, 0, cols - 1))
        row = int(np.clip((bounds.maxy - y) / resolution_m, 0, rows - 1))
        return float(elevation[row, col])

    return sample


# ---------------------------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------------------------


def run_controls(model: FrozenModel, root: Path, tile_ids: dict[str, str]) -> list[dict]:
    """What the experiment's own treatment costs, measured on real tiles.

    Three runs per tile, each removing one thing:

    ``full``
        Every channel, as the model was trained and evaluated. The reference.
    ``terrain only``
        Buildings and land use zeroed — the experiment's channel policy, applied to real ground.
    ``terrain only, smoothed``
        The same, over a DEM blurred at 4 m. Every valley, ridge and slope survives a 4 m blur;
        what does not survive is the metre-scale signature of a road — its cutting, its
        embankment, its graded bench. If the prediction dies here, the model is reading roads that
        are already in the terrain rather than inferring where roads should go.

    Without these, a blind world that produces nothing proves nothing.
    """
    from scipy.ndimage import gaussian_filter

    from ..pipeline.store import read_tile

    rows = []
    for site, tid in sorted(tile_ids.items()):
        bundle = read_tile(root, tid)
        read = bundle.tile.read
        full = model.predict(bundle.stack, read)

        dem = bundle.stack[model.spec.index_of("elevation")].astype(np.float32)
        terrain_only = model.predict(
            terrain_stack(Raster(dem, read, model.card.crs), model.spec), read
        )
        smoothed = model.predict(
            terrain_stack(
                Raster(np.ascontiguousarray(gaussian_filter(dem, 4.0)), read, model.card.crs),
                model.spec,
            ),
            read,
        )
        rows.append({
            "site": site,
            "tile_id": tid,
            "full_channels": round(float(full.coverage), 8),
            "terrain_only": round(float(terrain_only.coverage), 8),
            "terrain_only_smoothed_4m": round(float(smoothed.coverage), 8),
            "cost_of_zeroing": _ratio(terrain_only.coverage, full.coverage),
            "cost_of_smoothing": _ratio(smoothed.coverage, terrain_only.coverage),
        })
    return rows


def _ratio(a: float, b: float) -> float | None:
    return None if b <= 0 else round(float(a / b), 6)


# ---------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------


def _png(rgba: np.ndarray, decimate: int, *, levels: int = 32) -> bytes:
    """Decimate and posterise before encoding.

    A 4 km hillshade at full depth is a megabyte of PNG per panel, and ninety-six of those is a
    repository nobody wants to clone. Quantising to 32 levels is invisible on shaded relief and
    roughly thirds the file: PNG codes runs of equal bytes, and continuous noise has none.
    """
    from ..viz.image import png_bytes

    from .demo import _decimate

    out = _decimate(rgba, decimate).copy()
    step = 256 // levels
    out[..., :3] = (out[..., :3] // step) * step
    return png_bytes(out)


def world_panels(run: WorldRun, *, decimate: int = 6) -> list[DemoStage]:
    """The panels the experiment asks for, in pipeline order.

    Panels 3 to 6 share a canvas, scale and coordinate system, so what the procedural layer did is
    a visual difference rather than a claim in a table.
    """
    from ..geo.terrain import hillshade, slope
    from ..viz.image import colourise, shade

    world, pred = run.world, run.prediction
    bounds = world.bounds
    resolution = world.params.resolution_m
    raster = world.raster(run.roads.crs)
    relief = hillshade(raster).data
    shape = world.elevation.shape
    d = run.roads.diagnostics
    p = pred.probability

    # A field whose 99.9th percentile is 0.04 renders as black on a 0-1 scale. Both scalings are
    # shown: the honest one, and one stretched to the field's own top, labelled as stretched.
    stretch = max(float(np.percentile(p, 99.99)), 1e-4)

    return [
        DemoStage(
            "1. terrain",
            _png(shade(relief), decimate),
            "Synthetic. Generated from parameters and a seed, with no reference to any tile, "
            "raster or road network in the corpus.",
            f"{world.params.size_m / 1000:.0f} km square, "
            f"{world.elevation.min():.0f}-{world.elevation.max():.0f} m",
        ),
        DemoStage(
            "1b. slope",
            _png(colourise(slope(raster, as_percent=True).data, cmap="magma",
                           vmin=0.0, vmax=60.0), decimate),
            "Grade in percent, cut at 60. The 12% road limit is dark purple; anything bright is "
            "terrain no road may cross directly.",
            f"mean {float(slope(raster, as_percent=True).data.mean()):.1f}%",
        ),
        DemoStage(
            "2. ML probability (true scale)",
            _png(colourise(p, cmap="magma", vmin=0.0, vmax=1.0), decimate),
            "The frozen model's raw output, on the 0-1 scale its threshold lives on.",
            f"max {p.max():.3f}, mean {p.mean():.5f}, "
            f"{float((p >= pred.threshold).mean()):.4%} of cells above threshold "
            f"{pred.threshold}",
        ),
        DemoStage(
            "2b. ML probability (stretched)",
            _png(colourise(p, cmap="magma", vmin=0.0, vmax=stretch), decimate),
            "The same field rescaled to its own 99.99th percentile, so a faint response can be "
            "distinguished from no response at all. Not a threshold change.",
            f"stretched to {stretch:.4f}",
        ),
        DemoStage(
            "3. raw ML graph",
            _graph_png(run.raw, bounds, shape, resolution, decimate=decimate),
            "Vectorised exactly as predicted. Red is a dead end, blue a junction.",
            d.candidates.describe(),
        ),
        DemoStage(
            "4. repaired graph",
            _graph_png(run.repaired, bounds, shape, resolution, decimate=decimate),
            "After connectivity repair and junction processing. Deterministic.",
            d.repair.describe(),
        ),
        DemoStage(
            "5. final roads",
            _graph_png(run.roads.graph, bounds, shape, resolution, decimate=decimate),
            "After grade enforcement and smoothing.",
            d.terrain.describe(),
        ),
        DemoStage(
            "6. roads on terrain",
            overlay_png(relief, run.roads.graph, bounds, resolution, decimate=decimate),
            "The final network over the ground that produced it.",
            "Does it follow the terrain, or merely sit on it?",
        ),
    ]


# ---------------------------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------------------------

_CSS = """
body{background:#0f1115;color:#dfe3ea;font:15px/1.6 system-ui,sans-serif;margin:0 auto;
     padding:32px;max-width:1500px}
h1{font-size:26px;margin:0 0 6px} h2{font-size:20px;margin:44px 0 6px;
   border-bottom:1px solid #262c38;padding-bottom:7px} h3{font-size:16px;margin:26px 0 8px}
.sub{color:#8b93a1;margin-bottom:20px}
p{max-width:76ch} li{max-width:74ch}
.note{border-left:3px solid #4b6ea8;background:#151b25;padding:12px 16px;border-radius:0 6px 6px 0;
      margin:18px 0;color:#b9c6da}
.warn{border-left-color:#c08a3e;background:#231d13;color:#dcbd8a}
.bad{border-left-color:#b5544f;background:#241618;color:#e0aaa6}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin:14px 0}
figure{margin:0;background:#161a21;border:1px solid #232833;border-radius:8px;overflow:hidden}
img{width:100%;display:block;background:#000;image-rendering:pixelated}
figcaption{padding:9px 11px}
.t{font-weight:600;margin-bottom:3px}.c{color:#98a2b3;font-size:13px}
.s{font-size:12px;font-family:ui-monospace,monospace;margin-top:6px;color:#7f8899;
   word-break:break-word}
.scroll{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 11px;text-align:right;border-bottom:1px solid #21262f;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:#93a0b3;font-weight:600;text-align:right}
tbody tr:hover{background:#161b23}
code{background:#1a1f28;padding:1px 5px;border-radius:3px;font-size:13px}
pre{background:#151920;border:1px solid #202632;border-radius:6px;padding:12px;
    overflow-x:auto;white-space:pre-wrap;word-break:break-word}
ol,ul{max-width:78ch}
.zero{color:#6d7684}
"""


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="zero">{c}</td>' if c in ("0", "-", "0.0", "None") else f"<td>{c}</td>"
            for c in row
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _panels(stages: list[DemoStage]) -> str:
    return '<div class="grid">' + "".join(
        f'<figure><img alt="{s.title}" src="{_uri(s.png)}">'
        f'<figcaption><div class="t">{s.title}</div><div class="c">{s.caption}</div>'
        f'<div class="s">{s.stats}</div></figcaption></figure>'
        for s in stages
    ) + "</div>"


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stage_rows(result: WorldResult) -> list[list]:
    rows = []
    for m in result.metrics:
        rows.append([
            m["stage"], m["edges"], m["junctions"], m["dead_ends"], m["components"],
            _fmt(m["total_length_m"] / 1000.0), _fmt(m["mean_edge_length_m"], 1),
            _fmt(m["dead_end_ratio"], 3), _fmt(m["road_density_km_per_km2"]),
            _fmt(m["intersection_density_per_km2"]), _fmt(m["orientation_entropy"], 3),
            _fmt(m["sinuosity_median"], 3), _fmt(m["max_grade_pct"], 1),
            _fmt(m["mean_grade_pct"], 1), m["grade_violations"],
            " ".join(f"{k}:{v}" for k, v in m["degree_histogram"].items()) or "-",
        ])
    return rows


STAGE_HEADERS = [
    "stage", "edges", "junctions", "dead ends", "components", "length km", "mean edge m",
    "dead-end ratio", "density km/km2", "junctions/km2", "orient. entropy", "sinuosity",
    "max grade %", "mean grade %", "violations", "degree histogram",
]


def write_world_report(run: WorldRun, destination: Path, *, decimate: int = 4) -> Path:
    """One world: every panel, every stage, and the bundle beside it."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    r = run.result

    stages = world_panels(run, decimate=decimate)
    (destination / "params.json").write_text(
        json.dumps({"params": r.params, "terrain_notes": r.terrain_notes}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(json.dumps(r.to_dict(), indent=2) + "\n",
                                              encoding="utf-8")
    export_bundle(run.roads, destination)

    page = destination / "world.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>blind generation — {r.name}</title><style>{_CSS}</style>"
        f"<h1>{r.name}</h1>"
        f"<div class='sub'>{r.archetype} &middot; seed {r.seed} &middot; "
        f"elevation {r.elevation_range_m[0]:.0f}-{r.elevation_range_m[1]:.0f} m</div>"
        "<div class='note'>This terrain was generated procedurally from the parameters below, "
        "independently of the training and evaluation corpus, and contains no provided road "
        "network. APLS, TOPO and pixel F1 are not applicable: there is no ground truth.</div>"
        + _panels(stages)
        + "<h2>Measurements</h2>" + _table(STAGE_HEADERS, _stage_rows(r))
        + f"<h2>Terrain</h2><ul>{''.join(f'<li>{n}</li>' for n in r.terrain_notes)}</ul>"
        + f"<pre class='s'>{json.dumps(r.params, indent=1)}</pre>",
        encoding="utf-8",
    )
    return page


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")


def save_images(stages: list[DemoStage], destination: Path, name: str) -> list[dict]:
    """Write panels as files as well as embedding them.

    The master report is then rebuildable from disk without re-running the model, which matters
    because the prose in it is written *after* reading the numbers — and a report that needs a GPU
    to regenerate is a report that quietly stops matching its own data.
    """
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, s in enumerate(stages):
        path = destination / f"{name}__{i}_{_slug(s.title)}.png"
        path.write_bytes(s.png)
        saved.append({"title": s.title, "caption": s.caption, "stats": s.stats,
                      "file": path.name})
    return saved


ASSESSMENT: dict[str, str] = {
    "intro": """
<p>The frozen model was given twelve heightfields it has never seen, generated from parameters and
a seed, containing no roads and derived from no tile in the corpus. It was given the terrain
channels and nothing else. The question was what roads it would propose.</p>
<p><b>It proposed almost nothing.</b> Six of the twelve worlds produced not one cell above the
model's threshold across 16 km&sup2;; seven produced no extractable edge at all; <b>nine finished
with no road whatsoever</b>. Across all twelve worlds — 192 km&sup2; — the total output is
<b>1.04 km</b> of disconnected road, in an area where the real corpus carries hundreds of
kilometres. This section reports what that means, and the controls are what allow it to mean
anything.</p>
""",
    "controls": """
<p>Three explanations were available for an empty world, and the controls separate them.</p>
<p><b>Zeroing the building and land-use channels costs a factor of two, not a factor of a
thousand.</b> On the held-out plain the same tile goes from 19.98% coverage with every channel to
9.84% with terrain alone. That is a real handicap and it is not the explanation: 9.84% is still a
dense road network, and the synthetic worlds sit near 0.005%.</p>
<div class="note bad"><b>Blurring the real terrain by 4 m destroys the prediction.</b> A 4 m
Gaussian leaves every valley, ridge, spur and slope in place — mean roughness barely moves,
0.0111 to 0.0070 — and coverage falls from 9.84% to <b>0.00035%</b>, a factor of 28,000. On the
mountain tile it falls by a factor of 32.</div>
<p>What a 4 m blur removes is not landform. It is the metre-scale signature of the road itself:
the cutting, the embankment, the graded bench, the flat strip bounded by two breaks of slope. The
model is reading roads that are <em>already present in the terrain</em>. It is a road detector
operating on a DEM, not a road planner operating on a landscape.</p>
<p>This was not a hypothesis chosen to fit the result. The competing one — that synthetic ground
is too smooth, lacking the metre-scale noise of a LiDAR-derived DEM — was tested directly by
adding Gaussian micro-relief of 0.05 to 0.4 m to a synthetic window. It made the response
<em>worse</em>, from a peak of 0.110 to 0.028. Roughness is not what the model wants; road-shaped
geometry is.</p>
""",
    "worlds": """
<p>One configuration produced all twelve rows: the same generation seed, the same threshold, the
same repair, validation, grade and geometry settings. Nothing was tuned per world and no world was
re-run to get a better picture.</p>
<p>The plain is the most informative failure. It is <em>gentler</em> than Hamamatsu — mean slope
3.9% against 6.5% — and on Hamamatsu, with the same terrain-only channel policy, the model covers
9.84% of the tile. On three synthetic plains its peak probability across 16 million cells is
0.026, against a threshold of 0.45. The difference between those two cases is not the landform and
not the channels. It is that one of them has roads built into the ground and the other does
not.</p>
""",
    "archetype:mountain_valley": """
<p>The strongest of the four, which is faint praise: three worlds, three to six candidate edges,
and at most 520 m of final road in 16 km&sup2; — one of the three loses everything it proposed to
grade enforcement and finishes empty. Assessed against the questions asked of this
archetype — do roads follow valleys, avoid direct climbs, switchback, cross ridges sensibly — none
can be answered, because there are no roads to assess. What can be said is where the sub-threshold
response falls: the stretched panels show thin arcs along the <b>break of slope where the valley
floor meets its wall</b>, which is the closest thing this terrain contains to the geometry of a
road bench. The model is looking for the right shape and finding it in the only place a natural
landform imitates it.</p>
""",
    "archetype:coastal": """
<p>Two worlds produced a handful of candidate edges and one produced nothing; of those two, one
survives to 0.20 km of final road and the other is emptied by grade enforcement. The response is concentrated
on the coastal terrace — the flattest ground — and this is the archetype with the strongest slope
ordering, 16 to 18 times more probability mass in the flattest quintile than the steepest. The
direction is right. The magnitude is four hundred times below the threshold.</p>
<p>One limitation is this experiment's own doing: the sea is terrain only. The land-use water
channel is zero for every world, so the model has no way to know the flat ground at &minus;40 m is
water rather than a plain. It did not place roads there, but it did not place roads anywhere, so
that is not evidence of judgement.</p>
""",
    "archetype:plain": """
<p>Not one cell above threshold, on all three seeds, with a peak probability of 0.026. The mean probability is higher
here than on any other archetype — around 0.01 against 0.0005 — but it is a uniform haze with no
structure, and it <em>rises</em> with slope rather than falling, the reverse of every other
archetype. A diffuse field that is slightly denser on hill flanks is not a road network being
suppressed by a threshold. It is a model with nothing to say.</p>
""",
    "archetype:basin": """
<p>Two seeds produce not one cell above threshold and the third produces 0.00004% of cells, none
of which survives extraction. The question this archetype was built to ask — do roads
concentrate in the basin and connect toward the lower passes — cannot be answered. The
sub-threshold field does concentrate on the basin floor, eight to nine times the steep-ground
level, and the passes are not distinguishable from the rest of the rim at any scaling.</p>
""",
    "limitations": """
<ul>
<li><b>Ten of fifteen channels were withheld</b>, and the control shows that costs a factor of
two. The headline result does not depend on it, but every absolute number here would be roughly
twice as large with buildings and land use supplied.</li>
<li><b>The sea is not marked as water.</b> Coastal worlds give the model flat ground below zero
with no water channel. A game integrating this module would supply that channel.</li>
<li><b>Four archetypes, three seeds, one model.</b> These conclusions are about
<code>road_v1</code>, a 12-epoch U-Net trained on 106 tiles. They are not about learned road
generation in general.</li>
<li><b>The synthetic terrain is smoother than its real counterparts</b> — mean slope 57% against
Kawanehon's 75%, 39% against Izu's 47%, 3.9% against Hamamatsu's 6.5%. It is inside the corpus's
range and gentler, which if anything should have made prediction easier.</li>
<li><b>No ground truth exists</b>, so no accuracy statement is made or is possible. Coverage,
probability and structural counts are all that is reported.</li>
</ul>
""",
    "ml_generalization": """
<p><b>No.</b> The frozen model does not produce meaningful road proposals from terrain it has never
seen. Nine of twelve worlds finish empty; the other three yield 0.2 to 0.5 km of disconnected
fragments each, 1.04 km in total across 192 km&sup2;. There is no reading of these numbers under
which the output is usable.</p>
""",
    "terrain_understanding": """
<p><b>Weakly, and in the right direction on three archetypes out of four.</b> Mean probability by
slope quintile is ordered — the flattest fifth of the ground carries 4 to 18 times the probability
mass of the steepest, monotonically, on mountain, coastal and basin worlds. On plains the ordering
inverts and the field is a structureless haze.</p>
<p>So the model has learned <em>roads prefer flat ground</em>. That is a real relationship and it
survives to unseen terrain. It is also, on its own, worth very little: the response never
approaches the operating threshold, and "flat" does not distinguish a road corridor from a field.
Valleys, ridges, coastlines, basins and passes produce no identifiable response beyond what their
slope alone explains.</p>
""",
    "procedural_correction": """
<p><b>It cannot help, and that is not a defect in it.</b> Repair bridges gaps up to 45 m, drops
components under 60 m and snaps dead ends within 25 m. Given four candidate edges in 16 km&sup2;,
kilometres apart, there is nothing within reach of any of those thresholds — across all twelve
worlds the repair pass bridged nothing, snapped nothing and pruned nothing.</p>
<p>On real tiles the same layer is worth three to four times fewer components and roughly half the
dead ends. The difference is not the layer's strength; it is that a corrector needs something to
correct. Invariant 5 says ML proposes and procedural disposes, and this experiment locates the
boundary of that division exactly: <b>the procedural system can repair a bad network and cannot
manufacture a missing one.</b></p>
""",
    "game_suitability": """
<p><b>No.</b> Not from terrain alone, with this checkpoint. A game handing this module a fresh
heightfield receives an empty world.</p>
<p>The module around the model is a different matter and is not implicated by this result. Given a
plausible probability field it produces grade-legal, connected, smoothed, engine-ready geometry —
demonstrated on real tiles, where the same code turns a 102-component scatter into a 22-component
network with every grade violation rerouted rather than deleted. What is missing is the
proposal.</p>
""",
    "training_decision": """
<p><b>Yes — this is a specific failure the procedural system cannot address.</b> But "train
longer on more tiles" is the wrong response, and the corpus curve is not the relevant lever here.
The model is not underfitting; it is solving a different task from the intended one, and doing it
well enough that no amount of the same data will redirect it.</p>
<p>The task it learned is <em>find the roads in this DEM</em>, which is available at 1 m because a
road is a visible earthwork. The task wanted is <em>propose roads for this landscape</em>. Three
changes would separate them, in the order they seem worth trying:</p>
<ol>
<li><b>Train at a resolution where road earthworks are invisible.</b> The project already plans a
1 m working tier with augmentation simulating 30 m sources, for a different reason — VIRTUAL
SHIZUOKA's 0.5 m does not exist nationally. At 30 m a 6 m bench is sub-pixel, so the shortcut
closes and the model must key on landform. This is the cheapest test: the augmentation is already
specified, and the blur control above is a preview of what the model does when it is applied — it
should be run as a training input, not as a destructive test.</li>
<li><b>Supply demand, not just terrain.</b> Roads exist to connect destinations. With buildings and
land use zeroed there is no demand signal at all, and even with them the model is given
<em>where the buildings are</em> rather than <em>where people want to go</em>. A generator for
games can legitimately take settlement seeds as an input, because the game knows where its towns
are.</li>
<li><b>Measure this properly, not incidentally.</b> The blur control should become a standing
evaluation. Any future checkpoint that claims to generate rather than reconstruct must survive it,
and this one takes four seconds to run.</li>
</ol>
<p>Until one of those changes is made, further training against the current objective will produce
a better road detector, which is a useful thing and not the thing this project's generation goal
requires.</p>
""",
    "conclusion": """
<p>The experiment answered its question, and the answer is negative and specific.</p>
<p><code>road_v1</code> reconstructs roads from the terrain they are cut into. Blur that terrain by
four metres, leaving every landform intact, and the prediction collapses by four orders of
magnitude. Give it a landscape with no roads in it and it returns 1.04 km across 192 km&sup2; —
not a bad network, not a noisy one, an empty one on nine of twelve worlds.</p>
<p>This does not retract Phase 4. Beating a non-learned prior on APLS and TOPO on held-out real
tiles is a reconstruction result and it stands as one. What it does retract is any reading of
Phase 4 as evidence about <em>generation</em>. The two were never the same claim, and this is the
first instrument that could tell them apart.</p>
<p>It also explains the Phase 5 sweep retrospectively. Quantile-mapping a slope distribution
preserves the fine geometry and moves the magnitudes, so the response tracked size and not
direction; shuffling the channel destroys the geometry, so the prediction collapsed. Both
behaviours are exactly what a detector of metre-scale road shape would do, and neither is what a
model reasoning about landform would do.</p>
<p>The procedural half of the module is not implicated. It was built to dispose of what the model
proposes, and it does that measurably well on real tiles. It has nothing to dispose of here.</p>
"""
}
"""The written conclusions, composed after the numbers existed. Kept in the module so that
regenerating the report reproduces the report, rather than an empty shell of one."""


def write_master_report(
    path: Path,
    *,
    results: list[dict],
    controls: list[dict],
    model_card,
    images_dir: Path,
    assessment: dict[str, str] | None = None,
) -> Path:
    """The whole experiment on one page."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assessment = assessment or ASSESSMENT
    rel = images_dir.name

    def section(key: str, default: str = "") -> str:
        text = assessment.get(key, default)
        return f"<div>{text}</div>" if text else ""

    control_rows = [
        [
            c["site"], c["tile_id"],
            f"{c['full_channels']:.4%}", f"{c['terrain_only']:.4%}",
            f"{c['terrain_only_smoothed_4m']:.4%}",
            "-" if c["cost_of_zeroing"] is None else f"x{c['cost_of_zeroing']:.3f}",
            "-" if c["cost_of_smoothing"] is None else f"x{c['cost_of_smoothing']:.5f}",
        ]
        for c in controls
    ]

    summary_rows = []
    for r in results:
        final = next(m for m in r["metrics"] if m["stage"] == "FINAL")
        raw = next(m for m in r["metrics"] if m["stage"] == "RAW ML")
        summary_rows.append([
            r["name"], r["archetype"], r["seed"],
            f"{r['elevation_range_m'][0]:.0f}-{r['elevation_range_m'][1]:.0f}",
            f"{r['probability']['coverage_at_threshold']:.5%}",
            f"{r['probability']['max']:.3f}",
            raw["edges"], raw["components"],
            final["edges"], final["junctions"], final["components"],
            _fmt(final["total_length_m"] / 1000.0),
            _fmt(final["dead_end_ratio"], 3),
            r["reroutes"], r["deleted_no_route"],
        ])

    body = [
        f"<h1>Blind generation on synthetic terrain</h1>",
        "<div class='sub'>What the frozen model infers from ground it has never seen</div>",
        "<div class='note'>These terrain worlds were procedurally generated independently of the "
        "training and evaluation corpus and contain no provided road network. APLS, TOPO and "
        "pixel F1 are <b>not applicable</b> — there is no ground truth for a place that does not "
        "exist, and inventing one to score against would be inventing the answer.</div>",
        section("intro"),
        "<h2>1. What was run</h2>",
        f"<p>Four archetypes &times; three seeds = {len(results)} worlds, each 4 km square at "
        "1 m/px, generated from parameters and a seed. One generation configuration throughout: "
        "no per-world thresholds, no hand-placed roads, no tuning against results.</p>",
        f"<pre class='s'>{model_card.describe()}</pre>",
        f"<p>Channels supplied: the five terrain channels "
        f"(<code>{'</code>, <code>'.join(TERRAIN_CHANNEL_NAMES)}</code>) and "
        "<code>valid</code>. The ten building and land-use channels are held at the stack's "
        "nodata value for every world — they are exactly the settlement information the "
        "experiment must not supply.</p>",
        "<h2>2. Controls: what the treatment itself costs</h2>",
        "<p>A blind world that produces nothing proves nothing on its own. The same treatment is "
        "applied to <em>real</em> tiles, where the full-channel answer is known, so the three "
        "possible explanations can be separated. Coverage is the fraction of cells above the "
        "model's threshold.</p>",
        _table(
            ["site", "tile", "full channels", "terrain only", "terrain only, 4 m blur",
             "cost of zeroing", "cost of blurring"],
            control_rows,
        ),
        section("controls"),
        "<h2>3. All worlds</h2>",
        _table(
            ["world", "archetype", "seed", "elevation m", "coverage", "max p",
             "raw edges", "raw comp", "final edges", "final junctions", "final comp",
             "final km", "dead-end ratio", "rerouted", "deleted"],
            summary_rows,
        ),
        section("worlds"),
    ]

    for archetype in ARCHETYPES:
        rows = [r for r in results if r["archetype"] == archetype]
        if not rows:
            continue
        body.append(f"<h2>4. {archetype.replace('_', ' ')}</h2>")
        body.append(section(f"archetype:{archetype}"))
        for r in rows:
            body.append(f"<h3>{r['name']}</h3>")
            body.append(
                '<div class="grid">' + "".join(
                    f'<figure><img alt="{i["title"]}" src="{rel}/{i["file"]}">'
                    f'<figcaption><div class="t">{i["title"]}</div>'
                    f'<div class="c">{i["caption"]}</div>'
                    f'<div class="s">{i["stats"]}</div></figcaption></figure>'
                    for i in r.get("images", [])
                ) + "</div>"
            )
            body.append(_table(STAGE_HEADERS, _stage_rows(WorldResult(**{
                k: v for k, v in r.items() if k in WorldResult.__dataclass_fields__
            }))))
            if r.get("observations"):
                body.append("<ul>" + "".join(f"<li>{o}</li>" for o in r["observations"]) + "</ul>")

    for key, heading in (
        ("limitations", "5. Failures and limitations"),
        ("ml_generalization", "6. Can the frozen model generalise?"),
        ("terrain_understanding", "7. Does the probability field respond to terrain?"),
        ("procedural_correction", "8. How much does the procedural system recover?"),
        ("game_suitability", "9. Is the output usable in a game world?"),
        ("training_decision", "10. Does this justify another training cycle?"),
        ("conclusion", "11. Conclusion"),
    ):
        if assessment.get(key):
            body.append(f"<h2>{heading}</h2>{assessment[key]}")

    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>JapGo — blind generation</title>"
        f"<style>{_CSS}</style>" + "".join(body),
        encoding="utf-8",
    )
    return path


TERRAIN_CHANNEL_NAMES = ("elevation", "slope", "aspect_sin", "aspect_cos", "roughness")
