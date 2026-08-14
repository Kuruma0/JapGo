"""Phase 7 — the one entry point a game-world generator calls.

Everything else in this package is a stage. This is the pipeline, and it exists so that a caller
needs to know terrain, bounds and a seed — not thresholds, not fold names, not what a TileBundle
is, and not that a U-Net was involved.

    terrain + bounds + seed
        -> road probability          (frozen model)
        -> candidate graph           (extraction, grade annotation)
        -> repaired graph            (connectivity)
        -> validated graph           (junction and geometry rules)
        -> terrain-legal graph       (reroute, switchback, or refuse)
        -> splines + junctions       (geometry)
        -> GeneratedRoads

Every stage after the model is deterministic, so the same inputs and the same seed give the same
network — the property the whole design is arranged around. The seed is threaded through and
recorded even though the current stages do not sample from it: a generator that cannot say what
seed produced a world cannot reproduce one, and adding the field later means every stored world
loses its provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.roads import RoadGraph
from ..geo.tiling import Bounds
from .candidates import CandidateReport, extract_candidates
from .geometry import GeometrySpec, JunctionPoint, RoadSpline, build_geometry
from .inference import FrozenModel
from .repair import RepairReport, RepairSpec, repair
from .terrain import TerrainReport, TerrainSpec, enforce_grade
from .validate import ValidationReport, ValidationSpec, enforce


@dataclass(frozen=True)
class GenerationParams:
    """Everything a caller may tune, in one place, all of it recorded in the output."""

    seed: int = 0
    threshold: float | None = None
    """Probability cut. ``None`` takes the frozen model's calibrated default."""

    elevation_datum_m: float = 0.0
    """Added to every emitted elevation, to turn a relative heightfield into an absolute one.

    The stack's ``elevation`` channel is **tile-relative** — ``raster_stack.yaml`` subtracts each
    tile's own mean, deliberately, so the model learns slope rather than "this is a mountain
    region". Grade is a difference and is unaffected by that offset, but absolute placement is
    not: exporting the raw channel puts roads tens of metres underground in an engine. Pass the
    tile's mean elevation here, or supply an absolute raster as ``elevation``.
    """

    repair: RepairSpec = field(default_factory=RepairSpec)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    geometry: GeometrySpec = field(default_factory=GeometrySpec)


@dataclass
class GenerationDiagnostics:
    """What each stage did. The reason a disappointing world can be explained rather than guessed at."""

    candidates: CandidateReport
    repair: RepairReport
    validation: ValidationReport
    terrain: TerrainReport

    def describe(self) -> str:
        return "\n".join([
            f"  candidates  {self.candidates.describe()}",
            f"  repair      {self.repair.describe()}",
            f"  validation  {self.validation.describe().splitlines()[0]}",
            f"  terrain     {self.terrain.describe()}",
        ])


@dataclass
class GeneratedRoads:
    """A finished road network and the record of how it was made."""

    graph: RoadGraph
    splines: list[RoadSpline]
    junctions: list[JunctionPoint]
    bounds: Bounds
    crs: str
    seed: int
    diagnostics: GenerationDiagnostics
    elevation_reference: str = "tile-relative"
    """Whether emitted heights are absolute or relative to the tile mean. Recorded rather than
    assumed, because an importer cannot tell by looking and a road at -92 m looks like a bug in
    the engine rather than a datum mismatch."""

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.splines)

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "crs": self.crs,
            "bounds": list(self.bounds.as_tuple()),
            "roads": len(self.splines),
            "junctions": len(self.junctions),
            "total_length_m": round(self.total_length_m, 1),
            "components": len(self.graph.connected_components()),
            "dead_end_ratio": round(self.graph.dead_end_ratio, 3),
            "elevation_reference": self.elevation_reference,
        }


def _sampler(elevation: np.ndarray, bounds: Bounds, resolution_m: float, *, datum: float = 0.0):
    rows, cols = elevation.shape

    def sample(x: float, y: float) -> float:
        col = int(np.clip((x - bounds.minx) / resolution_m, 0, cols - 1))
        row = int(np.clip((bounds.maxy - y) / resolution_m, 0, rows - 1))
        return float(elevation[row, col]) + datum

    return sample


def generate_roads(
    model: FrozenModel,
    channels: np.ndarray,
    bounds: Bounds,
    *,
    params: GenerationParams | None = None,
    elevation: np.ndarray | None = None,
) -> GeneratedRoads:
    """Terrain and world channels in, a game-ready road network out.

    ``elevation`` defaults to the stack's own elevation channel. It is separable because a game
    may hold its heightfield at a different resolution than the channels it feeds the model, and
    forcing the two together would make that awkward for no gain.
    """
    params = params or GenerationParams()
    spec = model.spec
    supplied = elevation is not None
    if elevation is None:
        elevation = channels[spec.index_of("elevation")]

    prediction = model.predict(channels, bounds, threshold=params.threshold)
    graph, candidates = extract_candidates(
        prediction, elevation=elevation, grade_limit_pct=params.terrain.max_grade_pct
    )
    graph, repair_report = repair(graph, params.repair)
    graph, validation = enforce(graph, params.validation)
    graph, terrain_report = enforce_grade(
        graph, elevation, bounds, model.card.resolution_m, params.terrain
    )

    datum = params.elevation_datum_m
    splines, junctions = build_geometry(
        graph,
        _sampler(elevation, bounds, model.card.resolution_m, datum=datum),
        spec=params.geometry,
    )
    return GeneratedRoads(
        graph=graph,
        splines=splines,
        junctions=junctions,
        bounds=bounds,
        crs=prediction.crs,
        seed=params.seed,
        diagnostics=GenerationDiagnostics(
            candidates=candidates, repair=repair_report,
            validation=validation, terrain=terrain_report,
        ),
        elevation_reference=(
            "absolute" if supplied or params.elevation_datum_m else "tile-relative"
        ),
    )


def export_bundle(roads: GeneratedRoads, destination: Path) -> Path:
    """Write an engine-agnostic bundle: GeoJSON geometry plus a JSON manifest.

    GeoJSON because every engine and every GIS tool reads it, and because invariant 1 says the
    core emits an interchange format rather than engine types. A Unity or Unreal adapter consumes
    this; it does not reach into the generator.
    """
    import json

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [
                [round(x, 3), round(y, 3), round(z, 3)]
                for (x, y), z in zip(s.points, s.elevations, strict=True)
            ]},
            "properties": {
                "id": s.edge_id, "road_class": s.road_class,
                "width_m": s.width_m, "grade_pct": s.grade_pct,
                "length_m": round(s.length_m, 1),
            },
        }
        for s in roads.splines
    ]
    (destination / "roads.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "crs": roads.crs, "features": features}, indent=1)
        + "\n", encoding="utf-8",
    )
    (destination / "junctions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "crs": roads.crs, "features": [
            {"type": "Feature",
             "geometry": {"type": "Point",
                          "coordinates": [round(j.position[0], 3), round(j.position[1], 3),
                                          round(j.elevation, 3)]},
             "properties": {"id": j.node_id, "degree": j.degree, "incident": j.incident}}
            for j in roads.junctions
        ]}, indent=1) + "\n", encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps({
            **roads.summary(),
            "model": asdict(roads.diagnostics.candidates),
            "diagnostics": roads.diagnostics.describe(),
        }, indent=2) + "\n", encoding="utf-8",
    )
    return destination
