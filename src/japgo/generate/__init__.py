"""The game-world road generation module.

Separate from :mod:`japgo.model` on purpose. That package answers research questions — folds,
priors, APLS, corpus curves — and a game engine should never need to import any of it. This one
takes terrain in and produces a road network out, with the frozen model as one stage inside it.

The pipeline the rest of this package builds out::

    terrain / world channels
        -> FrozenModel.predict          road probability   (japgo.generate.inference)
        -> extract_graph                raw road graph     (japgo.model.extract, reused)
        -> connectivity repair          coherent graph
        -> junction + terrain validation
        -> road geometry
        -> engine-ready bundle

Invariant 5 governs the whole thing: **ML proposes, procedural disposes.** The model is a proposal
system and never the final authority. Everything downstream of `predict` is deterministic, which
is what makes a seed reproduce a world.
"""

from __future__ import annotations

from .candidates import CandidateReport, extract_candidates
from .demo import DemoStage, build_demo, overlay_png, provenance_note, write_demo
from .geometry import GeometrySpec, JunctionPoint, RoadSpline, build_geometry
from .inference import DEFAULT_THRESHOLD, FrozenModel, ModelCard, RoadPrediction
from .pipeline import (
    GeneratedRoads,
    GenerationDiagnostics,
    GenerationParams,
    export_bundle,
    generate_roads,
    local_frame,
)
from .repair import RepairReport, RepairSpec, largest_component, repair
from .synthetic import ARCHETYPES, SyntheticWorld, TerrainParams, params_for, synthesise
from .world import WindowPlan, predict_world, terrain_stack
from .terrain import TerrainReport, TerrainSpec, enforce_grade
from .validate import ValidationReport, ValidationSpec, enforce, validate

__all__ = [
    "DEFAULT_THRESHOLD",
    "CandidateReport",
    "DemoStage",
    "build_demo",
    "overlay_png",
    "provenance_note",
    "write_demo",
    "FrozenModel",
    "ModelCard",
    "RepairReport",
    "RepairSpec",
    "RoadPrediction",
    "extract_candidates",
    "largest_component",
    "repair",
    "TerrainReport",
    "TerrainSpec",
    "enforce_grade",
    "ValidationReport",
    "ValidationSpec",
    "enforce",
    "validate",
    "GeometrySpec",
    "JunctionPoint",
    "RoadSpline",
    "build_geometry",
    "ARCHETYPES",
    "SyntheticWorld",
    "TerrainParams",
    "params_for",
    "synthesise",
    "WindowPlan",
    "predict_world",
    "terrain_stack",
    "GeneratedRoads",
    "GenerationDiagnostics",
    "GenerationParams",
    "export_bundle",
    "local_frame",
    "generate_roads",
]
