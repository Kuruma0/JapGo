"""Phase 5 — the counterfactual sensitivity sweep.

This is the project's actual thesis, and the roadmap says so: *"Until the counterfactual sweep
demonstrates that, nothing else measured matters."* Every other number so far establishes that the
model can reconstruct a road network. None of them shows that **environmental information is doing
the work** — a model could score well by learning "roads are where the buildings are" and ignoring
terrain entirely, which is risk R2.

The experiment is the one §1.3 specifies. Hold the model fixed, perturb a single environmental
channel on a real tile, re-predict, extract the graph, and measure how road *structure* moves. The
prediction is directional and was written down before the project had a model:

    raise slope and the network should thin, lose intersections, and wind more;
    raise built land and road density should rise with it.

Three properties make this a test rather than a demonstration.

**One channel at a time.** Perturbing several at once cannot attribute the response. The terrain
channels are collinear (slope, roughness and relief rank together across the corpus), so a slope
perturbation moves its companions physically but the sweep still reports which *input* was touched.

**A null perturbation is included.** Re-running the model on the unmodified tile must reproduce the
baseline exactly. Without that control, a "response" could be inference nondeterminism.

**Direction, not magnitude.** The success criterion is that the network moves the way a geographer
would predict, not that it moves by a particular amount. Reporting an effect size that nothing
calibrates would invite over-reading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.structure import road_structure
from ..pipeline.channels import load_stack_spec
from ..pipeline.store import read_tile

log = logging.getLogger(__name__)

RESPONSES = ("road_density_km_per_km2", "intersection_density_per_km2", "sinuosity_median")
"""The structural measures the sweep reads. Deliberately few, and each with a written-down
expectation — a sweep over every metric would find *something* moving by chance."""


@dataclass(frozen=True)
class Perturbation:
    """One counterfactual: what is changed, and what should happen if the thesis holds."""

    name: str
    channel: str
    factor: float
    expect: dict[str, str]
    """Response name -> ``"up"``, ``"down"`` or ``"flat"``, fixed in advance."""

    def apply(self, stack: np.ndarray, index: int) -> np.ndarray:
        out = stack.copy()
        out[index] = np.clip(out[index] * self.factor, 0.0, None)
        return out


#: The sweep as run. Expectations come from site-selection.md and research doc §1.3, both written
#: before any model existed — which is what makes them predictions rather than rationalisations.
DEFAULT_SWEEP: tuple[Perturbation, ...] = (
    Perturbation("null", "slope", 1.0, dict.fromkeys(RESPONSES, "flat")),
    Perturbation(
        "flatten", "slope", 0.25,
        {"road_density_km_per_km2": "up", "intersection_density_per_km2": "up",
         "sinuosity_median": "down"},
    ),
    Perturbation(
        "steepen", "slope", 3.0,
        {"road_density_km_per_km2": "down", "intersection_density_per_km2": "down",
         "sinuosity_median": "up"},
    ),
    Perturbation(
        "unbuild", "landuse_built", 0.0,
        {"road_density_km_per_km2": "down", "intersection_density_per_km2": "down",
         "sinuosity_median": "flat"},
    ),
)


@dataclass
class SweepResult:
    perturbation: str
    channel: str
    factor: float
    baseline: dict[str, float]
    perturbed: dict[str, float]
    expect: dict[str, str]
    tiles: int = 0
    notes: list[str] = field(default_factory=list)

    def direction(self, response: str, *, tolerance: float = 0.05) -> str:
        """Which way the response actually moved, with a dead band for noise."""
        before, after = self.baseline.get(response), self.perturbed.get(response)
        if before is None or after is None or np.isnan(before) or np.isnan(after):
            return "n/a"
        if before == 0:
            return "flat" if after == 0 else "up"
        change = (after - before) / abs(before)
        if abs(change) < tolerance:
            return "flat"
        return "up" if change > 0 else "down"

    def agrees(self, response: str) -> bool:
        return self.direction(response) == self.expect.get(response)

    @property
    def score(self) -> float:
        """Fraction of responses that moved as predicted."""
        checked = [r for r in self.expect if self.direction(r) != "n/a"]
        return sum(self.agrees(r) for r in checked) / len(checked) if checked else float("nan")


def _predict(model, stack: np.ndarray):
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(stack[None]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(x)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def run_sweep(
    root: Path,
    model,
    tile_ids: list[str],
    *,
    perturbations: tuple[Perturbation, ...] = DEFAULT_SWEEP,
    threshold: float = 0.5,
    limit: int | None = None,
) -> list[SweepResult]:
    """Perturb, re-predict, extract, and compare structure — one channel at a time."""
    from .extract import ExtractionSpec, extract_graph

    spec = load_stack_spec()
    extraction = ExtractionSpec(threshold=threshold)
    chosen = tile_ids[:limit] if limit else tile_ids
    model.eval()

    bundles = [read_tile(root, t) for t in chosen]
    baseline_structures = []
    for bundle in bundles:
        graph = extract_graph(
            _predict(model, bundle.stack), bundle.tile.read, bundle.manifest.crs,
            spec=extraction, tile_id=bundle.tile.id,
        )
        baseline_structures.append(road_structure(graph, bundle.tile))
    baseline = _mean(baseline_structures)

    results = []
    for perturbation in perturbations:
        if perturbation.channel not in spec.names:
            log.warning("sweep: no channel %r in the stack; skipping", perturbation.channel)
            continue
        index = spec.index_of(perturbation.channel)

        structures = []
        for bundle in bundles:
            altered = perturbation.apply(bundle.stack, index)
            graph = extract_graph(
                _predict(model, altered), bundle.tile.read, bundle.manifest.crs,
                spec=extraction, tile_id=bundle.tile.id,
            )
            structures.append(road_structure(graph, bundle.tile))

        results.append(
            SweepResult(
                perturbation=perturbation.name,
                channel=perturbation.channel,
                factor=perturbation.factor,
                baseline=baseline,
                perturbed=_mean(structures),
                expect=perturbation.expect,
                tiles=len(bundles),
            )
        )
    return results


def _mean(structures: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.nanmean([s.get(name, np.nan) for s in structures]))
        for name in RESPONSES
    }
