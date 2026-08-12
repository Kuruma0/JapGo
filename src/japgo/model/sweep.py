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

    def apply(
        self, stack: np.ndarray, index: int, *, bounds: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, float]:
        """Return the perturbed stack and the fraction of pixels the bounds had to clamp.

        ``bounds`` is the range the channel actually takes across the corpus. Without it the first
        sweep multiplied slope by 3.0 and produced terrain steeper than anything in training, so
        the model degraded rather than responded — flatten and steepen both *reduced* predicted
        road density, when the whole design declares them opposite. That measures out-of-
        distribution robustness, not environmental response.

        Clamping keeps the counterfactual answerable. The clamped fraction is returned rather than
        hidden, because a perturbation that clamps most of the tile has been neutered and the
        result should say so.
        """
        out = stack.copy()
        altered = out[index] * self.factor

        clamped = 0.0
        if bounds is not None:
            low, high = bounds
            outside = (altered < low) | (altered > high)
            clamped = float(outside.mean())
            altered = np.clip(altered, low, high)

        out[index] = np.clip(altered, 0.0, None)
        return out, clamped


@dataclass(frozen=True)
class QuantileMap:
    """Reshape a channel onto another *real* site's distribution.

    The scaling perturbations could not be made honest. Multiplying slope by 3.0 invents terrain
    that exists nowhere; clamping it back to the corpus range produces a saturated uniform field,
    which is no more natural. Both measure how the model behaves on inputs it was never shown.

    Quantile mapping asks the question the thesis actually poses: *give this coastal tile the slope
    distribution of the mountain valley — does the predicted network become more valley-like?* Every
    value the model sees is one some real tile really had, and the shape of the distribution is
    preserved rather than stretched. There is nothing out-of-distribution left to blame.
    """

    name: str
    channel: str
    reference_site: str
    expect: dict[str, str]

    #: Kept so :class:`SweepResult` can report the two uniformly.
    factor: float = float("nan")

    def apply(
        self, stack: np.ndarray, index: int, *, reference: np.ndarray | None = None, **_
    ) -> tuple[np.ndarray, float]:
        if reference is None or reference.size == 0:
            return stack.copy(), 0.0

        out = stack.copy()
        source = out[index]
        # Map by rank: a pixel at the 40th percentile of this tile takes the reference's 40th.
        levels = np.linspace(0.0, 100.0, 256)
        source_q = np.percentile(source, levels)
        reference_q = np.percentile(reference, levels)
        # np.interp needs a strictly increasing x; ties in a flat tile would otherwise break it.
        source_q = np.maximum.accumulate(source_q + np.arange(source_q.size) * 1e-9)
        out[index] = np.interp(source, source_q, reference_q).astype(stack.dtype)
        return out, 0.0


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
    clamped: float = 0.0
    """Mean fraction of pixels pushed outside the corpus range and clipped back.

    Read it before reading the directions: above roughly a third, the perturbation has been
    neutered by the bounds and a 'flat' response means the input barely moved."""
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


FLATTER_EXPECTATION = {
    "road_density_km_per_km2": "up",
    "intersection_density_per_km2": "up",
    "sinuosity_median": "down",
}
STEEPER_EXPECTATION = {
    "road_density_km_per_km2": "down",
    "intersection_density_per_km2": "down",
    "sinuosity_median": "up",
}


def quantile_sweep(references: dict[str, float], held_out_median: float) -> tuple:
    """Swaps between real sites' slope distributions, with expectations set against *home*.

    The first version ranked the two reference sites against each other and called the lower one
    "flatten". That is wrong whenever the held-out site sits outside their range: sweeping the
    Hamamatsu plain, both references are steeper than home, so the arm labelled "flatten" in fact
    steepened the tile and was scored against an expectation it could not meet. The comparison has
    to be against the distribution being replaced, not against the other replacement.
    """
    out: list = [Perturbation("null", "slope", 1.0, dict.fromkeys(RESPONSES, "flat"))]
    for site, median in sorted(references.items(), key=lambda kv: kv[1]):
        flatter = median < held_out_median
        out.append(
            QuantileMap(
                f"slope_of_{site}", "slope", site,
                dict(FLATTER_EXPECTATION if flatter else STEEPER_EXPECTATION),
            )
        )
    return tuple(out)


def reference_values(root: Path, tile_ids: list[str], index: int, *, cap: int = 2_000_000):
    """Channel values from a reference site, sampled for the quantile map."""
    chunks = []
    for tile_id in tile_ids:
        b = read_tile(root, tile_id)
        chunks.append(b.stack[index][b.stack[-1] > 0.5].ravel())
    if not chunks:
        return np.empty(0)
    values = np.concatenate(chunks)
    if values.size > cap:
        values = np.random.default_rng(0).choice(values, cap, replace=False)
    return values


def observed_range(bundles, index: int, *, low: float = 1.0, high: float = 99.0):
    """The percentile range a channel occupies across the swept tiles.

    Percentiles rather than min/max so a single outlier pixel cannot licence a perturbation the
    corpus does not support.
    """
    values = np.concatenate([b.stack[index][b.stack[-1] > 0.5].ravel() for b in bundles])
    if values.size == 0:
        return None
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def run_sweep(
    root: Path,
    model,
    tile_ids: list[str],
    *,
    perturbations: tuple[Perturbation, ...] = DEFAULT_SWEEP,
    threshold: float = 0.5,
    limit: int | None = None,
    in_distribution: bool = True,
    site_tiles: dict[str, list[str]] | None = None,
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
        bounds = observed_range(bundles, index) if in_distribution else None

        extra: dict = {}
        if isinstance(perturbation, QuantileMap):
            tiles = (site_tiles or {}).get(perturbation.reference_site)
            if not tiles:
                log.warning(
                    "sweep: no tiles for reference site %r; skipping %s",
                    perturbation.reference_site, perturbation.name,
                )
                continue
            extra["reference"] = reference_values(root, tiles, index)
            bounds = None      # the reference *is* the distribution; clamping would distort it

        structures, clamps = [], []
        for bundle in bundles:
            altered, clamped = perturbation.apply(
                bundle.stack, index, bounds=bounds, **extra
            )
            clamps.append(clamped)
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
                clamped=float(np.mean(clamps)) if clamps else 0.0,
            )
        )
    return results


def _mean(structures: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.nanmean([s.get(name, np.nan) for s in structures]))
        for name in RESPONSES
    }
