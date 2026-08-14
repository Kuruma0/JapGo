"""Phase 8 — judge the module by what a game needs, not by what a paper needs.

APLS answers "did this reproduce a specific real network", which is the right question for
reconstruction and the wrong one for a generator. A game does not need Atami's actual roads; it
needs roads that are **valid**, **plausible for their terrain**, and **different between different
environments**. Those are three separate questions and this module asks them separately.

The comparison that matters most is *raw ML against the finished network*, both measured against
reality. Every earlier number in this project scored the model's output directly, which understates
the product by exactly the amount the procedural layer contributes — and the procedural layer is
where the dead-end defect gets fixed. Reporting only one of them tells you almost nothing about
whether the module works.

Nothing here is a pass/fail gate. The bands are wide and stated, because "close enough to look
right" has never been calibrated against a person looking at a generated world, and pretending
otherwise would put a number on a judgement nobody has made yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.structure import road_structure
from ..core.roads import RoadGraph

#: What a player notices. Density and junction spacing set how a place reads, orientation entropy
#: separates grid from organic, sinuosity is what makes a mountain road look like one, and the
#: dead-end ratio is what makes a network feel connected or feel like rubble.
GAME_MEASURES = (
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "orientation_entropy",
    "sinuosity_median",
    "dead_end_ratio",
)


@dataclass
class StageStats:
    """One stage of the pipeline, in the terms a game cares about."""

    name: str
    edges: int = 0
    components: int = 0
    dead_end_ratio: float = 0.0
    junctions: int = 0
    length_km: float = 0.0
    over_grade: int = 0

    def describe(self) -> str:
        return (
            f"{self.name:<12} {self.edges:>5} edges  {self.components:>4} comp  "
            f"{self.junctions:>4} junc  {self.length_km:>6.2f} km  "
            f"dead ends {self.dead_end_ratio:>5.0%}  over-grade {self.over_grade:>3}"
        )


def stage_stats(name: str, graph: RoadGraph, tile, *, grade_limit_pct: float = 12.0) -> StageStats:
    s = road_structure(graph, tile)
    return StageStats(
        name=name,
        edges=len(graph.edges),
        components=len(graph.connected_components()),
        dead_end_ratio=graph.dead_end_ratio,
        junctions=sum(1 for n in graph.nodes if graph.degree(n) >= 3),
        length_km=graph.total_length_m / 1000.0,
        over_grade=sum(
            1 for e in graph.edges.values()
            if e.grade_pct is not None and e.grade_pct > grade_limit_pct
        ),
    )


@dataclass
class SiteEvaluation:
    """Everything measured for one environment."""

    site: str
    tiles: int = 0
    stages: list[StageStats] = field(default_factory=list)
    real: dict[str, float] = field(default_factory=dict)
    generated: dict[str, float] = field(default_factory=dict)

    def ratio(self, measure: str) -> float:
        real = self.real.get(measure, float("nan"))
        return self.generated.get(measure, float("nan")) / real if real else float("nan")

    def within(self, measure: str, factor: float = 2.0) -> bool:
        r = self.ratio(measure)
        return not np.isnan(r) and 1 / factor <= r <= factor

    @property
    def plausibility(self) -> float:
        checked = [m for m in GAME_MEASURES if not np.isnan(self.ratio(m))]
        return sum(self.within(m) for m in checked) / len(checked) if checked else float("nan")

    def describe(self) -> str:
        lines = [f"{self.site}  ({self.tiles} tiles)"]
        lines += [f"  {s.describe()}" for s in self.stages]
        lines.append("  structural plausibility, generated against real:")
        for m in GAME_MEASURES:
            r = self.ratio(m)
            mark = "ok " if self.within(m) else "OFF"
            lines.append(
                f"    {mark} {m:<32} real {self.real.get(m, float('nan')):8.3f}  "
                f"generated {self.generated.get(m, float('nan')):8.3f}   x{r:.2f}"
            )
        lines.append(f"  plausible on {self.plausibility:.0%} of measures")
        return "\n".join(lines)


def ordering_across(sites: list[SiteEvaluation], measure: str):
    """Whether generated networks rank the environments the way reality does.

    The single most important number for a generation product. A model that is wrong everywhere
    but keeps the archetypes apart is useful — it needs calibration. One that is right on average
    and cannot tell a mountain valley from a plain is not, whatever its averages say, because
    environment-specific output is the entire value over existing procedural tools.
    """
    usable = [s for s in sites if not np.isnan(s.real.get(measure, float("nan")))]
    real_order = [s.site for s in sorted(usable, key=lambda s: s.real[measure])]
    gen_order = [s.site for s in sorted(usable, key=lambda s: s.generated.get(measure, 0.0))]
    return real_order == gen_order, real_order, gen_order


def summarise(sites: list[SiteEvaluation]) -> str:
    lines = ["", "archetype ordering — is the output environment-specific?"]
    kept = 0
    for measure in GAME_MEASURES:
        agrees, real_order, gen_order = ordering_across(sites, measure)
        kept += agrees
        lines.append(
            f"  {'ok ' if agrees else 'NO '} {measure:<32}"
            f" real {' < '.join(s[:9] for s in real_order)}"
            f"   generated {' < '.join(s[:9] for s in gen_order)}"
        )
    lines.append(f"\n{kept}/{len(GAME_MEASURES)} measures keep the archetypes in the right order")
    return "\n".join(lines)
