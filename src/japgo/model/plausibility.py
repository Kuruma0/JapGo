"""Is the generated network *plausible for its environment*, even where it is not *correct*?

APLS and TOPO ask whether the model reproduced a specific real network. That is the right question
for reconstruction and the wrong one for generation: a game does not need Atami's actual roads, it
needs roads that read as belonging to Atami's terrain. Research doc §38 says this outright — "a
structurally excellent network that differs from the real one is a success" — and §16.2 asks for
distributional distance between generated and real morphometrics as a first-class measure.

This module answers that narrower question in two ways, and the second matters more.

**Per-metric distributional agreement.** For each structural measure, how does the distribution
over predicted tiles compare with the distribution over real ones? A ratio near 1 means the
generated networks have the right character on that axis, whatever they did to any single tile.

**Archetype ordering.** Real road density falls plain → coast → valley, and sinuosity rises the
other way. If the predictions preserve that ordering, the model is producing environment-specific
output even where every individual tile is wrong. If they do not, the output is generic — and for
a generation product that is the more damaging failure, because it is the whole value proposition.

Ordering is scored on *ranks*, not values. A model whose densities are uniformly half of reality
is far more useful to a generator than one whose densities are right on average and unordered
between archetypes: the first needs a scale factor, the second needs a different model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.structure import road_structure

#: The measures a player would notice. Deliberately not the full §16.2 suite — density and junction
#: spacing govern how a place reads, orientation entropy separates grid from organic, and sinuosity
#: is what makes a mountain road look like one.
PLAUSIBILITY_METRICS = (
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "orientation_entropy",
    "sinuosity_median",
    "dead_end_ratio",
)


@dataclass(frozen=True)
class MetricAgreement:
    metric: str
    real: float
    predicted: float

    @property
    def ratio(self) -> float:
        return self.predicted / self.real if self.real else float("nan")

    @property
    def plausible(self) -> bool:
        """Within a factor of two, in either direction.

        A loose bar on purpose. Nothing calibrates what "close enough to look right" means yet, and
        a tight threshold would imply a precision this project has not earned. Factor-of-two is the
        level at which a difference stops being a scale factor and starts being a different kind of
        place.
        """
        return not np.isnan(self.ratio) and 0.5 <= self.ratio <= 2.0

    def describe(self) -> str:
        mark = "ok " if self.plausible else "OFF"
        return (
            f"{mark} {self.metric:<32} real {self.real:8.3f}   predicted {self.predicted:8.3f}"
            f"   x{self.ratio:.2f}"
        )


@dataclass
class SitePlausibility:
    site: str
    tiles: int
    agreements: list[MetricAgreement] = field(default_factory=list)

    @property
    def score(self) -> float:
        checked = [a for a in self.agreements if not np.isnan(a.ratio)]
        return sum(a.plausible for a in checked) / len(checked) if checked else float("nan")

    def value(self, metric: str, *, predicted: bool) -> float:
        for a in self.agreements:
            if a.metric == metric:
                return a.predicted if predicted else a.real
        return float("nan")


def compare(predicted_graphs, real_graphs, tiles, site: str) -> SitePlausibility:
    """Median structural measures over a site's tiles, predicted against real."""
    predicted = [road_structure(g, t) for g, t in zip(predicted_graphs, tiles, strict=True)]
    real = [road_structure(g, t) for g, t in zip(real_graphs, tiles, strict=True)]

    out = SitePlausibility(site=site, tiles=len(tiles))
    for metric in PLAUSIBILITY_METRICS:
        out.agreements.append(
            MetricAgreement(
                metric=metric,
                real=float(np.nanmedian([s[metric] for s in real])),
                predicted=float(np.nanmedian([s[metric] for s in predicted])),
            )
        )
    return out


def ordering_preserved(sites: list[SitePlausibility], metric: str) -> tuple[bool, list[str], list[str]]:
    """Whether predictions rank the archetypes the way reality does, on one metric.

    The generation-relevant test. Returns ``(agrees, real_order, predicted_order)`` so a
    disagreement can be read rather than merely counted.
    """
    usable = [s for s in sites if not np.isnan(s.value(metric, predicted=False))]
    real_order = [s.site for s in sorted(usable, key=lambda s: s.value(metric, predicted=False))]
    pred_order = [s.site for s in sorted(usable, key=lambda s: s.value(metric, predicted=True))]
    return real_order == pred_order, real_order, pred_order
