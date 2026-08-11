"""Associating environment with road structure, defensibly.

Phase 3's exit criterion asks for a *ranked, quantified* list of which environmental features
predict road structure, **with the null results stated**. That last clause drives the design: a
feature that turns out not to matter has to come back as a confident "no", not as an absence from
the table. Silence and a null result look identical in a ranking that only reports winners.

Three choices worth defending, because each has an obvious alternative that is wrong here:

**Spearman, not Pearson.** The expected relationships are monotone but nowhere near linear —
road density against slope should fall off sharply and then flatten, not descend in a straight
line. Rank correlation measures the monotone part without pretending to know the functional form.

**Cluster bootstrap over sites, not tiles.** This is the same principle as invariant 4, applied to
inference instead of training. Adjacent tiles share a 256 m halo and are drawn from the same
settlement, so resampling tiles independently would treat 100 tiles from three sites as 100
independent observations. They are closer to three. Resampling whole sites keeps the confidence
interval honest, and it will be *wide* — correctly so. A narrow interval computed the naive way
would be a fabrication.

**An explicit "insufficient" verdict.** With three MVP sites the cluster bootstrap has very little
to work with. Rather than emit a confident-looking interval from two or three clusters, the study
says so. docs/decision-log.md already records that one tile is an anecdote; this is the machinery
that stops the corpus version of that mistake being made quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import ENVIRONMENTAL_FEATURES
from .structure import ROAD_STRUCTURE_METRICS

MIN_OBSERVATIONS = 8
"""Below this many tiles, no verdict is offered at all."""

MIN_CLUSTERS = 3
"""Below this many independent sites, a confidence interval is not meaningful.

Set to 3 because that is exactly the MVP configuration — the study should be able to *run* on the
three sites and still tell you the interval is thin, rather than refusing or, worse, pretending.
"""

NEGLIGIBLE_RHO = 0.3
"""The largest rank correlation the study is willing to call "no relationship".

Below this, a monotone association explains too little to matter for the §1.3 thesis. An interval
lying entirely inside ±0.3 has therefore *excluded* anything worth acting on, which is a real
finding. An interval that merely straddles zero has excluded nothing.
"""


@dataclass(frozen=True)
class TileObservation:
    """One tile's paired predictor and response vectors."""

    tile_id: str
    site: str
    features: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class Association:
    """One feature-versus-metric relationship, with its uncertainty and its verdict."""

    feature: str
    metric: str
    rho: float
    n: int
    clusters: int
    ci: tuple[float, float] | None
    verdict: str
    """``supported`` | ``null`` | ``inconclusive`` | ``insufficient``.

    The distinction between the middle two is the one that matters and the one that is easiest to
    get wrong. ``null`` means the interval is tight around zero — the study looked and there is
    nothing there. ``inconclusive`` means the interval spans zero *and* something substantial —
    the study looked and cannot yet say. Collapsing them would let "we need more sites" masquerade
    as "this feature does not matter", which is precisely the claim Phase 3 is supposed to settle.
    """

    @property
    def strength(self) -> float:
        """Magnitude, for ranking. NaN sorts last rather than raising."""
        return 0.0 if np.isnan(self.rho) else abs(self.rho)

    def describe(self) -> str:
        interval = "n/a" if self.ci is None else f"[{self.ci[0]:+.2f}, {self.ci[1]:+.2f}]"
        return (
            f"{self.feature:<28} {self.metric:<32} rho={self.rho:+.3f}  95% CI {interval:<18} "
            f"n={self.n:<4} sites={self.clusters}  {self.verdict}"
        )


@dataclass
class Study:
    """The full cross-product of predictors against responses."""

    associations: list[Association] = field(default_factory=list)
    tiles: int = 0
    sites: tuple[str, ...] = ()

    def ranked(self) -> list[Association]:
        """Strongest first. Ties broken by name so the report is reproducible (invariant 8)."""
        return sorted(
            self.associations,
            key=lambda a: (-a.strength, a.feature, a.metric),
        )

    def supported(self) -> list[Association]:
        return [a for a in self.ranked() if a.verdict == "supported"]

    def nulls(self) -> list[Association]:
        return [a for a in self.ranked() if a.verdict == "null"]

    def inconclusive(self) -> list[Association]:
        return [a for a in self.ranked() if a.verdict == "inconclusive"]

    def report(self, limit: int | None = None) -> str:
        """A plain-text findings table: what held, what did not, and what is still open."""

        def section(title: str, rows: list[Association], empty: str) -> list[str]:
            out = [f"{title} ({len(rows)}):"]
            out.extend(f"  {a.describe()}" for a in (rows[:limit] if limit else rows))
            if not rows:
                out.append(f"  {empty}")
            elif limit and len(rows) > limit:
                out.append(f"  ... and {len(rows) - limit} more")
            return out

        lines = [
            "Phase 3 — environmental predictors of road structure",
            f"{self.tiles} tiles across {len(self.sites)} sites: {', '.join(self.sites) or 'none'}",
            "",
        ]

        lines += section(
            "Supported", self.supported(), "none — no association's interval excluded zero"
        )
        lines += ["", *section(
            f"Null results — interval inside ±{NEGLIGIBLE_RHO}, so the study looked and there is "
            "nothing there",
            self.nulls(),
            "none",
        )]
        lines += ["", *section(
            "Inconclusive — interval spans zero and something substantial. NOT a null result",
            self.inconclusive(),
            "none",
        )]

        insufficient = [a for a in self.associations if a.verdict == "insufficient"]
        if insufficient:
            lines += [
                "",
                f"Insufficient data ({len(insufficient)}): a feature or metric was absent, or "
                "there were too few tiles or independent sites to estimate an interval at all.",
            ]

            # Show the point estimates anyway, clearly marked. They are not findings and must
            # never be quoted as such — but a run that computed them and displayed nothing is
            # useless for the thing a small run is actually for, which is checking that the
            # pipeline produces sane numbers before spending days building a corpus.
            estimated = [a for a in self.ranked() if a.verdict == "insufficient" and a.strength]
            if estimated:
                shown = estimated[: limit or 10]
                lines += [
                    "",
                    "  Strongest point estimates among them — NOT findings, no interval was "
                    "estimated:",
                    *(
                        f"    {a.feature:<28} {a.metric:<32} rho={a.rho:+.3f}  n={a.n}"
                        for a in shown
                    ),
                ]

        n_sites = len(self.sites)
        if n_sites < MIN_CLUSTERS:
            lines += [
                "",
                f"Note: {n_sites} site{'' if n_sites == 1 else 's'} — below the {MIN_CLUSTERS} "
                "needed to estimate an interval at all, so every row above is unverdicted "
                "regardless of how many tiles it covers. Pass --split, or build more sites.",
            ]
        elif n_sites == MIN_CLUSTERS and self.inconclusive():
            lines += [
                "",
                f"Note: {n_sites} sites is the floor for an interval at all, so intervals here "
                "are wide by construction. The inconclusive rows are limited by the number of "
                "independent sites, not the number of tiles — adding tiles to the same three "
                "sites will not narrow them much.",
            ]

        return "\n".join(lines)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, tie-aware, over pairs where both values are present.

    Hand-rolled rather than taken from scipy so that the analysis package carries no dependency
    the provenance layer does not already have — the same reasoning that keeps the MVT reader in
    :mod:`japgo.sources.meshindex` free of protobuf.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    both = ~(np.isnan(x) | np.isnan(y))
    if both.sum() < 3:
        return float("nan")

    rx, ry = _rank(x[both]), _rank(y[both])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denominator <= 0:
        # One side is constant — every value tied. Undefined, not zero.
        return float("nan")
    return float((rx * ry).sum() / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering that is not in the data."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _cluster_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile CI from resampling whole clusters with replacement."""
    unique = np.unique(clusters)
    if unique.size < MIN_CLUSTERS:
        return None

    rng = np.random.default_rng(seed)
    index_by_cluster = [np.flatnonzero(clusters == c) for c in unique]

    estimates = []
    for _ in range(iterations):
        picked = rng.integers(0, unique.size, size=unique.size)
        rows = np.concatenate([index_by_cluster[p] for p in picked])
        rho = spearman(x[rows], y[rows])
        if not np.isnan(rho):
            estimates.append(rho)

    if len(estimates) < iterations // 4:
        # Mostly degenerate resamples — usually one site dominating. No usable interval.
        return None
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def _verdict(rho: float, ci: tuple[float, float] | None) -> str:
    """Classify one association from its interval, not from its point estimate.

    A strong rho with an interval spanning zero is not a finding, and a weak rho with a tight
    interval is.
    """
    if ci is None or np.isnan(rho):
        return "insufficient"
    if ci[0] > 0 or ci[1] < 0:
        return "supported"
    if abs(ci[0]) <= NEGLIGIBLE_RHO and abs(ci[1]) <= NEGLIGIBLE_RHO:
        return "null"
    return "inconclusive"


def correlate(
    observations: list[TileObservation],
    *,
    features: tuple[str, ...] = ENVIRONMENTAL_FEATURES,
    metrics: tuple[str, ...] = ROAD_STRUCTURE_METRICS,
    iterations: int = 2000,
    seed: int = 0,
    min_observations: int = MIN_OBSERVATIONS,
) -> Study:
    """Run every predictor against every response and rank what comes back.

    ``seed`` is explicit and defaulted so a study is re-runnable from its config alone
    (invariant 8). Two runs over the same tiles must produce the same table, or the table is not
    evidence of anything.
    """
    sites = tuple(sorted({o.site for o in observations}))
    study = Study(tiles=len(observations), sites=sites)
    if not observations:
        return study

    clusters = np.array([o.site for o in observations])

    for feature in features:
        x = np.array([o.features.get(feature, float("nan")) for o in observations])
        for metric in metrics:
            y = np.array([o.metrics.get(metric, float("nan")) for o in observations])

            paired = ~(np.isnan(x) | np.isnan(y))
            n = int(paired.sum())
            present_clusters = int(np.unique(clusters[paired]).size) if n else 0
            rho = spearman(x, y)

            ci = None
            if n >= min_observations and present_clusters >= MIN_CLUSTERS:
                ci = _cluster_bootstrap_ci(
                    x[paired],
                    y[paired],
                    clusters[paired],
                    iterations=iterations,
                    seed=seed,
                )

            verdict = _verdict(rho, ci)

            study.associations.append(
                Association(
                    feature=feature,
                    metric=metric,
                    rho=rho,
                    n=n,
                    clusters=present_clusters,
                    ci=ci,
                    verdict=verdict,
                )
            )

    return study
