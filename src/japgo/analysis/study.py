"""Running the Phase 3 study over a built corpus.

The glue between the tile store and :mod:`japgo.analysis.correlate`. Kept separate from both so
that the statistics can be tested without a corpus and the corpus can be read without importing
the statistics.

Site membership comes from the split definition, not from a tile's coordinates. The split is
already the project's authority on which tiles belong together (invariant 4), and deriving
grouping a second way here would eventually disagree with it — at which point the confidence
intervals would be wrong in a way nobody would notice.
"""

from __future__ import annotations

from pathlib import Path

from ..pipeline.splits import SplitDefinition
from ..pipeline.store import list_tiles, read_tile
from .correlate import Study, TileObservation, correlate
from .features import coverage, environmental_features
from .structure import road_structure

MIN_COVERAGE = 0.5
"""Tiles observed over less than half their core do not vote.

Their feature values are defined but dominated by whatever the fill happens to be adjacent to.
Better to lose a tile than to let it cast a vote it cannot support.
"""

UNASSIGNED_SITE = "unassigned"


def observe(
    root: Path,
    *,
    split: SplitDefinition | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[list[TileObservation], list[str]]:
    """Read every tile under ``root`` into paired predictor/response vectors.

    Returns the observations and a list of human-readable skips — an analysis that silently drops
    half its corpus is indistinguishable from one that had half the corpus to begin with.
    """
    site_of: dict[str, str] = {}
    if split is not None:
        for name, site in split.sites.items():
            for tile_id in site.tiles:
                site_of[tile_id] = name

    observations: list[TileObservation] = []
    skipped: list[str] = []

    for tile_id in list_tiles(root):
        bundle = read_tile(root, tile_id)

        observed = coverage(bundle)
        if observed < min_coverage:
            skipped.append(f"{tile_id}: coverage {observed:.0%} below {min_coverage:.0%}")
            continue

        if bundle.roads is None:
            skipped.append(f"{tile_id}: no road graph — nothing to correlate against")
            continue

        observations.append(
            TileObservation(
                tile_id=tile_id,
                site=site_of.get(tile_id, UNASSIGNED_SITE),
                features=environmental_features(bundle),
                metrics=road_structure(bundle.roads, bundle.tile),
            )
        )

    return observations, skipped


def run_study(
    root: Path,
    *,
    split_path: Path | None = None,
    iterations: int = 2000,
    seed: int = 0,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[Study, list[str]]:
    """Read a corpus and correlate it. The one command Phase 3's exit criterion needs."""
    split = SplitDefinition.read(split_path) if split_path and Path(split_path).is_file() else None
    observations, skipped = observe(root, split=split, min_coverage=min_coverage)
    study = correlate(observations, iterations=iterations, seed=seed)
    return study, skipped
