"""Patch sampling over a built corpus.

Phase 4 trains on 512² crops rather than whole tiles, because that is what closes the 16 GB budget
(research doc §20.2) — a ~4× activation saving, and the largest single one available.

Deliberately numpy-only. Torch lives in the `ml` extra and the provenance and pipeline layers must
stay installable without it; keeping the index here means patch selection can be tested in CI on a
machine with no GPU stack at all.

**On leakage.** Crops are taken over the full read extent, halo included, so the network sees the
context the halo exists to provide. Halo pixels are shared with the *neighbouring tile*, which is
why invariant 4 forbids random tile splits — but here splits are by site and the MVP sites are
50–124 km apart, so no halo pixel can be shared across a split boundary. :func:`assert_no_overlap`
checks that rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geo.tiling import parse_tile_id
from ..pipeline.splits import Split, SplitDefinition
from ..pipeline.store import list_tiles, read_tile

DEFAULT_CROP = 512
MIN_VALID_FRACTION = 0.25
"""A crop with less real observation than this is mostly fill, and training on fill teaches the
model to reproduce voids. Cheaper to drop it here than to explain a metric in Phase 5."""


@dataclass
class Patch:
    tile_id: str
    row: int
    col: int


@dataclass
class Fold:
    """One train/evaluate division, named for what it holds out."""

    name: str
    train_sites: tuple[str, ...]
    held_out: str
    train_tiles: list[str] = field(default_factory=list)
    eval_tiles: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.name}: train {'+'.join(self.train_sites)} ({len(self.train_tiles)} tiles) "
            f"-> held out {self.held_out} ({len(self.eval_tiles)} tiles)"
        )


def sites_of(split: SplitDefinition) -> dict[str, list[str]]:
    """Site name -> tile ids, from the split definition."""
    return {name: sorted(site.tiles) for name, site in split.sites.items()}


def configured_fold(split: SplitDefinition) -> Fold:
    """The train/val/test assignment as written in ``config/sites.yaml``."""
    by_site = sites_of(split)
    train, held = [], []
    train_sites, held_site = [], "?"
    for name, tiles in by_site.items():
        assignment = {split.assignment.get(t) for t in tiles}
        if Split.TEST in assignment:
            held, held_site = tiles, name
        elif Split.TRAIN in assignment:
            train += tiles
            train_sites.append(name)
    return Fold("configured", tuple(train_sites), held_site, train, held)


def leave_one_site_out(split: SplitDefinition) -> list[Fold]:
    """One fold per site, each holding that site out entirely.

    This is what makes the first Phase 4 objective reachable. The configured split trains on the
    Hamamatsu plain alone, where slope barely varies — a model cannot learn a response to terrain
    from data that holds terrain nearly constant, so "does it learn environment-responsive road
    structure?" would be unanswerable rather than answered negatively. Rotating the held-out site
    gives every fold both steep and flat training ground while still evaluating on an **unseen
    archetype**, which §16.1 requires and a random tile split would destroy.
    """
    by_site = sites_of(split)
    folds = []
    for held in sorted(by_site):
        others = tuple(s for s in sorted(by_site) if s != held)
        folds.append(
            Fold(
                name=f"holdout_{held}",
                train_sites=others,
                held_out=held,
                train_tiles=[t for s in others for t in by_site[s]],
                eval_tiles=by_site[held],
            )
        )
    return folds


def assert_no_overlap(fold: Fold, *, buffer_tiles: int = 1) -> None:
    """Fail if any training tile is within ``buffer_tiles`` of a held-out tile.

    Adjacent tiles share halo pixels, so a training tile touching an evaluation tile leaks input
    directly across the split. Checked rather than assumed: the sites happen to be far apart, but
    "happens to be" is not a property anyone should rely on after the corpus grows.
    """
    train = [parse_tile_id(t) for t in fold.train_tiles]
    held = [parse_tile_id(t) for t in fold.eval_tiles]
    if not train or not held:
        return

    shared = {t.id for t in train} & {t.id for t in held}
    if shared:
        raise ValueError(f"{fold.name}: {len(shared)} tile(s) in both sides: {sorted(shared)[:3]}")

    for a in train:
        for b in held:
            if max(abs(a.ix - b.ix), abs(a.iy - b.iy)) <= buffer_tiles:
                raise ValueError(
                    f"{fold.name}: {a.id} (train) is within {buffer_tiles} tile(s) of {b.id} "
                    "(held out); they share halo pixels"
                )


def index_patches(
    root: Path,
    tile_ids: list[str],
    *,
    crop: int = DEFAULT_CROP,
    stride: int | None = None,
    min_valid: float = MIN_VALID_FRACTION,
) -> list[Patch]:
    """Every crop position worth training on, as a flat list.

    Precomputed so an epoch is a deterministic permutation of a fixed set rather than a stream of
    random draws — an experiment that cannot be re-run from its seed is a failed experiment
    (invariant 8).
    """
    stride = stride or crop // 2
    patches: list[Patch] = []

    for tile_id in tile_ids:
        bundle = read_tile(root, tile_id)
        valid = bundle.channel("valid")
        rows, cols = valid.shape
        if rows < crop or cols < crop:
            continue
        for r in range(0, rows - crop + 1, stride):
            for c in range(0, cols - crop + 1, stride):
                if valid[r : r + crop, c : c + crop].mean() >= min_valid:
                    patches.append(Patch(tile_id, r, c))
    return patches


class PatchLoader:
    """Reads patches, caching whole tiles because a tile is the unit of I/O.

    A 1512² × 19 float32 tile is ~165 MB; a corpus of them is not, so the cache is bounded and
    least-recently-used. Without it every patch re-reads and re-decompresses its tile.
    """

    def __init__(self, root: Path, *, max_tiles: int = 4) -> None:
        self.root = Path(root)
        self.max_tiles = max_tiles
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._order: list[str] = []

    def tile(self, tile_id: str) -> tuple[np.ndarray, np.ndarray]:
        if tile_id in self._cache:
            self._order.remove(tile_id)
            self._order.append(tile_id)
            return self._cache[tile_id]

        bundle = read_tile(self.root, tile_id)
        if bundle.targets is None:
            raise ValueError(f"{tile_id} carries no targets; it is not trainable")
        entry = (bundle.stack, bundle.targets)

        self._cache[tile_id] = entry
        self._order.append(tile_id)
        while len(self._order) > self.max_tiles:
            self._cache.pop(self._order.pop(0), None)
        return entry

    def read(self, patch: Patch, crop: int = DEFAULT_CROP) -> tuple[np.ndarray, np.ndarray]:
        stack, targets = self.tile(patch.tile_id)
        r, c = patch.row, patch.col
        return (
            stack[:, r : r + crop, c : c + crop].copy(),
            targets[:, r : r + crop, c : c + crop].copy(),
        )


def corpus_tiles(root: Path) -> list[str]:
    return list_tiles(root)
