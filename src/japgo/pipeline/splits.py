"""Geographic train/validation/test splits.

Invariant 4: **splits are geographic, never random**. A random tile split is not a weak evaluation
here, it is an invalid one, for a concrete reason — tiles are 1 km core plus a 256 m halo, so
adjacent tiles literally share input pixels. A model evaluated across a random split has seen part
of its own test set as input.

The wider problem is spatial autocorrelation: random cross-validation on geospatial tasks can
report up to ~40% more optimistic scores than a spatial split (research doc §16.1). The failure is
silent and flattering, which is the worst combination.

So this module does three things:

1. Assigns tiles to splits by **site membership**, not by tile.
2. Discards a **buffer zone** of at least one tile between differently-assigned regions.
3. **Validates** the result, so the invariant is checkable rather than merely intended.

Point 3 is the one that matters. An invariant nothing can check is a comment.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..geo.tiling import Tile, parse_tile_id

MIN_BUFFER_TILES = 1
"""Adjacent tiles share halo pixels, so zero buffer leaks input data across the split boundary."""


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    BUFFER = "buffer"
    """Discarded. Present in the definition so the discard is auditable rather than invisible."""


class SplitError(RuntimeError):
    """Raised when a split would leak geography between folds."""


@dataclass(frozen=True)
class Site:
    """A named geographic region — the unit splits are assigned to."""

    name: str
    archetype: str
    tiles: frozenset[str]

    def __len__(self) -> int:
        return len(self.tiles)


@dataclass
class SplitDefinition:
    """An assignment of tiles to splits, with the buffer zone made explicit."""

    assignment: dict[str, Split] = field(default_factory=dict)
    sites: dict[str, Site] = field(default_factory=dict)
    buffer_tiles: int = MIN_BUFFER_TILES

    def tiles_in(self, split: Split) -> list[str]:
        return sorted(t for t, s in self.assignment.items() if s is split)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(s.value for s in self.assignment.values()))

    def split_of(self, tile_id: str) -> Split | None:
        return self.assignment.get(tile_id)

    def archetypes_in(self, split: Split) -> set[str]:
        """Archetypes represented in a split.

        Held-out sets must cover unseen *archetypes*, not merely unseen cities — otherwise
        "generalisation" only demonstrates transfer between similar places (research doc §16.1).
        """
        assigned = set(self.tiles_in(split))
        return {
            site.archetype
            for site in self.sites.values()
            if site.tiles & assigned
        }

    # ---------------------------------------------------------------------------------------

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "buffer_tiles": self.buffer_tiles,
            "sites": {
                name: {"archetype": site.archetype, "tiles": sorted(site.tiles)}
                for name, site in self.sites.items()
            },
            "assignment": {t: s.value for t, s in sorted(self.assignment.items())},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> SplitDefinition:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            assignment={t: Split(s) for t, s in data["assignment"].items()},
            sites={
                name: Site(name=name, archetype=v["archetype"], tiles=frozenset(v["tiles"]))
                for name, v in data.get("sites", {}).items()
            },
            buffer_tiles=int(data.get("buffer_tiles", MIN_BUFFER_TILES)),
        )


# ------------------------------------------------------------------------------------------------


def _chebyshev(a: Tile, b: Tile) -> int:
    """Tile-grid distance. Chebyshev because diagonal neighbours share halo pixels too."""
    return max(abs(a.ix - b.ix), abs(a.iy - b.iy))


def make_split(
    sites: dict[str, Site],
    assignment: dict[str, Split],
    *,
    buffer_tiles: int = MIN_BUFFER_TILES,
) -> SplitDefinition:
    """Assign tiles by site, then carve out the buffer zone.

    ``assignment`` maps **site name** to split — deliberately not tile id, because assigning
    individual tiles is the mistake this module exists to prevent.
    """
    if buffer_tiles < MIN_BUFFER_TILES:
        raise SplitError(
            f"buffer_tiles={buffer_tiles} is below the minimum of {MIN_BUFFER_TILES}. Adjacent "
            "tiles share halo pixels, so a zero buffer leaks input data across the boundary."
        )

    unknown = set(assignment) - set(sites)
    if unknown:
        raise SplitError(f"assignment references unknown sites: {sorted(unknown)}")

    missing = set(sites) - set(assignment)
    if missing:
        raise SplitError(
            f"sites with no split assignment: {sorted(missing)}. Assign every site explicitly; "
            "an unassigned site silently disappears from the dataset."
        )

    tile_split: dict[str, Split] = {}
    for site_name, split in assignment.items():
        for tile_id in sites[site_name].tiles:
            tile_split[tile_id] = split

    buffered = _apply_buffer(tile_split, buffer_tiles)
    return SplitDefinition(assignment=buffered, sites=dict(sites), buffer_tiles=buffer_tiles)


def _apply_buffer(tile_split: dict[str, Split], buffer_tiles: int) -> dict[str, Split]:
    """Reassign to BUFFER any tile within ``buffer_tiles`` of a differently-assigned tile."""
    parsed = {tid: parse_tile_id(tid) for tid in tile_split}
    out = dict(tile_split)

    by_split: dict[Split, list[tuple[str, Tile]]] = {}
    for tid, split in tile_split.items():
        by_split.setdefault(split, []).append((tid, parsed[tid]))

    for tid, tile in parsed.items():
        mine = tile_split[tid]
        for other_split, members in by_split.items():
            if other_split is mine:
                continue
            if any(_chebyshev(tile, other) <= buffer_tiles for _, other in members):
                out[tid] = Split.BUFFER
                break

    return out


def validate_split(definition: SplitDefinition) -> list[str]:
    """Check a split for geographic leakage. Returns problems; empty means clean.

    This is the function that turns invariant 4 from a comment into a property.
    """
    problems: list[str] = []
    evaluated = (Split.TRAIN, Split.VAL, Split.TEST)

    parsed = {
        tid: parse_tile_id(tid)
        for tid, split in definition.assignment.items()
        if split in evaluated
    }

    # 1. No two differently-assigned tiles closer than the buffer.
    by_split: dict[Split, list[tuple[str, Tile]]] = {}
    for tid, tile in parsed.items():
        by_split.setdefault(definition.assignment[tid], []).append((tid, tile))

    for split_a in evaluated:
        for split_b in evaluated:
            if split_a.value >= split_b.value:
                continue
            for tid_a, tile_a in by_split.get(split_a, []):
                for tid_b, tile_b in by_split.get(split_b, []):
                    distance = _chebyshev(tile_a, tile_b)
                    if distance <= definition.buffer_tiles:
                        problems.append(
                            f"{split_a.value} tile {tid_a} is {distance} tile(s) from "
                            f"{split_b.value} tile {tid_b}; buffer is "
                            f"{definition.buffer_tiles}. Adjacent tiles share halo pixels."
                        )
                        break
                else:
                    continue
                break

    # 2. A site must not span more than one evaluated split.
    for site in definition.sites.values():
        splits = {
            definition.assignment[t]
            for t in site.tiles
            if definition.assignment.get(t) in evaluated
        }
        if len(splits) > 1:
            problems.append(
                f"site {site.name!r} spans {sorted(s.value for s in splits)}; splits are assigned "
                "per site, never per tile"
            )

    # 3. Test must exercise an archetype, and ideally one train does not monopolise.
    test_archetypes = definition.archetypes_in(Split.TEST)
    if not test_archetypes and definition.tiles_in(Split.TEST):
        problems.append("test split has tiles but no identifiable archetype")

    return problems


def assert_valid(definition: SplitDefinition) -> None:
    """Raise if a split would leak geography."""
    problems = validate_split(definition)
    if problems:
        raise SplitError(
            "split would leak geography between folds:\n  " + "\n  ".join(problems)
        )
