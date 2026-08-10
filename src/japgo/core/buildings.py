"""Building representation and the type taxonomy.

Three levels, deliberately separated so that ML operates on the middle one (research doc §9):

1. **Instance** — this module. One building, its geometry and attributes.
2. **Morphometric vector** — derived, computed by momepy (Phase 3).
3. **Archetype** — learned by clustering (Phase 6), never authored.

Every attribute carries its provenance, so a model is never trained on its own predictions by
accident.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_TAXONOMY_PATH = Path("config/building_taxonomy.yaml")


class LabelSource(StrEnum):
    """Where an attribute came from. Priority order per research doc §10."""

    PLATEAU = "plateau"
    """Authoritative municipal attribute."""

    OSM = "osm"
    """Crowd-sourced; uneven. Gap-filling and out-of-coverage training labels only."""

    PREDICTED = "predicted"
    """Model output. Never a training target."""

    UNKNOWN = "unknown"


class Taxonomy(BaseModel):
    """The versioned building type taxonomy."""

    model_config = ConfigDict(frozen=True)

    taxonomy_version: int
    coarse: list[str]
    fine: dict[str, dict]
    plateau_usage: dict[str, dict]
    osm_building: dict[str, object]

    @field_validator("osm_building", "plateau_usage", mode="before")
    @classmethod
    def _keys_must_be_strings(cls, v: object) -> object:
        """Reject YAML boolean coercion of tag keys.

        YAML 1.1 resolves bare ``yes``/``no``/``on``/``off`` to booleans. ``building=yes`` is the
        most common building tag in OSM, so an unquoted key silently breaks the most frequent
        case. Failing loudly here beats mis-labelling at scale.
        """
        if isinstance(v, dict):
            bad = [k for k in v if not isinstance(k, str)]
            if bad:
                raise ValueError(
                    f"non-string keys {bad!r} — quote them in the YAML "
                    "(YAML 1.1 coerces bare yes/no/on/off to booleans)"
                )
        return v

    def coarse_for(self, fine: str) -> str:
        entry = self.fine.get(fine)
        if entry is None:
            return "unknown"
        return str(entry.get("coarse", "unknown"))

    def from_plateau_usage(self, code: str) -> tuple[str, bool]:
        """Map a PLATEAU usage code to a fine type.

        Returns ``(fine_type, was_mapped)``. An unmapped code resolves to ``unknown`` with
        ``was_mapped=False`` so the caller can surface it — an unmapped code is a taxonomy gap to
        close, not a data error to swallow.
        """
        entry = self.plateau_usage.get(str(code).strip())
        if entry is None:
            return "unknown", False
        return str(entry["fine"]), True

    def from_osm_building(self, value: str) -> tuple[str, bool]:
        entry = self.osm_building.get(str(value).strip().lower())
        if entry is None:
            return "unknown", False
        if isinstance(entry, dict):
            return str(entry["fine"]), True
        return str(entry), True

    @property
    def low_confidence_codes(self) -> list[str]:
        """PLATEAU codes seeded by inference rather than verified documentation.

        Reconcile these against the shipped codelist label on first real ingest.
        """
        return sorted(
            code
            for code, entry in self.plateau_usage.items()
            if entry.get("confidence") != "high"
        )


def _find_taxonomy(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_TAXONOMY_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find {DEFAULT_TAXONOMY_PATH} walking up from {here}")


@lru_cache(maxsize=4)
def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or _find_taxonomy()
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Building(BaseModel):
    """One building instance.

    Geometry is a footprint ring in the tile's metric CRS: ``[(x, y), ...]``, closed. Heights are
    metres. Nothing here describes appearance — materials, textures and meshes are applied at
    export time against a declared LOD (invariant 10).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    """Registry source id. Present on every feature so Collective Database status is preserved."""

    footprint: list[tuple[float, float]]

    height_m: float | None = None
    storeys_above_ground: int | None = None
    storeys_below_ground: int | None = None

    fine_type: str = "unknown"
    coarse_type: str = "unknown"
    type_source: LabelSource = LabelSource.UNKNOWN

    year_of_construction: int | None = None
    raw_usage_code: str | None = None

    lod: int | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return len(self.footprint) >= 4 and self.footprint[0] == self.footprint[-1]

    @property
    def area_m2(self) -> float:
        """Shoelace area of the footprint. Metric CRS is assumed and enforced upstream."""
        pts = self.footprint
        if len(pts) < 3:
            return 0.0
        total = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=False):
            total += x0 * y1 - x1 * y0
        return abs(total) / 2.0

    @property
    def perimeter_m(self) -> float:
        pts = self.footprint
        if len(pts) < 2:
            return 0.0
        return sum(
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False)
        )

    @property
    def estimated_height_m(self) -> float | None:
        """Height, falling back to storey count where a measured height is absent.

        3.0 m per storey is the conventional Japanese residential assumption. Recorded as an
        estimate rather than silently written into ``height_m`` so downstream code can tell a
        measurement from a guess.
        """
        if self.height_m is not None:
            return self.height_m
        if self.storeys_above_ground:
            return self.storeys_above_ground * 3.0
        return None
