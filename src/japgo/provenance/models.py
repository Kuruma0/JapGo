"""Typed model of the dataset provenance registry.

The registry (``data/provenance/registry.yaml``) is executable policy, not documentation. These
models exist so that a malformed or under-specified entry fails loudly at load time rather than
silently permitting an ingest it should have blocked.

See ``docs/data-provenance.md`` for field semantics and ``docs/phase0-research.md`` §6 for the
licensing analysis the tiers encode.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UsageTier(StrEnum):
    """How far a source is permitted to travel through the system."""

    PUBLIC = "public"
    """May be used anywhere, including in published artifacts."""

    INTERNAL_RESEARCH_ONLY = "internal-research-only"
    """May be used for development and evaluation; blocked from public artifacts."""

    QUARANTINED = "quarantined"
    """May not be ingested at all. Recorded so the exclusion is auditable."""


class OutputRole(StrEnum):
    """Whether a source's geometry may appear in shipped output.

    Distinct from :class:`UsageTier` on purpose. Training on OSM is permitted and its predictions
    are unencumbered; *transforming* OSM geometry into shipped output is a Derivative Database
    carrying ODbL share-alike. See research doc §6.1c.
    """

    REDISTRIBUTABLE_CORE = "redistributable_core"
    TRAINING_ONLY = "training_only"


class ShareAlike(StrEnum):
    """Copyleft family. Two incompatible families in one derivative database is the worst
    available licensing outcome, so the family is recorded rather than remembered."""

    NONE = "none"
    ODBL = "odbl"
    CC_BY_SA = "cc-by-sa"


# Closed vocabulary for `commercial_restrictions`. Free text here would make the
# non-commercial exclusion check unenforceable, which is how it was originally written and why
# the first validation run produced a false positive on aw3d30.
COMMERCIAL_OK = "none"
COMMERCIAL_UNRESOLVED = {"unresolved", "ambiguous"}


class Source(BaseModel):
    """One external dataset or media source."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    name: str
    source: str
    url: str

    license: str
    share_alike: ShareAlike = ShareAlike.NONE

    geographic_coverage: str = ""
    resolution: str = ""

    allowed_uses: list[str] = Field(default_factory=list)
    commercial_restrictions: str = COMMERCIAL_OK

    attribution_required: bool = False
    attribution_string: str | None = None

    training_restrictions: str = ""
    redistribution_restrictions: str = ""
    derivative_restrictions: str = ""

    usage_tier: UsageTier
    output_role: OutputRole | None = None
    layer_isolation: str = "not_required"

    version_pin: str | None = None

    @field_validator("commercial_restrictions", mode="before")
    @classmethod
    def _strip_inline_comment(cls, v: object) -> object:
        """YAML inline comments after a scalar are part of the value when quoted oddly.

        Normalising here keeps :meth:`is_commercially_clear` a simple equality test rather than a
        fuzzy string match — the distinction that made the original check unenforceable.
        """
        if isinstance(v, str):
            return v.split("#")[0].strip()
        return v

    @property
    def is_commercially_clear(self) -> bool:
        return self.commercial_restrictions.strip().lower() == COMMERCIAL_OK

    @property
    def is_ingestible(self) -> bool:
        return self.usage_tier is not UsageTier.QUARANTINED

    @property
    def may_ship_geometry(self) -> bool:
        return self.output_role is OutputRole.REDISTRIBUTABLE_CORE


class RegistryPolicy(BaseModel):
    """Project-wide settings that change tier and role meaning across the whole registry.

    Kept explicit rather than implied, because changing one of these invalidates assignments in
    every entry rather than in one.
    """

    model_config = ConfigDict(extra="allow")

    commercial_intent: bool = True
    """When true, no source with commercial restrictions may be ingested at any tier."""

    modes_supported: list[str] = Field(default_factory=lambda: ["reconstruction", "generation"])


class Registry(BaseModel):
    """The parsed registry."""

    model_config = ConfigDict(extra="allow")

    registry_version: int
    last_reviewed: str | None = None
    reviewed_by: str | None = None
    sources: list[Source]

    policy: RegistryPolicy = Field(default_factory=RegistryPolicy)

    @field_validator("last_reviewed", mode="before")
    @classmethod
    def _date_to_string(cls, v: object) -> object:
        """YAML resolves unquoted ``2026-08-10`` to a ``datetime.date``.

        Normalising to ISO text keeps the field comparable and JSON-serialisable without forcing
        registry authors to quote dates.
        """
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        return v

    @field_validator("sources")
    @classmethod
    def _unique_ids(cls, v: list[Source]) -> list[Source]:
        seen: set[str] = set()
        dupes = {s.id for s in v if s.id in seen or seen.add(s.id)}  # type: ignore[func-returns-value]
        if dupes:
            raise ValueError(f"duplicate source ids in registry: {sorted(dupes)}")
        return v

    def get(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def require(self, source_id: str) -> Source:
        """Look up a source, refusing unknown ids.

        This is the enforcement point for the project's first invariant: a source with no registry
        entry may not be read by any pipeline stage.
        """
        found = self.get(source_id)
        if found is None:
            known = ", ".join(sorted(s.id for s in self.sources))
            raise UnregisteredSourceError(
                f"source {source_id!r} has no registry entry and may not be used. "
                f"Add an entry to data/provenance/registry.yaml first. Known sources: {known}"
            )
        return found

    @property
    def ingestible(self) -> list[Source]:
        return [s for s in self.sources if s.is_ingestible]

    @property
    def redistributable_core(self) -> list[Source]:
        return [s for s in self.sources if s.may_ship_geometry]


class UnregisteredSourceError(LookupError):
    """Raised when a pipeline stage references a source with no registry entry."""


class ProvenanceViolation(RuntimeError):
    """Raised when an operation would violate registry policy."""
