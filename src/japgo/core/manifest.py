"""Tile manifests.

Every tile carries a manifest recording which sources contributed to it, at which versions, under
which registry. This is what makes the pipeline auditable and what lets the exporter decide
whether an artifact may be redistributed under attribution alone.

The manifest is also the reproducibility record required by research doc §44: an experiment that
cannot be re-run from its recorded inputs is a failed experiment regardless of its numbers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..provenance import OutputRole, Registry, SourceGate

SCHEMA_VERSION = 1


class SourceRecord(BaseModel):
    """One source's contribution to a tile."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    version: str | None = None
    layers: list[str] = Field(default_factory=list)
    retrieved_at: str | None = None
    note: str | None = None


class TileManifest(BaseModel):
    """Provenance record for a single tile."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    tile_id: str
    zone: int
    crs: str

    core_size_m: float
    halo_m: float

    sources: list[SourceRecord] = Field(default_factory=list)

    registry_hash: str | None = None
    preprocessing_version: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def source_ids(self) -> list[str]:
        return [s.source_id for s in self.sources]

    def add(self, record: SourceRecord) -> None:
        if record.source_id in self.source_ids:
            raise ValueError(
                f"source {record.source_id!r} already recorded for tile {self.tile_id}; "
                "merge layers into the existing record rather than adding a second one"
            )
        self.sources.append(record)

    # ---------------------------------------------------------------------------------------

    def redistribution_class(self, registry: Registry) -> str:
        """Classify what may be done with artifacts derived from this tile.

        ``attribution-only`` — every contributing source is in the redistributable core.
        ``share-alike``      — at least one training_only (ODbL) source contributed.

        Computed rather than declared, so it cannot drift from what actually went in.
        """
        for sid in self.source_ids:
            src = registry.get(sid)
            if src is None or src.output_role is not OutputRole.REDISTRIBUTABLE_CORE:
                return "share-alike"
        return "attribution-only"

    def attribution(self, gate: SourceGate) -> list[str]:
        return gate.attribution_for(self.source_ids)

    def verify(self, gate: SourceGate) -> None:
        """Re-check this tile against current registry policy.

        Called at export time. A tile that was legitimate when built can become illegitimate if a
        licence is later reclassified, and the failure should surface then rather than never.
        """
        for sid in self.source_ids:
            gate.assert_ingestible(sid)

    # ---------------------------------------------------------------------------------------

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> TileManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: manifest schema_version {data.get('schema_version')!r} "
                f"!= supported {SCHEMA_VERSION}"
            )
        return cls.model_validate(data)
