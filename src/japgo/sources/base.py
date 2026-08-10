"""The source adapter contract.

Every adapter passes through the provenance gate before reading a byte. That is enforced here, in
:meth:`SourceAdapter.open`, rather than left to each adapter to remember.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..core.manifest import SourceRecord
from ..provenance import Source, SourceGate


@dataclass
class ReadResult:
    """What an adapter returns: data plus the provenance record describing it."""

    layers: dict[str, list] = field(default_factory=dict)
    record: SourceRecord | None = None
    warnings: list[str] = field(default_factory=list)

    def count(self, layer: str) -> int:
        return len(self.layers.get(layer, []))

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.layers.values())


class SourceAdapter(abc.ABC):
    """Base class for dataset adapters.

    Subclasses declare ``source_id`` and implement :meth:`read`. They must not bypass
    :meth:`open`, which is where the licensing gate is applied.
    """

    source_id: str
    provides: tuple[str, ...] = ()

    def __init__(self, gate: SourceGate) -> None:
        self.gate = gate
        self._source: Source | None = None

    def open(self, *, version: str | None = None) -> Source:
        """Obtain permission to read this source.

        Raises :class:`~japgo.provenance.UnregisteredSourceError` if the source has no registry
        entry, or :class:`~japgo.provenance.ProvenanceViolation` if policy forbids the read.
        """
        src = self.gate.assert_ingestible(self.source_id)
        if version is not None:
            self.gate.assert_version(self.source_id, version)
        self._source = src
        return src

    @property
    def source(self) -> Source:
        if self._source is None:
            raise RuntimeError(
                f"{type(self).__name__}.open() must be called before reading — it is where the "
                "provenance gate is applied"
            )
        return self._source

    def make_record(self, *, layers: list[str], version: str | None = None, note: str | None = None) -> SourceRecord:
        return SourceRecord(
            source_id=self.source_id,
            version=version or self.source.version_pin,
            layers=sorted(layers),
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            note=note,
        )

    @abc.abstractmethod
    def read(self, path: Path, **kwargs) -> ReadResult:
        """Read a source file into core representation objects."""
