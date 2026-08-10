"""Loading the provenance registry and gating source access through it."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable
from pathlib import Path

import yaml

from .checks import Finding, check_export, errors
from .models import ProvenanceViolation, Registry, Source

DEFAULT_REGISTRY_PATH = Path("data/provenance/registry.yaml")


def find_registry(start: Path | None = None) -> Path:
    """Locate the registry by walking up from ``start`` to the project root."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_REGISTRY_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"could not find {DEFAULT_REGISTRY_PATH} walking up from {here}. "
        "The provenance registry is required before any source may be read."
    )


def load_registry(path: Path | None = None) -> Registry:
    """Parse and validate the registry."""
    path = path or find_registry()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return Registry.model_validate(raw)


@functools.lru_cache(maxsize=1)
def _cached(path_str: str, mtime: float) -> Registry:
    return load_registry(Path(path_str))


def get_registry(path: Path | None = None) -> Registry:
    """Load the registry, cached on (path, mtime) so edits are picked up without a restart."""
    path = path or find_registry()
    return _cached(str(path), path.stat().st_mtime)


def registry_hash(path: Path | None = None) -> str:
    """Content hash of the registry, for recording in experiment configs and export manifests.

    Reproducibility (research doc §44) requires knowing which licensing policy was in force when
    an artifact was produced, not merely which datasets were used.
    """
    path = path or find_registry()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class SourceGate:
    """The enforcement point for invariant 2: no source may be read without a registry entry.

    Pipeline stages ask the gate for permission rather than reading the registry themselves, so
    that the rule lives in one place and cannot be partially applied.

    >>> gate = SourceGate(load_registry(Path("data/provenance/registry.yaml")))
    >>> gate.assert_ingestible("plateau").id
    'plateau'
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def assert_ingestible(self, source_id: str) -> Source:
        """Permit reading a source, or explain precisely why not."""
        src = self.registry.require(source_id)  # raises UnregisteredSourceError

        if not src.is_ingestible:
            detail = getattr(src, "decision", None) or getattr(src, "notes", "")
            raise ProvenanceViolation(
                f"source {source_id!r} is {src.usage_tier.value} and may not be ingested. "
                f"{str(detail).strip()[:300]}"
            )

        if self.registry.policy.commercial_intent and not src.is_commercially_clear:
            raise ProvenanceViolation(
                f"source {source_id!r} has commercial restrictions "
                f"({src.commercial_restrictions!r}) and commercial_intent is set. "
                "It may not be ingested at any tier, including for experiments."
            )

        return src

    def assert_version(self, source_id: str, resolved_version: str) -> None:
        """Refuse an ingest at a version other than the pinned one.

        Guards the AW3D30 trap: v4.1 excludes Japan, so fetching "latest" silently yields no data.
        """
        src = self.registry.require(source_id)
        if src.version_pin and str(resolved_version) != str(src.version_pin):
            raise ProvenanceViolation(
                f"source {source_id!r} is pinned to version {src.version_pin} but "
                f"{resolved_version!r} was resolved. "
                + (str(getattr(src, "warning", "")).strip()[:300])
            )

    def assert_exportable(
        self,
        contributing_ids: Iterable[str],
        *,
        redistribution_class: str = "attribution-only",
    ) -> list[Finding]:
        """Refuse to emit an artifact that violates policy. Returns findings when clean."""
        found = check_export(
            self.registry, contributing_ids, redistribution_class=redistribution_class
        )
        blocking = errors(found)
        if blocking:
            joined = "\n  ".join(str(f) for f in blocking)
            raise ProvenanceViolation(
                f"export as {redistribution_class!r} refused:\n  {joined}"
            )
        return found

    def attribution_for(self, contributing_ids: Iterable[str]) -> list[str]:
        """Assemble the attribution block for an artifact.

        Automatic because attribution assembled by hand is attribution that will eventually be
        wrong.
        """
        lines = []
        for sid in sorted(set(contributing_ids)):
            src = self.registry.require(sid)
            if src.attribution_required and src.attribution_string:
                text = " ".join(src.attribution_string.split())
                if text not in lines:
                    lines.append(text)
        return lines
