"""Registry policy checks.

The eight checks specified in ``docs/data-provenance.md``. Checks 1-5 were specified in Phase 0
revision 1; 6-8 were added once commercial intent and the reconstruction/generation split were
settled.

Each check returns findings rather than raising, so that ``japgo provenance check`` can report
everything wrong at once instead of one item per run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .models import (
    COMMERCIAL_UNRESOLVED,
    OutputRole,
    Registry,
    ShareAlike,
    Source,
    UsageTier,
)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    source_id: str | None
    message: str

    def __str__(self) -> str:
        where = f" [{self.source_id}]" if self.source_id else ""
        return f"{self.severity.value.upper():7} {self.check}{where}: {self.message}"


# --------------------------------------------------------------------------------------------
# Check 1 — coverage
# --------------------------------------------------------------------------------------------


def check_coverage(registry: Registry, referenced_ids: Iterable[str]) -> list[Finding]:
    """Every source id referenced by a tile manifest must resolve to a registry entry."""
    findings = []
    known = {s.id for s in registry.sources}
    for sid in sorted(set(referenced_ids)):
        if sid not in known:
            findings.append(
                Finding(
                    "coverage",
                    Severity.ERROR,
                    sid,
                    "referenced by a manifest but has no registry entry",
                )
            )
    return findings


# --------------------------------------------------------------------------------------------
# Check 2 — tier gating
# --------------------------------------------------------------------------------------------


def check_tier_gating(registry: Registry, contributing_ids: Iterable[str]) -> list[Finding]:
    """A public artifact may not depend on a non-public source, transitively."""
    findings = []
    for sid in sorted(set(contributing_ids)):
        src = registry.get(sid)
        if src is None:
            continue  # reported by check_coverage
        if src.usage_tier is UsageTier.QUARANTINED:
            findings.append(
                Finding(
                    "tier-gating",
                    Severity.ERROR,
                    sid,
                    "is quarantined and may not be ingested at all",
                )
            )
        elif src.usage_tier is UsageTier.INTERNAL_RESEARCH_ONLY:
            findings.append(
                Finding(
                    "tier-gating",
                    Severity.ERROR,
                    sid,
                    "is internal-research-only and may not reach a public artifact",
                )
            )
    return findings


# --------------------------------------------------------------------------------------------
# Check 3 — share-alike compatibility
# --------------------------------------------------------------------------------------------

_INCOMPATIBLE_PAIRS = {frozenset({ShareAlike.ODBL, ShareAlike.CC_BY_SA})}


def check_share_alike(registry: Registry, combined_ids: Iterable[str]) -> list[Finding]:
    """No derivative database may mix incompatible copyleft families."""
    families: dict[ShareAlike, list[str]] = {}
    for sid in sorted(set(combined_ids)):
        src = registry.get(sid)
        if src is None or src.share_alike is ShareAlike.NONE:
            continue
        families.setdefault(src.share_alike, []).append(sid)

    findings = []
    present = set(families)
    for pair in _INCOMPATIBLE_PAIRS:
        if pair <= present:
            involved = sorted(sid for f in pair for sid in families[f])
            findings.append(
                Finding(
                    "share-alike",
                    Severity.ERROR,
                    None,
                    f"incompatible copyleft families combined: "
                    f"{'+'.join(sorted(f.value for f in pair))} via {involved}",
                )
            )
    return findings


# --------------------------------------------------------------------------------------------
# Check 4 — attribution completeness
# --------------------------------------------------------------------------------------------


def check_attribution(registry: Registry, contributing_ids: Iterable[str] | None = None) -> list[Finding]:
    """Every source requiring attribution must supply the exact string to emit.

    Attribution assembled by hand is attribution that will eventually be wrong.
    """
    sources = (
        [s for s in registry.sources if s.id in set(contributing_ids)]
        if contributing_ids is not None
        else registry.ingestible
    )
    return [
        Finding(
            "attribution",
            Severity.ERROR,
            s.id,
            "attribution_required is true but attribution_string is missing",
        )
        for s in sources
        if s.attribution_required and not s.attribution_string
    ]


# --------------------------------------------------------------------------------------------
# Check 5 — vintage confirmation
# --------------------------------------------------------------------------------------------

_VINTAGE_HINT = "verify per vintage"


def check_vintage(registry: Registry) -> list[Finding]:
    """A source whose license varies by vintage must record the confirmed vintage."""
    findings = []
    for s in registry.ingestible:
        text = f"{s.license} {getattr(s, 'license_note', '')}".lower()
        if _VINTAGE_HINT in text:
            vintage = getattr(s, "vintage", None) or s.version_pin
            if not vintage:
                findings.append(
                    Finding(
                        "vintage",
                        Severity.ERROR,
                        s.id,
                        "license varies by vintage but no vintage/version_pin is recorded; "
                        "treat as quarantined until confirmed",
                    )
                )
    return findings


# --------------------------------------------------------------------------------------------
# Check 6 — output_role gating
# --------------------------------------------------------------------------------------------


def check_output_role(registry: Registry, contributing_ids: Iterable[str]) -> list[Finding]:
    """An attribution-only export may not contain geometry from a training_only source.

    This is what mechanically prevents OSM geometry from reaching a commercial reconstruction
    (research doc §6.1c).
    """
    findings = []
    for sid in sorted(set(contributing_ids)):
        src = registry.get(sid)
        if src is None:
            continue
        if src.output_role is OutputRole.TRAINING_ONLY:
            findings.append(
                Finding(
                    "output-role",
                    Severity.ERROR,
                    sid,
                    "is training_only; its geometry may not appear in redistributed output "
                    "(ODbL share-alike would attach)",
                )
            )
    return findings


def check_output_role_coverage(registry: Registry) -> list[Finding]:
    """Every ingestible source must declare an output_role, so the gate above can never be
    bypassed by omission."""
    return [
        Finding(
            "output-role-coverage",
            Severity.ERROR,
            s.id,
            "is ingestible but declares no output_role",
        )
        for s in registry.ingestible
        if s.output_role is None
    ]


# --------------------------------------------------------------------------------------------
# Check 7 — non-commercial exclusion
# --------------------------------------------------------------------------------------------


def check_commercial(registry: Registry) -> list[Finding]:
    """With commercial intent set, no restricted source may be ingested at any tier.

    Research-only data has a habit of becoming load-bearing, so this admits no experiment
    exemption.
    """
    if not registry.policy.commercial_intent:
        return []

    findings = []
    for s in registry.ingestible:
        value = s.commercial_restrictions.strip().lower()
        if s.is_commercially_clear:
            continue
        severity = Severity.ERROR
        detail = (
            "commercial restrictions are unresolved"
            if value in COMMERCIAL_UNRESOLVED
            else f"has commercial restrictions: {s.commercial_restrictions!r}"
        )
        findings.append(Finding("commercial", severity, s.id, f"{detail}; quarantine it"))
    return findings


# --------------------------------------------------------------------------------------------
# Check 8 — version pinning
# --------------------------------------------------------------------------------------------


def check_version_pin(registry: Registry, resolved: dict[str, str] | None = None) -> list[Finding]:
    """Sources with a pinned version must be ingested at that version.

    Currently only ``aw3d30``, where fetching "latest" (v4.1) returns no data over Japan.
    """
    findings = []
    resolved = resolved or {}
    for s in registry.ingestible:
        if not s.version_pin:
            continue
        actual = resolved.get(s.id)
        if actual is None:
            continue  # nothing ingested yet; not an error
        if str(actual) != str(s.version_pin):
            findings.append(
                Finding(
                    "version-pin",
                    Severity.ERROR,
                    s.id,
                    f"pinned to version {s.version_pin} but {actual} was resolved",
                )
            )
    return findings


# --------------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------------


def check_registry(registry: Registry) -> list[Finding]:
    """Run every check that needs no external context — the registry auditing itself."""
    findings: list[Finding] = []
    findings += check_output_role_coverage(registry)
    findings += check_commercial(registry)
    findings += check_attribution(registry)
    findings += check_vintage(registry)
    findings += check_share_alike(registry, [s.id for s in registry.redistributable_core])
    return findings


def check_export(
    registry: Registry,
    contributing_ids: Iterable[str],
    *,
    redistribution_class: str = "attribution-only",
) -> list[Finding]:
    """Run every check relevant to emitting an artifact from a known set of sources."""
    ids = list(contributing_ids)
    findings: list[Finding] = []
    findings += check_coverage(registry, ids)
    findings += check_tier_gating(registry, ids)
    findings += check_share_alike(registry, ids)
    findings += check_attribution(registry, ids)
    if redistribution_class == "attribution-only":
        findings += check_output_role(registry, ids)
    return findings


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity is Severity.ERROR]
