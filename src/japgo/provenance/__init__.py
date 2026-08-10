"""Dataset provenance: the gate every pipeline stage passes through.

A source with no registry entry may not be read. That rule is enforced here rather than
documented elsewhere, because ``docs/data-provenance.md``'s requirement — "do not silently
incorporate questionable data" — is only real if something can actually stop it.
"""

from .checks import (
    Finding,
    Severity,
    check_attribution,
    check_commercial,
    check_coverage,
    check_export,
    check_output_role,
    check_registry,
    check_share_alike,
    check_tier_gating,
    check_version_pin,
    check_vintage,
    errors,
)
from .models import (
    OutputRole,
    ProvenanceViolation,
    Registry,
    ShareAlike,
    Source,
    UnregisteredSourceError,
    UsageTier,
)
from .registry import (
    SourceGate,
    find_registry,
    get_registry,
    load_registry,
    registry_hash,
)

__all__ = [
    "Finding",
    "OutputRole",
    "ProvenanceViolation",
    "Registry",
    "Severity",
    "ShareAlike",
    "Source",
    "SourceGate",
    "UnregisteredSourceError",
    "UsageTier",
    "check_attribution",
    "check_commercial",
    "check_coverage",
    "check_export",
    "check_output_role",
    "check_registry",
    "check_share_alike",
    "check_tier_gating",
    "check_version_pin",
    "check_vintage",
    "errors",
    "find_registry",
    "get_registry",
    "load_registry",
    "registry_hash",
]
