# Data provenance

Spec §4 requires a provenance record for every external dataset and a registry that keeps the
training pipeline auditable. This document defines how that works here.

The registry is [data/provenance/registry.yaml](../data/provenance/registry.yaml). It is intended to
be **executable policy**, not documentation. §4's actual requirement — *"do not silently incorporate
questionable data into the training dataset"* — is only real if something can stop it.

## The rule

> A source with no registry entry may not be read by any pipeline stage. Ever.

No exceptions for "just testing" or "just this one tile". A source that is worth downloading is
worth a ten-line entry, and the entry is what makes the download defensible six months later when
nobody remembers where the file came from.

## Fields

The spec's §4 field list, plus three operational fields it implies but does not name.

### Required by spec §4

| Field | Meaning |
| --- | --- |
| `name`, `source`, `url` | Identity and origin |
| `license`, `license_url` | The license, named precisely (`ODbL-1.0`, not "open") |
| `geographic_coverage` | Including known gaps — see `plateau`, whose rural coverage is thin |
| `resolution` | Native resolution or vector detail level |
| `allowed_uses` | Enumerated, from a closed vocabulary |
| `commercial_restrictions` | `none`, a description, or `unresolved` |
| `attribution_required` / `attribution_string` | The exact string exports must carry |
| `derivative_restrictions` | What the license says about derived works |
| `training_restrictions` | What it says — or fails to say — about ML training |
| `redistribution_restrictions` | What may be passed on, and under what terms |

### Added operational fields

**`usage_tier`** — the enforcement hook. One of:

- `public` — may be used anywhere, including in published artifacts.
- `internal-research-only` — may be used for development and evaluation, but the exporter must
  refuse to emit any public artifact that transitively depends on it.
- `quarantined` — may not be ingested at all. Recorded so the exclusion is auditable rather than
  forgotten, and so nobody re-litigates it from scratch in three months.

**`share_alike`** — the copyleft family: `none`, `odbl`, `cc-by-sa`. Two incompatible copyleft
families fused into one derivative database is the worst available licensing outcome and the one
hardest to unwind. Recording the family lets a check catch it mechanically instead of relying on
memory. Currently `odbl` and `cc-by-sa` must never co-occur in a derivative database.

**`layer_isolation`** — `required`, `recommended`, or `not_required`. Whether the source must be
kept in its own keyed layer to preserve **Collective Database** status under OSMF guidance.

**`output_role`** *(added v2)* — `redistributable_core` or `training_only`. Whether a source's
geometry may appear in **shipped output**, as distinct from whether we may **train** on it. The two
are genuinely different questions once commercial use is in scope: model predictions trained on OSM
are unencumbered, but a reconstruction transformed *from* OSM geometry is a Derivative Database
carrying ODbL share-alike. See research doc §6.1c.

This is the field that keeps the commercial path open. The `redistributable_core` set — PLATEAU,
NLNI, VIRTUAL SHIZUOKA, AW3D30, e-Stat, Sentinel-2 — covers terrain, buildings with semantic type,
roads, land use and population under attribution-only terms.

This last field is the one that reaches furthest into the architecture. Under OSMF's Collective
Database guideline, share-alike applies only to the OSM-derived parts of a collection — *provided
the parts remain distinguishable*. That translates into a hard schema rule:

> Every vector feature carries its `source_id`. OSM and non-OSM attributes are never flattened into
> a single merged table.

It is much cheaper to design for this now than to re-derive a corpus later.

## Checks to implement in Phase 1

These are specified now so the pipeline is built with them rather than around them:

1. **Coverage** — every `source_id` referenced by any tile manifest resolves to a registry entry.
2. **Tier gating** — export of a `public` artifact fails if any contributing source is
   `internal-research-only` or `quarantined`. This is what keeps the unresolved GSI question
   (research doc §6.3) from becoming a release incident.
3. **Share-alike compatibility** — no derivative database mixes `odbl` and `cc-by-sa`.
4. **Attribution completeness** — every export carries the `attribution_string` of every
   contributing source, assembled automatically. Attribution assembled by hand is attribution that
   will eventually be wrong.
5. **Vintage confirmation** — any entry whose license is marked *verify per vintage* (currently
   `nlni_landuse`) must have a confirmed vintage recorded, or it is treated as `quarantined`.
6. **`output_role` gating** — an export declaring `redistribution_class: attribution-only` fails if
   any contributing source is `training_only`. This is what mechanically prevents OSM geometry from
   reaching a commercial reconstruction.
7. **Non-commercial exclusion** — because `commercial_intent` is true project-wide, ingest fails for
   any source whose `commercial_restrictions` is not `none`. No exceptions for experiments;
   research-only data has a habit of becoming load-bearing.
8. **Version pinning** — ingest asserts the pinned version where one exists. Currently `aw3d30`,
   where fetching "latest" (v4.1) returns **no data over Japan**.

## Reviewing

Reviewed 2026-08-10 following the four Phase 0 decisions. The policy block at the top of the
registry records them: commercial intent is true, GSI bulk use is avoided, and reconstruction is the
primary mode.

Re-review whenever a source is added, a license changes, a new vintage is ingested, or one of those
policy settings changes — the last of these invalidates tier and `output_role` assignments across
the whole file, not just one entry.
