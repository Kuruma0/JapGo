# Decision log

Findings and decisions that are not derivable from the code or the git history, recorded because
session transcripts do not travel between machines.

[docs/phase0-research.md](phase0-research.md) holds the *reasoning*; this holds *what happened when
the reasoning met real data*. Where the two disagree, this file is later and wins.

Entries are newest last.

---

## 2026-08-10 — Phase 0 closed with four owner decisions

| Question | Decision | Consequence |
| --- | --- | --- |
| GSI Survey Act (§6.3) | **Avoid bulk GSI use** | GSI quarantined outright, not merely gated |
| Commercial use | **Yes, potentially** | Non-commercial data barred at every tier; output path must stay attribution-only |
| Hardware | **16 GB AMD RX 6800** | Patch-based training, fp16 (never bf16 — RDNA2 has no bf16 hardware) |
| Ordering | **Reconstruction first** | Output path built PLATEAU-primary from the start |

The GSI decision looked like a downgrade and was the opposite. Avoiding it forced the search that
found **VIRTUAL SHIZUOKA at 0.5 m under CC BY 4.0** — ten times finer than the 5 m GSI product being
avoided, with no approval procedure. Risk R1 closed at a net gain.

**Site selection.** Shizuoka chosen because it is the only place in Japan offering all four of:
0.5 m open terrain, PLATEAU coverage across the needed archetypes, all three archetypes inside one
CRS zone, and publication already in JGD2011 Plane Rectangular Zone 8. The region was chosen by
where the best openly-licensed data exists, not chosen first and then made to work.

---

## 2026-08-11 — First contact with real PLATEAU data

**PLATEAU packages are ~15 GB per municipality.** Atami — one of the *smaller* MVP cities — ships a
15 GB CityGML ZIP, because a package bundles LOD2 textures, disaster models, terrain and 3D Tiles
alongside the buildings. Three MVP sites would exceed 45 GB before any terrain.

*Decision:* read archives remotely. The CDN answers HTTP range requests and a ZIP keeps its
directory at the end, so `japgo.sources.fetch` fetches only the members needed. Measured: listed all
69 building GML members and extracted three while fetching **1.4 MB, 0.0096% of the archive**.

**The `0001` construction-year sentinel.** PLATEAU writes `0001` for "year unknown", and **38% of
Atami buildings carry it**. Parsed literally it becomes the year 1 — a cohort that would drag every
median in the §16 development-age analysis and skew any age-versus-morphology relationship, silently
and plausibly. Implausible years are now discarded; the real Atami range is 1975–2013 with a sensible
decade curve.

**All 18 usage codes reconciled** against the codelist shipped inside the real package, including
the six previously inferred from the code block's pattern. Those inferences were correct. The
taxonomy now admits no code at medium confidence, enforced by test.

Otherwise the adapter handled real CityGML unchanged: 188 buildings, zero unmapped codes, median
footprint 73.4 m² and median height 8.7 m.

---

## 2026-08-11 — VIRTUAL SHIZUOKA distribution differs from the Phase 0 survey

Phase 0 recorded "LAS files, 300 MB – 5.6 GB". The current distribution is **a vector-tile index**
whose features carry a mesh id and that mesh's download URL, with terrain published as a pre-gridded
product: plain `x y z` text at 0.5 m, already in Zone 8, roughly 400 × 300 m and ~2 MB zipped per
mesh.

*Consequence:* terrain for a specific place costs megabytes, needs no reprojection, and needs no
re-derivation. `japgo.sources.meshindex` reads the index; the MVT reader is hand-rolled rather than
taking a protobuf dependency for two field types, and is tested against bytes from an independent
encoder written from the spec.

**The text is ~58× the raster it becomes.** One tile's terrain is ~30 meshes and 431 MB extracted,
against a 7.5 MB raster. `TerrainFetcher` streams each mesh, grids it in memory and caches only the
raster — verified **bit-identical** to the disk route (max difference 0.000000000 m over 2,220,608
cells).

| | Text on disk | Cached raster |
| --- | --- | --- |
| Per tile | 431 MB | 7.5 MB |
| 100-tile site | 43.1 GB | 0.75 GB |
| Rebuild | minutes + re-download | 0.06 s |

---

## 2026-08-11 — First real tile, and the first empirical support for the thesis

Tile `z08_x000053_y-00108` (Atami): 100% terrain coverage, 134 PLATEAU buildings, OSM road targets
from Overpass, elevation 78–352 m across 1.5 km, **90.8% of the tile above the 12% road grade limit**.

**The road metrics discriminate archetypes:**

| | Atami (real) | Synthetic grid town |
| --- | --- | --- |
| Orientation entropy | **0.872** | 0.281 |
| Sinuosity p90 | 1.504 | ~1.0 |
| Road density | 4.93 km/km² | 14.29 km/km² |
| Dead-end ratio | 20.3% | 0% |

**The environmental relationship is measured, not assumed:**

- median slope where built — **14.1%**
- median slope where unbuilt — **54.1%**
- above 30% grade — 25% of built cells versus 78% of unbuilt

Development concentrates on the flattest ground available. This is the project's central hypothesis,
visible in one real tile before any model exists. It is *encouraging, not conclusive* — one tile is
an anecdote, and Phase 3 needs enough tiles for this to be statistical.

---

## Corrections — things believed and then disproved

Recorded because the wrong version is the intuitive one and will otherwise be re-derived.

**A centroid test is invalid for polylines.** Comparing the road target raster's centroid against
the road vectors' showed a 61 m offset. The *test* was wrong: polyline vertex density varies ~10×
between curves and straights, so a vertex-mean is not the geometric centre of a line. Valid for
building rings; invalid for road geometry. Roads are checked by distance-to-centreline instead.

**The residual was not a rasterisation artefact.** Having found a 7 m tail, the first explanation —
"an artefact of measuring against a rasterised hairline; these cells are inside the carriageway" —
was contradicted by the data: against true geometry they sat at 3.37–3.44 m, outside a 2.75 m
half-width. The real explanation is arithmetic: `all_touched=True` burns any cell the polygon clips,
so a cell centre can lie up to half a cell diagonal beyond the edge. 2.75 + 0.69 = 3.44, matching
exactly. The bound is now pinned by test.

**LAS parsing was not the build bottleneck.** Optimising it first moved a full-site build only
2m13 → 1m50. Profiling showed the real cost was `_grid_points` computing indices over all 8.7 M
points regardless of how many fell in the tile. Measure before optimising.

**YAML 1.1 coerces a bare `yes` key to boolean `true`** — and `building=yes` is the single most
common building tag in OSM. Unquoted, the *most frequent* case mis-keys silently. Keys are quoted and
a validator rejects non-string keys outright.

**`sliding_window_view` + a `nan*` reduction is O(n·w²)** and materialises a view w² times the DEM
(~0.5 s per tile at 1512²). Replaced with summed-area tables: O(n), and flat in window size.

---

## Open items

- **Phase 0.5 GPU spike unrun.** `scripts/gpu_spike.py` is the one-command go/no-go. See §20.1.
- **One real tile is not a dataset.** Phase 3's analysis needs enough tiles across all three sites
  for the relationships above to be statistical rather than anecdotal.
- **e-Stat population adapter** not yet written; its channels are not in the raster stack.
- **Aerial imagery** not yet wired. VIRTUAL SHIZUOKA publishes an ortho index (`ORTHO_INDEX` in
  `meshindex`) alongside the terrain one; the adapter path does not read it yet.
