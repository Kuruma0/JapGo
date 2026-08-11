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

## 2026-08-11 — Phase 0.5 GPU spike: pass, and R1b closes

The spike ran on the Linux training machine and the stack came up unmodified. Risk R1b — carried
since Phase 0 as the project's largest schedule risk, downgraded twice on argument — is now closed
on evidence.

| | |
| --- | --- |
| Verdict | **PASS** |
| Backend | rocm, torch 2.9.1+rocm6.4, HIP 6.4 |
| Device | gfx1030 / RX 6800, 17.16 GB reported |
| fp16 autocast | working |
| Peak memory | **4.03 GB** at 512² × 15 channels, batch 8 |
| Step time | 467 ms, 17.1 samples/s |

**The 16 GB budget closes with room to spare.** §20.2 estimated 6–9 GB for the U-Net baseline with
gradient checkpointing; the spike measures 4.03 GB *without* it, at the real crop size and the real
stack depth. Checkpointing is therefore a lever still in reserve rather than a prerequisite, which
matters for the SAM-class encoder route (§12) estimated at 10–13 GB.

**No system ROCm libraries were needed.** The machine carries only the HIP runtime and OpenCL
(`amdrocm-* 7.14.0~pre3`, no MIOpen or rocBLAS); the wheel bundles its own. What the system has to
supply is the `amdgpu` kernel driver and membership of `render`/`video`. Worth knowing before
anyone tries to "fix" a future GPU problem by reinstalling ROCm.

**Wheel selection is a real constraint, and it is checkable without downloading.** Official
PyTorch ROCm wheels have been dropping older gfx targets. gfx1030 kernels are present in the
rocm6.4 through rocm7.2 channels, confirmed by range-reading each wheel's ZIP central directory
rather than fetching 4–6 GB to find out — the same trick `ArchiveFetcher` uses on PLATEAU. Do that
check before any future upgrade; the failure mode if the target is missing is a runtime error deep
in a training run, not an install error.

**The spike's own API nearly produced a false negative.** It called
`torch.cuda.amp.GradScaler`, deprecated since torch 2.4. On a torch that has removed it, the
`except` block would have caught the `AttributeError` and reported **FAIL** — a code failure
indistinguishable, in the report, from a hardware one. Now uses `torch.amp.GradScaler("cuda", …)`.
A go/no-go script needs to be more robust than the thing it is testing.

Timings, for planning: first run 2m32 (MIOpen compiling kernels for these shapes), 19 s warm.

---

## 2026-08-11 — Remote ingest exists; first multi-tile corpus built

Until now `japgo tiles build` could only read staged local files, and `TerrainFetcher` /
`ArchiveFetcher` — written and tested for exactly this — were referenced by nothing outside
`japgo.sources`. There was also no Overpass client at all, so the road layer, which is the Phase 3
response variable and the Phase 4 target, had no automated path. A corpus was therefore not one
command away; it was a missing component.

Now wired: `japgo.sources.overpass` (one query per region, cached, gated on
`assert_training_only_use` *before* the request), `japgo.sources.jismesh` (JIS X 0410 decoding, so
PLATEAU selection is by mesh code), and `japgo.pipeline.remote` composing all three behind the
`TileInputSource` protocol. `RegionBuilder.build_from` now holds the build loop so the staged and
remote paths share coverage gating, manifests, attribution and error isolation.

**First real corpus: 15 tiles over Atami** (`japgo tiles build izu_coast --remote`), 5 skipped for
coverage below 50% — correctly, they are sea.

| | |
| --- | --- |
| PLATEAU | 10 of 69 members, **4.6 MB — 0.0306% of the 15 GB archive**, 5,580 buildings |
| Roads | 1,124 edges, 0.7 MB, one Overpass query |
| Terrain | 138 meshes, streamed and cached as raster |
| On disk | 496 MB of tiles, 141 MB of cache — against ~43 GB had the sources been staged |

A rebuild makes **zero** network requests: every fetch is cached by content.

**The one validated tile reproduces.** `z08_x000053_y-00108` comes back at 100% coverage with a
274.4 m elevation range against the 78–352 m recorded, and 294 buildings over the read extent —
which is 129 scaled to the core, against the 134 recorded. The environmental relationship holds:
median slope **22.5% where built against 48.7% where not**.

**Road metrics differ from the earlier record, and the reason is the measurement convention, not
the data.** Recorded 4.93 km/km²; `japgo.analysis` reports 2.968. The earlier figure was the read
extent over the read area (recomputed: 5.121, the residual being same-day OSM drift), whereas the
study measures the **core** over the core area. Dead-end ratio is lower for the same reason plus
the halo-aware counting that is the point of `structure.py`. Neither number is wrong; they answer
different questions, and only the core-based one is comparable between tiles.

*Two bugs found by checking the output rather than trusting it:* a fresh `TerrainFetcher` per tile
re-fetched the mesh index every tile, and the remote path handed every tile the **whole region's**
road graph where the staged path clips per tile — so `bundle.roads` meant different things
depending on which provider built the tile. Both fixed, the second now covered by a test asserting
the two paths agree.

*And one near miss worth recording:* the first run aimed `--limit 10` at the southernmost 10 of 30
tiles, which sit just south of where Atami's PLATEAU coverage starts at lat 35.025. It would have
produced ten clean-looking tiles with every building channel zero — indistinguishable downstream
from "this area has no buildings", and the study would have reported a real-looking null for built
form. Empty member selection is now a loud warning.

---

## 2026-08-11 — VIRTUAL SHIZUOKA is several surveys, not one dataset

Extending beyond Atami exposed two assumptions that held only because the first site was in the
2019 survey area. Both were silent failures, not errors.

**There is no prefecture-wide index.** Phase 0 recorded coverage as "effectively the whole
prefecture", which is true of the *data* and false of the *endpoints*: each survey is published as
its own dataset with its own vector-tile index. `japgo.sources.meshindex` knew only the 2019
富士山南東部・伊豆東部 index, so a coverage probe returned **0 meshes for both Hamamatsu and
Kawanehon** while PLATEAU coverage for the same extents was fine (87 and 46 members).

Zero meshes does not look like a missing endpoint downstream. It looks like a tile with no
terrain, and the builder skips it for low coverage — so an entire site would come back "built: 0"
with nothing in the log to say why. `MeshIndex` now consults every published grid index and
deduplicates by mesh number; off-coverage tiles 403, which already read as empty.

| Site | 2019 index | 2025 中・西部 index |
| --- | --- | --- |
| Atami | 6.5 KB | 403 |
| Hamamatsu | 403 | 6.1 KB |
| Kawanehon | 403 | 6.6 KB |

**The two surveys do not use the same Grid text format**, which is the more dangerous of the two::

    2019 富士山南東部・伊豆東部   50000.250 -105299.750 225.858       x y z, space separated
    2025 中・西部                1,-40399.75,-101400.25,754.30,1     seq,x,y,z,flag, commas

Here the failure was loud — `np.fromstring(sep=" ")` raised on every mesh — but it did not have to
be. Read positionally as three columns, the 5-column form would have taken the **sequence number
as an easting**, producing a well-formed raster of the wrong place. The parser now sniffs the
layout per file.

The 2025 form's 5th column is 0 or 1 and is **not** a nodata flag: on mesh 08NC3989 both values
carry plausible elevations over the same 417–755 m range (0: 175,523 posts; 1: 304,477). Filtering
on it would punch holes in the DEM, so all posts are kept.

---

## 2026-08-11 — The NLNI land-use vintage in the registry does not exist **[FLAG]**

The registry pins `nlni_landuse` at *"Spec v3.1, FY2021 data"*. Checking the distribution before
wiring the adapter: the authoritative datalist offers L03-b in vintages **06, 09, 14, 16, 76, 87,
91, 97** — newest **FY2016**. The sibling products agree (L03-a same set; L03-b-c only `-16`).
There is no FY2021 land-use mesh. The Phase 0 entry appears to have taken the *specification*
version for the data vintage.

This is not a coding blocker — the fetch is a URL pattern away, confirmed live at
`L03-b-16_{5237,5238,5239}-jgd_GML.zip`, the three primary meshes covering the MVP sites. It is a
**provenance blocker**, and deliberately so. §6.5 records the license per product *per vintage*
because the download site warns terms differ between versions, and data-provenance check 5 treats
an unconfirmed vintage as quarantined. Ingesting FY2016 under an entry that claims FY2021 would
defeat the one mechanism the project has against exactly this.

Left unwired pending an owner decision: confirm the FY2016 terms and re-pin the registry, or leave
land use out. Until then the four `landuse_*` channels are zero, and the study reports every
land-use association as `insufficient` rather than inventing a null.

---

## 2026-08-11 — Three sites, and the first Phase 3 result

31 tiles across all three MVP archetypes, every layer from the published sources with nothing
staged: Hamamatsu 8, Kawanehon 8, Atami 15. Split validates with no geographic leakage. Land use
is in, at the FY2016 vintage (see the entry below). Attribution assembles from all four sources.

**The archetypes separate, and they separate the way the environment says they should.** Medians
per site:

| | slope p50 | >12% grade | built | forest | km/km² | int/km² | orient H | sinuosity | dead-end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hamamatsu plain | 3.2% | 7.6% | 26.9% | 0% | 27.62 | 246.5 | 0.81 | 1.00 | 5% |
| Izu coast | 48.5% | 74.6% | 0.9% | 59.3% | 5.09 | 17.0 | 0.73 | 1.02 | 15% |
| Kawanehon valley | 75.7% | 89.7% | 0.6% | 86.2% | 3.68 | 8.5 | 0.47 | 1.05 | 8% |

**27 associations are supported** — interval excluding zero under a bootstrap that resamples
sites, not tiles. The strongest:

| Predictor | Response | rho | 95% CI |
| --- | --- | --- | --- |
| landuse_built_frac | intersection density | **+0.950** | [+0.26, +0.96] |
| landuse_built_frac | road density | +0.925 | [+0.40, +0.94] |
| slope_p90 | intersection density | **−0.906** | [−0.93, −0.55] |
| relief_m | sinuosity median | +0.862 | [+0.41, +0.92] |
| slope_above_limit_frac | road density | −0.814 | [−0.87, −0.38] |

That is the project's thesis, measured rather than assumed: **steeper ground carries less road and
fewer intersections, and greater relief makes the roads that exist wind more.** It is the first
result in the project that is statistical rather than anecdotal.

**Two predictions from [site-selection.md](site-selection.md) did not hold**, which is the part
worth keeping:

- *"Flat suburban → low orientation entropy (grid)"* is **wrong**. Hamamatsu scores 0.81, the
  *highest* of the three. A Japanese suburban plain is not a US-style grid at street level, and
  27.6 km/km² of local road runs in many directions. Orientation entropy did not discriminate
  plain from coast (0.81 vs 0.73) at all.
- *"Mountain valley → dead-end ratio far above the plain"* holds only weakly: Kawanehon 8% against
  the plain's 5%, but **below** the Izu coast's 15%. Coastal constraint produces more cul-de-sacs
  than valley constraint does.

Entropy did cleanly isolate the valley at 0.47, which is the *other* prediction for that site —
"strong alignment with contours and the river corridor". A corridor forces one bearing, so low
entropy is the correct signature. The measure works; the expectation attached to the plain did not.

**Zero null results at this corpus size.** With 31 tiles nothing could be shown *absent* — 68
associations landed `inconclusive`, intervals spanning zero. At the time this was recorded as a
limitation of having three sites, with the claim that adding tiles would not narrow the intervals.

**That claim was wrong, and the expanded corpus disproves it** — see the 81-tile entry below.

**Collinearity limits attribution.** `slope_median`, `slope_p90`, `roughness_mean` and `relief_m`
derive from one surface and rank nearly together, so the ranking cannot say which of them carries
the effect. `landuse_built_frac` topping the table over `slope_p90` should be read as "built land
and road density go together", not as land use out-explaining terrain.

Fetch cost for the whole corpus, against ~334 GB of source archives:

| | |
| --- | --- |
| PLATEAU | Atami 4.6 MB (0.031%), Hamamatsu 40.9 MB (**0.015%** of 269 GB), Kawanehon cached |
| Land use | 33.3 MB, three primary meshes |
| Roads | 3.4 MB, three Overpass queries |
| Corpus | 962 MB of tiles, 1.0 GB of cache |

---

## 2026-08-11 — The land-use "built" channel contained the target

Found in the pre-Phase-4 review, and the reason to do that review before training rather than
after. `config/landuse.yaml` grouped NLNI code **0901 道路** into `landuse_built` — a model
**input** — while `road_mask` is what the model predicts. The answer was inside the question.

Worse than a generic leak: 0901 marks *road corridors*, so the leaked signal sits exactly on the
wide arterials, which are the easiest roads to predict and the ones a baseline would be judged on.

**It did not manufacture the Phase 3 result.** Measured before removing it: 0901 is about 1% of
cells in the three MVP primary meshes (Hamamatsu 4,488 of ~640k; Atami 720 of ~380k), and none
fell in the sampled tiles at all. Rebuilding without it moved the headline from rho +0.952 to
**+0.950**, and supported associations from 28 to **27**. The site-level separation that drives it
(`landuse_built` 0.93 in Hamamatsu against 0.00 in Kawanehon) is genuine built-up land.

Fixed at `landuse_version: 2`: `road` now falls through to "other" — all four channels zero. A
hole is neutral; a leak is not. `railway` stays, because rail is environmental context (§8) and is
not the target. The adapter now stamps `landuse_v<n>` into the source record, so a tile built under
a different grouping is distinguishable — the grouping decides what the channel *means*, and
invariant 8 cannot hold across a silent config change.

**Read the ranking with care even now.** `landuse_built_frac` topping `slope_p90` says "built land
and roads co-occur", which is close to definitional and is not the project's thesis. The thesis is
the terrain rows: slope suppressing density and relief driving sinuosity.

---

## 2026-08-11 — 81 tiles, and the nulls appear

The corpus went 31 → **81 tiles**, balanced across the archetypes rather than deepening the one
that was easiest to extend: Hamamatsu 8 → 26, Kawanehon 8 → 30, Atami 15 → 25. Split validates
with no geographic leakage, every tile carries targets, and all 81 share one registry hash, one
stack version and `landuse_v2`.

**The Phase 3 exit criterion is now actually satisfiable.** At 31 tiles the study returned 27
supported and **zero** null results; at 81 it returns **44 supported and 8 nulls**.

| Predictor | Response | rho | 95% CI |
| --- | --- | --- | --- |
| landuse_built_frac | intersection density | +0.950 | [+0.59, +0.96] |
| slope_above_limit_frac | intersection density | **−0.928** | [−0.95, −0.37] |
| roughness_mean_m | intersection density | −0.924 | [−0.93, −0.32] |
| slope_median_pct | intersection density | −0.907 | [−0.91, −0.20] |
| slope_above_limit_frac | sinuosity median | +0.882 | [+0.23, +0.89] |

And, for the first time, relationships that are *absent* rather than merely unproven — interval
inside ±0.3:

| Predictor | Response | rho | 95% CI |
| --- | --- | --- | --- |
| roughness_mean_m | component count | −0.221 | [−0.27, +0.08] |
| landuse_water_frac | dead-end ratio | −0.001 | [−0.24, +0.13] |
| landuse_agricultural_frac | sinuosity p90 | −0.040 | [−0.23, +0.05] |

**This corrects a claim made at 31 tiles**, that intervals were limited by having three sites and
that adding tiles would not narrow them. Tripling the tiles produced eight nulls where there were
none. The reason: the bootstrap resamples *sites*, but the statistic inside each resample is
computed over every tile in the resampled sites, so more tiles per site reduces within-resample
noise. Between-site disagreement sets the floor on interval width; tile count still moves it.
Adding sites remains the stronger lever, but "more tiles will not help" was wrong.

Fetch cost stayed negligible against ~334 GB of archives: Hamamatsu 30.4 MB (**0.0113%** of
269 GB) for 101,427 buildings and 13,989 road edges; Kawanehon 3.2 MB (0.0064% of 50 GB).

---

## 2026-08-11 — Phase 4 baseline: 3/3 folds clear the floor on unseen archetypes

The dull U-Net of spec §51, trained leave-one-site-out so every fold evaluates on an archetype it
has never seen. Scored against two non-learned priors on the same held-out tiles, each at its own
best threshold.

| Held out (unseen) | Model F1 | Constant prior | Built-proximity prior | |
| --- | --- | --- | --- | --- |
| izu_coast — coastal | **0.524** (P 0.497 / R 0.555) | 0.074 | 0.185 | PASS |
| hamamatsu_plain — flat | **0.447** (P 0.319 / R 0.748) | 0.000 | 0.284 | PASS |
| kawanehon_valley — mountain | **0.335** (P 0.365 / R 0.309) | 0.049 | 0.161 | PASS |

Every fold beats **both** floors, including built-proximity — so the model has learned more than
"roads are where the town is", which was the whole point of including that prior. The difficulty
ordering is the one [site-selection.md](site-selection.md) predicted: the mountain valley is
hardest, and it is the site the document names as where the thesis is falsifiable.

### The control: terrain in the training set is what makes the difference

Same held-out site, same architecture, same seed, same epoch count. The only change is whether
steep ground appears in training.

| Training set | Held out | Model F1 | Verdict |
| --- | --- | --- | --- |
| Hamamatsu only — flat (the configured split) | Kawanehon | **0.142** | PARTIAL — loses to built-proximity at 0.161 |
| Hamamatsu + Izu — flat and steep (LOSO) | Kawanehon | **0.335** | PASS |

**F1 more than doubles.** Trained on the plain alone the model does not even clear the
built-proximity floor: it learned "road where building", carried that to a mountain valley where
buildings are 0.6% of cells, and failed. Given steep ground to learn from, the same network on the
same target recovers roads at more than twice the score.

This is the first direct evidence that the model responds to environment rather than to built
form alone, and it settles by measurement what was previously an argument for using LOSO. The
configured single-site split was not merely thin — on this evidence it cannot answer the question
Phase 4 exists to ask. It is retained as one fold of the LOSO scheme, not as the default.

**What this is not.** F1 against a known mask is a reconstruction score, not the §16.2 metric
suite: no APLS, no TOPO, no graph extraction yet, so "beats a non-learned prior on APLS/TOPO"
remains partly open. Precision is the weak side everywhere (0.10–0.50), meaning the model
over-paints roads — expected from a class-weighted loss at pos_weight 5–31, and the first thing a
threshold-calibrated or Dice-style objective should improve.

Runs are re-creatable: each checkpoint has a config beside it pinning fold, tile lists, crop,
batch, epochs, seed, stack version and registry hash `5cf7ff78cf16f0c3`.

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

- **One real tile is not a dataset.** Phase 3's analysis needs enough tiles across all three sites
  for the relationships above to be statistical rather than anecdotal. This is now the *only* thing
  standing between the project and a Phase 3 result: `japgo.analysis` is written and tested, and
  `japgo study` runs the moment a corpus exists.
- **Three sites is the floor, and it binds.** The cluster bootstrap resamples sites, so with three
  of them almost every interval is wide enough to be `inconclusive` rather than `null`. That is the
  honest answer, but it means the Phase 3 exit criterion's "with the null results stated" may be
  only partly satisfiable at MVP scale. Adding tiles will not fix it; adding *sites* would.
- **The terrain predictors are collinear.** Slope, relief and roughness derive from one surface and
  can be rank-identical across a corpus, in which case the ranked table cannot attribute an effect
  to one of them rather than another. Worth a collinearity pass before the ranking is quoted as a
  result.
- **e-Stat population adapter** not yet written; its channels are not in the raster stack.
- **Aerial imagery** not yet wired. VIRTUAL SHIZUOKA publishes an ortho index (`ORTHO_INDEX` in
  `meshindex`) alongside the terrain one; the adapter path does not read it yet.
