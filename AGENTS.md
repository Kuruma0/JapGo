# JapGo — working agreement

Read [docs/phase0-research.md](docs/phase0-research.md) before making architectural decisions. This
file is the short version: the invariants that must not be violated without an explicit decision to
change them.

**This file is the entry point.** If you are an assistant or agent picking this project up on a new
machine, read this file and then the research document; between them they carry every decision and
the reasoning behind it. Session transcripts do not travel between machines — the repository is the
handover.

**No AI-assistant attribution anywhere in this repository** — not in commit messages, not in
filenames, not in file contents. The project is published as the author's own work. Do not add
co-author trailers or tool branding. Before pushing, check with
`git log --format='%b' | grep -i <vendor>` and a `git grep` over tracked files.

## Current phase

**Phase 3 has a result. Phase 4 does not, yet.** **74 tiles** across all three MVP archetypes
(stack v2), built from the published sources with nothing staged. Phase 3 reports **44 supported
associations and 8 nulls** — steeper ground carries less road and fewer intersections; greater
relief makes roads wind.

Phase 4's exit criterion — beat a non-learned prior on APLS/TOPO on an unseen archetype — is **not
cleanly met**, and five measured configurations say the objective is not why. Mean APLS across the
folds: v1 width+BCE 0.080, **v2 width+Dice 0.092**, v3 centreline 0.036, v4 +distance weighting
0.037, v5 +true positive weight 0.031.

**v2 is the reference configuration** — width target, Dice, positive weight capped at 5, no
distance weighting. Defaults are pinned to it and covered by a test so they cannot drift to
whatever the last experiment used; every alternative stays reachable from the CLI. The three
changes after v2 were regressions.

Junction inflation went **4.78× → 0.99×** across those changes and APLS never followed, so junction
count was a symptom throughout. **Do not tune the objective further** without a reason beyond "the
last thing did not work" — five configurations put APLS in 0.015–0.19 regardless of target, loss
composition or class weighting. The untested candidates are the **corpus** (74 tiles, ~800
patches/fold; every fold looks under-fitted) and **node placement**, which nothing measures yet:
APLS needs junctions in the right *places*, and every metric so far constrains their *number*.

The control that matters, and it survives every change so far: trained on the flat plain alone the
model *loses* to the prior on the mountain valley; trained on flat **and** steep it wins. Same
site, same seed. The model responds to environment, not just to built form.

Done (420 tests):

| Package | Contents |
| --- | --- |
| `japgo.provenance` | The gate. 8 policy checks, `SourceGate` as the single enforcement point |
| `japgo.geo` | CRS, tile grid, `Raster`, terrain derivatives (slope/aspect/roughness/hillshade) |
| `japgo.core` | Tile manifest, `Building`, `Taxonomy`, `RoadGraph` + hierarchy/LOD |
| `japgo.sources` | Adapter contract; PLATEAU, VIRTUAL SHIZUOKA, NLNI, OSM; remote archive and mesh fetchers |
| `japgo.pipeline` | Channel spec, rasterisation, `TileAssembler`, store, geographic splits, region builder |
| `japgo.viz` | Phase 2 alignment reports — self-contained HTML, no plotting dependency |
| `japgo.analysis` | Phase 3 study: environmental predictors, road-structure responses, cluster-bootstrapped ranking |
| ingest | `sources.overpass` (roads), `sources.jismesh` (PLATEAU member selection), `pipeline.remote` (no staging) |

Geospatial stack verified on Windows/py3.13 — GDAL 3.12.4 via wheels, no toolchain build. The
Python code is platform-clean (no OS-specific imports, no path literals, no case collisions); the
**training machine runs Linux**, which is also where gfx1030 ROCm support is supported rather than
community-maintained.

**The GPU stack is verified, not assumed** (2026-08-11). `torch 2.9.1+rocm6.4` on gfx1030, fp16
autocast working, **4.03 GB peak** at 512² × 15 channels batch 8 — inside budget without gradient
checkpointing. R1b closed. Re-run `python scripts/gpu_spike.py` after any torch upgrade, and check
the wheel still ships gfx1030 kernels before taking one.

Session transcripts do not travel between machines. [docs/decision-log.md](docs/decision-log.md)
records what happened when the reasoning met real data — read it alongside the research document.

Next: **a soft centreline target, then re-run the sweep**. Neither target is right — width
over-paints, hairline under-detects. Distance-transform weighting, so a near-miss is penalised in
proportion to distance, is what the original note proposed and what stack v2 only half-implemented.

**The sweep has produced its first real answer, and it is negative.** `japgo sweep --mode
quantile` gives a held-out tile another real site's slope distribution, value for value — nothing
out of distribution left to blame. Swept on all three folds, the model uses the slope channel
intensively and still gets the geography wrong. Changing the values while keeping the pattern
(quantile map), the response tracks the *size* of the change and not its direction — six for six on
magnitude, three for six on direction, which is chance. Keeping the values and destroying the
pattern (shuffle), the prediction **collapses to near zero**. So it is neither ignoring terrain nor
reading it as a bare magnitude: it has fitted a narrow joint distribution and degrades under any
departure from it.

This is **risk R2, caught by the only instrument that could catch it**. Beating the priors on three
archetypes was compatible with a model that had learned terrain properly; the sweep shows this one
did not. Most likely magnitude, not geography — slope is scale-normalised, and a network keying on
activation magnitude behaves exactly like this. Do not read Phase 4's pass as evidence the model
uses terrain correctly.

Before trusting Phase 3's ranking as an attribution, note that the terrain predictors are collinear
(slope/relief/roughness rank together). Use `--scheme loso`, not the configured single-site split —
see the decision log for why that distinction is worth F1 0.142 against 0.335.

Still unwritten: the **e-Stat population** adapter and the **aerial imagery** path (`ORTHO_INDEX`
in `meshindex` is published but unread).

Settled parameters: commercial use is in scope, with the long-term emphasis on **generated**
environments and reconstruction kept legally clean alongside; GSI is avoided; the MVP region is
**Shizuoka Prefecture**; terrain is **VIRTUAL SHIZUOKA at 0.5 m** (CC BY 4.0) with AW3D30 v3.1
nationally; target hardware is a **16 GB AMD RX 6800 on Linux**; **reconstruction before
generation**.

## Invariants

1. **No engine types in the core.** Nothing under the core model may import or reference Unity or
   Unreal types. Engines consume a versioned interchange bundle. If logic exists in both exporters,
   it belongs in the core.

2. **No source without a registry entry.** See [docs/data-provenance.md](docs/data-provenance.md).
   A pipeline stage that reads an unregistered source is a bug, not a shortcut.

3. **`source_id` on every vector feature.** Never flatten OSM and non-OSM attributes into one
   merged table. This preserves Collective Database status and is a licensing requirement, not a
   style preference (research doc §6.1).

3b. **OSM geometry never reaches shipped output.** OSM and Overture are `training_only`. Shipped
   reconstruction geometry comes from the redistributable core (PLATEAU, NLNI, VIRTUAL SHIZUOKA,
   AW3D30). Training on OSM is fine — predictions are unencumbered; *transforming* OSM geometry into
   a product is not (research doc §6.1c). Build the output path PLATEAU-primary from the start;
   swapping later means re-deriving the corpus. This holds even though generation is the long-term
   commercial focus: the point is to keep reconstruction shippable without a redesign.

3d. **Generated output must be measurably novel.** The "predictions are unencumbered" position
   depends on output being new rather than re-emitted training data. The nearest-neighbour novelty
   check (research doc §6.1d, §16.2) is not optional instrumentation — a generator that memorises
   fails the project's own generalisation standard *and* thins the licensing argument.

3c. **No non-commercial data, ever, at any tier.** Commercial use is in scope. Research-only
   datasets do not enter the pipeline even for one-off experiments.

4. **Geographic splits only.** Never randomly split tiles between train and test. Splits are by
   city/region with a discarded buffer zone. Adjacent tiles share halo pixels — a random split is
   not a weak evaluation, it is an invalid one.

5. **ML proposes, procedural disposes.** Grade limits, turning radii, intersection validity,
   hierarchy consistency, bridge/tunnel necessity and connectivity repair are deterministic. Do not
   ask a model to satisfy a constraint that can be enforced.

6. **The halo is not optional.** Tiles are 1 km core + 256 m context halo, read but not predicted
   into. Adding it later invalidates every cached sample.

7. **Metric CRS internally.** JGD2011 Japan Plane Rectangular CS for all computation; WGS84 only at
   I/O boundaries. Every morphological measure is in metres.

8. **Every experiment reproducible from its config.** Dataset version, registry hash, preprocessing
   version, model version, seeds, split definition. An experiment that cannot be re-run from its
   config is a failed experiment regardless of its numbers.

9. **Design for 16 GB VRAM.** Patch-based training (512² crops), fp16 with loss scaling (**not
   bf16** — RDNA2 has no bf16 hardware), gradient checkpointing, gradient accumulation, and a
   disciplined ~12-channel raster stack. If something genuinely needs more, reformulate it
   patch-wise or rent a GPU for that run — do not assume a bigger card.

10. **Structure and appearance stay separate.** The core carries footprints, heights, types,
   orientations, setbacks. Never materials, textures or meshes. Appearance is applied at export
   time against a declared LOD level. This is what lets higher-quality visuals be added later
   without rebuilding anything.

## Phase gates

Do not skip these. Each exists because the failure it prevents is expensive and silent.

- **Phase 2 before Phase 4.** No modelling until a human has visually confirmed every layer is
  spatially aligned. Vector/raster misalignment corrupts training without producing errors.
- **Phase 4 baseline before Phase 5.** The dull U-Net baseline establishes the floor. Anything
  fancier must beat it on held-out cities.
- **Phase 5 sensitivity sweep.** The project's actual thesis is that environmental information
  changes the road network. Until the counterfactual sweep demonstrates that, nothing else measured
  matters.

## Things that look like good ideas and are not

- **Adding street-level imagery early.** Licensing is unresolved and CC BY-SA conflicts with ODbL
  (research doc §6.4). Phase 9, after review.
- **Training on one city.** Tokyo dominance is a named risk (R4). Stratify by archetype, not area.
- **Judging output by how it looks.** Spec §38 and research doc §16.2 both reject pixel similarity
  as a primary metric. A structurally excellent network that differs from the real one is a success.
- **Porting to Rust early.** Python + GDAL is correct through Phase 4. Port what profiling proves
  is hot — probably tessellation and seam-merge, not the model.
- **Adding raster channels casually.** The stack is a memory budget, not just a feature list
  (§20.2). At 512² float32, each channel is ~1 MB per sample before batching. Every new channel
  needs a reason, and it needs an entry in `config/raster_stack.yaml` — never hardcoded in the
  assembler.
- **Sliding-window neighbourhood ops.** `sliding_window_view` + a `nan*` reduction is O(n·w²) and
  materialises a view w² times the DEM. Use summed-area tables — see `terrain._box_sum`.
- **Extracting source archives to disk.** Both major sources punish it. A PLATEAU municipality ZIP
  is ~15 GB for ~200 MB of building GML — read it remotely with `ArchiveFetcher`, which range-reads
  the members it needs (measured: 0.0096% of the archive). VIRTUAL SHIZUOKA's Grid text is ~58× the
  raster it becomes — use `TerrainFetcher`, which grids in memory and caches the raster. Writing the
  intermediates out costs 43 GB per 100-tile site instead of 0.75 GB.
- **Letting the model depend on 0.5 m terrain.** VIRTUAL SHIZUOKA is far finer than anything
  available for the rest of Japan. Terrain enters the model at a 1 m working tier with augmentation
  that simulates 30 m sources. Keep the raw 0.5 m for validation and high-detail export.
- **Fetching AW3D30 "latest".** v4.1 excludes Japan. Pin v3.1.
- **Assuming the GPU stack works.** Run `scripts/gpu_spike.py` before Phase 4 depends on it. On
  Linux gfx1030 is a supported ROCm configuration so this should be a confirmation; the historical
  warnings about Windows wheels and WSL2 no longer apply to the training machine.
- **Windows-only dependencies.** Linux is a likely training target. Keep paths, subprocess calls and
  GPU-specific code portable from the start; the whole geospatial stack is cross-platform already.
