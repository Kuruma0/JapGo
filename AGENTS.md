# JapGo — working agreement

Read [docs/phase0-research.md](docs/phase0-research.md) before making architectural decisions. This
file is the short version: the invariants that must not be violated without an explicit decision to
change them.

## Current phase

**Phase 1 (data pipeline), in progress.** Phase 0 closed with no open questions.

Done (143 tests):

| Package | Contents |
| --- | --- |
| `japgo.provenance` | The gate. 8 policy checks, `SourceGate` as the single enforcement point |
| `japgo.geo` | CRS, tile grid, `Raster`, terrain derivatives (slope/aspect/roughness/hillshade) |
| `japgo.core` | Tile manifest, `Building`, `Taxonomy` |
| `japgo.sources` | Adapter contract, PLATEAU (CityGML), VIRTUAL SHIZUOKA (LAS) |
| `japgo.pipeline` | Channel spec, rasterisation, `TileAssembler`, Zarr/Parquet store |

Geospatial stack verified on Windows/py3.13 — GDAL 3.12.4 via wheels, no toolchain build.

Next: **NLNI land use** and **e-Stat population** adapters, an **OSM adapter** (training-only —
it must not reach the shipped-geometry path), then the **Phase 2 visualiser**, which gates all
modelling work. The **Phase 0.5 GPU spike** runs in parallel and blocks nothing before Phase 4.

Settled parameters: commercial use is in scope, with the long-term emphasis on **generated**
environments and reconstruction kept legally clean alongside; GSI is avoided; the MVP region is
**Shizuoka Prefecture**; terrain is **VIRTUAL SHIZUOKA at 0.5 m** (CC BY 4.0) with AW3D30 v3.1
nationally; target hardware is a **16 GB AMD RX 6800**, Windows first with an approved **Linux
dual-boot** fallback; **reconstruction before generation**.

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
- **Assuming the GPU stack works.** gfx1030 on Windows is community-supported only, and ROCm on
  WSL2 does not reliably work on this card. Run the Phase 0.5 spike before Phase 4 depends on it —
  and **timebox the Windows attempt to one day**. A Linux dual-boot is an approved destination where
  gfx1030 is supported; do not spend a week debugging a community build to avoid an OS install.
- **Windows-only dependencies.** Linux is a likely training target. Keep paths, subprocess calls and
  GPU-specific code portable from the start; the whole geospatial stack is cross-platform already.
