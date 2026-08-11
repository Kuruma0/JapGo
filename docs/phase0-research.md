# Phase 0 — Technical Research Document

**Project:** JapGo — AI-powered geographic, environmental, architectural and road-network
understanding system
**Date:** 2026-08-10
**Status:** Revision 2. Covers the 23 topics required by the master specification §47.

> **Revision 2 incorporates four decisions from the project owner:** avoid bulk GSI use;
> design for eventual commercial use from the start; plan for a **16 GB AMD RX 6800** rather than a
> 24 GB NVIDIA card; **reconstruction before generation**. These are not cosmetic — they changed the
> terrain source, the MVP region, the licensing posture, and the Phase 4 plan. Sections materially
> revised: §4.1, §6, §12, §19, §20, §21, §23.

Every recommendation in this document is marked **[DECIDE]** (a decision I am proposing and will act
on unless overruled), **[DEFER]** (deliberately left open until a benchmark settles it), or
**[FLAG]** (a risk or ambiguity that needs a human answer).

---

## 1. Problem definition

### 1.1 What this is not

Three framings must be rejected explicitly, because each one leads to an architecture that cannot
grow into the stated goal:

| Rejected framing | Why it fails |
| --- | --- |
| "Procedural road generator" | Produces networks that are self-consistent but environmentally arbitrary. Adding terrain later means rewriting the generator. |
| "Road detection from satellite imagery" | Solves *transcription*, not *understanding*. A perfect detector cannot answer "what road belongs here?" for an area with no roads yet. |
| "Scatter buildings beside roads" | Buildings become decoration. The causal arrow in reality runs both ways, and a decoration pipeline can never be inverted. |

### 1.2 What it is

The system learns a conditional distribution over **built environment structure** given
**environmental context**:

```
P( road graph, parcels, building types, building placement | terrain, hydrology, land use,
                                                             rail, population, existing built form )
```

Two operating modes fall out of the same model:

- **Reconstruction** — context is a real place; produce a clean, engine-ready representation of
  what is actually there (with LOD control).
- **Generation** — context is a real or synthetic environment with the road/building layer removed
  or absent; produce a plausible one.

A third mode, **reverse reasoning** (infer land use / settlement type / development era from road
and building pattern), is not an MVP goal but is architecturally free if the representation is
symmetric. It is the cheapest available *validation* signal for whether the model learned structure
or texture, so it is worth keeping reachable.

### 1.3 Success criterion

The MVP succeeds if, holding the model fixed, **changing the environmental input changes the road
network in the direction a geographer would predict**. Concretely: raise terrain slope in a tile and
the generated network should shift from grid toward valley-following and switchback forms; raise
building density and road density should rise with it. This is a *sensitivity* test, not a pixel
test, and it is the honest measure of whether environmental information is doing any work.

---

## 2. Existing research

### 2.1 Road graph extraction from imagery (mature)

This subfield is well developed and should be **reused, not reinvented**. Two lineages:

- **Iterative / agent-based** — the model walks the graph vertex by vertex. `RoadTracer`,
  `RNGDet`, `RNGDet++` (transformer + imitation learning; handles complex intersections well).
  Strong topology, expensive inference.
- **Global / one-shot** — the model predicts the whole graph in one pass. `DeepRoadMapper`,
  `Sat2Graph` (graph-tensor encoding), `SAM-Road` (Segment Anything backbone + lightweight
  transformer GNN), and `SAM-Road++` / `samroadplus` (CVPR 2025), which adds node-guided resampling
  to fix the train/inference mismatch in SAM-Road.

The CVPR 2025 work also released a **Global-Scale road extraction dataset** spanning ~13,800 km²,
roughly 20× the largest prior public dataset, and reports SOTA F1 on City-Scale and SpaceNet.
Global methods have displaced iterative ones on efficiency grounds.

**Relevance:** This gives us reconstruction (§1.2 mode 1) largely off the shelf, and — more
importantly — a **pretrained encoder of "what built-up land looks like from above"** that we can
reuse as one branch of the environmental encoder.

### 2.2 Urban layout / road network generation (active, less settled)

- **Conditional diffusion on road rasters** — generate road imagery conditioned on a context
  image, then vectorize. Simple, works, but topology is an afterthought.
- **Vector/graph-native** — `BlockPlanner` (vectorized dual-layer graph for urban blocks) and
  successors. Fine-grained control and semantic reasoning; harder to train.
- **Multimodal, context-informed diffusion for urban morphology synthesis** — closest published
  analogue to this project's ambition.
- **`RoBus`** — a multimodal dataset explicitly pairing **road networks with building layouts** for
  controllable generation. This is the single most directly relevant public dataset to §15's
  road→parcel→building pipeline.
- **`CityGen`**, GAN-based site-layout work, ControlNet-style semantic-map conditioning, and
  Stable-Diffusion LoRAs fine-tuned on city road structure.

**Assessment:** the generative side is *not* solved, and no published system does the full
terrain→road→parcel→building chain with architectural semantics. That is where this project's
contribution actually lies. Everything upstream of it should be borrowed.

### 2.3 Urban morphometrics (mature, underused by ML people)

`momepy` (PySAL) implements systematic urban form measurement across six character families —
dimension, shape, spatial distribution, intensity, connectivity, diversity — plus **morphological
tessellation**, a Voronoi-based partition of built-up area from building footprints.

**This is the most important find in the entire survey.** Morphological tessellation gives us
**parcels without parcel data** — a defensible, reproducible proxy for the land-parcel layer that
§15 requires and that Japan does not publish openly at national scale. It also gives us the
building-morphology feature vector of §8 essentially for free, computed identically at train and
inference time.

---

## 3. Existing open-source projects

Recorded in the format required by §31. Fuller entries live in
[data/provenance/registry.yaml](../data/provenance/registry.yaml); this is the shortlist.

| Name | Purpose | License | Verdict |
| --- | --- | --- | --- |
| **GDAL/OGR** | Raster+vector I/O, warping, reprojection | MIT-style | Adopt. Non-negotiable foundation. |
| **GeoPandas / Shapely / pyproj** | Vector dataframes, geometry ops, CRS | BSD-3 / MIT | Adopt. |
| **Rasterio** | Rasters as numpy, windowed reads | BSD-3 | Adopt. |
| **OSMnx** | OSM → cleaned, topologically correct street graphs | MIT | Adopt for road graph ingest. Do not hand-roll OSM way→graph logic. |
| **momepy** (PySAL) | Urban morphometrics + morphological tessellation | BSD-3 | Adopt. See §2.3. |
| **NetworkX** | Graph algorithms | BSD-3 | Adopt for correctness; profile before trusting at city scale. |
| **PyTorch** | ML | BSD-3 | Adopt. |
| **PyTorch Geometric** | GNN layers | MIT | Adopt when the graph branch lands. |
| **samroadplus / SAM-Road++** | SOTA satellite→road-graph | check repo | Integrate as reconstruction baseline + pretrained aerial encoder. **[FLAG]** confirm license before any redistribution. |
| **esmini** (+ `esminiRMLib`) | OpenDRIVE road manager, `.xodr` validation, plotting | MPL-2.0 | Adopt as the OpenDRIVE *validator* in CI. |
| **scenariogeneration** | Python → OpenDRIVE/OpenSCENARIO XML | MPL-2.0 | Adopt as the `.xodr` writer. Do not write XML by hand. |
| **awesome-openx** | Curated index of ASAM OpenX tooling | — | Reference. |
| **spacv** / `spatialCV` | Spatial cross-validation splitters | open | Adopt for §29 splits. |
| **PDAL, WhiteboxTools, RichDEM** | Point cloud / terrain derivatives | open | Evaluate for slope/aspect/flow accumulation. |

**Do not reinvent:** OSM parsing, CRS transforms, DEM resampling, raster↔vector conversion, graph
algorithms, tessellation, OpenDRIVE serialization.

---

## 4. Available datasets

Japan is unusually well served. Ranked by value to this project.

### 4.1 Tier 1 — the backbone

**Project PLATEAU (MLIT).** National open 3D city models in **CityGML** with the i-UR application
domain extension, ~250 cities, distributed through the G-Spatial Information Center. Crucially these
are *semantic* models — buildings carry **usage, year of construction, and urban planning
attributes**, not just geometry.

This is the highest-value dataset in the project by a wide margin, because it directly supplies the
three things that are normally the hardest labels to obtain:

- **building type** (§9) — from the usage attribute, no classifier needed for training labels;
- **building height / floors** (§8) — from LOD1/LOD2 geometry;
- **development age** (§16) — from year of construction, which makes the historical-development
  research direction empirically testable rather than speculative.

**OpenStreetMap.** Road network, rail, water, land use polygons, POIs. Japan coverage is good and
improving; over 236 municipalities' PLATEAU building datasets have been prepared in ODbL-compatible
form for OSM integration.

**VIRTUAL SHIZUOKA point clouds (Shizuoka Prefecture).** Aerial LiDAR plus mobile mapping,
published through the G-Spatial Information Center as LAS, **already organised into a 0.5 m grid
DEM**, in **JGD2011 / Plane Rectangular CS Zone 8**. Released under a **CC BY 4.0 / ODbL dual
license**. Coverage spans southeast Fuji + east Izu, central/west Shizuoka, northwest Shizuoka, and
Fuji + east Shizuoka — effectively the whole prefecture.

This is the most consequential find of Revision 2. Having been told to avoid bulk GSI use, the
expected outcome was a downgrade from GSI's 5 m DEM to a 30 m global product — a serious loss for
mountain and coastal work. Instead, this is **0.5 m: ten times finer than the GSI data we are
avoiding**, with clean commercial licensing and no approval procedure. The dual license also lets us
**elect CC BY 4.0 and decline ODbL**, keeping terrain entirely out of the share-alike analysis.

It comes with a practical cost: files average ~300 MB and the largest is 5.2 GB, so Phase 1 needs
real point-cloud tiling infrastructure (PDAL) rather than naive whole-file reads.

**ALOS AW3D30 v3.1 (JAXA).** 30 m global DSM under **CC BY 4.0**, commercial use permitted. Note
the version trap: **v4.1 explicitly excludes Japan**; Japanese coverage remains at **v3.1**. Pin the
version — a pipeline that silently fetches "latest" gets no data over Japan.

**GSI (Geospatial Information Authority).** 5 m/10 m DEM and the seamless aerial photo tile mosaic.
**Excluded from the MVP by owner decision** (see §6.3). Retained in the registry as a
`quarantined` future option should the 複製承認 route ever be worth taking.

**国土数値情報 / National Land Numerical Information (MLIT).** Land use mesh (`L03-b` fine mesh,
`L03-a` 3rd mesh, `L03-b-c` detailed mesh; spec v3.1, FY2021 data), plus administrative boundaries,
rail, stations, public facilities, flood hazard zones. Shapefile / GeoJSON / JPGIS.

**Census mesh population (e-Stat).** Population and household counts on the standard mesh grid.

### 4.2 Tier 2 — useful, with caveats

**Overture Maps Foundation.** Buildings and Transportation themes are convenient
cloud-native (GeoParquet) alternatives to raw OSM extraction. **They are ODbL**, same as OSM —
adopting Overture buys ergonomics, not license relief. Places and Divisions themes are
CDLA-Permissive-2.0.

**Sentinel-2 / Landsat.** Free, global, permissive, but 10 m GSD is too coarse for building
morphology. Useful for regional-scale context bands (vegetation, built-up index) only.

**Copernicus DEM GLO-30.** 30 m, free, attribution-based. **[FLAG]** A small set of countries is
withheld from the free public release and I could not confirm from the documentation whether Japan
is affected. Given AW3D30 v3.1 already covers Japan under CC BY 4.0, this is a redundant fallback,
not a dependency — verify coverage only if it is ever actually needed.

**Imagery for Shizuoka.** With GSI tiles excluded, high-resolution aerial imagery is the one input
that genuinely degrades. Sentinel-2 at 10 m cannot support building morphology. Options, in order of
preference: (a) PLATEAU LOD2 textures where available, (b) prefectural/municipal open orthophotos
published via G-Spatial Information Center — **to be surveyed in Phase 1**, (c) accept that the MVP
runs without high-resolution RGB. Option (c) is survivable: the aerial branch is one of several, and
0.5 m LiDAR-derived terrain plus PLATEAU footprints carry most of the structural signal. **[FLAG]**
Worth knowing early whether (b) exists for the chosen municipalities.

**Mapillary / KartaView street-level imagery.** Imagery is CC BY-SA 4.0. See §6.4 for a
significant caveat.

### 4.3 Tier 3 — Phase 9 material

Open driving/dashcam corpora, Mapillary Vistas (semantic segmentation) and Traffic Sign datasets,
academic road-scene datasets. Japan-specific open driving data exists but is thin and often
**research-use-only** — e.g. the Meijo University GNSS/IMU Odaiba set permits research use by both
non-profit and for-profit organisations but **forbids copying or modifying the dataset for use in a
commercial product**. That restriction pattern is typical and is exactly why the registry exists.

**[DECIDE]** Street-level and video sources are excluded from the MVP training set entirely. They
enter no earlier than Phase 9, and only after a per-source license review.

---

## 5. Open-source media

Beyond datasets, the spec asks about openly licensed photographs and 3D data for the eventual
architecture-generation stage. Findings, briefly:

- **Wikimedia Commons** — large volumes of CC BY-SA / public domain Japanese architecture and
  streetscape photography, with per-file licenses that vary. Usable as *reference and evaluation*
  material; per-file license checking makes bulk training use expensive.
- **PLATEAU LOD2 textured models** — the best open source of Japanese building appearance, already
  under the same license as the rest of PLATEAU.
- **Photogrammetry / 3D repositories** — heterogeneous licensing, low signal-to-noise. Deprioritise.

**[DECIDE]** For architectural appearance, PLATEAU LOD2 is the primary source. Photo corpora are
evaluation-only until there is a concrete need.

---

## 6. Licensing — the decisive constraint

This section is deliberately long. Licensing determines what the system may *ship*, and getting it
wrong late is far more expensive than getting it right now.

### 6.1 The OSM/ODbL question, resolved

The OSM Foundation's community guidance answers the question this project actually has:

- A training set that is a **substantial extraction** from OSM **is a Derivative Database**, and
  must be made available under ODbL **if publicly used**.
- Models trained on such a set **must attribute OSM** in documentation where a user would look —
  the model README or download page.
- **Predictions made using such a model are not implicated by ODbL.**

That last clause is the load-bearing one. It means:

> The generated environments this system outputs are **not** forced under ODbL by the fact that the
> model was trained on OSM. The training set and the attribution obligation are.

**[DECIDE]** Project licensing posture:

1. Training corpora derived from OSM are treated as ODbL Derivative Databases. If we publish them,
   we publish under ODbL. Internally-held corpora carry the same metadata regardless.
2. Model artifacts carry an OSM attribution notice in the README and model card.
3. Generated outputs are unencumbered by ODbL. We will still emit a provenance manifest with every
   export, because being able to *prove* that is worth more than the file it costs.
4. Non-OSM sources are kept in **separately keyed layers** so the corpus is a **Collective
   Database**, not a Derivative one, wherever that is achievable. Under OSMF guidance, share-alike
   then applies only to the OSM-derived parts. This is a concrete architectural requirement, not
   paperwork: **do not merge OSM and non-OSM attributes into a single flattened table.**

### 6.2 PLATEAU

Licensed under **PDL 1.0** (Public Data License), which the site policy states is **compatible with
CC BY 4.0**, with ODC-BY and ODbL offered as alternatives. Attribution required; modifications must
be marked as modified and must not be presented as MLIT's own work. No explicit commercial-use or
ML-training restriction. The site policy does note that some content carries restrictions under
individual statutes, particularly surveying law — which leads directly to the next item.

### 6.1b Commercial intent changes the risk calculus **[DECIDE]**

The owner has confirmed the project **may eventually be used commercially**, and asked that the
pipeline be designed for that from the beginning rather than retrofitted.

Two consequences. The first is easy: no dataset whose terms are non-commercial or research-only may
enter the pipeline at any tier, even for experiments. Research-only data has a way of becoming load
bearing. The registry already excludes these; the rule is now absolute rather than prudent.

The second is not easy, and it is the most important finding in this revision.

### 6.1c The reconstruction-output problem **[FLAG]**

Commercial intent and reconstruction-first, combined, create a tension that neither creates alone.

Recall §6.1: model *predictions* are not implicated by ODbL. That cleanly protects **generation**
mode — synthetic environments are unencumbered.

But **reconstruction** mode is different. Reconstructing a real Japanese place from OSM produces
output that is a **transformation of OSM data**, not merely a prediction informed by it. A cleaned,
LOD-filtered, engine-ready road graph of a real town is, on the plain reading of ODbL, a
**Derivative Database**. Publicly using it — including shipping it in a commercial product —
triggers share-alike on that output.

So the two modes have genuinely different licensing characters, and the mode the owner chose to
build first is the encumbered one:

| Mode | Output | ODbL status |
| --- | --- | --- |
| Generation | Model predictions over synthetic context | Not implicated |
| Reconstruction | Transformation of real OSM data | Derivative Database → share-alike |

**[DECIDE]** Mitigation, adopted now because retrofitting it later means re-deriving the corpus:

1. **Separate the training path from the reconstruction path.** OSM may be used freely for
   *training* — that route is settled by §6.1 and ends in unencumbered predictions.
2. **For reconstruction output intended for redistribution, prefer non-ODbL sources.** PLATEAU is
   PDL 1.0 (CC BY 4.0-compatible) and carries roads, buildings and land use. 国土数値情報 is CC BY
   4.0-compatible. VIRTUAL SHIZUOKA can be elected as CC BY 4.0. **A reconstruction built from
   PLATEAU + NLNI + VIRTUAL SHIZUOKA, with no OSM geometry in the output, is commercially
   redistributable under attribution alone.**
3. **Track this per output artifact.** Every export's provenance manifest already records
   contributing sources; the exporter gains a `redistribution_class` derived from them —
   `attribution-only` or `share-alike`.

This is achievable rather than aspirational precisely because PLATEAU is unusually rich: it has the
road, building and land-use layers that would otherwise force OSM into the output path. It also
means **PLATEAU coverage becomes a hard constraint on MVP site selection**, not merely a
convenience — which independently reinforces the Shizuoka choice in §23.

OSM remains valuable and stays in the project. It is simply confined to training, evaluation and
gap-filling, rather than being the geometry that ships.

**Owner confirmation (2026-08-10):** both modes will eventually be supported, with the long-term
commercial emphasis on **generated** environments, and the reconstruction pipeline kept legally
clean regardless so that no redesign is needed later. Nothing above relaxes — the PLATEAU-primary
output path stays mandatory. What changes is the *reason* it is worth the effort: it is no longer
protecting the main commercial artifact, it is protecting the option to ship reconstruction at all.
That is still worth paying for now, because the cost of building it in is small and the cost of
retrofitting it is a corpus re-derivation.

### 6.1d Memorisation — the one place the commercial emphasis creates new exposure **[FLAG]**

If generated environments are the eventual commercial product, the §6.1 shield — *predictions are
not implicated by ODbL* — becomes the single most load-bearing legal proposition in the project. It
is worth understanding where it could thin out.

That argument rests on the output being genuinely *new*, not a re-emission of training data. A
generative model that memorises can reproduce a specific real neighbourhood's layout closely enough
that calling it a "prediction" rather than a "substantial reproduction" becomes a much harder
argument to make. This is not a licensing question so much as a *modelling* question with licensing
consequences, and it is entirely within our control.

**[DECIDE]** Add a **nearest-neighbour novelty check** to the metric suite (§16.2): for each
generated tile, find the closest training tile under a structural similarity measure and record the
distance. Flag and inspect anything suspiciously close.

The reason to like this is that it is not a compliance tax — it is a metric we should want anyway.
A generator that scores well on realism *because* it is regurgitating training tiles is a failed
generator by the project's own standards (§39: "should not merely memorise Tokyo"). The novelty
check measures generalisation and reduces legal exposure with the same number. Dual-purpose metrics
are rare enough to take when they appear.

**Not a substitute for advice.** If and when this commercialises, the ODbL analysis in §6.1/§6.1c
deserves review by someone qualified. Everything here is a reading of published OSMF guidance, which
is the right basis for engineering decisions and the wrong basis for a launch.

### 6.3 GSI — resolved by decision

GSI content is under the **Public Data License 1.0**, requiring source citation. But GSI's core
products — 基盤地図情報 including the DEMs, and the map/photo tiles — are **基本測量成果 (basic
survey results)** under the **Survey Act (測量法)**, and reproduction/use of survey results
formally requires **approval (複製承認 / 使用承認)** from the originating survey organisation.

The rules have been relaxed: whereas dead-copying basic survey results was previously not
approvable, reproduction of survey results provided via GSI's website — including GSI tiles — **is
now approvable even as a dead copy**. Download requires user registration.

So the position is "permitted, subject to a procedure", not "public domain". For real-time tile use
in an application, attribution alone (`国土地理院` / `地理院タイル` plus a link) is the stated
requirement. For **bulk download of tiles or DEM into a training corpus**, this is a different act
with a different rule.

**Decision (owner, 2026-08-10): avoid bulk GSI use.** GSI-derived terrain becomes an optional future
component contingent on obtaining the appropriate permissions. The MVP must not be blocked by it.

**Implemented as:** `gsi_dem` and `gsi_tiles` are set to `usage_tier: quarantined` — not merely
gated but un-ingestable. This is deliberately stricter than the decision requires. An
`internal-research-only` tier would permit GSI data into development caches, and experience says
that data becomes load-bearing before anyone notices. Quarantine removes the possibility.

I had recommended filing the 複製承認 application. The owner's decision is the better one, because
the terrain research it was protecting turned out to be unnecessary: **VIRTUAL SHIZUOKA supplies
0.5 m terrain — an order of magnitude better than the 5 m GSI product — under CC BY 4.0** (§4.1).
The concern that motivated my recommendation was that avoiding GSI would degrade the
mountain-village archetype. It does the opposite.

Terrain sourcing after this decision:

| Scope | Source | License | Resolution |
| --- | --- | --- | --- |
| MVP region (Shizuoka) | VIRTUAL SHIZUOKA | CC BY 4.0 (elected from dual) | **0.5 m** |
| Rest of Japan | ALOS AW3D30 **v3.1** | CC BY 4.0 | 30 m |
| Redundant fallback | Copernicus GLO-30 | Attribution | 30 m, Japan coverage unverified |

### 6.4 Mapillary / KartaView **[FLAG]**

Both platforms license contributed imagery **CC BY-SA 4.0**. However, Mapillary is operated by Meta
and its platform Terms of Use layer additional conditions on API access and bulk use on top of the
imagery license. The interaction between "the imagery is CC BY-SA" and "the ToS restricts what you
may do with the API" is a genuinely unsettled question that has been argued in public forums, and I
am not able to resolve it from documentation alone.

Additionally, CC BY-**SA** is share-alike. If Mapillary-derived imagery features were fused into the
core representation, the same Collective-vs-Derivative analysis as §6.1 would apply — with a
*different* and incompatible share-alike license. Two mutually incompatible copyleft licenses in one
derived database is the worst available outcome.

**[DECIDE]** Street-level imagery is out of scope until Phase 9 (this independently reinforces the
§4.3 decision), and when it returns, KartaView is preferred over Mapillary purely because its
licensing story is simpler.

### 6.5 国土数値情報

Generally CC BY 4.0-compatible under the Government of Japan's standard terms, **but the download
site explicitly warns that license conditions differ between versions**, and older archived products
may carry different terms. MLIT address data is redistributed by Overture under CC BY 4.0, which is
a useful confirmation of the modern terms.

**[DECIDE]** The registry records the license **per product per vintage**, never per site. Any
product whose terms cannot be confirmed for the specific vintage we downloaded is quarantined.

### 6.6 Summary posture

Now expressed in the two columns that actually matter given commercial intent: may we *train* on
it, and may the *output* be redistributed commercially under attribution alone?

| Source | License | Training | Commercial redistribution of derived output |
| --- | --- | --- | --- |
| PLATEAU | PDL 1.0 (≈CC BY 4.0) | Yes | **Yes — attribution only** |
| 国土数値情報 | CC BY 4.0-ish, per vintage | Yes, if vintage confirmed | **Yes — attribution only** |
| VIRTUAL SHIZUOKA | CC BY 4.0 / ODbL dual → **elect CC BY 4.0** | Yes | **Yes — attribution only** |
| ALOS AW3D30 v3.1 | CC BY 4.0 | Yes | **Yes — attribution only** |
| e-Stat mesh population | Gov. standard terms | Yes | Yes |
| Sentinel-2 | Copernicus open | Yes | Yes |
| OSM | ODbL | Yes — predictions unencumbered | **Only if no OSM geometry is in the output** (§6.1c) |
| Overture (bldg/transport) | ODbL | Yes | Same as OSM |
| GSI DEM / tiles | PDL 1.0 + Survey Act | **No — quarantined** | No |
| Mapillary / KartaView | CC BY-SA 4.0 + ToS | **No** | No |
| Research driving datasets | Often non-commercial | **No** | No |

The top block is the **redistributable core**: PLATEAU + NLNI + VIRTUAL SHIZUOKA + AW3D30 covers
terrain, buildings with semantic type and construction year, roads, land use and population, and
every one of them ships under attribution alone. That the project can reach its MVP entirely inside
that block — with OSM as a training and evaluation input rather than shipped geometry — is the
single most useful structural result of this revision.

---

## 7. Data provenance

Implemented, not just specified — see [data/provenance/registry.yaml](../data/provenance/registry.yaml)
and [docs/data-provenance.md](data-provenance.md).

Design: every source is one YAML entry carrying the fields §4 of the spec requires, plus three
operational fields the spec implies but does not name:

- `usage_tier` — `public` | `internal-research-only` | `quarantined`. The dataset builder refuses
  to ingest anything not explicitly tiered, and the exporter refuses to emit public artifacts that
  transitively depend on a non-`public` source.
- `share_alike` — the copyleft family, so incompatible pairs (ODbL + CC BY-SA) are detected
  mechanically rather than by memory.
- `layer_isolation` — whether the source must be kept in its own layer to preserve Collective
  Database status (§6.1).

The point of making this executable rather than documentary is that §4's requirement — "do not
silently incorporate questionable data" — is only real if something can actually stop it.

---

## 8. Environmental representation

**[DECIDE]** A **fixed-extent geographic tile** is the unit of everything: dataset sample, model
input, cache entry, and export chunk.

- **Projection:** JGD2011 / Japan Plane Rectangular CS (EPSG:6669–6687, zone per region) for metric
  work; WGS84 only at I/O boundaries. Metric CRS is required — every morphological measure in §8 of
  the spec is in metres. Conveniently, VIRTUAL SHIZUOKA is *already published* in JGD2011 Plane
  Rectangular **Zone 8**, which is the correct zone for the MVP region — no reprojection of the
  largest and most awkward input.

- **Terrain resolution harmonisation [DECIDE].** The MVP region has 0.5 m terrain; the rest of Japan
  has 30 m. A model trained only on 0.5 m data may learn to depend on detail that will not exist
  outside Shizuoka (risk R1e). Therefore terrain enters the model at a **common working resolution
  of 1 m in the micro tier**, with 0.5 m data downsampled rather than natively consumed, and
  **training augmentation deliberately degrades terrain to simulate 30 m sources**. The 0.5 m data
  remains available for validation and for the eventual high-detail export path — it is an asset for
  output quality, not a crutch for the model.
- **Tile size:** 1 km × 1 km core with a **256 m context halo** that is read but never predicted
  into. The halo is what prevents roads from dead-ending at tile seams, and it must exist from day
  one because retrofitting it invalidates every cached sample.
- **Raster stack** (1 m/px core, 1024² per tile): elevation, slope, aspect, terrain roughness,
  distance-to-water, land use (one-hot), population density, building footprint mask, building
  height, distance-to-rail, distance-to-station, aerial RGB.
- **Vector layers:** roads, buildings, water, rail, land use polygons, admin boundaries, each
  retaining its `source_id` for §7 layer isolation.
- **Graph layer:** the road network (§20 of the spec).
- **Derived morphometrics:** momepy tessellation cells + per-building and per-cell morphometric
  vectors.

**Multi-scale (§27).** The same tile is materialised at three resolutions — 1 m (micro), 8 m
(neighbourhood), 64 m (regional, with a much larger footprint). The model reads all three. This is
the cheapest possible implementation of hierarchical context and it avoids a separate regional
pipeline.

**Storage.** Zarr for raster stacks (chunked, compressed, cloud-ready), GeoParquet for vectors,
and a versioned manifest per tile recording every contributing `source_id` and dataset version.

---

## 9. Architectural representation

Three levels, deliberately separated so that ML operates on the middle one:

1. **Instance** — one building: footprint polygon, height, floors, roof form, orientation,
   setback, frontage road id, plot (tessellation cell) id.
2. **Morphometric vector** — the §8 measurables computed by momepy: area, perimeter, aspect ratio,
   compactness, orientation, shared-wall ratio, neighbour distance, cell coverage ratio, local
   density at several radii.
3. **Archetype** — a *learned* cluster in morphometric × type × context space (§10 of the spec).

**[DECIDE]** Archetypes are discovered by clustering, not authored. The named examples in the spec
("Japanese Suburban Detached Home", "Dense Urban Apartment", "Rural Farmhouse") become *validation
targets*: if unsupervised clustering on PLATEAU + OSM morphometrics does not produce clusters
recognisably matching them, that is evidence the feature set is wrong — a genuinely useful negative
signal, and one we can check in Phase 3 before any generative model exists.

---

## 10. Building classification

**[DECIDE]** A two-level extensible taxonomy: a stable coarse level (residential / commercial /
industrial / institutional / transport / agricultural / other) and an open fine level seeded from
the spec's §9 list.

Label sources, in priority order: PLATEAU building usage attribute → OSM `building=*` /
`shop=*` / `amenity=*` tags → predicted. Each label carries its provenance and confidence, so a
model is never trained on its own predictions by accident.

The taxonomy lives in a versioned config file, not in code. Adding "greenhouse" must not require a
code change or a retrain of anything upstream of the classifier head.

---

## 11. Road graph representation

Per spec §20, with the additions experience says are needed:

```
Node   : id, position (x,y,z), kind {intersection|junction|endpoint|bridge_head|tunnel_mouth},
         degree, control {none|signal|stop|roundabout}, elevation_source
Edge   : id, u, v, geometry (polyline, metric CRS),
         road_class, width_m, lane_count, oneway, grade_pct, curvature_profile,
         structure {at_grade|bridge|tunnel|embankment|cut},
         surface, source_id, confidence
Graph  : crs, tile_id, lod_level, provenance_manifest
```

Design notes:

- **Undirected topology, directed attributes.** Oneway is an edge attribute, not a topology
  decision. Everything downstream (hierarchy, tessellation, export) is easier this way.
- **`confidence` on every edge** is mandatory. Without it the visualiser (§37) cannot show what the
  model is unsure about, and §37 exists precisely to answer "is the AI actually learning?"
- **LOD is a property of the graph, not of the export.** The five levels of spec §23 are produced by
  filtering on `road_class`, which means one model produces all five and they are guaranteed
  mutually consistent.
- **Elevation is carried, not baked.** Bridges and tunnels are edge attributes with a structure
  type; the exporter decides what geometry that becomes.

---

## 12. ML approaches

**[DEFER]** — this is the decision the spec explicitly says to benchmark rather than assume, and I
agree. But the candidate set should be narrow and the first entry should be dull:

| Candidate | Role |
| --- | --- |
| **Baseline: multi-channel U-Net → road probability raster → skeletonise → graph** | The mandatory dull baseline of spec §51. Establishes the floor. Anything fancier must beat it. |
| **SAM-Road++ style: aerial-pretrained encoder + transformer GNN** | Strongest reconstruction candidate; also our pretrained encoder. |
| **Conditional diffusion on the raster stack** | Best generative-diversity candidate; weakest topology. |
| **Graph-native generative (BlockPlanner lineage)** | Best topology; hardest to train; highest risk. |

**Memory-constrained ranking.** Given §20's 16 GB budget, the candidates do not start equal. The
U-Net baseline and the pretrained-encoder route both fit comfortably at 512² patches. The diffusion
decoder does not fit at full tile resolution and would need either a patch-based reformulation or
cloud training. That is a legitimate input to the benchmark — a model that cannot be trained on the
available hardware is not a candidate, it is a wish — but it must not be used to quietly pre-decide
the outcome. **If diffusion wins on a patch-based comparison, that is a finding worth renting a GPU
for.**

**[DECIDE]** The encoder is **multi-branch with a shared fusion trunk** — a raster branch (terrain +
imagery + density), a vector/morphometric branch, and later a graph branch — fused by
cross-attention. The specific *decoder* is what gets benchmarked. Committing to the encoder shape
now is safe and gives every candidate the same inputs, which is the only way the benchmark means
anything.

**Where ML should *not* be used:** road grade compliance, turning radii, intersection geometry
validity, hierarchy consistency, bridge/tunnel necessity, network connectivity repair. These are
constraints, not patterns. Spec §30's instruction — "if a deterministic GIS/procedural solution is
superior, use it" — applies to all of them.

---

## 13. Procedural approaches

The procedural half of the hybrid (§22) does four jobs:

1. **Graph extraction** — probability field → centreline → cleaned graph. Skeletonisation plus
   graph simplification; OSMnx's consolidation logic is the reference behaviour.
2. **Constraint enforcement** — reject or reroute segments violating max grade, min radius,
   intersection angle limits; insert bridges where an edge crosses water and tunnels where it would
   cut more than *n* metres below terrain. Deterministic and testable.
3. **Hierarchy assignment** — a network-analytic pass (centrality + connectivity + land-use
   context) that labels arterials/collectors/locals consistently. ML may propose; this decides.
4. **Parcel derivation** — momepy morphological tessellation from block polygons and building
   seeds, giving §15's parcel layer.

**[DECIDE]** The interface between halves is a **road probability/corridor field plus a hierarchy
prior** — not a mesh, not a final graph. This is the narrowest waist that lets either half be
replaced independently, which spec §60.9 requires.

---

## 14. Multi-modal learning

Deferred to Phase 9 as a *training* input for the reasons in §6.4, but the representation must not
preclude it. Two requirements on today's design:

- The fusion trunk accepts a **variable set of present modalities** with learned missing-modality
  embeddings, so adding a street-imagery branch later is an additive change.
- Every tile manifest has a slot for street-level and video observations, empty for now.

That is the whole cost of keeping §19 reachable, and it is cheap enough to pay immediately.

---

## 15. Training methodology

**Task framing.** Masked environmental completion. Given the full stack with the road layer (and
later the building layer) masked out over the core tile, reconstruct it. This single framing yields
both reconstruction and generation, and it is self-supervised over the whole of Japan — no manual
labelling anywhere in the loop.

**Curriculum.** Flat urban → suburban → mixed → coastal → mountainous. Terrain-dominated cases last,
because they are where a naive model fails most informatively.

**Sampling.** Stratify by environmental archetype (spec §5), not by area. Tokyo would otherwise
supply most tiles and the model would learn Tokyo — precisely the failure §39 names.

**Reproducibility (§44).** One config file per experiment pinning dataset version, source registry
hash, preprocessing version, model version, seeds, and split definition. An experiment that cannot
be re-run from its config is treated as a failed experiment.

---

## 16. Validation

### 16.1 Splits (§29)

**[DECIDE]** Splits are by **city/region**, never by tile. Random tile splits are indefensible here:
spatial autocorrelation means random CV can be up to ~40% optimistic on geospatial tasks, and with a
256 m halo, adjacent tiles literally share input pixels. `spacv`-style block/cluster splitters are
used, with a **buffer zone** of at least one tile discarded between folds.

Held-out sets must additionally cover **unseen archetypes**, not just unseen cities — otherwise
"generalisation" only proves transfer between similar places.

### 16.2 Metrics

- **Topology:** APLS and TOPO (standard in road-graph extraction), plus connectivity, node-degree
  distribution, dead-end ratio, intersection density.
- **Geometry:** segment length distribution, curvature distribution, orientation entropy (a good
  grid-vs-organic discriminator), grade distribution.
- **Environmental compatibility:** fraction of network exceeding max grade, unbridged water
  crossings, settlement coverage, land-use compatibility.
- **Morphological:** distributional distance between generated and real morphometric vectors —
  building density, footprint size, setback, type mix.
- **Sensitivity (§1.3):** the counterfactual sweep. Perturb one environmental channel, measure the
  response. This is the metric that actually tests the project's thesis, and I am not aware of it
  being standard practice in this literature — which is a reason to adopt it, not to avoid it.
- **Novelty (§6.1d):** nearest-neighbour distance from each generated tile to its closest training
  tile under a structural similarity measure. Low distance means the generator is memorising rather
  than generalising — a modelling failure and a licensing exposure in the same number.

**Explicitly rejected:** pixel-level similarity as a primary metric. A structurally excellent
network that differs from the real one scores badly, and per spec §38 that would be the wrong
answer.

---

## 17. Unity architecture

Thin exporter, zero logic. Core emits a **stable, versioned interchange bundle** (GeoParquet +
JSON manifest + optional glTF). A C# package reads it and produces Unity Splines, meshes, terrain
data, and prefab placements.

**[DECIDE]** No ML and no GIS in Unity. The Unity package must not be able to change generated
output — if it can, the same scene will differ between Unity and Unreal, which defeats §36's
requirement not to duplicate logic.

## 18. Unreal architecture

Same bundle, C++/Blueprint plugin, mapping to Unreal Splines, Landscape, procedural meshes, and
**PCG-compatible point/attribute sets**. PCG is the natural fit for the roadside-environment layer.

**[DECIDE]** Both exporters are built against a **conformance test suite**: the same bundle in, a
fixed set of assertions about node positions, edge counts and hierarchy out. Divergence between
engines becomes a test failure rather than a discovery six months later.

Also emit **OpenDRIVE `.xodr`** via `scenariogeneration`, validated in CI with `esmini`'s road
manager. This is the cheapest possible external correctness check on our own topology — a
third-party parser either accepts our network or does not.

---

## 19. Performance

Tiling makes the whole system embarrassingly parallel. Targets and tactics:

- Preprocessing parallel per tile; Zarr chunk = tile.
- Inference batched over tiles; halo overlap resolved by a deterministic seam-merge pass.
- Hierarchical generation: regional pass (64 m) fixes arterials, then local passes (1 m) infill,
  conditioned on the regional result. This is also what makes long-range structure possible at all.
- Aggressive caching keyed by `(tile_id, source_versions, preprocessing_version)`.
- Never load a region whole (spec §43).

**[DEFER]** Rust or C++ for the core. Python + numpy/GDAL is correct for Phase 1–4. Port only what
profiling proves is hot — most likely tessellation and seam-merge, not the model.

## 20. Hardware requirements

**Confirmed environment: AMD Radeon RX 6800, 16 GB VRAM, Linux.** (Revision 2 recorded Windows 11
here; the training machine moved to Linux on 2026-08-11 — see the update note below.) Plan around
it; do not assume
a larger card appears.

### 20.1 The real constraint is the software stack, not the VRAM **[FLAG]**

16 GB is workable — §20.2 shows the budget closes. The genuine risk is that **this is an AMD RDNA2
card (gfx1030) on Windows**, which is the least-supported quadrant of the PyTorch ecosystem:

- AMD's official PyTorch wheels for Windows target **RDNA3 (RX 7000) and RDNA4 (RX 9000) only**.
  gfx1030 is supported by the HIP SDK but **not** by the official Windows PyTorch wheels.
- **ROCm on WSL2 does not reliably work on the RX 6800** — there is a long-standing open issue.
- Working routes exist but are community-maintained: semi-official ROCm nightly builds compiled for
  gfx103x (reported working on RX 6800), community UV-managed PyTorch+ROCm setups for gfx1030, and
  building ROCm 7.x from source on Windows.
- **On Linux the picture is materially better** — gfx1030 appears in ROCm's supported configurations
  (Ubuntu 22.04/24.04, RHEL 9.6) and ROCm 7 is production on Linux.

This is a schedule risk, not a capability risk, and it is worth confronting immediately rather than
discovering it in Phase 4.

> **Update 2026-08-11: the training machine runs Linux, and the spike has now passed.** The risk
> described below was specific to gfx1030 *on Windows*; the Windows timebox and the dual-boot
> fallback are both moot. The stack came up unmodified on `torch 2.9.1+rocm6.4` with fp16 autocast
> working and **4.03 GB peak at 512² × 15 channels, batch 8** — comfortably inside the budget
> §20.2 sets out below, and without the gradient checkpointing that estimate assumed. **R1b is
> closed.** Everything from here to the end of §20.1 is retained as the record of a resolved risk,
> not as live guidance. Measurements and wheel-selection notes are in
> [decision-log.md](decision-log.md).

**[DECIDE] Add a Phase 0.5 "GPU spike"** — a half-day, before any modelling work: install a
candidate stack, train a small U-Net on random tensors, confirm fp16 works and memory reporting is
sane.

**The owner has confirmed a Linux dual-boot is acceptable**, which turns this from an open risk into
a bounded one: there is a known-good destination, and the only question is whether the Windows route
saves the trouble of getting there. That changes how the spike should be run:

1. **Timebox the Windows attempt to one day.** Community ROCm nightlies for gfx103x are reported
   working on the RX 6800, and if it comes up quickly it is worth having. If it does not, stop —
   do not debug a community build for a week when a supported path exists.
2. **Dual-boot Linux + ROCm** is the fallback and the expected destination. gfx1030 appears in
   ROCm's supported configurations on Ubuntu 22.04/24.04 and RHEL 9.6, with ROCm 7 in production.
3. Rented cloud GPU for individual training runs that exceed 16 GB (realistically only a diffusion
   decoder at full resolution).
4. `torch-directml` is now struck from the list. It only existed as an option because there was no
   acceptable Linux path; there is.

**[DECIDE] Consequence for tooling:** since Linux is a likely target, the pipeline must not acquire
Windows-only dependencies. This costs nothing today — GDAL, PyTorch, PDAL and the whole geospatial
stack are cross-platform — but it does mean paths, subprocess calls and any GPU-specific code should
be written portably from the start rather than ported later.

Phases 1–3 are CPU/GIS work and are **completely unaffected**, so this resolves in parallel with real
progress. That is why the spike is scheduled early but blocks nothing.

### 20.2 Designing for 16 GB

Per the owner's instruction to optimise for memory efficiency rather than assume more VRAM:

| Technique | Effect |
| --- | --- |
| **Patch-based training** — 512² crops from 1024² tiles, full tiles only at inference | Largest single saving; ~4× activation reduction |
| **Mixed precision (fp16)** | ~2× on activations. Note RDNA2 has **no bf16 hardware** — use fp16 with loss scaling, not bf16 |
| **Gradient checkpointing** on encoder blocks | Trades ~30% compute for large activation savings |
| **Gradient accumulation** | Decouples effective batch size from VRAM |
| **Channel discipline** — a curated ~12-channel stack, not every raster we can compute | Cheapest saving available, and it improves interpretability |
| **Frozen pretrained encoder** in early experiments | Removes encoder optimiser state entirely |
| **Tiled/patch inference with halo overlap** | Already required for seam handling (§8); doubles as a memory strategy |

**Budget check** — U-Net baseline, 512² × 12 channels, batch 8, fp16, checkpointing: roughly 6–9 GB.
Comfortable. A SAM-class pretrained encoder fine-tuned at 512² with a frozen backbone: roughly
10–13 GB. Tight but feasible. **A full diffusion decoder at 1024² is the case that does not fit** —
which is a reason to treat the diffusion candidate in §12 as the one requiring a patch-based
reformulation or cloud training, and to weight the benchmark accordingly.

- **Phase 1–3:** any modern workstation; disk is the constraint. Budget ~2–5 TB — VIRTUAL SHIZUOKA
  point clouds alone are large (files up to 5.2 GB).
- **Inference:** single GPU; CPU-only fallback viable for the entire procedural half.

---

## 21. Risks

| # | Risk | Severity | Mitigation |
| --- | --- | --- | --- |
| R1 | ~~GSI Survey Act blocks release~~ **Closed.** Owner decision + VIRTUAL SHIZUOKA removed this risk entirely, at a net *gain* in terrain resolution | — | Closed 2026-08-10 |
| R1b | ~~ROCm/gfx1030 training stack may not come up~~ **Closed 2026-08-11.** Spike passed on torch 2.9.1+rocm6.4: fp16 working, 4.03 GB peak at 512²×15 batch 8 | — | Closed. See [decision-log.md](decision-log.md) |
| R1f | Generated output memorises and reproduces a specific real place, weakening the "predictions are unencumbered" position (§6.1d) | Medium | Nearest-neighbour novelty check in the §16.2 metric suite; it doubles as a generalisation measure |
| R1c | Reconstruction output derived from OSM carries ODbL share-alike, conflicting with commercial intent (§6.1c) | High | PLATEAU/NLNI-primary output path; OSM confined to training; `redistribution_class` on every export |
| R1d | ~~High-resolution aerial imagery has no confirmed open source after GSI exclusion~~ **Closed 2026-08-10.** VIRTUAL SHIZUOKA ships orthorectified imagery *and* bare-earth point clouds under the same licence and CRS | — | See [site-selection.md](site-selection.md) |
| R1e | VIRTUAL SHIZUOKA is one prefecture — the 0.5 m terrain advantage does not generalise nationally | Medium | Accept a two-tier terrain model (0.5 m MVP, 30 m national); do not let the model learn to depend on 0.5 m detail |
| R2 | Model learns "urban ⇒ dense grid" and ignores terrain — i.e. learns texture, not structure | High | The §16.2 sensitivity sweep detects this directly; archetype-stratified sampling reduces it |
| R3 | Scope. The spec describes a multi-year system; effort disperses across 60 sections and nothing ships | High | Hard MVP definition (§23 below); each phase gated on a measurable result |
| R4 | Tokyo dominance in training data | Medium | Archetype stratification; region-held-out evaluation |
| R5 | PLATEAU coverage is city-centric — rural and mountain villages are thinnest nationally | ~~Medium~~ **Low for the MVP** | Downgraded 2026-08-10: Shizuoka coverage is 22 cities + 12 towns including Kawanehon (mountain valley) and the Izu coastal towns. Still applies when the project leaves this prefecture |
| R6 | Two incompatible copyleft licenses (ODbL + CC BY-SA) fuse into one corpus | Medium | §6.4 exclusion; `share_alike` field checked mechanically |
| R7 | Generated networks are locally plausible but globally incoherent (no through-routes) | Medium | Hierarchical generation (§19); connectivity metrics as first-class |
| R8 | Vector/raster misalignment silently corrupts training | Medium | Phase 2 visualiser exists specifically to catch this *before* Phase 4 |

---

## 22. Limitations

Stated plainly, because a research system that overstates its reach is worse than one that does not
exist:

- The system learns **correlation in built form**, not causation. It cannot know *why* a road was
  built; it can only know what usually accompanies what.
- Japan-trained models will transfer poorly to other countries without retraining. Land-use policy,
  parcel law and vehicle norms are baked into the data.
- Open data lags reality by months to years. New development will be missing.
- Historical development inference (§16 of the spec) is bounded by PLATEAU's construction-year
  attribute coverage; where that is absent, it is speculation.
- **The MVP will not produce visually impressive output.** It will produce measurably
  environment-responsive output. Those are different things and conflating them is how this class of
  project dies.

---

## 23. Development roadmap

Each phase has an exit criterion. A phase is not finished when the code runs; it is finished when
the criterion is met.

| Phase | Content | Exit criterion |
| --- | --- | --- |
| **0. Research** | This document + provenance registry | ✅ Reviewed; GSI decision made; commercial intent, hardware and ordering settled |
| **0.5 GPU spike** | Stand up a working PyTorch stack on the RX 6800 (§20.1) | A toy U-Net trains in fp16 on the GPU, or a fallback is chosen. Runs *in parallel* with Phase 1 |
| **1. Data pipeline** | Ingest OSM, PLATEAU, DEM, land use, population, rail → tiled representation | One command produces a valid, manifest-carrying tile set for the MVP area |
| **2. Visualisation** | Standalone layer-toggling viewer | A human can confirm every layer is spatially aligned. **Gate: no modelling before this passes.** |
| **3. Road analysis** | Morphometrics + correlation study across archetypes | A ranked, quantified list of which environmental features actually predict road structure — with the null results stated |
| **4. First ML prototype** | U-Net baseline → probability → graph | Beats a non-learned prior on APLS/TOPO on a held-out city |
| **5. Env-conditioned generation** | Full hybrid; benchmark the §12 candidates | **Passes the sensitivity sweep** (§1.3). This is the project's real proof point |
| **6. Building understanding** | Classification, morphometrics, archetype discovery | Discovered clusters match §9's named archetypes without being told about them |
| **7. Road + building generation** | Road → parcel → type → placement → geometry | Generated environments are internally coherent (no warehouses on residential lanes) |
| **8. Engine integration** | Unity + Unreal exporters + OpenDRIVE | Same bundle, both engines, conformance suite green |
| **9. Multi-modal** | Street imagery, video, historical maps | Measurable improvement over Phase 5 baseline — otherwise dropped |

### MVP definition (spec §57)

**[DECIDE] Region: Shizuoka Prefecture.** Revision 1 left the area open. The Revision 2 findings
select it almost mechanically, and for a reason worth stating plainly: **the MVP region should be
chosen by where the best openly-licensed data exists, not chosen first and then made to work with
whatever data is available.** Shizuoka is the only place in Japan I found that offers all four of:

1. **0.5 m open terrain** under CC BY 4.0 (VIRTUAL SHIZUOKA), prefecture-wide.
2. **PLATEAU coverage** across multiple municipalities — required for the §6.1c redistributable
   output path, not merely convenient.
3. **All three target archetypes within one prefecture**, sharing one CRS zone and one terrain
   source:
   - *flat suburban* — the Hamamatsu / Shizuoka city plains;
   - *mountain valley with railway* — the Ōi River valley, a textbook case of terrain-forced
     road/rail/settlement co-location;
   - *coastal constrained* — the Izu peninsula towns, where terrain pins settlement against the sea.
4. Already published in **JGD2011 Plane Rectangular Zone 8**.

Three contrasting sites rather than one, because a single area cannot demonstrate environmental
responsiveness at all — the entire thesis is about *variation*.

Exact municipal selection is a Phase 1 task, gated on confirming PLATEAU coverage per municipality.

**MVP delivers:** ingest → tiled environmental representation → visualiser → road analysis →
baseline ML reconstruction + procedural constraint pass → engine-independent road graph → export
(GeoJSON + OpenDRIVE + Unity bundle).

**MVP does not deliver:** buildings beyond footprints and morphometrics, architecture generation,
street imagery, full-city scale, photorealism.

**MVP must prove exactly one thing:** that environmental information measurably changes the road
network. Everything else in this document is downstream of that result.

### Reconstruction first — confirmed, with one adjustment

The owner confirmed reconstruction before generation: learn from real environments and demonstrate
measurable reproduction of terrain/building/land-use/road relationships, then use that
representation as the foundation for generation.

The roadmap already assumed this, so the ordering is unchanged. One adjustment follows from §6.1c:
because reconstruction is the licence-encumbered mode, **the reconstruction output path must be
built on PLATEAU/NLNI geometry from Phase 1**, not retrofitted after the model works. Building it
OSM-first and swapping later would mean re-deriving the corpus and re-validating every metric.

There is also a genuine methodological benefit to this ordering that is worth naming: reconstruction
has **ground truth**. Generation does not. Establishing the environmental representation against a
measurable target first means that when generation later produces something odd, we can tell whether
the representation or the generator is at fault. Doing it the other way round makes every failure
ambiguous.

**Where the commercial value sits.** The owner has indicated the long-term commercial emphasis is on
**generated** environments, with reconstruction supported as well. That does not change the build
order — reconstruction remains the foundation, for the ground-truth reason above — but it does
change how the roadmap should be read:

- Phases 1–4 build the **representation and its validation**. They are infrastructure. Judging them
  by how commercially interesting they look is a category error.
- **Phase 5 is where commercial value begins**, and Phases 6–7 are where it compounds.
- Phase 5's exit criterion — the sensitivity sweep — is therefore not merely a research checkpoint.
  It is the gate that says the foundation is sound enough to build a product on. Passing it weakly
  and moving on would be the single most expensive mistake available in this plan.

### Designing for later visual quality

The owner asked that higher-quality visual generation be addable later without rebuilding the
system. Concretely, this requires exactly three things, all cheap now and expensive later:

1. **Geometry and appearance stay separate.** The core model carries structure — footprints,
   heights, types, orientations, setbacks. It never carries materials, textures or meshes. Appearance
   is a function applied to structure at export time.
2. **The LOD ladder is a first-class axis** (spec §34): footprint → simple 3D → archetype →
   architectural features → detailed procedural → asset replacement. Every consumer declares the
   level it wants. Nothing in the core assumes a level.
3. **Archetypes are the hand-off point.** A building archetype is what a future high-quality
   generator consumes. Getting the archetype vocabulary right in Phase 6 is what makes Level 4–5
   possible without touching Phases 1–5.

The 0.5 m VIRTUAL SHIZUOKA terrain also matters here: it is far more detail than the *model* needs,
but it is exactly the detail a high-quality *export* will want later. Keeping the raw data alongside
the downsampled training tier (§8) costs disk and preserves that option.

---

## Decisions received (2026-08-10)

All four open questions from Revision 1 are answered and incorporated.

| # | Question | Decision | Effect |
| --- | --- | --- | --- |
| 1 | GSI Survey Act | **Avoid bulk GSI use**; treat as optional future component | §6.3 rewritten; GSI quarantined; VIRTUAL SHIZUOKA adopted at 0.5 m — a net improvement |
| 2 | Commercial use | **Yes, potentially**; design for it from the start | §6.1b/§6.1c added; redistributable-core output path defined; R1c raised |
| 3 | Hardware | **16 GB AMD RX 6800**, optimise for memory | §20 rewritten; Phase 0.5 GPU spike added; R1b raised |
| 4 | Ordering | **Reconstruction first** | Roadmap confirmed; output path must be PLATEAU-primary from Phase 1 |

### Follow-up decisions received (2026-08-10)

| # | Question | Decision | Effect |
| --- | --- | --- | --- |
| 5 | Linux dual-boot acceptable? | **Yes** | R1b downgraded High→Medium; Windows attempt timeboxed to one day; DirectML struck; cross-platform tooling required |
| 6 | Redistribution scope | **Both modes; long-term commercial emphasis on generation; keep reconstruction legally clean regardless** | §6.1c strictness unchanged; §6.1d added on memorisation; novelty metric added to §16.2 |

## Open questions remaining

**None.** The last one — aerial imagery for Shizuoka (§4.2, R1d) — was closed on 2026-08-10 during
Phase 1: VIRTUAL SHIZUOKA ships orthorectified imagery and bare-earth point clouds alongside the
terrain, under the same licence and CRS. Site selection is recorded in
[site-selection.md](site-selection.md).

Everything remaining is a matter of doing the work rather than deciding what the work is.

---

## Sources

- [Project PLATEAU (MLIT)](https://www.mlit.go.jp/plateau/en/) · [PLATEAU site policy](https://www.mlit.go.jp/plateau/site-policy/) · [Integrating PLATEAU data into OSM, SotM 2025](https://pretalx.com/sotm2025/talk/N89YSU/) · [OGC discussion paper on the PLATEAU ecosystem](https://docs.ogc.org/dp/25-032.html) · [Standard Data Product Specification for 3D City Models (ISPRS)](https://isprs-annals.copernicus.org/articles/X-4-W6-2025/129/2025/isprs-annals-X-4-W6-2025-129-2025.pdf)
- [GSI Website Terms of Use](https://www.gsi.go.jp/ENGLISH/page_e30286.html) · [GSI tile list](https://maps.gsi.go.jp/development/ichiran.html) · [GSI tile usage](https://maps.gsi.go.jp/development/siyou.html) · [基盤地図情報 download service](https://service.gsi.go.jp/kiban/) · [基盤地図情報 FAQ](https://www.gsi.go.jp/kiban/faq.html) · [地図の利用手続パンフレット](https://www.gsi.go.jp/common/000223838.pdf) · [Japan elevation data guide](https://www.gpxz.io/blog/japan-dem-guide)
- [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/) · [土地利用細分メッシュ L03-b](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-b.html)
- [OSMF Licence/Community Guidelines](https://osmfoundation.org/wiki/Licence/Community_Guidelines) · [Collective Database Guideline](https://osmfoundation.org/wiki/License/Community_Guidelines/Collective_Database_Guideline_Guideline) · [OSMF Attribution Guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines) · [OSM Wiki: ODbL](https://wiki.openstreetmap.org/wiki/Open_Database_License)
- [Overture attribution and licensing](https://docs.overturemaps.org/attribution/) · [Overture buildings guide](https://docs.overturemaps.org/guides/buildings/)
- [Mapillary CC BY-SA for open data](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data) · [Mapillary licensing discussion](https://forum.mapillary.com/t/licensing-issues-for-using-mapillary-images-on-wikimedia-commons/3821) · [KartaView](https://en.wikipedia.org/wiki/KartaView)
- [Towards Satellite Image Road Graph Extraction (CVPR 2025)](https://arxiv.org/html/2411.16733) · [samroadplus](https://github.com/earth-insights/samroadplus) · [Sat2Graph](https://www.semanticscholar.org/paper/Sat2Graph:-Road-Graph-Extraction-through-Encoding-He-Bastani/c428a60015a5d074bc257892e1660ad7bff4a42b) · [Deep learning road extraction review (ISPRS J.)](https://www.sciencedirect.com/science/article/pii/S0924271625002758) · [GLD-Road](https://www.sciencedirect.com/science/article/abs/pii/S0924271625002886)
- [RoBus dataset](https://arxiv.org/pdf/2407.07835) · [Context-informed multimodal diffusion for urban morphology](https://arxiv.org/pdf/2409.17049) · [CityGen](https://arxiv.org/pdf/2312.01508) · [Controllable 3D urban layout generation](https://arxiv.org/html/2509.23804) · [Generating urban road networks with conditional diffusion](https://www.researchgate.net/publication/381502461_Generating_Urban_Road_Networks_with_Conditional_Diffusion_Models)
- [momepy](https://github.com/pysal/momepy) · [momepy morphological tessellation](http://docs.momepy.org/en/stable/user_guide/elements/tessellation.html) · [momepy urban type detection](http://docs.momepy.org/en/stable/examples/clustering.html)
- [ASAM OpenDRIVE](https://www.asam.net/standards/detail/opendrive/) · [esmini](https://github.com/esmini/esmini) · [awesome-openx](https://github.com/benediktschwab/awesome-openx)
- [Spatial cross-validation for GeoAI](https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf) · [spatialCV](https://github.com/geoai-lab/spatialCV) · [Spatial+ CV method](https://www.sciencedirect.com/science/article/pii/S1569843223001887) · [Data leakage in online mapping datasets](https://arxiv.org/pdf/2312.06420)
- [Meijo GNSS/IMU open dataset (Odaiba, Tokyo)](https://github.com/MeijoMeguroLab/Open_data)

**Added in Revision 2:**

- [VIRTUAL SHIZUOKA 富士山南東部・伊豆東部 point cloud](https://www.geospatial.jp/ckan/dataset/shizuoka-2019-pointcloud) · [中・西部](https://www.geospatial.jp/ckan/dataset/virtual-shizuoka-mw) · [北西部 release](https://front.geospatial.jp/data/2025/10/6726/) · [VIRTUAL SHIZUOKA overview (Shizuoka Prefecture)](https://www.pref.shizuoka.jp/machizukuri/1049255/1052183.html)
- [ALOS AW3D30 (JAXA)](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/) · [AW3D30 v4.1 product description](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/data/aw3d30v4.1_product_e_1.0.pdf) · [AW3D30 v4.1 in Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1)
- [Copernicus DEM collection description](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) · [COP-DEM GLO-30 licence](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf) · [Copernicus DEM on AWS Open Data](https://registry.opendata.aws/copernicus-dem/)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-7.0.0/compatibility/compatibility-matrix.html) · [ROCm/PyTorch install for gfx1030](https://github.com/patientx/ComfyUI-Zluda/issues/431) · [ROCm on Windows for gfx1030](https://github.com/ssubedir/RCOm-windows-gfx1030) · [ROCm on WSL2 + RX 6800 issue](https://github.com/ROCm/ROCm/issues/3371) · [Building ROCm 7.1 + PyTorch on Windows for unsupported GPUs](https://medium.com/@guinmoon/building-rocm-7-1-and-pytorch-on-windows-for-unsupported-gpus-my-hands-on-guide-0758d2d2b334)
