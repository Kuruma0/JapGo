# MVP site selection

**Resolved 2026-08-10 (Phase 1).** Closes the last open question from Phase 0.

The research document chose Shizuoka Prefecture as the MVP region on the strength of VIRTUAL
SHIZUOKA's 0.5 m open terrain, but left two things unconfirmed: whether PLATEAU actually covers the
municipalities we need, and whether open high-resolution aerial imagery exists after the GSI
exclusion. Both are now answered, and both better than expected.

## PLATEAU coverage in Shizuoka is exceptional

The national caveat — that PLATEAU is city-centric and thin in rural and mountain areas, logged as
risk R5 — **does not bind in this prefecture**. Coverage is 22 cities and 12 towns, most with both
FY2022 and FY2023 vintages, and it includes the rural and coastal municipalities that are usually
missing:

| Archetype | Municipalities with PLATEAU coverage |
| --- | --- |
| Flat suburban | Hamamatsu 浜松市, Iwata 磐田市, Fukuroi 袋井市, Kakegawa 掛川市, Fujieda 藤枝市 |
| Mountain valley | **Kawanehon 川根本町** (FY2023), Shimada 島田市 (FY2023), Izu 伊豆市 |
| Coastal constrained | Atami 熱海市, Ito 伊東市, Shimoda 下田市, Higashi-Izu 東伊豆町, Nishi-Izu 西伊豆町, Minami-Izu 南伊豆町, Matsuzaki 松崎町 |

## Imagery: risk R1d is closed

VIRTUAL SHIZUOKA does not only ship point clouds. It also ships **orthorectified imagery**,
**ground-filtered (bare-earth) point clouds**, 1 m contours and water polygons — under the same
CC BY 4.0 / ODbL dual licence, in the same CRS, over the same area.

That resolves the one genuinely unresolved input after the GSI exclusion. It also removes a subtler
hazard: with only a DSM available, slope and grade measures would have been computed over tree
canopy and rooftops. Bare-earth returns make terrain-derived features trustworthy, which matters
because §12 assigns grade compliance to the deterministic half of the system, where it is expected
to be exact.

Shizuoka Prefecture separately publishes FY2019 and FY2021 orthophotos, but only within
市街化区域 (urbanisation promotion areas) — which excludes exactly the mountain villages we care
about. VIRTUAL SHIZUOKA's imagery is preferred on coverage, licence and CRS alignment alike.

## The three MVP sites

Selected to maximise *contrast* in the environmental variables the model is supposed to respond to,
while holding data sources, licence and CRS zone constant across all three. Holding those constant
matters: if the three sites differed in data provenance as well as terrain, a difference in output
could not be attributed to the environment.

### 1. Hamamatsu / Iwata plain — the easy case

Flat alluvial plain, regular blocks, medium-density detached housing with some mid-rise. Rail and
arterial grid. This is the case where a naive model should already do well; if it does not, the
pipeline is broken rather than the hypothesis.

*Expected structure:* high road density, high intersection density, low orientation entropy (grid),
near-zero grade.

### 2. Kawanehon 川根本町 / upper Ōi River valley — the terrain-dominated case

The most important of the three, and the reason the region choice works. A steep valley with the
Ōigawa Railway running through it, settlement pinned to narrow valley floor and terraces, roads
forced to follow contours with switchbacks where they climb.

This is where the project's thesis is falsifiable. A model that has learned "urban ⇒ dense grid"
and nothing else will produce a grid here, and the sensitivity sweep will catch it. It is also the
case that most depends on the 0.5 m bare-earth terrain being correct.

*Expected structure:* low road density, high curvature, strong alignment with contours and the
river corridor, dead-end ratio far above the plain, settlement clustered near the railway.

### 3. Izu coastal towns (Atami / Ito / Higashi-Izu) — the constrained-boundary case

Terrain meeting sea. Settlement compressed into a narrow coastal strip and up steep slopes behind
it; roads run parallel to the coast because nothing else is possible. Atami in particular is
famously steep, with dense development on gradients that would be undeveloped elsewhere.

Included because it tests a *different* terrain response from the valley: not "follow the corridor"
but "fill the only available land". A model that has learned one but not the other has learned a
rule rather than a relationship.

*Expected structure:* strongly anisotropic orientation, high density despite high slope, sharp
density falloff inland, elevation-correlated road hierarchy.

## Why not one site

The MVP must prove that environmental information measurably changes the road network. A single
site cannot demonstrate a *change* — there is nothing to contrast against. Three sites sharing
everything except terrain and settlement pattern are the minimum configuration in which the
sensitivity sweep means anything.

## Confirmed before Phase 1 ingest

- [x] PLATEAU coverage exists for all three archetypes
- [x] Open high-resolution imagery exists (VIRTUAL SHIZUOKA ortho)
- [x] Bare-earth terrain available, not just DSM
- [x] All sources in the redistributable core — output is attribution-only
- [x] Single CRS zone (JGD2011 Plane Rectangular Zone 8) across all three sites
- [ ] Exact tile extents per site — pending first ingest run
- [ ] PLATEAU vintage selected per municipality (FY2023 preferred where available)

## Sources

- [PLATEAU open data portal](https://www.mlit.go.jp/plateau/open-data/) · [G-Spatial PLATEAU portal](https://front.geospatial.jp/plateau_portal_site/) · [Shizuoka City FY2023](https://www.geospatial.jp/ckan/dataset/plateau-22100-shizuoka-shi-2023) · [Numazu PLATEAU page](https://www.city.numazu.shizuoka.jp/shisei/office/ichiran/toshikei/machiseisaku/plateau/)
- [VIRTUAL SHIZUOKA central/west](https://www.geospatial.jp/ckan/dataset/virtual-shizuoka-mw) · [SE Fuji / E Izu](https://www.geospatial.jp/ckan/dataset/shizuoka-2019-pointcloud) · [Shizuoka Prefecture datasets on G-Spatial](https://www.geospatial.jp/ckan/dataset?organization=shizuokapref)
- [Shizuoka Prefecture open data portal](https://opendata.pref.shizuoka.jp/)
