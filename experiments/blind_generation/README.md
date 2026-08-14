# Blind generation on synthetic terrain

**Question.** Can the frozen model produce a plausible road network for terrain it has never seen,
given only the terrain?

**Answer.** No. Six of twelve worlds produced not one cell above the model's threshold, seven
produced no extractable edge, and nine finished with no road at all. Across 192 km² the total
output is **1.04 km** of disconnected fragments. `road_v1` reconstructs roads from the earthworks
they are cut into; it does not propose roads for a landscape.

Read [report.html](report.html) — it carries the panels, the tables and the reasoning. This file
covers what is here and how to rebuild it.

## The control that makes the result mean something

An empty world proves nothing on its own. Three explanations were live, and each is separable by
applying the same treatment to a **real** tile whose full-channel answer is known:

| treatment | hamamatsu (held out) | kawanehon (trained) |
| --- | --- | --- |
| every channel | 19.98% | 4.64% |
| terrain channels only | 9.84% (×0.49) | 3.23% (×0.70) |
| terrain only, DEM blurred 4 m | **0.00035%** (×0.00004) | **0.102%** (×0.032) |

Withholding the building and land-use channels costs a factor of two. Blurring the terrain by four
metres — which leaves every valley, ridge and slope intact — costs a factor of 28,000. What a 4 m
blur removes is the metre-scale signature of the road itself.

The competing hypothesis, that synthetic ground is too smooth, was tested and rejected: adding
0.05–0.4 m of Gaussian micro-relief to a synthetic window made the response *worse*, from a peak of
0.110 to 0.028.

## Reproducing it

```bash
japgo blind
```

Twelve worlds, two controls, about six minutes on the RX 6800. Every world is regenerated from its
parameters and seed, so the heightfields are not stored — `worlds/<name>/params.json` is the
complete recipe. To rebuild only the report from saved metrics and images, with no GPU:

```bash
japgo blind --report-only
```

## Layout

```
report.html            the experiment, with every panel and table
metrics/all.json       controls and all twelve worlds, machine-readable
images/                every panel as a file, so the report rebuilds without the model
worlds/<name>/
    params.json        terrain parameters and seed — the complete recipe
    metrics.json       per-stage measurements for this world
    world.html         self-contained page for this world alone
    roads.geojson      engine-ready bundle, such as it is
    junctions.geojson
    manifest.json
```

## Conditions

- Four archetypes — mountain valley, coastal, plain, basin — three seeds each, 4 km square at
  1 m/px.
- Terrain generated from parameters and a seed, with no reference to any tile, raster or road
  network in the corpus.
- **Terrain channels only.** The ten building and land-use channels are held at the stack's nodata
  value for every world; they are exactly the settlement information the experiment must not
  supply.
- One generation configuration throughout — same threshold, same repair, validation, grade and
  geometry settings, same generation seed. No world was tuned or re-run for a better picture.
- **APLS, TOPO and pixel F1 are not applicable.** There is no ground truth for a place that does
  not exist, and no artificial "correct" network was constructed to score against.
