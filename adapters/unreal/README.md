# JapGo Roads — Unreal plugin

Imports a JapGo road bundle as a spline road network, at runtime or from a Blueprint.

## Install

Copy `JapGoRoads/` into your project's `Plugins/` directory and rebuild. Runtime module, no
third-party dependencies — parsing uses Unreal's built-in `Json` module.

## Use it

From a Blueprint, an Editor Utility Widget, or the level Blueprint:

**Import Bundle Directory** → *Directory* = the folder holding `roads.geojson`,
`junctions.geojson` and `manifest.json`. It returns an `AJapGoRoadNetwork`, or null with
*Out Error* set.

From C++:

```cpp
FJapGoImportOptions Options;
FString Error;
AJapGoRoadNetwork* Network =
    UJapGoRoadImporter::ImportBundleDirectory(this, BundleDirectory, Options, Error);
```

`ImportBundleText` takes the three documents as strings, for bundles streamed from a pak file, a
download, or a world generated moments earlier.

## What you get

An `AJapGoRoadNetwork` actor carrying one `USplineComponent` per road, plus:

- `Roads` — id, class, width in metres, grade in percent, and the spline itself
- `Junctions` — id, degree, position, incident road ids
- `Seed`, `Crs`, `Origin`, `Summary` — provenance, so the actor can say which world it is

**Splines, not meshes.** Every Unreal project already has a road mesher it likes — spline meshes,
Landmass, something from the marketplace, something in-house — and all of them take a
`USplineComponent`. Generating geometry here would mean this plugin having opinions about art
direction, which the core spends an invariant avoiding. What the bundle carries is structure, and
structure is exactly a spline plus metadata.

Tangents are left to Unreal's auto-curve by default. The exporter has already smoothed each
centreline under a displacement cap and resampled it evenly, so the points are curve-ready;
forcing linear tangents puts the extractor's corners back.

## Not compiled

Unreal is not installed on the development machine, so this plugin has never been built or run.
The parsing, frame maths and API shapes were reasoned through carefully; the compiler has not
seen them.
