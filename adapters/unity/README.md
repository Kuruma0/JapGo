# JapGo Roads — Unity package

Imports a JapGo road bundle into a Unity scene, at edit time or at runtime.

## Install

Package Manager → **Add package from disk…** → pick
`adapters/unity/com.japgo.roads/package.json`. Or add to your project's `Packages/manifest.json`:

```json
"com.japgo.roads": "file:../../JapGo/adapters/unity/com.japgo.roads"
```

Requires Unity 2021.3 or later and `com.unity.nuget.newtonsoft-json`, which Package Manager pulls
in automatically.

## Use it from the editor

**Window → JapGo → Import road bundle**, point it at a bundle directory, press Import. Options
cover mesh generation, material, width scale, surface offset and junction markers. The result is
registered with undo, and "Save as prefab" also writes the generated meshes into an asset next to
it — a prefab referencing in-memory meshes looks fine until the project is reopened.

## Use it at runtime

```csharp
using JapGo.Roads;

var bundle = RoadBundle.Load(Path.Combine(Application.streamingAssetsPath, "world"));
var roads  = RoadNetworkBuilder.Build(bundle, new RoadBuildOptions
{
    GenerateMesh = true,
    RoadMaterial = myRoadMaterial,
});

Debug.Log(bundle.Describe());   // "880 roads, 347 junctions, 46.28 km, seed 42, absolute heights"
```

`RoadBundle.Parse(roadsJson, junctionsJson, manifestJson)` takes strings instead, for bundles that
arrive from an addressable, a web request, or a world generated moments earlier.

## What you get

```
JapGo <bundle name>          RoadNetworkInfo — seed, CRS, origin, summary
├── Roads
│   ├── e0042 (residential)  RoadSegment + MeshFilter/MeshRenderer, or a LineRenderer
│   └── …
└── Junctions
    └── n0117 (degree 3)     an empty transform at the node
```

`RoadSegment` keeps the validated centreline alongside the mesh, so a project with its own road
mesher can turn `GenerateMesh` off and build from `RoadSegment.Centreline` instead.

The generated mesh is a flat ribbon — the centreline offset by half the carriageway width and
lifted a few centimetres. No crown, no kerbs, no junction fill, on purpose: the core carries
structure and never appearance, and anything prettier is your project's art direction rather than
an importer's business.

## Not compiled

Unity is not installed on the development machine, so this package has never been built or run.
The parsing, frame maths and API shapes were reasoned through carefully; the compiler has not
seen them.
