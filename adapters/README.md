# Engine adapters

Two importers for the interchange bundle: one for Unity, one for Unreal. Both are consumers. They
read `roads.geojson`, `junctions.geojson` and `manifest.json` and build scene objects; neither
imports any Python, and the core imports neither of them.

That direction is invariant 1, and it is the reason these live outside `src/japgo/`. The core
carries structure — where a road goes, how wide it is, what class it is, what its grade is — and
never appearance. An adapter turns that into whatever its engine wants.

| | Unity | Unreal |
| --- | --- | --- |
| form | UPM package `com.japgo.roads` | plugin `JapGoRoads` |
| builds | ribbon meshes or `LineRenderer` | `USplineComponent` per road |
| runtime | yes | yes |
| editor entry | **Window → JapGo → Import road bundle** | Blueprint: `Import Bundle Directory` |
| dependency | `com.unity.nuget.newtonsoft-json` | none (built-in `Json` module) |

Both work at runtime, not only at edit time. That is deliberate: the product is procedural world
generation, and a game that builds terrain while it runs has to build its roads the same way. The
Unity editor window is a thin wrapper over the same runtime call.

## Producing a bundle

```bash
japgo demo --site hamamatsu_plain --datum 300
```

writes `runs/demo/hamamatsu_plain/` with the three files. Any directory holding those three is a
bundle. `--datum` matters: without it the exported heights are relative to the source tile's mean
and the roads import tens of metres underground. Both adapters warn when they see a
`tile-relative` bundle rather than failing, because such a bundle is still correct if you place it
against the matching terrain yourself.

## The one thing worth reading before editing either

The bundle is east/north/up metres, right-handed, with a `local_frame.origin` to subtract. Each
adapter does two things with that, and both are easy to get wrong in ways nothing will catch.

**Subtract the origin in double precision.** Projected coordinates in JGD2011 run past 100 km,
where a 32-bit float holds about a centimetre. Geometry built from raw eastings visibly shimmers
as the camera moves. Subtract first, cast once.

**Permute the axes, and count the handedness flip.** Geographic (east, north, up) is right-handed.
Unity is left-handed y-up, so east→x, up→y, north→z — one swap, one flip, lands correctly. Unreal
is left-handed z-up in centimetres, so north→x, east→y, up→z — again one swap. Map east→x and
north→y in Unreal and you get a mirror world in which every junction still meets and every road
still follows its valley. There is no visual tell. The reasoning depends on the source really
being right-handed east/north/up, which
`tests/test_generate_pipeline.py::test_the_declared_axes_are_the_ones_the_geometry_actually_uses`
pins down.

The permutation is intentionally *not* in the core. It is not shared logic being duplicated
between the two importers — it is the entire difference between them, and hoisting it would mean
the core knowing what Unity is.

## What is verified, and what is not

**Not compiled.** Neither Unity nor Unreal is installed on the development machine, so neither
adapter has been built or run against an engine. Treat both as first drafts: the parsing, the
frame maths and the API shapes have been reasoned through carefully, and the compilers have never
seen them.

What *is* checked automatically is the contract between the exporter and these importers —
`test_the_bundle_carries_everything_the_engine_adapters_read` fails if the bundle stops carrying
a field either adapter reads. That guards the failure mode that would otherwise be silent: rename
`width_m` and nothing throws, every road just quietly imports at the 5 m default.
