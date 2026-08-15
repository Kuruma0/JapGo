// Turning a parsed bundle into scene objects.
//
// Works at runtime, not only in the editor, because the product this adapter serves is procedural
// world generation: a game that generates terrain at runtime and asks for roads needs to build
// them at runtime too. The editor window in Editor/ is a convenience wrapper around exactly this.
//
// The mesh is a flat ribbon: the centreline offset left and right by half the carriageway width,
// lifted a few centimetres so it does not z-fight the terrain. Deliberately no crown, no kerbs, no
// junction fill — the core carries structure and never appearance (invariant 10), and a ribbon is
// the least this can do while still being a road. Anything prettier belongs in the project's own
// materials and mesh work, not in an importer that would then have opinions about art direction.

using System.Collections.Generic;
using UnityEngine;

namespace JapGo.Roads
{
    [System.Serializable]
    public sealed class RoadBuildOptions
    {
        [Tooltip("Generate a ribbon mesh per road. Off gives empty transforms with the centreline " +
                 "on a LineRenderer, which is what a project with its own road mesher wants.")]
        public bool GenerateMesh = true;

        [Tooltip("Metres above the sampled ground. The exporter already lifts by 0.15 m; this is " +
                 "on top of that, for terrain that does not match the source DEM exactly.")]
        public float SurfaceOffset = 0.02f;

        [Tooltip("Material for generated road meshes. Null uses Unity's default.")]
        public Material RoadMaterial;

        [Tooltip("Create an empty marker at every junction, named by node id.")]
        public bool CreateJunctionMarkers = true;

        [Tooltip("Widen every carriageway by this factor. 1 is the exporter's width.")]
        public float WidthScale = 1f;
    }

    public static class RoadNetworkBuilder
    {
        /// <summary>Build a road network under a new GameObject and return its root.</summary>
        public static GameObject Build(RoadBundle bundle, RoadBuildOptions options = null,
                                       Transform parent = null, string name = "JapGo Roads")
        {
            options ??= new RoadBuildOptions();

            if (bundle.Frame.IsRelativeElevation)
            {
                // Not an exception: a relative bundle is still usable if the caller knows to place
                // it against the matching terrain. But silence here means roads tens of metres
                // underground and a bug hunt in the wrong place.
                Debug.LogWarning(
                    "[JapGo] This bundle's heights are tile-relative, not absolute. Roads will sit " +
                    "around the source tile's mean elevation — pass elevation_datum_m when " +
                    "exporting, or offset this object yourself.");
            }

            var root = new GameObject(name);
            if (parent != null)
            {
                root.transform.SetParent(parent, false);
            }

            var info = root.AddComponent<RoadNetworkInfo>();
            info.Seed = bundle.Seed;
            info.Crs = bundle.Frame.Crs;
            info.OriginEast = bundle.Frame.OriginEast;
            info.OriginNorth = bundle.Frame.OriginNorth;
            info.Summary = bundle.Describe();

            var roads = new GameObject("Roads");
            roads.transform.SetParent(root.transform, false);
            foreach (var road in bundle.Roads)
            {
                BuildRoad(road, roads.transform, options);
            }

            if (options.CreateJunctionMarkers && bundle.Junctions.Count > 0)
            {
                var junctions = new GameObject("Junctions");
                junctions.transform.SetParent(root.transform, false);
                foreach (var junction in bundle.Junctions)
                {
                    var marker = new GameObject($"{junction.Id} (degree {junction.Degree})");
                    marker.transform.SetParent(junctions.transform, false);
                    marker.transform.localPosition = junction.Position;
                }
            }

            return root;
        }

        static void BuildRoad(RoadSpline road, Transform parent, RoadBuildOptions options)
        {
            if (road.Points.Length < 2)
            {
                return;     // a one-point road is a node the exporter should not have emitted
            }

            var go = new GameObject($"{road.Id} ({road.RoadClass})");
            go.transform.SetParent(parent, false);

            var component = go.AddComponent<RoadSegment>();
            component.Id = road.Id;
            component.RoadClass = road.RoadClass;
            component.WidthMetres = road.WidthMetres;
            component.LengthMetres = road.LengthMetres;
            component.GradePercent = road.GradePercent ?? float.NaN;
            component.Centreline = road.Points;

            if (!options.GenerateMesh)
            {
                var line = go.AddComponent<LineRenderer>();
                line.useWorldSpace = false;
                line.positionCount = road.Points.Length;
                line.SetPositions(road.Points);
                line.widthMultiplier = road.WidthMetres * options.WidthScale;
                return;
            }

            var filter = go.AddComponent<MeshFilter>();
            var renderer = go.AddComponent<MeshRenderer>();
            filter.sharedMesh = Ribbon(road, options);
            if (options.RoadMaterial != null)
            {
                renderer.sharedMaterial = options.RoadMaterial;
            }
        }

        /// <summary>A flat ribbon along the centreline, half a width to each side.</summary>
        static Mesh Ribbon(RoadSpline road, RoadBuildOptions options)
        {
            var points = road.Points;
            var half = road.WidthMetres * options.WidthScale * 0.5f;
            var lift = Vector3.up * options.SurfaceOffset;

            var vertices = new Vector3[points.Length * 2];
            var uv = new Vector2[points.Length * 2];
            var triangles = new List<int>((points.Length - 1) * 6);

            var travelled = 0f;
            for (var i = 0; i < points.Length; i++)
            {
                // Tangent by central difference, so interior vertices bisect their corner rather
                // than following one segment and pinching the other.
                var ahead = points[Mathf.Min(i + 1, points.Length - 1)];
                var behind = points[Mathf.Max(i - 1, 0)];
                var tangent = ahead - behind;

                // The normal is taken in the horizontal plane only. A road banked to follow the
                // terrain's normal looks correct on a hillside and wrong everywhere else, and the
                // core carries no camber information to bank it by.
                var flat = new Vector3(tangent.x, 0f, tangent.z);
                if (flat.sqrMagnitude < 1e-8f)
                {
                    flat = Vector3.forward;
                }
                var side = Vector3.Cross(Vector3.up, flat.normalized) * half;

                if (i > 0)
                {
                    travelled += Vector3.Distance(points[i - 1], points[i]);
                }

                vertices[i * 2] = points[i] - side + lift;
                vertices[i * 2 + 1] = points[i] + side + lift;
                uv[i * 2] = new Vector2(0f, travelled / Mathf.Max(road.WidthMetres, 0.01f));
                uv[i * 2 + 1] = new Vector2(1f, travelled / Mathf.Max(road.WidthMetres, 0.01f));

                if (i == 0)
                {
                    continue;
                }
                var a = (i - 1) * 2;
                triangles.AddRange(new[] { a, a + 2, a + 1, a + 1, a + 2, a + 3 });
            }

            var mesh = new Mesh { name = $"road_{road.Id}" };
            // Long roads at 10 m spacing stay well under 65k vertices, but a caller that resampled
            // finer should not hit a silent truncation.
            if (vertices.Length > 65000)
            {
                mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            }
            mesh.vertices = vertices;
            mesh.uv = uv;
            mesh.triangles = triangles.ToArray();
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
