// The interchange bundle, as C# data. Parsing only — nothing here touches the scene.
//
// The bundle is three files written by `japgo.generate.export_bundle`: roads.geojson,
// junctions.geojson and manifest.json. It is GeoJSON because every engine and every GIS tool
// reads it, and because the core emits an interchange format rather than engine types. This
// adapter consumes that format; it does not reach into the generator, and the generator knows
// nothing about Unity.
//
// Coordinates arrive as east/north/up metres in a projected CRS, right-handed, with six-figure
// magnitudes. Two consequences, both handled in Frame.cs rather than here: the origin has to be
// subtracted before anything reaches a float32 vertex buffer, and east/north/up has to be
// permuted into Unity's left-handed y-up convention.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace JapGo.Roads
{
    /// <summary>Where the bundle sits and what its axes mean. Written by the core.</summary>
    [Serializable]
    public sealed class LocalFrame
    {
        public double OriginEast;
        public double OriginNorth;
        public double OriginUp;
        public float SizeEast;
        public float SizeNorth;
        public string Crs = "";
        public string Units = "m";
        public string Handedness = "right";

        /// <summary>"absolute" or "tile-relative".</summary>
        public string ElevationReference = "absolute";

        /// <summary>
        /// True when heights are measured from the source tile's mean rather than a datum.
        /// A tile-relative bundle placed as-is puts roads tens of metres underground, and an
        /// importer cannot tell by looking — which is exactly why the core records it.
        /// </summary>
        public bool IsRelativeElevation =>
            string.Equals(ElevationReference, "tile-relative", StringComparison.OrdinalIgnoreCase);

        public static LocalFrame FromManifest(JObject manifest)
        {
            var frame = manifest["local_frame"] as JObject;
            if (frame == null)
            {
                throw new RoadBundleException(
                    "manifest.json has no local_frame block. It was written by an older version " +
                    "of the exporter; re-export the bundle rather than guessing an origin.");
            }

            var origin = (JArray)frame["origin"];
            var size = (JArray)frame["size_m"];
            var result = new LocalFrame
            {
                OriginEast = origin[0].Value<double>(),
                OriginNorth = origin[1].Value<double>(),
                OriginUp = origin.Count > 2 ? origin[2].Value<double>() : 0.0,
                SizeEast = size[0].Value<float>(),
                SizeNorth = size[1].Value<float>(),
                Crs = frame.Value<string>("crs") ?? "",
                Units = frame.Value<string>("units") ?? "m",
                Handedness = frame.Value<string>("handedness") ?? "right",
                ElevationReference = frame.Value<string>("elevation_reference") ?? "absolute",
            };

            // Refuse rather than silently scale. A bundle in feet would import at 3.28x and look
            // plausible until someone measured a carriageway.
            if (result.Units != "m")
            {
                throw new RoadBundleException(
                    $"bundle units are '{result.Units}'; this importer assumes metres.");
            }
            return result;
        }
    }

    /// <summary>One road: a centreline with a width and a class.</summary>
    [Serializable]
    public sealed class RoadSpline
    {
        public string Id = "";
        public string RoadClass = "unknown";
        public float WidthMetres = 5f;
        public float LengthMetres;

        /// <summary>Grade in percent, or null when the exporter did not record one.</summary>
        public float? GradePercent;

        /// <summary>Centreline in engine space, origin already subtracted.</summary>
        public Vector3[] Points = Array.Empty<Vector3>();
    }

    /// <summary>A node where three or more roads meet.</summary>
    [Serializable]
    public sealed class Junction
    {
        public string Id = "";
        public int Degree;
        public Vector3 Position;
        public string[] Incident = Array.Empty<string>();
    }

    public sealed class RoadBundleException : Exception
    {
        public RoadBundleException(string message) : base(message) { }
    }

    /// <summary>A parsed bundle, ready to build from.</summary>
    public sealed class RoadBundle
    {
        public LocalFrame Frame { get; private set; } = new LocalFrame();
        public List<RoadSpline> Roads { get; } = new List<RoadSpline>();
        public List<Junction> Junctions { get; } = new List<Junction>();

        /// <summary>Seed that produced the world. Recorded so a scene can say where it came from.</summary>
        public int Seed { get; private set; }

        public float TotalLengthMetres { get; private set; }

        /// <summary>Read a bundle directory containing the three exported files.</summary>
        public static RoadBundle Load(string directory)
        {
            string Read(string name)
            {
                var path = Path.Combine(directory, name);
                if (!File.Exists(path))
                {
                    throw new RoadBundleException(
                        $"{name} not found in '{directory}'. A bundle is roads.geojson, " +
                        "junctions.geojson and manifest.json together.");
                }
                return File.ReadAllText(path);
            }

            return Parse(Read("roads.geojson"), Read("junctions.geojson"), Read("manifest.json"));
        }

        /// <summary>
        /// Parse from strings, so a game can stream a bundle from anywhere — an addressable, a
        /// web request, a runtime-generated world — without touching the file system.
        /// </summary>
        public static RoadBundle Parse(string roadsJson, string junctionsJson, string manifestJson)
        {
            var manifest = JObject.Parse(manifestJson);
            var bundle = new RoadBundle
            {
                Frame = LocalFrame.FromManifest(manifest),
                Seed = manifest.Value<int?>("seed") ?? 0,
                TotalLengthMetres = manifest.Value<float?>("total_length_m") ?? 0f,
            };

            foreach (var feature in Features(roadsJson, "roads.geojson"))
            {
                var properties = feature["properties"] as JObject ?? new JObject();
                var coordinates = (JArray)feature["geometry"]["coordinates"];
                var points = new Vector3[coordinates.Count];
                for (var i = 0; i < coordinates.Count; i++)
                {
                    points[i] = Frame3.ToUnity((JArray)coordinates[i], bundle.Frame);
                }

                bundle.Roads.Add(new RoadSpline
                {
                    Id = properties.Value<string>("id") ?? "",
                    RoadClass = properties.Value<string>("road_class") ?? "unknown",
                    WidthMetres = properties.Value<float?>("width_m") ?? 5f,
                    LengthMetres = properties.Value<float?>("length_m") ?? 0f,
                    GradePercent = properties.Value<float?>("grade_pct"),
                    Points = points,
                });
            }

            foreach (var feature in Features(junctionsJson, "junctions.geojson"))
            {
                var properties = feature["properties"] as JObject ?? new JObject();
                var incident = properties["incident"] as JArray ?? new JArray();
                var ids = new string[incident.Count];
                for (var i = 0; i < incident.Count; i++)
                {
                    ids[i] = incident[i].Value<string>();
                }

                bundle.Junctions.Add(new Junction
                {
                    Id = properties.Value<string>("id") ?? "",
                    Degree = properties.Value<int?>("degree") ?? 0,
                    Position = Frame3.ToUnity((JArray)feature["geometry"]["coordinates"], bundle.Frame),
                    Incident = ids,
                });
            }

            return bundle;
        }

        static IEnumerable<JToken> Features(string json, string what)
        {
            var root = JObject.Parse(json);
            if (root.Value<string>("type") != "FeatureCollection")
            {
                throw new RoadBundleException($"{what} is not a GeoJSON FeatureCollection.");
            }
            return root["features"] as JArray ?? new JArray();
        }

        public string Describe()
        {
            return string.Format(
                CultureInfo.InvariantCulture,
                "{0} roads, {1} junctions, {2:0.00} km, seed {3}, {4} heights",
                Roads.Count, Junctions.Count, TotalLengthMetres / 1000f, Seed,
                Frame.IsRelativeElevation ? "tile-relative" : "absolute");
        }
    }
}
