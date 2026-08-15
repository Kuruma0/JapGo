// What the built scene objects remember about where they came from.
//
// An imported world with no provenance is a pile of meshes. These two components keep the seed,
// the CRS and the origin attached to the objects themselves, so a scene can answer "which world
// is this and how do I regenerate it" months later without a note in a wiki.

using UnityEngine;

namespace JapGo.Roads
{
    /// <summary>Provenance for one imported bundle. Added to the root object.</summary>
    public sealed class RoadNetworkInfo : MonoBehaviour
    {
        [Tooltip("Generation seed. With the same terrain and parameters it reproduces this world.")]
        public int Seed;

        [Tooltip("Source coordinate reference system, e.g. EPSG:6676.")]
        public string Crs;

        [Tooltip("Projected easting of this object's local zero.")]
        public double OriginEast;

        [Tooltip("Projected northing of this object's local zero.")]
        public double OriginNorth;

        [TextArea]
        public string Summary;
    }

    /// <summary>One road, with the attributes the core carries and no more.</summary>
    public sealed class RoadSegment : MonoBehaviour
    {
        public string Id;
        public string RoadClass;
        public float WidthMetres;
        public float LengthMetres;

        [Tooltip("End-to-end grade in percent. NaN when the exporter recorded none.")]
        public float GradePercent;

        [Tooltip("Centreline in local space — the geometry the core validated, before meshing.")]
        public Vector3[] Centreline;
    }
}
