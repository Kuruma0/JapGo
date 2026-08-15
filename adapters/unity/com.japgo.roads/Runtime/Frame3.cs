// The one genuinely engine-specific piece: east/north/up to Unity's axes.
//
// The core deliberately stops short of this. It emits east/north/up metres, right-handed, plus an
// origin, and says so in the manifest. Permuting those onto a particular engine's axes is not
// shared logic being duplicated between the two importers — it is precisely the difference
// between them, and putting it in the core would mean the core knowing what Unity is.
//
// Unity is left-handed with y up. Mapping east to x, up to y and north to z takes the geographic
// right-handed triple (east, north, up) to (x, z, y), one swap — which flips handedness exactly
// once and therefore lands on Unity's, with no mirroring. Get this wrong by mapping north to z
// and east to x while leaving handedness alone and every junction still meets, every road still
// follows its valley, and the whole world is a mirror image of itself. There is no visual tell.

using Newtonsoft.Json.Linq;
using UnityEngine;

namespace JapGo.Roads
{
    public static class Frame3
    {
        /// <summary>
        /// A bundle coordinate to a Unity position: origin subtracted, axes permuted.
        /// </summary>
        /// <remarks>
        /// The subtraction happens in double precision and the result is cast once. Projected
        /// coordinates in JGD2011 run past 100 km, and float32 holds about 1 cm of precision
        /// there; a vertex buffer built from raw eastings visibly shimmers as the camera moves.
        /// Subtracting first keeps everything within a few kilometres of zero, where float32 has
        /// sub-millimetre precision to spare.
        /// </remarks>
        public static Vector3 ToUnity(double east, double north, double up, LocalFrame frame)
        {
            return new Vector3(
                (float)(east - frame.OriginEast),
                (float)(up - frame.OriginUp),
                (float)(north - frame.OriginNorth));
        }

        public static Vector3 ToUnity(JArray coordinate, LocalFrame frame)
        {
            var up = coordinate.Count > 2 ? coordinate[2].Value<double>() : 0.0;
            return ToUnity(coordinate[0].Value<double>(), coordinate[1].Value<double>(), up, frame);
        }

        /// <summary>The inverse, for writing positions back out or querying source data.</summary>
        public static Vector3d ToWorld(Vector3 unity, LocalFrame frame)
        {
            return new Vector3d(
                unity.x + frame.OriginEast,
                unity.z + frame.OriginNorth,
                unity.y + frame.OriginUp);
        }
    }

    /// <summary>A double-precision triple, for coordinates that must not lose their origin.</summary>
    public readonly struct Vector3d
    {
        public readonly double East;
        public readonly double North;
        public readonly double Up;

        public Vector3d(double east, double north, double up)
        {
            East = east;
            North = north;
            Up = up;
        }

        public override string ToString() => $"({East:0.###}, {North:0.###}, {Up:0.###})";
    }
}
