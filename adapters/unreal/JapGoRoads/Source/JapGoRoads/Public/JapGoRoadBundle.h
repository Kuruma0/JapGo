// The interchange bundle, as Unreal types. Parsing only — nothing here spawns anything.
//
// The bundle is three files written by `japgo.generate.export_bundle`: roads.geojson,
// junctions.geojson and manifest.json. GeoJSON because every engine and every GIS tool reads it,
// and because the core emits an interchange format rather than engine types. This plugin consumes
// that format; the generator knows nothing about Unreal.
//
// Coordinates arrive as east/north/up metres in a projected CRS, right-handed, with six-figure
// magnitudes. Both consequences are handled in FJapGoFrame: the origin has to come off before
// anything reaches a float, and east/north/up has to be permuted into Unreal's left-handed
// z-up centimetres.

#pragma once

#include "CoreMinimal.h"
#include "JapGoRoadBundle.generated.h"

/**
 * Where the bundle sits and what its axes mean. Written by the core, never guessed here.
 */
USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoLocalFrame
{
    GENERATED_BODY()

    /** Projected easting, northing and up of the bundle's local zero, in metres. */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FVector Origin = FVector::ZeroVector;

    /** Extent in metres, east by north. */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FVector2D SizeMetres = FVector2D::ZeroVector;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString Crs;

    /** "absolute" or "tile-relative". */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString ElevationReference = TEXT("absolute");

    /**
     * True when heights are measured from the source tile's mean rather than a datum. A relative
     * bundle placed as-is puts roads tens of metres underground, and an importer cannot tell by
     * looking — which is exactly why the core records it.
     */
    bool IsRelativeElevation() const
    {
        return ElevationReference.Equals(TEXT("tile-relative"), ESearchCase::IgnoreCase);
    }
};

/** One road: a centreline with a width and a class. */
USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoRoad
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString Id;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString RoadClass = TEXT("unknown");

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float WidthMetres = 5.f;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float LengthMetres = 0.f;

    /** End-to-end grade in percent. NaN when the exporter recorded none. */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float GradePercent = 0.f;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    bool bHasGrade = false;

    /** Centreline in Unreal space (cm), origin already subtracted. */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    TArray<FVector> Points;
};

/** A node where three or more roads meet. */
USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoJunction
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString Id;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    int32 Degree = 0;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FVector Position = FVector::ZeroVector;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    TArray<FString> Incident;
};

/** A parsed bundle, ready to build from. */
USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoRoadBundle
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FJapGoLocalFrame Frame;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    TArray<FJapGoRoad> Roads;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    TArray<FJapGoJunction> Junctions;

    /** Generation seed. With the same terrain and parameters it reproduces this world. */
    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    int32 Seed = 0;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float TotalLengthMetres = 0.f;

    FString Describe() const;
};

/**
 * Bundle to Unreal space.
 *
 * The core deliberately stops short of this. It emits east/north/up metres, right-handed, plus an
 * origin, and says so in the manifest. Permuting those onto a particular engine's axes is not
 * shared logic being duplicated between importers — it is precisely the difference between them,
 * and putting it in the core would mean the core knowing what Unreal is.
 *
 * Unreal is left-handed with z up, x forward and y right, in centimetres. Mapping north to x and
 * east to y takes the geographic right-handed triple (east, north, up) to (north, east, up), one
 * swap, which flips handedness exactly once and therefore lands on Unreal's with no mirroring.
 * Get it wrong by mapping east to x and north to y and every junction still meets, every road
 * still follows its valley, and the whole world is a mirror image. There is no visual tell.
 */
struct JAPGOROADS_API FJapGoFrame
{
    /** Metres to Unreal centimetres. */
    static constexpr double UnrealUnitsPerMetre = 100.0;

    /**
     * The subtraction happens in double precision and the result is cast once. Projected
     * coordinates in JGD2011 run past 100 km; a float holds about 1 cm of precision there, and a
     * spline built from raw eastings visibly jitters. Subtracting first keeps everything within a
     * few kilometres of zero.
     */
    static FVector ToUnreal(double East, double North, double Up, const FJapGoLocalFrame& Frame)
    {
        return FVector(
            (North - Frame.Origin.Y) * UnrealUnitsPerMetre,
            (East - Frame.Origin.X) * UnrealUnitsPerMetre,
            (Up - Frame.Origin.Z) * UnrealUnitsPerMetre);
    }
};

/** Reads bundles. Static because a bundle has no identity worth keeping around. */
class JAPGOROADS_API FJapGoBundleReader
{
public:
    /** Read a directory containing the three exported files. */
    static bool LoadFromDirectory(const FString& Directory, FJapGoRoadBundle& OutBundle,
                                  FString& OutError);

    /**
     * Parse from strings, so a game can stream a bundle from anywhere — a pak file, a web
     * request, a world generated moments ago — without touching the file system.
     */
    static bool Parse(const FString& RoadsJson, const FString& JunctionsJson,
                      const FString& ManifestJson, FJapGoRoadBundle& OutBundle, FString& OutError);
};
