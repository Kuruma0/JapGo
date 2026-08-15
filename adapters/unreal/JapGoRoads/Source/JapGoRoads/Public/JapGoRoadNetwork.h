// Turning a parsed bundle into an actor with spline roads.
//
// Splines rather than meshes, deliberately. Unreal projects already have a road mesher they like —
// the built-in spline mesh workflow, Landmass, a marketplace tool, or their own — and every one of
// them takes a USplineComponent. Generating geometry here would mean this plugin having opinions
// about art direction, which the core spent ten invariants avoiding (invariant 10: structure and
// appearance stay separate). What the bundle carries is structure: where the road goes, how wide
// it is, what class it is, what its grade is. That is exactly a spline plus metadata.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "JapGoRoadBundle.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "JapGoRoadNetwork.generated.h"

class USplineComponent;

USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoImportOptions
{
    GENERATED_BODY()

    /** Add a spline component per road. Off imports metadata only, for a custom builder. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "JapGo")
    bool bCreateSplines = true;

    /**
     * Centimetres added to every point, above the exporter's own 0.15 m camber lift. For terrain
     * that does not match the source DEM exactly.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "JapGo")
    float SurfaceOffset = 2.f;

    /**
     * Leave spline tangents to Unreal's auto-curve. The exporter has already smoothed the
     * centreline under a displacement cap and resampled it evenly, so the points are curve-ready;
     * forcing linear tangents throws that away and puts the corners back.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "JapGo")
    bool bAutoTangents = true;

    /** Skip roads shorter than this, in metres. 0 imports everything. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "JapGo")
    float MinimumLengthMetres = 0.f;
};

/** One road's structure, kept alongside its spline. */
USTRUCT(BlueprintType)
struct JAPGOROADS_API FJapGoRoadInfo
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString Id;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    FString RoadClass;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float WidthMetres = 5.f;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    float GradePercent = 0.f;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    bool bHasGrade = false;

    UPROPERTY(BlueprintReadOnly, Category = "JapGo")
    USplineComponent* Spline = nullptr;
};

/**
 * A whole imported road network. Keeps its provenance: an imported world with no record of its
 * seed or origin is a pile of splines nobody can regenerate.
 */
UCLASS(BlueprintType)
class JAPGOROADS_API AJapGoRoadNetwork : public AActor
{
    GENERATED_BODY()

public:
    AJapGoRoadNetwork();

    /** Build splines from a parsed bundle. Safe to call at runtime. */
    UFUNCTION(BlueprintCallable, Category = "JapGo")
    void BuildFrom(const FJapGoRoadBundle& Bundle, const FJapGoImportOptions& Options);

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo")
    TArray<FJapGoRoadInfo> Roads;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo")
    TArray<FJapGoJunction> Junctions;

    /** Generation seed. With the same terrain and parameters it reproduces this world. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo|Provenance")
    int32 Seed = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo|Provenance")
    FString Crs;

    /** Projected easting and northing of this actor's local zero, in metres. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo|Provenance")
    FVector Origin = FVector::ZeroVector;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "JapGo|Provenance")
    FString Summary;
};

/** Blueprint entry points, so importing needs no C++ in the consuming project. */
UCLASS()
class JAPGOROADS_API UJapGoRoadImporter : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Read a bundle directory and spawn a network actor. Returns null and fills OutError on
     * failure — a silent no-op here looks exactly like a world with no roads in it, which after
     * the blind-generation experiment is a genuinely possible answer and must not be confused
     * with a failed import.
     */
    UFUNCTION(BlueprintCallable, Category = "JapGo",
              meta = (WorldContext = "WorldContextObject"))
    static AJapGoRoadNetwork* ImportBundleDirectory(UObject* WorldContextObject,
                                                    const FString& Directory,
                                                    const FJapGoImportOptions& Options,
                                                    FString& OutError);

    /** Parse a bundle held in memory — streamed, downloaded, or generated moments ago. */
    UFUNCTION(BlueprintCallable, Category = "JapGo",
              meta = (WorldContext = "WorldContextObject"))
    static AJapGoRoadNetwork* ImportBundleText(UObject* WorldContextObject,
                                               const FString& RoadsJson,
                                               const FString& JunctionsJson,
                                               const FString& ManifestJson,
                                               const FJapGoImportOptions& Options,
                                               FString& OutError);
};
