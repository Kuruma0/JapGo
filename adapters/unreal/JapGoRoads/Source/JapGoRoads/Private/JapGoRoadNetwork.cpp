#include "JapGoRoadNetwork.h"

#include "Components/SplineComponent.h"
#include "Engine/World.h"
#include "JapGoRoadsModule.h"

AJapGoRoadNetwork::AJapGoRoadNetwork()
{
    PrimaryActorTick.bCanEverTick = false;
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void AJapGoRoadNetwork::BuildFrom(const FJapGoRoadBundle& Bundle,
                                  const FJapGoImportOptions& Options)
{
    Roads.Reset();
    Junctions = Bundle.Junctions;
    Seed = Bundle.Seed;
    Crs = Bundle.Frame.Crs;
    Origin = Bundle.Frame.Origin;
    Summary = Bundle.Describe();

    if (Bundle.Frame.IsRelativeElevation())
    {
        // Not a failure: a relative bundle is usable if the caller knows to place it against the
        // matching terrain. But silence means roads tens of metres underground and a bug hunt in
        // the wrong place.
        UE_LOG(LogJapGoRoads, Warning,
               TEXT("Bundle heights are tile-relative, not absolute. Roads will sit around the "
                    "source tile's mean elevation - pass elevation_datum_m when exporting, or "
                    "offset this actor yourself."));
    }

    const FVector Lift(0.f, 0.f, Options.SurfaceOffset);

    for (const FJapGoRoad& Road : Bundle.Roads)
    {
        if (Road.Points.Num() < 2 || Road.LengthMetres < Options.MinimumLengthMetres)
        {
            continue;
        }

        FJapGoRoadInfo Info;
        Info.Id = Road.Id;
        Info.RoadClass = Road.RoadClass;
        Info.WidthMetres = Road.WidthMetres;
        Info.GradePercent = Road.GradePercent;
        Info.bHasGrade = Road.bHasGrade;

        if (Options.bCreateSplines)
        {
            USplineComponent* Spline = NewObject<USplineComponent>(
                this, USplineComponent::StaticClass(),
                *FString::Printf(TEXT("Road_%s"), *Road.Id));
            Spline->SetupAttachment(RootComponent);
            Spline->RegisterComponent();
            Spline->SetMobility(EComponentMobility::Static);
            Spline->ClearSplinePoints(false);

            for (const FVector& Point : Road.Points)
            {
                Spline->AddSplinePoint(Point + Lift, ESplineCoordinateSpace::Local, false);
            }

            if (!Options.bAutoTangents)
            {
                const int32 Count = Spline->GetNumberOfSplinePoints();
                for (int32 Index = 0; Index < Count; ++Index)
                {
                    Spline->SetSplinePointType(Index, ESplinePointType::Linear, false);
                }
            }

            // Closed loops are never emitted: the exporter writes one open polyline per edge, and
            // a loop here would silently join a road's two ends across the map.
            Spline->SetClosedLoop(false, false);
            Spline->UpdateSpline();
            Info.Spline = Spline;
        }

        Roads.Add(MoveTemp(Info));
    }

    UE_LOG(LogJapGoRoads, Log, TEXT("Imported %s"), *Summary);
}

namespace
{
    AJapGoRoadNetwork* SpawnFrom(UObject* WorldContextObject, const FJapGoRoadBundle& Bundle,
                                 const FJapGoImportOptions& Options, FString& OutError)
    {
        UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(
                                      WorldContextObject, EGetWorldErrorMode::ReturnNull)
                                : nullptr;
        if (World == nullptr)
        {
            OutError = TEXT("no world context; call this from an actor, a level Blueprint or a "
                            "game instance.");
            return nullptr;
        }

        AJapGoRoadNetwork* Network = World->SpawnActor<AJapGoRoadNetwork>();
        if (Network == nullptr)
        {
            OutError = TEXT("failed to spawn the road network actor.");
            return nullptr;
        }

        Network->BuildFrom(Bundle, Options);
        return Network;
    }
}

AJapGoRoadNetwork* UJapGoRoadImporter::ImportBundleDirectory(UObject* WorldContextObject,
                                                             const FString& Directory,
                                                             const FJapGoImportOptions& Options,
                                                             FString& OutError)
{
    OutError.Reset();
    FJapGoRoadBundle Bundle;
    if (!FJapGoBundleReader::LoadFromDirectory(Directory, Bundle, OutError))
    {
        UE_LOG(LogJapGoRoads, Error, TEXT("%s"), *OutError);
        return nullptr;
    }
    return SpawnFrom(WorldContextObject, Bundle, Options, OutError);
}

AJapGoRoadNetwork* UJapGoRoadImporter::ImportBundleText(UObject* WorldContextObject,
                                                        const FString& RoadsJson,
                                                        const FString& JunctionsJson,
                                                        const FString& ManifestJson,
                                                        const FJapGoImportOptions& Options,
                                                        FString& OutError)
{
    OutError.Reset();
    FJapGoRoadBundle Bundle;
    if (!FJapGoBundleReader::Parse(RoadsJson, JunctionsJson, ManifestJson, Bundle, OutError))
    {
        UE_LOG(LogJapGoRoads, Error, TEXT("%s"), *OutError);
        return nullptr;
    }
    return SpawnFrom(WorldContextObject, Bundle, Options, OutError);
}
