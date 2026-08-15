#include "JapGoRoadBundle.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
    bool ReadText(const FString& Directory, const TCHAR* Name, FString& OutText, FString& OutError)
    {
        if (!FFileHelper::LoadFileToString(OutText, *FPaths::Combine(Directory, Name)))
        {
            OutError = FString::Printf(
                TEXT("%s not found in '%s'. A bundle is roads.geojson, junctions.geojson and "
                     "manifest.json together."), Name, *Directory);
            return false;
        }
        return true;
    }

    bool ParseObject(const FString& Json, const TCHAR* What, TSharedPtr<FJsonObject>& OutObject,
                     FString& OutError)
    {
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Json);
        if (!FJsonSerializer::Deserialize(Reader, OutObject) || !OutObject.IsValid())
        {
            OutError = FString::Printf(TEXT("%s is not valid JSON."), What);
            return false;
        }
        return true;
    }

    /** Features of a GeoJSON FeatureCollection, or false with a reason. */
    bool Features(const TSharedPtr<FJsonObject>& Root, const TCHAR* What,
                  const TArray<TSharedPtr<FJsonValue>>*& OutFeatures, FString& OutError)
    {
        FString Type;
        if (!Root->TryGetStringField(TEXT("type"), Type) || Type != TEXT("FeatureCollection"))
        {
            OutError = FString::Printf(TEXT("%s is not a GeoJSON FeatureCollection."), What);
            return false;
        }
        if (!Root->TryGetArrayField(TEXT("features"), OutFeatures))
        {
            OutError = FString::Printf(TEXT("%s has no features array."), What);
            return false;
        }
        return true;
    }

    FVector CoordinateToUnreal(const TArray<TSharedPtr<FJsonValue>>& Coordinate,
                               const FJapGoLocalFrame& Frame)
    {
        const double East = Coordinate.Num() > 0 ? Coordinate[0]->AsNumber() : 0.0;
        const double North = Coordinate.Num() > 1 ? Coordinate[1]->AsNumber() : 0.0;
        const double Up = Coordinate.Num() > 2 ? Coordinate[2]->AsNumber() : 0.0;
        return FJapGoFrame::ToUnreal(East, North, Up, Frame);
    }
}

FString FJapGoRoadBundle::Describe() const
{
    return FString::Printf(
        TEXT("%d roads, %d junctions, %.2f km, seed %d, %s heights"),
        Roads.Num(), Junctions.Num(), TotalLengthMetres / 1000.f, Seed,
        Frame.IsRelativeElevation() ? TEXT("tile-relative") : TEXT("absolute"));
}

bool FJapGoBundleReader::LoadFromDirectory(const FString& Directory, FJapGoRoadBundle& OutBundle,
                                           FString& OutError)
{
    FString RoadsText, JunctionsText, ManifestText;
    if (!ReadText(Directory, TEXT("roads.geojson"), RoadsText, OutError) ||
        !ReadText(Directory, TEXT("junctions.geojson"), JunctionsText, OutError) ||
        !ReadText(Directory, TEXT("manifest.json"), ManifestText, OutError))
    {
        return false;
    }
    return Parse(RoadsText, JunctionsText, ManifestText, OutBundle, OutError);
}

bool FJapGoBundleReader::Parse(const FString& RoadsJson, const FString& JunctionsJson,
                               const FString& ManifestJson, FJapGoRoadBundle& OutBundle,
                               FString& OutError)
{
    OutBundle = FJapGoRoadBundle();

    TSharedPtr<FJsonObject> Manifest;
    if (!ParseObject(ManifestJson, TEXT("manifest.json"), Manifest, OutError))
    {
        return false;
    }

    const TSharedPtr<FJsonObject>* FrameObject = nullptr;
    if (!Manifest->TryGetObjectField(TEXT("local_frame"), FrameObject))
    {
        OutError = TEXT("manifest.json has no local_frame block. It was written by an older "
                        "version of the exporter; re-export rather than guessing an origin.");
        return false;
    }

    FJapGoLocalFrame& Frame = OutBundle.Frame;
    const TArray<TSharedPtr<FJsonValue>>* Origin = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* Size = nullptr;
    if ((*FrameObject)->TryGetArrayField(TEXT("origin"), Origin) && Origin->Num() >= 2)
    {
        Frame.Origin = FVector((*Origin)[0]->AsNumber(), (*Origin)[1]->AsNumber(),
                               Origin->Num() > 2 ? (*Origin)[2]->AsNumber() : 0.0);
    }
    if ((*FrameObject)->TryGetArrayField(TEXT("size_m"), Size) && Size->Num() >= 2)
    {
        Frame.SizeMetres = FVector2D((*Size)[0]->AsNumber(), (*Size)[1]->AsNumber());
    }
    (*FrameObject)->TryGetStringField(TEXT("crs"), Frame.Crs);
    (*FrameObject)->TryGetStringField(TEXT("elevation_reference"), Frame.ElevationReference);

    // Refuse rather than silently scale. A bundle in feet would import at 3.28x and look
    // plausible until someone measured a carriageway.
    FString Units;
    if ((*FrameObject)->TryGetStringField(TEXT("units"), Units) && Units != TEXT("m"))
    {
        OutError = FString::Printf(
            TEXT("bundle units are '%s'; this importer assumes metres."), *Units);
        return false;
    }

    Manifest->TryGetNumberField(TEXT("seed"), OutBundle.Seed);
    double TotalLength = 0.0;
    if (Manifest->TryGetNumberField(TEXT("total_length_m"), TotalLength))
    {
        OutBundle.TotalLengthMetres = static_cast<float>(TotalLength);
    }

    TSharedPtr<FJsonObject> RoadsRoot, JunctionsRoot;
    if (!ParseObject(RoadsJson, TEXT("roads.geojson"), RoadsRoot, OutError) ||
        !ParseObject(JunctionsJson, TEXT("junctions.geojson"), JunctionsRoot, OutError))
    {
        return false;
    }

    const TArray<TSharedPtr<FJsonValue>>* RoadFeatures = nullptr;
    if (!Features(RoadsRoot, TEXT("roads.geojson"), RoadFeatures, OutError))
    {
        return false;
    }

    for (const TSharedPtr<FJsonValue>& Value : *RoadFeatures)
    {
        const TSharedPtr<FJsonObject> Feature = Value->AsObject();
        if (!Feature.IsValid())
        {
            continue;
        }

        const TSharedPtr<FJsonObject>* Geometry = nullptr;
        const TArray<TSharedPtr<FJsonValue>>* Coordinates = nullptr;
        if (!Feature->TryGetObjectField(TEXT("geometry"), Geometry) ||
            !(*Geometry)->TryGetArrayField(TEXT("coordinates"), Coordinates))
        {
            continue;
        }

        FJapGoRoad Road;
        const TSharedPtr<FJsonObject>* Properties = nullptr;
        if (Feature->TryGetObjectField(TEXT("properties"), Properties))
        {
            (*Properties)->TryGetStringField(TEXT("id"), Road.Id);
            (*Properties)->TryGetStringField(TEXT("road_class"), Road.RoadClass);

            double Number = 0.0;
            if ((*Properties)->TryGetNumberField(TEXT("width_m"), Number))
            {
                Road.WidthMetres = static_cast<float>(Number);
            }
            if ((*Properties)->TryGetNumberField(TEXT("length_m"), Number))
            {
                Road.LengthMetres = static_cast<float>(Number);
            }
            // grade_pct is null for edges the exporter could not measure. A null read as 0 would
            // report every unmeasured road as perfectly flat, so absence is kept as absence.
            const TSharedPtr<FJsonValue> Grade = (*Properties)->TryGetField(TEXT("grade_pct"));
            if (Grade.IsValid() && Grade->Type == EJson::Number)
            {
                Road.GradePercent = static_cast<float>(Grade->AsNumber());
                Road.bHasGrade = true;
            }
        }

        Road.Points.Reserve(Coordinates->Num());
        for (const TSharedPtr<FJsonValue>& Point : *Coordinates)
        {
            Road.Points.Add(CoordinateToUnreal(Point->AsArray(), Frame));
        }
        OutBundle.Roads.Add(MoveTemp(Road));
    }

    const TArray<TSharedPtr<FJsonValue>>* JunctionFeatures = nullptr;
    if (!Features(JunctionsRoot, TEXT("junctions.geojson"), JunctionFeatures, OutError))
    {
        return false;
    }

    for (const TSharedPtr<FJsonValue>& Value : *JunctionFeatures)
    {
        const TSharedPtr<FJsonObject> Feature = Value->AsObject();
        if (!Feature.IsValid())
        {
            continue;
        }

        const TSharedPtr<FJsonObject>* Geometry = nullptr;
        const TArray<TSharedPtr<FJsonValue>>* Coordinates = nullptr;
        if (!Feature->TryGetObjectField(TEXT("geometry"), Geometry) ||
            !(*Geometry)->TryGetArrayField(TEXT("coordinates"), Coordinates))
        {
            continue;
        }

        FJapGoJunction Junction;
        Junction.Position = CoordinateToUnreal(*Coordinates, Frame);

        const TSharedPtr<FJsonObject>* Properties = nullptr;
        if (Feature->TryGetObjectField(TEXT("properties"), Properties))
        {
            (*Properties)->TryGetStringField(TEXT("id"), Junction.Id);
            (*Properties)->TryGetNumberField(TEXT("degree"), Junction.Degree);

            const TArray<TSharedPtr<FJsonValue>>* Incident = nullptr;
            if ((*Properties)->TryGetArrayField(TEXT("incident"), Incident))
            {
                for (const TSharedPtr<FJsonValue>& Edge : *Incident)
                {
                    Junction.Incident.Add(Edge->AsString());
                }
            }
        }
        OutBundle.Junctions.Add(MoveTemp(Junction));
    }

    return true;
}
