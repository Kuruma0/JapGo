#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/** One log category for the plugin, so import problems are greppable in a shipping log. */
JAPGOROADS_API DECLARE_LOG_CATEGORY_EXTERN(LogJapGoRoads, Log, All);

class FJapGoRoadsModule : public IModuleInterface
{
};
