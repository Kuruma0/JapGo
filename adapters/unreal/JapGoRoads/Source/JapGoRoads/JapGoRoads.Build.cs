// Runtime module, not editor-only. The point of this plugin is worlds that are generated while
// the game runs, so the importer has to be available there.

using UnrealBuildTool;

public class JapGoRoads : ModuleRules
{
    public JapGoRoads(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Json",              // FJsonSerializer: no third-party JSON dependency needed
            "JsonUtilities",
            "Projects",
        });
    }
}
