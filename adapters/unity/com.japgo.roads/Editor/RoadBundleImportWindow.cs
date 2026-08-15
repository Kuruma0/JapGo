// Editor convenience over the runtime builder. Nothing happens here that a game could not do at
// runtime, which is deliberate: a procedural world generator builds its roads while it runs, and
// an importer that only worked at edit time would be the wrong shape for the product.
//
// Not a ScriptedImporter. A bundle is a directory of three files that only mean something
// together, and ScriptedImporter is per-file — it would fire three times and see a third of a
// world each time.

using System.IO;
using UnityEditor;
using UnityEngine;

namespace JapGo.Roads.Editor
{
    public sealed class RoadBundleImportWindow : EditorWindow
    {
        [SerializeField] string _directory = "";
        [SerializeField] RoadBuildOptions _options = new RoadBuildOptions();
        [SerializeField] bool _savePrefab;
        [SerializeField] string _prefabFolder = "Assets/JapGo";

        string _status = "";
        bool _statusIsError;

        [MenuItem("Window/JapGo/Import road bundle")]
        public static void Open()
        {
            GetWindow<RoadBundleImportWindow>(true, "JapGo road bundle", true).minSize =
                new Vector2(430f, 320f);
        }

        void OnGUI()
        {
            EditorGUILayout.LabelField("Bundle", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _directory = EditorGUILayout.TextField("Directory", _directory);
                if (GUILayout.Button("Browse", GUILayout.Width(70f)))
                {
                    var picked = EditorUtility.OpenFolderPanel(
                        "Bundle directory (roads.geojson, junctions.geojson, manifest.json)",
                        string.IsNullOrEmpty(_directory) ? Application.dataPath : _directory, "");
                    if (!string.IsNullOrEmpty(picked))
                    {
                        _directory = picked;
                        _status = "";
                    }
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Build", EditorStyles.boldLabel);
            _options.GenerateMesh = EditorGUILayout.Toggle(
                new GUIContent("Generate mesh", "Off leaves the centreline on a LineRenderer, for " +
                                                "projects with their own road mesher."),
                _options.GenerateMesh);
            _options.RoadMaterial = (Material)EditorGUILayout.ObjectField(
                "Road material", _options.RoadMaterial, typeof(Material), false);
            _options.WidthScale = EditorGUILayout.Slider("Width scale", _options.WidthScale, 0.25f, 4f);
            _options.SurfaceOffset = EditorGUILayout.FloatField(
                new GUIContent("Surface offset (m)", "Above the exporter's own 0.15 m camber lift."),
                _options.SurfaceOffset);
            _options.CreateJunctionMarkers = EditorGUILayout.Toggle(
                "Junction markers", _options.CreateJunctionMarkers);

            EditorGUILayout.Space();
            _savePrefab = EditorGUILayout.Toggle("Save as prefab", _savePrefab);
            using (new EditorGUI.DisabledScope(!_savePrefab))
            {
                _prefabFolder = EditorGUILayout.TextField("Prefab folder", _prefabFolder);
            }

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(_directory)))
            {
                if (GUILayout.Button("Import", GUILayout.Height(28f)))
                {
                    Import();
                }
            }

            if (!string.IsNullOrEmpty(_status))
            {
                EditorGUILayout.HelpBox(
                    _status, _statusIsError ? MessageType.Error : MessageType.Info);
            }
        }

        void Import()
        {
            try
            {
                var bundle = RoadBundle.Load(_directory);
                var name = $"JapGo {Path.GetFileName(_directory.TrimEnd('/', '\\'))}";
                var root = RoadNetworkBuilder.Build(bundle, _options, null, name);

                // Registered with the undo system rather than saved immediately: an import that
                // cannot be undone is one a level artist will only try once.
                Undo.RegisterCreatedObjectUndo(root, "Import JapGo road bundle");
                Selection.activeGameObject = root;

                if (_savePrefab)
                {
                    Directory.CreateDirectory(_prefabFolder);
                    var path = AssetDatabase.GenerateUniqueAssetPath(
                        Path.Combine(_prefabFolder, name + ".prefab").Replace('\\', '/'));
                    SaveMeshes(root, path);
                    PrefabUtility.SaveAsPrefabAssetAndConnect(root, path, InteractionMode.UserAction);
                    AssetDatabase.SaveAssets();
                }

                _status = bundle.Describe();
                _statusIsError = false;
            }
            catch (RoadBundleException error)
            {
                _status = error.Message;
                _statusIsError = true;
            }
        }

        /// <summary>
        /// Generated meshes live only in memory. A prefab referencing them saves a prefab full of
        /// missing references, which looks fine until the project is reopened.
        /// </summary>
        static void SaveMeshes(GameObject root, string prefabPath)
        {
            var container = ScriptableObject.CreateInstance<MeshContainer>();
            var meshPath = Path.ChangeExtension(prefabPath, ".meshes.asset");
            AssetDatabase.CreateAsset(container, meshPath);

            foreach (var filter in root.GetComponentsInChildren<MeshFilter>())
            {
                if (filter.sharedMesh != null && !AssetDatabase.Contains(filter.sharedMesh))
                {
                    AssetDatabase.AddObjectToAsset(filter.sharedMesh, container);
                }
            }
            AssetDatabase.SaveAssets();
        }

        class MeshContainer : ScriptableObject { }
    }
}
