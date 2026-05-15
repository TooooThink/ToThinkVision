// TTVSceneImporter.cs
// ToThinkVision → Unity Editor Scene Importer
//
// Usage:
// 1. Place this file in your Unity project under Assets/Editor/
// 2. Drag your exported JSON file (e.g. demo_unity_json.json) into the project
// 3. Menu: GameObject → ToThinkVision → Import Scene from JSON
// 4. Select the JSON file
// 5. The scene is built automatically with GameObjects, Colliders, Materials, Sprites
//
// Optional: If you have a .splat file, install UnityGaussianSplatting plugin first
// then drag the .splat into the scene separately.

using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections.Generic;

namespace ToThinkVision
{
    public static class TTVSceneImporter
    {
        [MenuItem("GameObject/ToThinkVision/Import Scene from JSON")]
        public static void ImportSceneMenu()
        {
            string jsonPath = EditorUtility.OpenFilePanel(
                "Select ToThinkVision JSON",
                "",
                "json"
            );
            if (string.IsNullOrEmpty(jsonPath)) return;
            ImportScene(jsonPath);
        }

        [MenuItem("GameObject/ToThinkVision/Import Splat File")]
        public static void ImportSplatMenu()
        {
            string splatPath = EditorUtility.OpenFilePanel(
                "Select .splat file",
                "",
                "splat"
            );
            if (string.IsNullOrEmpty(splatPath)) return;
            ImportSplatFile(splatPath);
        }

        public static void ImportScene(string jsonPath)
        {
            if (!File.Exists(jsonPath))
            {
                EditorUtility.DisplayDialog("TTV Import", $"File not found:\n{jsonPath}", "OK");
                return;
            }

            string json = File.ReadAllText(jsonPath);
            var data = JsonUtility.FromJson<TTVStructuredOutput>(json);

            if (data == null || data.objects == null)
            {
                EditorUtility.DisplayDialog("TTV Import", "Invalid JSON structure", "OK");
                return;
            }

            // Create root GameObject for the imported scene
            GameObject root = new GameObject($"TTV_{data.source_file}");
            Undo.RegisterCreatedObjectUndo(root, "Import TTV Scene");

            // Read metadata for scale calculation
            float sceneWidth = data.metadata != null ? data.metadata.width : 1920f;
            float pixelsPerUnit = 100f; // Unity default: 100 pixels = 1 world unit

            int objectsCreated = 0;

            foreach (var obj in data.objects)
            {
                if (obj.bbox == null) continue;

                // Create GameObject
                var go = new GameObject(obj.label_custom ?? obj.label);
                go.transform.SetParent(root.transform);

                // Calculate position in world space (center of bbox, Y flipped for Unity coords)
                float worldX = (obj.bbox.x + obj.bbox.w / 2f) / pixelsPerUnit;
                float worldY = -(obj.bbox.y + obj.bbox.h / 2f) / pixelsPerUnit; // Flip Y
                float worldZ = obj.bbox_3d != null ? obj.bbox_3d.z / pixelsPerUnit : 0f;
                go.transform.position = new Vector3(worldX, worldY, worldZ);

                // Scale from bbox size
                float scaleX = obj.bbox.w / pixelsPerUnit;
                float scaleY = obj.bbox.h / pixelsPerUnit;
                go.transform.localScale = new Vector3(scaleX, scaleY, 1f);

                // 3D position if available
                if (obj.bbox_3d != null)
                {
                    // Store 3D info in a custom component
                    var info = go.AddComponent<TTVObjectInfo>();
                    info.objectId = obj.id;
                    info.label = obj.label_custom ?? obj.label;
                    info.confidence = obj.confidence;
                    info.depthMeters = obj.bbox_3d.z;
                }

                // Add Collider
                if (IsGameRelated(obj.label))
                {
                    bool isTrigger = obj.label.Contains("door") || obj.label.Contains("effect");
                    if (isTrigger)
                    {
                        var col = go.AddComponent<BoxCollider2D>();
                        col.isTrigger = true;
                    }
                    else
                    {
                        go.AddComponent<BoxCollider2D>();
                    }
                }

                // Create material with dominant color
                if (!string.IsNullOrEmpty(obj.dominant_color))
                {
                    Color col = HexToColor(obj.dominant_color);
                    var mat = new Material(Shader.Find("Sprites/Default"));
                    mat.color = col;
                    var renderer = go.AddComponent<SpriteRenderer>();
                    renderer.material = mat;
                    renderer.color = col;

                    // Try to load sprite from crop image
                    string spritePath = FindSpriteAsset(jsonPath, obj.id);
                    if (!string.IsNullOrEmpty(spritePath) && File.Exists(spritePath))
                    {
                        Sprite sprite = AssetDatabase.LoadAssetAtPath<Sprite>(spritePath);
                        if (sprite != null)
                        {
                            renderer.sprite = sprite;
                            renderer.color = Color.white; // Let the sprite color through
                        }
                    }
                }

                // Add Rigidbody2D for NPCs and items
                if (obj.label == "game_npc")
                {
                    var rb = go.AddComponent<Rigidbody2D>();
                    rb.bodyType = RigidbodyType2D.Dynamic;
                }
                else if (obj.label == "game_item")
                {
                    var rb = go.AddComponent<Rigidbody2D>();
                    rb.bodyType = RigidbodyType2D.Static;
                    rb.isKinematic = true;
                }

                // Add z-index as sorting order
                go.GetComponent<SpriteRenderer>()?.sortingOrder = obj.z_index;

                objectsCreated++;
            }

            // Add 3D point cloud info if available
            if (data.point_cloud != null && data.point_cloud.points.Count > 0)
            {
                var cloudInfo = root.AddComponent<TTVPointCloudInfo>();
                cloudInfo.pointCount = data.point_cloud.points.Count;

                if (data.camera_poses.Count > 0)
                {
                    cloudInfo.cameraPoseCount = data.camera_poses.Count;
                }

                // Create a 3D bounds indicator
                var bounds = new GameObject("PointCloud_Bounds");
                bounds.transform.SetParent(root.transform);
                bounds.transform.position = new Vector3(0, 0, 0);

                EditorUtility.DisplayDialog(
                    "TTV Import Complete",
                    $"Imported {objectsCreated} objects from {data.source_file}\n" +
                    $"Point cloud: {data.point_cloud.points.Count} points\n" +
                    $"Camera poses: {data.camera_poses.Count} frames\n\n" +
                    $"Tip: If you have a .splat file, use GameObject > ToThinkVision > Import Splat File",
                    "OK"
                );
            }
            else
            {
                EditorUtility.DisplayDialog(
                    "TTV Import Complete",
                    $"Imported {objectsCreated} objects from {data.source_file}",
                    "OK"
                );
            }

            Selection.activeGameObject = root;
        }

        private static bool IsGameRelated(string label)
        {
            return label.StartsWith("game_") || label.StartsWith("embodied_");
        }

        private static Color HexToColor(string hex)
        {
            hex = hex.Replace("#", "");
            if (hex.Length != 6) return Color.magenta;
            float r = int.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float g = int.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float b = int.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            return new Color(r, g, b);
        }

        private static string FindSpriteAsset(string jsonPath, string objId)
        {
            // Look for PNG files next to the JSON with matching object ID
            string jsonDir = Path.GetDirectoryName(jsonPath);
            if (string.IsNullOrEmpty(jsonDir)) return null;

            // Search for crop or masked PNG files
            string[] patterns = new[] {
                $"{objId}_masked.png",
                $"{objId}_crop.png",
            };

            foreach (var pattern in patterns)
            {
                string candidate = Path.Combine(jsonDir, pattern);
                if (File.Exists(candidate))
                {
                    // Convert to Unity AssetDatabase path
                    string relativePath = MakeRelativePath("Assets", candidate);
                    if (!string.IsNullOrEmpty(relativePath)) return relativePath;
                }
            }

            return null;
        }

        private static string MakeRelativePath(string fromPath, string toPath)
        {
            if (string.IsNullOrEmpty(fromPath) || string.IsNullOrEmpty(toPath)) return null;

            // Normalize paths
            fromPath = Path.GetFullPath(fromPath).Replace('\\', '/');
            toPath = Path.GetFullPath(toPath).Replace('\\', '/');

            if (toPath.StartsWith(fromPath))
            {
                return "Assets" + toPath.Substring(fromPath.Length);
            }

            // Check if file is somewhere under Assets
            int assetsIndex = toPath.IndexOf("/Assets/");
            if (assetsIndex >= 0)
            {
                return toPath.Substring(assetsIndex + 1);
            }

            return null;
        }

        public static void ImportSplatFile(string splatPath)
        {
            if (!File.Exists(splatPath))
            {
                EditorUtility.DisplayDialog("TTV Splat Import", $"File not found:\n{splatPath}", "OK");
                return;
            }

            // Create a GameObject to hold the splat info
            var splatRoot = new GameObject("TTV_GaussianSplat");

            // Note: The actual .splat rendering requires the UnityGaussianSplatting plugin.
            // We create a placeholder and log instructions.

            var info = splatRoot.AddComponent<TTVSplatInfo>();
            info.splatFilePath = splatPath;
            info.importedAt = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

            // Try to determine file size / splat count
            long fileSize = new FileInfo(splatPath).Length;
            int splatCount = (int)(fileSize / 32); // Each splat = 32 bytes
            info.splatCount = splatCount;

            Debug.Log($"[TTV] Splat file loaded: {splatPath} ({splatCount} splats)");
            Debug.Log("[TTV] To render, install UnityGaussianSplatting: https://github.com/aras-p/UnityGaussianSplatting");

            Selection.activeGameObject = splatRoot;

            EditorUtility.DisplayDialog(
                "TTV Splat Info",
                $"Splat file: {Path.GetFileName(splatPath)}\n" +
                $"Splat count: {splatCount}\n\n" +
                $"To render this:\n" +
                $"1. Install UnityGaussianSplatting plugin\n" +
                $"2. Drag the .splat file into the scene\n" +
                $"3. Adjust camera FOV for best view",
                "OK"
            );
        }
    }

    // ─── Data Models (matching TTV JSON structure) ──────────

    [Serializable]
    public class TTVStructuredOutput
    {
        public string source_file;
        public string source_type;
        public TTVVideoMetadata metadata;
        public List<TTVObject> objects;
        public int frame_count;
        public TTVPointCloud point_cloud;
        public List<TTVCameraPose> camera_poses;
    }

    [Serializable]
    public class TTVVideoMetadata
    {
        public float fps;
        public int total_frames;
        public int width;
        public int height;
        public float duration_seconds;
    }

    [Serializable]
    public class TTVObject
    {
        public string id;
        public string label;
        public string label_custom;
        public float confidence;
        public TTVBBox2D bbox;
        public TTVBBox3D bbox_3d;
        public string dominant_color;
        public int z_index;
        public string text_content;
    }

    [Serializable]
    public class TTVBBox2D
    {
        public float x, y, w, h;
    }

    [Serializable]
    public class TTVBBox3D
    {
        public float x, y, z;
    }

    [Serializable]
    public class TTVPointCloud
    {
        public List<TTVVec3> points;
        public List<TTVVec3Int> colors;
    }

    [Serializable]
    public class TTVVec3
    {
        public float x, y, z;
    }

    [Serializable]
    public class TTVVec3Int
    {
        public int x, y, z;
    }

    [Serializable]
    public class TTVCameraPose
    {
        public int frame_idx;
        public TTVVec3 position;
    }

    // ─── Runtime Components ─────────────────────────────────

    public class TTVObjectInfo : MonoBehaviour
    {
        public string objectId;
        public string label;
        public float confidence;
        public float depthMeters;
    }

    public class TTVPointCloudInfo : MonoBehaviour
    {
        public int pointCount;
        public int cameraPoseCount;
    }

    public class TTVSplatInfo : MonoBehaviour
    {
        public string splatFilePath;
        public int splatCount;
        public string importedAt;
    }
}
