// TTVRuntimeLoader.cs
// ToThinkVision → Unity Runtime Scene Loader
//
// Usage:
// 1. Place this script on any GameObject in your scene
// 2. Set the jsonFilePath in the Inspector
// 3. Call LoadScene() at runtime via code, button click, or scene start
//
// Example:
//   GetComponent<TTVRuntimeLoader>().LoadScene();
//   // Or from a button:
//   public void OnLoadButton() { loader.LoadScene(); }

using UnityEngine;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace ToThinkVision
{
    public class TTVRuntimeLoader : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Path to the TTV JSON file (relative to StreamingAssets or absolute)")]
        public string jsonFilePath;

        [Tooltip("Pixels per Unity unit. Default 100 means 1920px = 19.2 world units")]
        public float pixelsPerUnit = 100f;

        [Tooltip("If true, adds BoxCollider2D to game objects")]
        public bool addColliders = true;

        [Tooltip("If true, adds Rigidbody2D to NPCs")]
        public bool addPhysics = true;

        [Header("Events")]
        public event Action<GameObject> OnObjectCreated;
        public event Action<int> OnLoadComplete;

        private GameObject _sceneRoot;

        public void LoadScene()
        {
            StartCoroutine(LoadSceneAsync());
        }

        public void LoadScene(string customPath)
        {
            jsonFilePath = customPath;
            LoadScene();
        }

        private IEnumerator LoadSceneAsync()
        {
            string fullPath = ResolvePath(jsonFilePath);

            if (!File.Exists(fullPath))
            {
                Debug.LogError($"[TTV] JSON file not found: {fullPath}");
                yield break;
            }

            string json = File.ReadAllText(fullPath);
            var data = JsonUtility.FromJson<TTVRuntimeData>(json);

            if (data == null || data.objects == null || data.objects.Count == 0)
            {
                Debug.LogError("[TTV] No objects found in JSON");
                yield break;
            }

            // Create root
            if (_sceneRoot != null)
                Destroy(_sceneRoot);

            _sceneRoot = new GameObject($"TTV_{data.source_file}");
            _sceneRoot.tag = "TTVScene";

            yield return null;

            int count = 0;
            foreach (var obj in data.objects)
            {
                if (obj.bbox == null) continue;

                var go = CreateObject(obj);
                if (go != null)
                {
                    go.transform.SetParent(_sceneRoot.transform);
                    OnObjectCreated?.Invoke(go);
                    count++;
                }

                // Yield every 10 objects to avoid frame stutter
                if (count % 10 == 0)
                    yield return null;
            }

            Debug.Log($"[TTV] Loaded {count} objects from {data.source_file}");
            OnLoadComplete?.Invoke(count);
        }

        private GameObject CreateObject(TTVRuntimeObjectData obj)
        {
            if (obj.bbox == null) return null;

            var go = new GameObject(obj.label_custom ?? obj.label);

            // Position
            float worldX = (obj.bbox.x + obj.bbox.w / 2f) / pixelsPerUnit;
            float worldY = -(obj.bbox.y + obj.bbox.h / 2f) / pixelsPerUnit;
            float worldZ = obj.bbox_3d != null ? obj.bbox_3d.z / pixelsPerUnit : 0f;
            go.transform.position = new Vector3(worldX, worldY, worldZ);

            // Scale
            float scaleX = obj.bbox.w / pixelsPerUnit;
            float scaleY = obj.bbox.h / pixelsPerUnit;
            go.transform.localScale = new Vector3(scaleX, scaleY, 1f);

            // Store info
            var info = go.AddComponent<TTVObjectInfo>();
            info.objectId = obj.id;
            info.label = obj.label_custom ?? obj.label;
            info.confidence = obj.confidence;
            info.depthMeters = obj.bbox_3d != null ? obj.bbox_3d.z : 0f;

            // Renderer with color
            Color objColor = Color.white;
            if (!string.IsNullOrEmpty(obj.dominant_color))
            {
                objColor = HexToColor(obj.dominant_color);
            }

            var renderer = go.AddComponent<SpriteRenderer>();
            var mat = new Material(Shader.Find("Sprites/Default"));
            mat.color = objColor;
            renderer.material = mat;
            renderer.color = objColor;
            renderer.sortingOrder = obj.z_index;

            // Collider
            if (addColliders && IsGameRelated(obj.label))
            {
                bool isTrigger = obj.label.Contains("door") || obj.label.Contains("effect");
                var col = go.AddComponent<BoxCollider2D>();
                col.isTrigger = isTrigger;
            }

            // Physics
            if (addPhysics)
            {
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
            }

            return go;
        }

        private string ResolvePath(string path)
        {
            // Try StreamingAssets first (for runtime builds)
            string streamingPath = Path.Combine(Application.streamingAssetsPath, path);
            if (File.Exists(streamingPath)) return streamingPath;

            // Try as absolute path
            if (Path.IsPathRooted(path) && File.Exists(path)) return path;

            // Try relative to Application.dataPath
            string relativePath = Path.Combine(Application.dataPath, path);
            if (File.Exists(relativePath)) return relativePath;

            return path; // Return as-is, let the file check fail
        }

        private bool IsGameRelated(string label)
        {
            return label.StartsWith("game_") || label.StartsWith("embodied_");
        }

        private Color HexToColor(string hex)
        {
            hex = hex.Replace("#", "");
            if (hex.Length != 6) return Color.magenta;
            float r = int.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float g = int.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            float b = int.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
            return new Color(r, g, b);
        }
    }

    // ─── Runtime Data Models (minimal subset needed at runtime) ──

    [Serializable]
    public class TTVRuntimeData
    {
        public string source_file;
        public string source_type;
        public TTVRuntimeMetadata metadata;
        public List<TTVRuntimeObjectData> objects;
        public int frame_count;
    }

    [Serializable]
    public class TTVRuntimeMetadata
    {
        public float fps;
        public int total_frames;
        public int width;
        public int height;
        public float duration_seconds;
    }

    [Serializable]
    public class TTVRuntimeObjectData
    {
        public string id;
        public string label;
        public string label_custom;
        public float confidence;
        public TTVRuntimeBBox2D bbox;
        public TTVRuntimeBBox3D bbox_3d;
        public string dominant_color;
        public int z_index;
        public string text_content;
    }

    [Serializable]
    public class TTVRuntimeBBox2D
    {
        public float x, y, w, h;
    }

    [Serializable]
    public class TTVRuntimeBBox3D
    {
        public float x, y, z;
    }
}
