# ToThinkVision Unity Integration

## Files

| File | Purpose |
|------|---------|
| `TTVSceneImporter.cs` | Unity Editor script — import JSON and build scene |
| `TTVRuntimeLoader.cs` | Runtime script — load JSON scenes during gameplay |

## Editor Import (Recommended for scene setup)

1. Copy `TTVSceneImporter.cs` to `Assets/Editor/` in your Unity project
2. Export your video from ToThinkVision with `unity_json` format
3. Copy the exported JSON + PNG files to a folder in your Unity project (e.g. `Assets/TTV_Data/`)
4. In Unity: **GameObject → ToThinkVision → Import Scene from JSON**
5. Select the JSON file
6. Scene is automatically created with:
   - GameObjects for each detected object
   - BoxCollider2D components
   - SpriteRenderer with dominant colors
   - Rigidbody2D for NPCs/items
   - Sprite textures (if PNG files are in the project)

## Runtime Loading (for dynamic scenes)

1. Copy `TTVRuntimeLoader.cs` to your project (any folder, NOT Editor/)
2. Place the JSON file in `Assets/StreamingAssets/`
3. Add `TTVRuntimeLoader` component to any GameObject
4. Set `Json File Path` to the filename
5. Call `LoadScene()` via code or UI button

```csharp
// Example: load on button click
public TTVRuntimeLoader loader;

public void OnLoadClicked()
{
    loader.LoadScene("demo_unity_json.json");
}

// Or listen to events
loader.OnLoadComplete += (count) => Debug.Log($"Loaded {count} objects");
```

## Splat Files (3D Gaussian Splatting)

1. Install the [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) plugin
2. In Unity: **GameObject → ToThinkVision → Import Splat File**
3. Select the `.splat` file

## JSON Data Fields Used by the Importer

The importer reads these fields from the TTV JSON:

```
objects[].id                    → Stored in TTVObjectInfo component
objects[].label                 → GameObject name
objects[].label_custom          → Custom name (if available)
objects[].bbox.x/y/w/h          → Position + Scale in world space
objects[].bbox_3d.z             → Z depth position
objects[].dominant_color        → Material color (hex)
objects[].z_index               → Sprite sorting order
```

## Notes

- **Coordinate system**: TTV uses top-left origin (image coords), Unity uses bottom-left.
  The importer flips Y automatically.
- **Scale**: Default `pixelsPerUnit = 100`. A 1920×1080 video becomes 19.2×10.8 world units.
- **Colliders**: Only `game_*` and `embodied_*` type objects get colliders.
- **Physics**: `game_npc` gets Dynamic Rigidbody2D, `game_item` gets Static.
