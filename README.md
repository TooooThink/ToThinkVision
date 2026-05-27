# ToThinkVision v2.1 — Video → Editable 4D Scene Decomposition Engine

Input any image or video (including AI-generated world model outputs), automatically decompose into **structured 2D + 3D + 4D scene data** with per-object meshes, **6DoF motion trajectories**, camera poses, and export to **25+ formats** (glTF/OBJ with UV+textures, Unity/UE5, After Effects, Photoshop, .splat Gaussian splats, **animated glTF·USD·Blender scenes**).

v2.1 adds **4D scene decomposition**: input a video, get each object's 3D geometry + **6DoF motion trajectory** + dynamic scene graph — ready to import into Unity/Unreal/Blender/AE for editing.

Open-source research & industrial project. All models run with automatic mock fallback — works without GPU.

---

## Quick Start

### Try it now (no GPU needed)

```bash
pip install -r requirements.txt
MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Full deployment (with NVIDIA GPU + model weights)

```bash
pip install -r requirements.txt
# Download v2 base model weights interactively:
chmod +x install_models.sh && ./install_models.sh
# [Optional] Install v2.1 advanced 4D models (CoTracker3/ObjectGS/Spann3R/Shape of Motion):
chmod +x install_models_v3.sh && ./install_models_v3.sh
#   All 4 have mock fallback — pipeline runs even if you skip them.
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Models auto-fallback to mock mode if weights are missing — never crashes.

---

## Pipeline: 9 Stages from Video to Editable 4D Scene

```
Input: Image / Video (.mp4 / .png) — including AI-generated world model outputs
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Perception — "What's in the scene?"              │
│   SAM 3 + OmniParser + Grounding DINO + StrongSORT + OCR  │
│   Output: 2D bbox + mask + track_id per object per frame  │
├──────────────────────────────────────────────────────────┤
│ Stage 2: Depth + 2D Completion                            │
│   Depth Pro → metric depth; LaMa inpaint for partial objs │
│   Output: Depth map (H×W meters) + completed masks        │
├──────────────────────────────────────────────────────────┤
│ Stage 3: 3D Reconstruction (Spann3R / MASt3R / VGGT)     │
│   Multi-view fusion → point cloud + camera poses          │
│   + per-object mesh + UV unwrap + texture bake            │
│   Output: 3D PC + poses + textured meshes (.obj/.gltf)    │
├──────────────────────────────────────────────────────────┤
│ Stage 4: Scene Understanding                              │
│   Ground plane (RANSAC) → gravity align → classification  │
├──────────────────────────────────────────────────────────┤
│ Stage 5: 3D Gaussian Splatting / ObjectGS (optional)     │
│   Scene-level 3DGS or per-object Gaussians                │
├──────────────────────────────────────────────────────────┤
│ Stage 6: 4D Trajectory Extraction ⭐                     │
│   Option A: Shape of Motion (end-to-end 4D from video)    │
│   Option B: ICP + PCA + B-spline + CoTracker3 enhance    │
│   Output: per-object 6DoF (pos + rot + scale + velocity)  │
├──────────────────────────────────────────────────────────┤
│ Stage 7: 4D Gaussian Splatting (HexPlane, optional) ⭐   │
│   Temporally-varying Gaussian scene representation        │
├──────────────────────────────────────────────────────────┤
│ Stage 8: Dynamic Scene Graph ⭐                          │
│   Time-varying spatial relations + interaction events     │
│   (collisions, pick-ups, put-downs, contact start/end)    │
├──────────────────────────────────────────────────────────┤
│ Stage 9: Animated Export ⭐                              │
│   Animated glTF (.glb) / USDA / Blender import script     │
│   + 4D Scene Graph JSON + AE Keyframes                    │
└──────────────────────────────────────────────────────────┘
```

---

## Core Models

| Model | Source | VRAM | Role |
|-------|--------|------|------|
| **SAM 3** | Meta AI | 12-24GB | Unified detection + segmentation + video tracking |
| **OmniParser v2** | Microsoft | 8-12GB | UI element detection + interactivity prediction |
| **Grounding DINO** | IDEA-Research | 4-8GB | Open-vocabulary object detection |
| **StrongSORT** | CVPR 2023 | CPU | ReID + Kalman filter + GMC multi-object tracking |
| **Depth Pro** | Apple (ICLR 2025) | 4-8GB | Metric depth estimation (meters), <1s inference |
| **MASt3R / VGGT** | NAVER / Meta | 24-48GB | Video → 3D point cloud + camera poses (SfM) |
| **Spann3R** ⭐ | 3DV 2025 | 24-48GB | 3D reconstruction with spatial memory (long sequences) |
| **ObjectGS** ⭐ | ICCV 2025 | 24GB | Per-object 3D Gaussian Splatting |
| **CoTracker3** ⭐ | Meta AI | 8-12GB | Dense point tracking (265×265) for accurate trajectories |
| **Shape of Motion** ⭐ | ICCV 2025 | 24GB | End-to-end 4D reconstruction from monocular video |
| **3D Gaussian Splatting** | Nerfstudio | 24GB | Video → photorealistic 3D Gaussian scene |

⭐ New in v2.1: advanced 3D/4D models. All models can be toggled off via environment variables or API form params. Auto-fallback to mock data if weights/repo missing.

---

## What You Get

### Per-Object Data

- **2D**: bbox, segmentation mask, contour, crop image
- **3D**: depth value, 3D position (bbox_3d), point cloud indices
- **Mesh**: triangle mesh (vertices/faces/normals), UV coordinates, baked texture (512×512 PNG), per-object OBJ file
- **Temporal**: appear/disappear frames, trajectory [{x, y, t}], velocity, depth over time
- **Appearance**: dominant color, color palette
- **Relations**: collision with other objects, relative positions (above/below/left/right)
- **Interaction**: clickable, scrollable, toggle state

### Global Scene Data

- **Point Cloud**: (N, 3) XYZ points + (N, 3) RGB colors + (N, 3) normals
- **Camera Poses**: per-frame intrinsics (3×3 K) + extrinsics (4×4 RT) + position + quaternion rotation
- **Gaussian Splats**: means, quaternions, scales, opacities, spherical harmonic coefficients
- **Scene Mesh**: combined OBJ with all object meshes, UV coordinates, textures, MTL materials

---

## Export Formats (25+)

### 3D Mesh & Scene
| Format | Extension | Description |
|--------|-----------|-------------|
| `gltf` | `.gltf` + `.bin` | glTF 2.0 with UV coordinates, embedded textures, PBR materials, camera pose nodes |
| `obj_3d` | `.obj` + `.mtl` | Wavefront OBJ with UV, MTL materials, texture references (map_Kd), camera pose comments |

### Game Engines
| Format | Extension | Description |
|--------|-----------|-------------|
| `unity_json` | `.json` | Unity scene with 3D meshes (MeshFilter + MeshRenderer + MeshCollider), textures, transforms |
| `ue_json` | `.json` | UE5 actors with StaticMeshComponent, MaterialInterface, transforms |
| `unity_splat` | `.splat` | 3D Gaussian splat binary for UnityGaussianSplatting plugin |
| `ue_splat` | `.splat` | 3D Gaussian splat binary for UnrealSplat plugin |
| `collision_json` | `.json` | Pure 2D/3D collision box data for physics |

### Animation & Design
| Format | Extension | Description |
|--------|-----------|-------------|
| `psd_static` | `.psd` | Photoshop PSD — each object as a transparent layer |
| `psd_animated` | `.psd` | Photoshop PSD — each frame as a Group with object layers |
| `ae_project` | `.jsx` + `.json` | After Effects ExtendScript — auto-create comp with tracked layers + camera |
| `ae_keyframes` | `.json` | AE keyframe timeline (position/scale/rotation/opacity) |
| `video_trajectory` | `.csv` | Per-frame object trajectory CSV |
| `pr_markers` | `.json` | Premiere Pro chapter markers with timecodes |

### UI & Embodied
| Format | Extension | Description |
|--------|-----------|-------------|
| `figma_json` | `.json` | Figma document structure (RECTANGLE/TEXT/FRAME nodes) |
| `html_css` | `.html` | Self-contained HTML with absolute-positioned elements |
| `ui_json` | `.json` | Simplified UI component list |
| `embodied_json` | `.json` | Scene objects with 3D pose + physical properties + interaction sequence |
| `robot_action` | `.json` | Robot approach/grasp action sequences |
| `pose_csv` | `.csv` | Per-frame 3D pose CSV |

### 4D Scene (v2.1 new) ⭐
| Format | Extension | Description |
|--------|-----------|-------------|
| `animated_gltf` | `.glb` | Animated glTF binary — per-object 3D mesh + keyframe animation (translation + rotation + scale) for Unity/Blender |
| `usd_scene` | `.usda` | USD text-format scene with time-sampled transforms, materials, animated camera for Unreal/Omniverse |
| `blender_scene` | `.py` + `.obj` + `.json` | Blender Python import script + per-object OBJ meshes + trajectory JSON |
| `scene_graph_json` | `.json` | Dynamic 4D scene graph with nodes (objects), edges (time-varying relations), interaction events |

### Universal
| Format | Extension | Description |
|--------|-----------|-------------|
| `full_json` | `.json` | Complete structured output with all 2D + 3D + 4D + mesh + camera + Gaussian data |

---

## Project Structure

```
ToThinkVision/
├── app/
│   ├── main.py                       # FastAPI server + REST API
│   ├── config.py                     # Global config (env vars, paths, thresholds)
│   ├── pipeline.py                   # ★ Core pipeline: orchestrates all models
│   ├── schemas.py                    # Pydantic data models (StructuredObject, Mesh3D, etc.)
│   ├── preprocessor.py               # Video frame extraction, image preprocessing
│   │
│   ├── models/                       # AI model wrappers (all with mock fallback)
│   │   ├── sam3.py                   # SAM 3: detection + segmentation + video tracking
│   │   ├── grounding_dino.py         # Grounding DINO: open-vocabulary detection
│   │   ├── omniparser.py             # OmniParser v2: UI element detection
│   │   ├── depth_pro.py              # Depth Pro: metric depth estimation
│   │   ├── mast3r.py                 # MASt3R/VGGT: 3D point cloud + camera poses
│   │   ├── gaussian_splatting.py     # 3DGS: training + .splat/.ply export
│   │   ├── strongsort_wrapper.py     # StrongSORT: multi-object tracking
│   │   ├── mesh_reconstruction.py    # ★ Per-object 3D mesh: depth→Poisson→UV→texture
│   │   ├── cotracker3.py             # ★ CoTracker3: dense point tracking (265×265)
│   │   ├── object_gs.py              # ★ ObjectGS: per-object 3D Gaussian Splatting
│   │   ├── spann3r.py                # ★ Spann3R: 3D reconstruction with spatial memory
│   │   ├── shape_of_motion.py        # ★ Shape of Motion: end-to-end 4D reconstruction
│   │   ├── trajectory_4d.py          # ★ 6DoF trajectory: ICP + PCA + B-spline
│   │   └── gaussian_splatting_4d.py  # ★ 4DGS with HexPlane temporal decomposition
│   │
│   ├── scene/                        # 4D scene understanding
│   │   ├── scene_graph_4d.py         # ★ Dynamic scene graph builder
│   │   └── world_model_adapter.py    # ★ AI-generated video detection + threshold tuning
│   │
│   ├── exporters/                    # Export modules
│   │   ├── base.py                   # Base exporter class
│   │   ├── gltf_exporter.py          # glTF 2.0 with UV + textures + PBR materials
│   │   ├── obj_exporter.py           # Wavefront OBJ + MTL with UV + textures
│   │   ├── game_exporter.py          # Unity JSON / UE JSON / Collision (3D mesh aware)
│   │   ├── ae_project_exporter.py    # After Effects ExtendScript (.jsx)
│   │   ├── psd_exporter.py           # Photoshop PSD (static + animated)
│   │   ├── ui_exporter.py            # Figma JSON / HTML-CSS / UI JSON
│   │   ├── video_exporter.py         # AE keyframes / trajectory CSV / PR markers
│   │   ├── embodied_exporter.py      # Robot actions / pose CSV
│   │   ├── splat_exporter.py         # .splat binary / .ply Gaussian params
│   │   ├── image_exporter.py         # Crops / masks / depth visualization
│   │   ├── animated_gltf_exporter.py # ★ Animated .glb with per-object keyframe anim
│   │   ├── usd_exporter.py           # ★ USDA with time-sampled transforms
│   │   ├── blender_exporter.py       # ★ Blender Python script + OBJ + trajectory JSON
│   │   └── manifest.py               # Export manifest (README.txt)
│   │
│   └── utils/                        # Utility functions
│       ├── camera.py                 # Camera intrinsics/extrinsics, coordinate transforms
│       ├── geometry.py               # Collision detection, relative position, z-index
│       ├── color.py                  # Dominant color extraction
│       ├── pointcloud.py             # Depth back-projection, normals, voxel filtering
│       ├── texture_bake.py           # UV unwrapping + multi-view texture baking
│       └── scene_understanding.py    # Ground plane (RANSAC), gravity alignment
│
├── docs/
│   └── GETTING_STARTED.md            # ★ Beginner's guide: knowledge map + learning path
│
├── static/
│   ├── index.html                    # Single-page web UI
│   └── style.css                     # Dark theme styling
│
├── tests/                            # 112 unit tests passing
│   ├── test_schemas.py               # Data structures + exporters + utils (19)
│   ├── test_exporters.py             # Export format validation (12)
│   ├── test_pipeline.py              # Tracker + Geometry + Color + end-to-end (20)
│   └── test_4d_scene.py              # ★ 4D trajectory + animated exporters + scene graph (63)
│
├── outputs/                          # Exported results
├── requirements.txt                  # Python dependencies
├── install_models.sh                 # Interactive model weight downloader (v2 base models)
├── install_models_v2.sh              # v2 advanced model downloader
├── install_models_v3.sh              # ★ v2.1 4D model installer (CoTracker3/ObjectGS/Spann3R/Shape of Motion)
├── README.md                         # English documentation (this file)
└── README_CN.md                      # Chinese documentation
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/api/formats` | List all export formats |
| `POST` | `/api/process` | Upload file + process + export |
| `GET` | `/api/download/{filename}` | Download result file |
| `GET` | `/api/health` | Health check + model status |

### Example: Process video, export glTF + Unity JSON

```bash
curl -X POST http://localhost:8000/api/process \
  -F "file=@demo.mp4" \
  -F 'export_formats=["gltf", "unity_json"]' \
  -F "mode=general"
```

### Example: Python direct call

```python
from app.pipeline import process_file
from app.exporters.gltf_exporter import GltfExporter

result = process_file("demo.mp4", mode="video")

# Check 3D data
print(f"Point cloud: {len(result.point_cloud.points)} points")
print(f"Camera poses: {len(result.camera_poses)} frames")
print(f"Objects with 3D mesh: {sum(1 for o in result.objects if o.mesh_3d)}")
print(f"Scene mesh: {result.scene_mesh_path}")

# Export glTF
exporter = GltfExporter()
gltf_path = exporter.export(result)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTV_MOCK_MODE` | `false` | `true` uses mock data (no GPU needed) |
| `TTV_DEVICE` | `cuda` | Inference device: `cuda` / `cpu` / `mps` |
| `TTV_MAX_VIDEO_FRAMES` | `300` | Max frames to extract from video |
| `TTV_MODEL_CACHE_DIR` | `~/.cache/tothinkvision` | Model weights directory |
| `TTV_MAX_UPLOAD_MB` | `500` | Max upload file size (MB) |
| `TTV_ENABLE_SAM3` | `true` | Enable SAM 3 |
| `TTV_ENABLE_OMNIPARSER` | `true` | Enable OmniParser v2 |
| `TTV_ENABLE_GROUNDING_DINO` | `true` | Enable Grounding DINO |
| `TTV_ENABLE_STRONGSORT` | `true` | Enable StrongSORT |
| `TTV_ENABLE_DEPTH_PRO` | `true` | Enable Depth Pro |
| `TTV_ENABLE_MAST3R` | `true` | Enable MASt3R / VGGT |
| `TTV_ENABLE_GAUSSIAN_SPLATTING` | `false` | Enable 3D Gaussian Splatting |
| **v2.1 advanced models** (need separate setup via `install_models_v3.sh`) | | |
| `TTV_ENABLE_COTRACKER3` | `true` | Enable CoTracker3 dense point tracking |
| `TTV_ENABLE_OBJECTGS` | `false` | Enable ObjectGS (needs repo clone) |
| `TTV_ENABLE_SPANN3R` | `false` | Enable Spann3R (needs repo clone) |
| `TTV_ENABLE_SHAPE_OF_MOTION` | `false` | Enable Shape of Motion (needs repo clone) |
| `TTV_ENABLE_4D_TRAJECTORY` | `true` | Enable 4D 6DoF trajectory extraction |
| `TTV_ENABLE_4DGS` | `false` | Enable 4D Gaussian Splatting (heavy, multi-GPU) |
| `TTV_ENABLE_SCENE_GRAPH` | `true` | Enable dynamic scene graph construction |
| `TTV_ENABLE_ANIMATED_EXPORT` | `true` | Enable animated glTF/USD/Blender export |
| `OBJECT_GS_PATH` | — | Path to ObjectGS repo clone |
| `SPANN3R_PATH` | — | Path to Spann3R repo clone |
| `SHAPE_OF_MOTION_PATH` | — | Path to Shape of Motion repo clone |
| `TTV_OUTPUT_DIR` | `./outputs` | Output directory |

---

## Running Tests

```bash
# Run all 112+ tests
MOCK_MODE=true pytest tests/ -v

# Run by module
MOCK_MODE=true pytest tests/test_schemas.py -v      # Data structures + exporters + utils
MOCK_MODE=true pytest tests/test_exporters.py -v    # Export format validation
MOCK_MODE=true pytest tests/test_pipeline.py -v     # Tracker + Geometry + pipeline
MOCK_MODE=true pytest tests/test_4d_scene.py -v     # ★ 4D trajectory + animated exporters + scene graph (63)
```

---

## Getting Started Learning

New to computer vision / 3D reconstruction? Read **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — a complete beginner's guide covering:

- The complete knowledge map (20 concepts, 6 layers)
- How each concept maps to project code
- Recommended learning path (8 stages with practice tasks)
- Top 5 research breakthrough opportunities
- Resource links (courses, papers, tools)

---

## FAQ

**Q: Can I run this without a GPU?**
A: Yes. Set `MOCK_MODE=true` to run the full pipeline with simulated data. Format validation and export logic don't need GPU at all.

**Q: What if I don't have enough VRAM?**
A: Disable heavy models via env vars: `TTV_ENABLE_MAST3R=false TTV_ENABLE_GAUSSIAN_SPLATTING=false`. Lightweight models (Grounding DINO, Depth Pro) run on 4-8GB.

**Q: Video processing is too slow?**
A: Limit frames with `TTV_MAX_VIDEO_FRAMES=60` or sample every N seconds with `TTV_FRAME_SAMPLE_INTERVAL=1.0`.

**Q: Model download failed?**
A: Auto-fallback to mock mode — service keeps running. Download weights manually to `TTV_MODEL_CACHE_DIR`.

**Q: How do I use the .splat file?**
A: Unity: install [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) plugin, drag `.splat` into scene. UE: use [UnrealSplat](https://github.com/mrquicksilver/UnrealSplat).

**Q: How do I import 3D meshes into Unity/Blender?**
A: glTF: drag into Blender directly, or use [gltf-viewer](https://gltf-viewer.donmccurdy.com) for web preview. OBJ: universal support — import into any 3D software. Unity JSON: needs companion C# importer script (see manifest instructions).

**Q: How do I install the new v2.1 models (CoTracker3, ObjectGS, Spann3R, Shape of Motion)?**
A: Run `chmod +x install_models_v3.sh && ./install_models_v3.sh` — it interactively clones repos, downloads weights, and sets env vars. CoTracker3 needs only weights (torch.hub auto-download); ObjectGS/Spann3R/Shape of Motion need full repo clones + submodules + their own dependencies. All 4 have mock fallback if not installed.

**Q: How do I use the animated glTF / USD / Blender exports?**
A: **Animated glTF (.glb)**: drag into Blender or [gltf-viewer](https://gltf-viewer.donmccurdy.com); import into Unity via `GameObject > Import Package` or into Unreal via glTF plugin. **USDA**: open in NVIDIA Omniverse or import into Unreal via USD plugin. **Blender export**: run the generated `.py` script inside Blender (`File > Open > scene.py`) — it auto-imports all per-object OBJ meshes and applies keyframe animation.

**Q: Can this handle AI-generated videos (Sora/Veo/Kling)?**
A: Yes. Tick the "World Model Video" option (or set `is_world_model_video=True` via API). The pipeline automatically: (1) detects AI-generated video characteristics via `WorldModelAdapter`, (2) relaxes ICP/deformation thresholds (2×/1.5×), (3) increases B-spline smoothing (+0.2–0.3) to absorb temporal jitter. Shape of Motion is the recommended backend for these — it's end-to-end and more robust to non-physical geometry.

**Q: What's the difference between 4D Trajectory and Shape of Motion?**
A: **4D Trajectory** (default on, CPU) is a pipeline approach: depth maps + masks + ICP alignment → per-object 6DoF. Works with any depth estimator. **Shape of Motion** (opt-in, 24GB GPU) is end-to-end 4D reconstruction from monocular video — jointly optimizes geometry + motion, no separate depth/tracking needed. Use Shape of Motion when available; fall back to 4D Trajectory when memory-limited.
