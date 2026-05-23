# ToThinkVision v2.0 — Universal Vision Structuring Engine

Input any image or video, automatically parse into **structured 2D + 3D scene data** with per-object meshes, camera poses, motion trajectories, and export to **20+ formats** (glTF/OBJ with UV+textures, Unity/UE5, After Effects, Photoshop, .splat Gaussian splats).

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
# Download model weights interactively:
chmod +x install_models.sh && ./install_models.sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Models auto-fallback to mock mode if weights are missing — never crashes.

---

## Pipeline: 5 Stages from Video to Editable 3D Scene

```
Input: Image / Video (.mp4 / .png)
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Perception — "What's in the scene?"              │
│   Detection → Segmentation → Tracking → OCR               │
│   Output: 2D bbox + mask + cross-frame track_id per object│
├──────────────────────────────────────────────────────────┤
│ Stage 2: Depth — "How far is each pixel?"                 │
│   Monocular depth estimation (per frame)                   │
│   Output: Depth map (H×W) in meters                       │
├──────────────────────────────────────────────────────────┤
│ Stage 3: 3D Reconstruction — "What does the real world    │
│   look like?"                                              │
│   Multi-view fusion → Point cloud → Per-object mesh       │
│   → UV unwrap → Multi-view texture bake → Mesh refinement  │
│   Output: 3D point cloud + camera poses + textured meshes  │
├──────────────────────────────────────────────────────────┤
│ Stage 4: Scene Understanding — "What kind of scene is     │
│   this?"                                                   │
│   Ground plane detection (RANSAC) → Gravity alignment      │
│   → Object classification → Physical properties            │
│   Output: Scene layout {floor, walls, furniture, ...}      │
├──────────────────────────────────────────────────────────┤
│ Stage 5: Export — "What can the target engine read?"       │
│   glTF 2.0 (UV+texture+PBR) / OBJ+MTL / Unity JSON        │
│   / UE JSON / AE ExtendScript / PSD / .splat              │
│   Output: Files ready for game engines, animation, design  │
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
| **Depth Pro** | Apple | 4-8GB | Metric depth estimation (meters), <1s inference |
| **MASt3R** | NAVER | 24-48GB | Video → 3D point cloud + camera poses (SfM) |
| **3D Gaussian Splatting** | Nerfstudio | 24GB | Video → photorealistic 3D Gaussian scene |

All models can be toggled off via environment variables. Auto-fallback to mock data if weights missing.

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

## Export Formats (20+)

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

### Universal
| Format | Extension | Description |
|--------|-----------|-------------|
| `full_json` | `.json` | Complete structured output with all 2D + 3D + mesh + camera + Gaussian data |

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
│   │   ├── mast3r.py                 # MASt3R: 3D point cloud + camera poses (SfM)
│   │   ├── gaussian_splatting.py     # 3DGS: training + .splat/.ply export
│   │   ├── strongsort_wrapper.py     # StrongSORT: multi-object tracking
│   │   └── mesh_reconstruction.py    # ★ Per-object 3D mesh: depth→Poisson→UV→texture
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
├── tests/                            # 51 unit tests (all pass)
│   ├── test_schemas.py               # Data structures + exporters + utils (19)
│   ├── test_exporters.py             # Export format validation (12)
│   └── test_pipeline.py              # Tracker + Geometry + Color + end-to-end (20)
│
├── outputs/                          # Exported results
├── requirements.txt                  # Python dependencies
├── install_models.sh                 # Interactive model weight downloader
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
| `TTV_ENABLE_MAST3R` | `true` | Enable MASt3R |
| `TTV_ENABLE_GAUSSIAN_SPLATTING` | `false` | Enable 3D Gaussian Splatting |
| `TTV_OUTPUT_DIR` | `./outputs` | Output directory |

---

## Running Tests

```bash
# Run all 51 tests
MOCK_MODE=true pytest tests/ -v

# Run by module
MOCK_MODE=true pytest tests/test_schemas.py -v      # Data structures + exporters
MOCK_MODE=true pytest tests/test_exporters.py -v    # Export format validation
MOCK_MODE=true pytest tests/test_pipeline.py -v     # Tracker + Geometry + pipeline
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
