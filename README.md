# ToThinkVision — Universal Vision Structuring Engine

**四合一通用视觉结构化引擎**

Input: Image (UI/Game/Anime/Real/AI) or Video
Core: Unified structured JSON → Objects, coordinates, hierarchy, color, text, motion trajectories, spatial relations, temporal info
Output: UI/Figma/HTML | Game/Unity/UE | Video/AE Keyframes | Embodied AI/Robot Data

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run server (mock mode for testing without GPU/models)
MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Open browser
# Visit: http://localhost:8000
```

## Full Setup (with GPU models)

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Start server (will use real models if weights are available)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Layer                           │
│  Image (JPG/PNG/WebP) / Video (MP4/AVI/MOV)             │
│  → Preprocessing → Video Frame Extraction (FFmpeg)       │
├─────────────────────────────────────────────────────────┤
│              Unified Intermediate Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐       │
│  │ SAM      │ │ Grounding│ │ OCR  │ │ Depth    │       │
│  │ Segment  │ │ DINO     │ │      │ │ Anything │       │
│  └──────────┘ └──────────┘ └──────┘ └──────────┘       │
│  → Cross-frame Tracking → Unified JSON Schema            │
├─────────────────────────────────────────────────────────┤
│                    Export Layer                          │
│  Module 1: UI     → Figma JSON / HTML+CSS / UI JSON     │
│  Module 2: Game   → Unity JSON / UE JSON / Collision    │
│  Module 3: Video  → AE Keyframes / Trajectory CSV       │
│  Module 4: Embodied → Robot Actions / Pose CSV           │
└─────────────────────────────────────────────────────────┘
```

## API Endpoints

### GET `/`
Frontend UI — upload, select format, process, download.

### GET `/api/formats`
List all available export formats.

### POST `/api/process`
Upload file and process it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | file | Image or video file |
| `export_format` | string | Output format (see `/api/formats`) |
| `mode` | string | Detection mode: general, ui, game, video, embodied |

### GET `/api/download/{filename}`
Download a processed output file.

### GET `/api/health`
Health check endpoint.

## Unified JSON Schema

Each detected object contains:

```json
{
  "id": "obj_0001",
  "label": "ui_button",
  "confidence": 0.92,
  "bbox": { "x": 100, "y": 200, "w": 120, "h": 40 },
  "contour": [{ "x": 100, "y": 200 }, ...],
  "bbox_3d": { "x": 160, "y": 220, "z": 2.5 },
  "depth_value": 180.0,
  "dominant_color": "#3b82f6",
  "z_index": 5,
  "text_content": "Submit",
  "temporal": {
    "frame_index": 10,
    "appear_frame": 0,
    "disappear_frame": -1,
    "trajectory": [{ "x": 160, "y": 220, "t": 0 }, ...],
    "velocity": { "vx": 0.5, "vy": 0.2 }
  },
  "relations": {
    "parent_id": null,
    "collision_with": ["obj_0002"],
    "relative_positions": [{ "target_id": "obj_0002", "relation": "above" }]
  },
  "interaction": {
    "type": "clickable",
    "clickable": true
  }
}
```

## Export Formats

### UI Module
| Format | File | Description |
|--------|------|-------------|
| `figma_json` | `.json` | Figma-compatible document structure |
| `html_css` | `.html` | Self-contained HTML with positioned elements |
| `ui_json` | `.json` | Simplified UI component JSON |

### Game Module
| Format | File | Description |
|--------|------|-------------|
| `unity_json` | `.json` | Unity GameObject hierarchy with colliders |
| `ue_json` | `.json` | Unreal Engine actor structure |
| `collision_json` | `.json` | Pure collision box data |

### Video Module
| Format | File | Description |
|--------|------|-------------|
| `ae_keyframes` | `.json` | After Effects keyframe timeline |
| `video_trajectory` | `.csv` | Per-frame object trajectory CSV |
| `pr_markers` | `.json` | Premiere Pro chapter markers |

### Embodied AI Module
| Format | File | Description |
|--------|------|-------------|
| `embodied_json` | `.json` | Full scene + interaction sequence |
| `robot_action` | `.json` | Robot approach/grasp action sequences |
| `pose_csv` | `.csv` | Per-frame 3D pose CSV |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTV_MOCK_MODE` | `false` | Use mock data instead of real models |
| `TTV_DEVICE` | `cuda` | Device for model inference |
| `TTV_MAX_VIDEO_FRAMES` | `300` | Max frames to extract from video |
| `TTV_MODEL_CACHE_DIR` | `~/.cache/tothinkvision` | Model weights directory |
| `TTV_MAX_UPLOAD_MB` | `500` | Max upload size in MB |

## Running Tests

```bash
# Run all tests (uses mock mode)
MOCK_MODE=true pytest tests/ -v

# Run specific test file
MOCK_MODE=true pytest tests/test_schemas.py -v
MOCK_MODE=true pytest tests/test_exporters.py -v
MOCK_MODE=true pytest tests/test_pipeline.py -v
```

## Dependencies

- **Segmentation**: Segment Anything (SAM)
- **Detection**: Grounding DINO (open-vocabulary)
- **OCR**: PaddleOCR
- **Depth**: Depth Anything
- **Tracking**: IoU-based Hungarian assignment
- **Video**: FFmpeg + OpenCV
- **Backend**: FastAPI + Pydantic
- **Frontend**: Vanilla HTML/CSS/JS

## Project Structure

```
ToThinkVision/
├── app/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Configuration
│   ├── schemas.py           # Unified JSON schema (Pydantic)
│   ├── pipeline.py          # Main pipeline orchestration
│   ├── preprocessor.py      # Image/video preprocessing
│   ├── models/
│   │   ├── segmentor.py     # SAM segmentation
│   │   ├── detector.py      # Grounding DINO detection
│   │   ├── ocr_engine.py    # PaddleOCR
│   │   ├── depth_estimator.py  # Depth Anything
│   │   └── tracker.py       # Cross-frame object tracker
│   ├── exporters/
│   │   ├── base.py          # Exporter base class
│   │   ├── ui_exporter.py   # UI → Figma/HTML/JSON
│   │   ├── game_exporter.py # Game → Unity/UE
│   │   ├── video_exporter.py # Video → AE/Trajectory
│   │   └── embodied_exporter.py # Embodied → Robot/Pose
│   └── utils/
│       ├── color.py         # Color extraction
│       ├── geometry.py      # Spatial relations
│       └── io.py            # File I/O
├── static/
│   ├── index.html           # Frontend UI
│   └── style.css
├── tests/
│   ├── test_schemas.py
│   ├── test_exporters.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.sh
└── README.md
```
