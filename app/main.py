"""FastAPI entry point — v2 with model selection and multi-format export."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.exporters.ae_project_exporter import AEProjectExporter
from app.exporters.embodied_exporter import EmbodiedExporter
from app.exporters.game_exporter import GameExporter
from app.exporters.gltf_exporter import GltfExporter
from app.exporters.manifest import build_manifest_for_export, ExportManifest
from app.exporters.obj_exporter import ObjExporter
from app.exporters.psd_exporter import PSDExporter
from app.exporters.splat_exporter import SplatExporter
from app.exporters.ui_exporter import UIExporter
from app.exporters.video_exporter import VideoExporter
from app.pipeline import process_file
from app.schemas import ExportFormat, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ToThinkVision v2 — Universal Vision Structuring Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = settings.base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Export format → exporter mapping
EXPORT_FORMAT_MAP: dict[str, type] = {
    # UI
    ExportFormat.FIGMA_JSON: lambda: UIExporter(ExportFormat.FIGMA_JSON),
    ExportFormat.HTML_CSS: lambda: UIExporter(ExportFormat.HTML_CSS),
    ExportFormat.UI_JSON: lambda: UIExporter(ExportFormat.UI_JSON),
    # Game 3D
    ExportFormat.UNITY_SPLAT: lambda: SplatExporter(ExportFormat.UNITY_SPLAT),
    ExportFormat.UE_SPLAT: lambda: SplatExporter(ExportFormat.UE_SPLAT),
    ExportFormat.GLTF: lambda: GltfExporter(),
    ExportFormat.OBJ_3D: lambda: ObjExporter(),
    ExportFormat.UNITY_JSON: lambda: GameExporter(ExportFormat.UNITY_JSON),
    ExportFormat.UE_JSON: lambda: GameExporter(ExportFormat.UE_JSON),
    ExportFormat.COLLISION_JSON: lambda: GameExporter(ExportFormat.COLLISION_JSON),
    # Video
    ExportFormat.AE_KEYFRAMES: lambda: VideoExporter(ExportFormat.AE_KEYFRAMES),
    ExportFormat.VIDEO_TRAJECTORY: lambda: VideoExporter(ExportFormat.VIDEO_TRAJECTORY),
    ExportFormat.PR_MARKERS: lambda: VideoExporter(ExportFormat.PR_MARKERS),
    ExportFormat.AE_PROJECT: lambda: AEProjectExporter(ExportFormat.AE_PROJECT),
    # PSD
    ExportFormat.PSD_STATIC: lambda: PSDExporter(ExportFormat.PSD_STATIC),
    ExportFormat.PSD_ANIMATED: lambda: PSDExporter(ExportFormat.PSD_ANIMATED),
    # Embodied
    ExportFormat.EMBODIED_JSON: lambda: EmbodiedExporter(ExportFormat.EMBODIED_JSON),
    ExportFormat.ROBOT_ACTION: lambda: EmbodiedExporter(ExportFormat.ROBOT_ACTION),
    ExportFormat.POSE_CSV: lambda: EmbodiedExporter(ExportFormat.POSE_CSV),
    # Universal
    ExportFormat.FULL_JSON: None,
    # 4D Scene (generated in-pipeline, not via exporter classes)
    ExportFormat.ANIMATED_GLTF: None,
    ExportFormat.USD_SCENE: None,
    ExportFormat.BLENDER_SCENE: None,
    ExportFormat.SCENE_GRAPH_JSON: None,
}


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>ToThinkVision v2</h1><p>Frontend not found.</p>")


@app.get("/api/formats")
async def list_formats():
    return {
        "models": {
            "sam3": {"label": "SAM 3", "description": "Detection + Segmentation + Tracking", "vram_gb": "12-24"},
            "omniparser": {"label": "OmniParser v2", "description": "UI Element Detection", "vram_gb": "8-12"},
            "grounding_dino": {"label": "Grounding DINO 1.6", "description": "Open-Vocabulary Detection", "vram_gb": "4-8"},
            "strongsort": {"label": "StrongSORT", "description": "Multi-Object Tracking", "vram_gb": "CPU"},
            "depth_pro": {"label": "Depth Pro", "description": "Metric Depth Estimation", "vram_gb": "4-8"},
            "mast3r": {"label": "MASt3R / VGGT", "description": "3D Point Cloud + Camera Pose", "vram_gb": "24-48"},
            "gaussian_splatting": {"label": "3D Gaussian Splatting", "description": "Photorealistic 3D Scene", "vram_gb": "24"},
            "cotracker3": {"label": "CoTracker3", "description": "Dense point tracking (265x265) for accurate trajectories", "vram_gb": "8-12"},
            "objectgs": {"label": "ObjectGS", "description": "Per-object 3D Gaussian Splatting (ICCV 2025)", "vram_gb": "24"},
            "spann3r": {"label": "Spann3R", "description": "3D reconstruction with spatial memory (3DV 2025)", "vram_gb": "24-48"},
            "shape_of_motion": {"label": "Shape of Motion", "description": "End-to-end 4D reconstruction (ICCV 2025)", "vram_gb": "24"},
            "trajectory_4d": {"label": "4D Trajectory", "description": "Per-object 6DoF motion extraction (ICP + PCA)", "vram_gb": "CPU"},
            "gaussian_splatting_4d": {"label": "4D Gaussian Splatting", "description": "Temporal scene rendering (HexPlane)", "vram_gb": "24-48"},
            "scene_graph_4d": {"label": "4D Scene Graph", "description": "Dynamic object relationships over time", "vram_gb": "CPU"},
        },
        "ui": [
            {"id": ExportFormat.FIGMA_JSON.value, "label": "Figma JSON"},
            {"id": ExportFormat.HTML_CSS.value, "label": "HTML/CSS"},
            {"id": ExportFormat.UI_JSON.value, "label": "UI JSON"},
        ],
        "game_3d": [
            {"id": ExportFormat.UNITY_SPLAT.value, "label": "Unity 3D Splat"},
            {"id": ExportFormat.UE_SPLAT.value, "label": "UE5 3D Splat"},
            {"id": ExportFormat.GLTF.value, "label": "glTF 3D"},
            {"id": ExportFormat.OBJ_3D.value, "label": "OBJ 3D"},
            {"id": ExportFormat.UNITY_JSON.value, "label": "Unity JSON (2D)"},
            {"id": ExportFormat.UE_JSON.value, "label": "UE JSON (2D)"},
        ],
        "video": [
            {"id": ExportFormat.AE_KEYFRAMES.value, "label": "AE Keyframes"},
            {"id": ExportFormat.AE_PROJECT.value, "label": "After Effects Project"},
            {"id": ExportFormat.VIDEO_TRAJECTORY.value, "label": "Trajectory CSV"},
            {"id": ExportFormat.PR_MARKERS.value, "label": "PR Markers"},
        ],
        "psd": [
            {"id": ExportFormat.PSD_STATIC.value, "label": "PSD (Static Layers)"},
            {"id": ExportFormat.PSD_ANIMATED.value, "label": "PSD (Animated Groups)"},
        ],
        "embodied": [
            {"id": ExportFormat.EMBODIED_JSON.value, "label": "Embodied JSON"},
            {"id": ExportFormat.ROBOT_ACTION.value, "label": "Robot Actions"},
            {"id": ExportFormat.POSE_CSV.value, "label": "Pose CSV"},
        ],
        "universal": [
            {"id": ExportFormat.FULL_JSON.value, "label": "Full Structured JSON"},
        ],
        "scene_4d": [
            {"id": ExportFormat.ANIMATED_GLTF.value, "label": "Animated glTF (Unity/Blender)"},
            {"id": ExportFormat.USD_SCENE.value, "label": "USD Scene (Unreal/Omniverse)"},
            {"id": ExportFormat.BLENDER_SCENE.value, "label": "Blender Scene"},
            {"id": ExportFormat.SCENE_GRAPH_JSON.value, "label": "4D Scene Graph JSON"},
        ],
    }


@app.post("/api/process")
async def process_upload(
    file: UploadFile = File(...),
    # Single format (backwards compatible) or multiple formats
    export_format: str = Form("full_json"),
    export_formats: str = Form(""),  # JSON list of format IDs, e.g. '["unity_splat", "psd_static"]'
    mode: str = Form("general"),
    # Model toggles (default: all enabled)
    enable_sam3: bool = Form(True),
    enable_omniparser: bool = Form(True),
    enable_grounding_dino: bool = Form(True),
    enable_strongsort: bool = Form(True),
    enable_depth_pro: bool = Form(True),
    enable_mast3r: bool = Form(True),
    enable_gaussian_splatting: bool = Form(False),
    # Advanced 3D/4D models
    enable_cotracker3: bool = Form(True),
    enable_objectgs: bool = Form(False),
    enable_spann3r: bool = Form(False),
    enable_shape_of_motion: bool = Form(False),
    # 4D Scene options
    enable_4d_trajectory: bool = Form(True),
    enable_4dgs: bool = Form(False),
    enable_scene_graph: bool = Form(True),
    enable_animated_export: bool = Form(True),
    is_world_model_video: bool = Form(False),
    # Video options
    max_video_frames: int = Form(300),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Build pipeline config from form fields
    config = PipelineConfig(
        mode=mode,
        enable_sam3=enable_sam3,
        enable_omniparser=enable_omniparser,
        enable_grounding_dino=enable_grounding_dino,
        enable_strongsort=enable_strongsort,
        enable_depth_pro=enable_depth_pro,
        enable_mast3r=enable_mast3r,
        enable_gaussian_splatting=enable_gaussian_splatting,
        enable_cotracker3=enable_cotracker3,
        enable_objectgs=enable_objectgs,
        enable_spann3r=enable_spann3r,
        enable_shape_of_motion=enable_shape_of_motion,
        enable_4d_trajectory=enable_4d_trajectory,
        enable_4dgs=enable_4dgs,
        enable_scene_graph=enable_scene_graph,
        enable_animated_export=enable_animated_export,
        is_world_model_video=is_world_model_video,
        max_video_frames=max_video_frames,
    )

    # Determine export formats: use export_formats list if provided, else fall back to single
    formats_to_export: list[str] = []
    if export_formats:
        try:
            formats_to_export = json.loads(export_formats)
        except json.JSONDecodeError:
            formats_to_export = [export_format]
    else:
        formats_to_export = [export_format]

    upload_id = uuid.uuid4().hex[:8]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ttv_{upload_id}_"))
    save_path = temp_dir / file.filename

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        structured_data = process_file(save_path, mode=mode, config=config)

        # Export to all requested formats
        export_results = []
        for fmt in formats_to_export:
            if fmt == ExportFormat.FULL_JSON.value:
                out_path = settings.output_dir / f"{Path(file.filename).stem}_{fmt}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(structured_data.model_dump_json(indent=2))
                mime_type = "application/json"
            else:
                exporter_cls = EXPORT_FORMAT_MAP.get(fmt)
                if exporter_cls is None:
                    raise HTTPException(status_code=400, detail=f"Unknown export format: {fmt}")
                exporter = exporter_cls()
                out_path = exporter.export(structured_data)
                mime_type = exporter.mime_type

            export_results.append({
                "format": fmt,
                "output_file": out_path.name,
                "output_path": str(out_path),
                "mime_type": mime_type,
            })

        # Build export manifest (lists all files + usage instructions)
        manifest = ExportManifest(file.filename)
        for er in export_results:
            manifest.add(er["output_path"], er["format"], er["format"])

        # Add auxiliary image files
        if structured_data.detection_overlay_png_path:
            manifest.add(structured_data.detection_overlay_png_path, "Detection visualization overlay", "Reference")
        if structured_data.depth_map_png_path:
            manifest.add(structured_data.depth_map_png_path, "Depth map visualization", "Reference")
        if structured_data.point_cloud_preview_png_path:
            manifest.add(structured_data.point_cloud_preview_png_path, "Point cloud preview", "Reference")

        # Add per-object image files
        for obj in structured_data.objects:
            if obj.crop_png_path:
                manifest.add(obj.crop_png_path, f"Object crop: {obj.label_custom or obj.label}", "PS/AE/Unity")
            if obj.mask_png_path:
                manifest.add(obj.mask_png_path, f"Object mask: {obj.label_custom or obj.label}", "PS/AE")

        manifest_path = manifest.generate()

        # Save 3D file paths
        ply_path = None
        splat_path = None
        if structured_data.ply_file_path:
            ply_path = structured_data.ply_file_path
        if structured_data.splat_file_path:
            splat_path = structured_data.splat_file_path

        # Add 3D files to manifest
        if structured_data.ply_file_path:
            manifest.add(structured_data.ply_file_path, "Point cloud PLY file (from MASt3R or 3DGS)", "Blender/Unity/All")
        if structured_data.splat_file_path:
            manifest.add(structured_data.splat_file_path, "3D Gaussian Splat binary (UnityGaussianSplatting)", "Unity/UE5")
        if structured_data.scene_mesh_path:
            manifest.add(structured_data.scene_mesh_path, "Combined scene mesh with per-object 3D models", "Blender/Unity/UE5")

        return {
            "status": "success",
            "source": file.filename,
            "source_type": structured_data.source_type,
            "objects_found": len(structured_data.objects),
            "frames_processed": structured_data.frame_count,
            "processing_time": structured_data.processing_time_seconds,
            "point_cloud_points": len(structured_data.point_cloud.points) if structured_data.point_cloud else 0,
            "camera_poses": len(structured_data.camera_poses),
            "gaussian_splats": len(structured_data.gaussian_splats.means) if structured_data.gaussian_splats else 0,
            "models_used": structured_data.model_versions,
            "exports": export_results,
            "ply_file_path": structured_data.ply_file_path,
            "splat_file_path": structured_data.splat_file_path,
            "scene_mesh_path": structured_data.scene_mesh_path,
            "objects_with_mesh": sum(1 for obj in structured_data.objects if obj.mesh_3d),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Processing failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = settings.output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(file_path), filename=filename)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "mock_mode": settings.mock_mode,
        "models": {
            "segmentation": settings.segmentation_model,
            "detection": settings.detection_model,
            "tracking": settings.tracking_model,
            "depth": settings.depth_model,
            "reconstruction": settings.reconstruction_model,
            "3dgs": settings.gaussian_splatting,
        },
    }
