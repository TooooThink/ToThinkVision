"""FastAPI entry point — serves frontend and handles upload/process/download."""

from __future__ import annotations

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
from app.exporters.embodied_exporter import EmbodiedExporter
from app.exporters.game_exporter import GameExporter
from app.exporters.ui_exporter import UIExporter
from app.exporters.video_exporter import VideoExporter
from app.pipeline import process_file
from app.schemas import ExportFormat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ToThinkVision — Universal Vision Structuring Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = settings.base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Export format → exporter mapping
EXPORT_FORMAT_MAP: dict[str, type] = {
    ExportFormat.FIGMA_JSON: lambda: UIExporter(ExportFormat.FIGMA_JSON),
    ExportFormat.HTML_CSS: lambda: UIExporter(ExportFormat.HTML_CSS),
    ExportFormat.UI_JSON: lambda: UIExporter(ExportFormat.UI_JSON),
    ExportFormat.UNITY_JSON: lambda: GameExporter(ExportFormat.UNITY_JSON),
    ExportFormat.UE_JSON: lambda: GameExporter(ExportFormat.UE_JSON),
    ExportFormat.COLLISION_JSON: lambda: GameExporter(ExportFormat.COLLISION_JSON),
    ExportFormat.AE_KEYFRAMES: lambda: VideoExporter(ExportFormat.AE_KEYFRAMES),
    ExportFormat.VIDEO_TRAJECTORY: lambda: VideoExporter(ExportFormat.VIDEO_TRAJECTORY),
    ExportFormat.PR_MARKERS: lambda: VideoExporter(ExportFormat.PR_MARKERS),
    ExportFormat.EMBODIED_JSON: lambda: EmbodiedExporter(ExportFormat.EMBODIED_JSON),
    ExportFormat.ROBOT_ACTION: lambda: EmbodiedExporter(ExportFormat.ROBOT_ACTION),
    ExportFormat.POSE_CSV: lambda: EmbodiedExporter(ExportFormat.POSE_CSV),
    ExportFormat.FULL_JSON: None,  # Special case: return raw structured output
}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>ToThinkVision</h1><p>Frontend not found. Check static/index.html</p>")


@app.get("/api/formats")
async def list_formats():
    """List available export formats grouped by category."""
    return {
        "ui": [
            {"id": ExportFormat.FIGMA_JSON.value, "label": "Figma JSON"},
            {"id": ExportFormat.HTML_CSS.value, "label": "HTML/CSS"},
            {"id": ExportFormat.UI_JSON.value, "label": "UI JSON"},
        ],
        "game": [
            {"id": ExportFormat.UNITY_JSON.value, "label": "Unity JSON"},
            {"id": ExportFormat.UE_JSON.value, "label": "UE JSON"},
            {"id": ExportFormat.COLLISION_JSON.value, "label": "Collision Boxes"},
        ],
        "video": [
            {"id": ExportFormat.AE_KEYFRAMES.value, "label": "AE Keyframes"},
            {"id": ExportFormat.VIDEO_TRAJECTORY.value, "label": "Trajectory CSV"},
            {"id": ExportFormat.PR_MARKERS.value, "label": "PR Markers"},
        ],
        "embodied": [
            {"id": ExportFormat.EMBODIED_JSON.value, "label": "Embodied JSON"},
            {"id": ExportFormat.ROBOT_ACTION.value, "label": "Robot Actions"},
            {"id": ExportFormat.POSE_CSV.value, "label": "Pose CSV"},
        ],
        "universal": [
            {"id": ExportFormat.FULL_JSON.value, "label": "Full Structured JSON"},
        ],
    }


@app.post("/api/process")
async def process_upload(
    file: UploadFile = File(...),
    export_format: str = Form("full_json"),
    mode: str = Form("general"),
):
    """Upload file, process it, and return the exported result."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file
    upload_id = uuid.uuid4().hex[:8]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ttv_{upload_id}_"))
    save_path = temp_dir / file.filename

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run pipeline
        structured_data = process_file(save_path, mode=mode)

        # Export
        if export_format == ExportFormat.FULL_JSON.value:
            # Return raw structured JSON
            out_path = settings.output_dir / f"{Path(file.filename).stem}_structured.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(structured_data.model_dump_json(indent=2))
            mime_type = "application/json"
        else:
            exporter_cls = EXPORT_FORMAT_MAP.get(export_format)
            if exporter_cls is None:
                raise HTTPException(status_code=400, detail=f"Unknown export format: {export_format}")
            exporter = exporter_cls()
            out_path = exporter.export(structured_data)
            mime_type = exporter.mime_type

        return {
            "status": "success",
            "source": file.filename,
            "source_type": structured_data.source_type,
            "objects_found": len(structured_data.objects),
            "frames_processed": structured_data.frame_count,
            "processing_time": structured_data.processing_time_seconds,
            "output_file": out_path.name,
            "output_path": str(out_path),
            "mime_type": mime_type,
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
    """Download a processed output file."""
    file_path = settings.output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(file_path), filename=filename)


@app.get("/api/health")
async def health():
    return {"status": "ok", "mock_mode": settings.mock_mode, "version": "1.0.0"}
