"""Configuration — v2 all models enabled by default."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = Path(__file__).resolve().parent.parent / "uploads"
    output_dir: Path = Path(__file__).resolve().parent.parent / "outputs"
    model_cache_dir: str = os.environ.get("MODEL_CACHE_DIR", str(Path.home() / ".cache" / "tothinkvision"))

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 500

    # Processing
    max_video_frames: int = 300
    frame_sample_interval: float = 0.0
    device: str = "cuda"
    mock_mode: bool = os.environ.get("MOCK_MODE", "false").lower() == "true"

    # GPU
    gpu_memory_gb: int = int(os.environ.get("TTV_GPU_MEMORY_GB", "24"))
    batch_size: int = 4

    # Model selection (all enabled by default, mock fallback if weights missing)
    segmentation_model: str = "sam3"
    detection_model: str = "auto"  # omniparser for ui mode, grounding_dino for others
    tracking_model: str = "botsort"
    depth_model: str = "depth_pro"
    reconstruction_model: str = "vggt"
    gaussian_splatting: bool = os.environ.get("TTV_ENABLE_3DGS", "false").lower() == "true"

    # Thresholds
    detection_threshold: float = 0.35
    segmentation_threshold: float = 0.5
    ocr_threshold: float = 0.4

    # 3D
    mast3r_sample_interval: int = 5  # Run MASt3R every N frames

    # PSD
    psd_max_objects_per_frame: int = 50

    model_config = {"env_prefix": "TTV_"}


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
