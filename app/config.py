"""Configuration management."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, configurable via environment variables."""

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
    frame_sample_interval: float = 0.0  # 0 = all frames; >0 = sample every N seconds
    device: str = "cuda"
    mock_mode: bool = os.environ.get("MOCK_MODE", "false").lower() == "true"

    # Model thresholds
    detection_threshold: float = 0.35
    segmentation_threshold: float = 0.5
    ocr_threshold: float = 0.4

    # UI export defaults
    html_minify: bool = True
    figma_version: str = "v2"

    model_config = {"env_prefix": "TTV_"}


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
