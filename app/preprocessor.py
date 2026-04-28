"""Input layer: image/video preprocessing and frame extraction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def preprocess_image(path: Path, max_size: int = 1920) -> tuple[np.ndarray, dict]:
    """Load and resize image if too large. Returns (image, info_dict)."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 1.0
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    info = {
        "original_width": w,
        "original_height": h,
        "width": img.shape[1],
        "height": img.shape[0],
        "scale": scale,
    }
    return img, info


def extract_frames(video_path: Path, output_dir: Path) -> tuple[list[Path], dict]:
    """Extract frames from video using FFmpeg. Returns (frame_paths, metadata)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    # Determine frame interval
    if settings.frame_sample_interval > 0:
        frame_interval = max(1, int(fps * settings.frame_sample_interval))
    else:
        max_frames = settings.max_video_frames
        frame_interval = max(1, total_frames // max_frames) if total_frames > max_frames else 1

    # Extract frames via FFmpeg (faster than OpenCV for batch)
    pattern = str(output_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    # Use FFmpeg to extract only sampled frames
    if frame_interval > 1:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{frame_interval}))",
            "-vsync", "vfr",
            "-q:v", "2",
            pattern,
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except FileNotFoundError:
        # FFmpeg not installed, fallback to OpenCV
        return _extract_frames_opencv(video_path, output_dir, frame_interval)

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "extracted_frames": len(frame_paths),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "frame_interval": frame_interval,
    }
    return frame_paths, metadata


def _extract_frames_opencv(
    video_path: Path, output_dir: Path, frame_interval: int
) -> tuple[list[Path], dict]:
    """Fallback frame extraction using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    frame_paths: list[Path] = []
    idx = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            idx += 1
        frame_idx += 1
    cap.release()

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "extracted_frames": len(frame_paths),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "frame_interval": frame_interval,
    }
    return frame_paths, metadata


def cleanup_frames(frame_dir: Path):
    """Remove extracted frame directory."""
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
