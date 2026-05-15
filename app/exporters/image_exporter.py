"""Export visual outputs from pipeline results: crops, masks, depth overlays, detection boxes."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import settings

logger = logging.getLogger(__name__)

# Color palette for detection overlay
DETECTION_COLORS = [
    (0, 255, 128), (255, 128, 0), (128, 0, 255), (255, 0, 128),
    (0, 128, 255), (128, 255, 0), (255, 64, 64), (64, 255, 64),
    (64, 64, 255), (255, 255, 64), (64, 255, 255), (255, 192, 0),
]


def _ensure_output_dir(name: str) -> Path:
    """Ensure an output subdirectory exists."""
    stem = Path(name).stem
    d = settings.output_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pil_save_png(path: Path, pil_img: Image.Image):
    """Save PIL image as PNG."""
    pil_img.save(str(path), format="PNG")


# ─── Export crop images ──────────────────────────────────────

def export_crop_image(img: np.ndarray, bbox: list[float], output_dir: Path, obj_id: str) -> Path:
    """Crop object from image and save as PNG. Returns path."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_img, w_img = img.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    if x2 <= x or y2 <= y:
        return None
    crop = img[y:y2, x:x2]
    pil = Image.fromarray(crop)
    path = output_dir / f"{obj_id}_crop.png"
    _pil_save_png(path, pil)
    return path


def export_mask_image(mask: np.ndarray, output_dir: Path, obj_id: str) -> Path:
    """Save binary mask as PNG. Returns path."""
    if mask is None:
        return None
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    path = output_dir / f"{obj_id}_mask.png"
    _pil_save_png(path, mask_img)
    return path


def export_mask_with_alpha(img: np.ndarray, mask: np.ndarray, output_dir: Path, obj_id: str) -> Path:
    """Crop object and apply mask as alpha channel. Returns PNG path with transparent background."""
    x, y, w, h = int(img.shape[1]), int(img.shape[0]), int(img.shape[1]), int(img.shape[0])
    pil = Image.fromarray(img)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    # Resize mask to match crop if needed
    if pil.size != mask_pil.size:
        mask_pil = mask_pil.resize(pil.size, Image.NEAREST)
    pil.putalpha(mask_pil)
    path = output_dir / f"{obj_id}_masked.png"
    _pil_save_png(path, pil)
    return path


# ─── Export detection overlay ────────────────────────────────

def export_detection_overlay(img: np.ndarray, detections: list[dict], output_dir: Path, source_name: str) -> Path:
    """Draw bounding boxes and labels on the image. Returns path."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Try to get a font, fallback to default
    font = None
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        color = DETECTION_COLORS[i % len(DETECTION_COLORS)]

        # Draw bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # Draw label background
        text = f"{label} {conf:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - text_h - 4, x + text_w + 4, y], fill=color)
        draw.text((x + 2, y - text_h - 2), text, fill=(0, 0, 0), font=font)

    path = output_dir / f"{Path(source_name).stem}_detection.png"
    _pil_save_png(path, pil)
    return path


# ─── Export depth visualization ──────────────────────────────

def export_depth_visualization(depth_map: np.ndarray, output_dir: Path, source_name: str) -> Path:
    """Convert depth map to colored visualization. Returns path."""
    if depth_map is None or depth_map.size == 0:
        return None
    # Normalize to 0-255
    d = depth_map.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min) * 255
    else:
        d = np.zeros_like(d)
    d_u8 = d.astype(np.uint8)
    # Apply colormap (JET-like)
    colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(colored)
    path = output_dir / f"{Path(source_name).stem}_depth.png"
    _pil_save_png(path, pil)
    return path


# ─── Export mask overlay (mask drawn on original image) ─────

def export_mask_overlay(img: np.ndarray, mask: np.ndarray, bbox: list[float],
                        color: tuple[int, int, int], output_dir: Path, obj_id: str) -> Path:
    """Draw mask overlay on the original image region."""
    h, w = img.shape[:2]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = color
    overlay = img.copy()
    overlay[mask > 0] = cv2.addWeighted(overlay[mask > 0], 0.5, mask_rgb[mask > 0], 0.5, 0)
    pil = Image.fromarray(overlay)
    path = output_dir / f"{obj_id}_mask_overlay.png"
    _pil_save_png(path, pil)
    return path


# ─── Export point cloud preview ──────────────────────────────

def export_point_cloud_preview(points: np.ndarray, colors: np.ndarray,
                               output_dir: Path, source_name: str,
                               max_points: int = 50000) -> Path:
    """Create a simple orthographic preview of point cloud."""
    if points is None or len(points) == 0:
        return None

    pts = np.array(points)[:max_points]
    clr = np.array(colors)[:max_points] if colors is not None else None

    # Normalize to 2D view (top-down: XZ plane)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    img_size = 800
    scale = min(img_size / max(x_max - x_min, 1e-6), img_size / max(z_max - z_min, 1e-6))

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cx = img_size // 2
    cy = img_size // 2

    for i in range(len(pts)):
        px = int(cx + (pts[i, 0] - (x_min + x_max) / 2) * scale)
        py = int(cy - (pts[i, 2] - (z_min + z_max) / 2) * scale)
        if 0 <= px < img_size and 0 <= py < img_size:
            c = tuple(clr[i].tolist()) if clr is not None else (255, 255, 255)
            canvas[py, px] = c

    # Dilate to make points visible
    canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)

    pil = Image.fromarray(canvas)
    path = output_dir / f"{Path(source_name).stem}_pointcloud.png"
    _pil_save_png(path, pil)
    return path
