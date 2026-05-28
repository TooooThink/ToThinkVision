"""Depth Anything wrapper for monocular depth estimation."""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_depth_model = None


def _get_depth_model():
    """Lazy-load Depth Anything model."""
    global _depth_model
    if _depth_model is not None:
        return _depth_model
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = "LiheYoung/depth-anything-small-hf"
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=settings.model_cache_dir)
        model = AutoModelForDepthEstimation.from_pretrained(model_name, cache_dir=settings.model_cache_dir)
        model.to(device=settings.device)
        _depth_model = {"processor": processor, "model": model}
        return _depth_model
    except ImportError:
        raise RuntimeError("transformers not installed. Install with: pip install transformers")


def estimate_depth(img: np.ndarray) -> np.ndarray:
    """Estimate depth map. Returns normalized depth array (same shape as image, single channel)."""
    depth_model = _get_depth_model()

    import torch
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(img)
    inputs = depth_model["processor"](images=pil_img, return_tensors="pt").to(settings.device)
    with torch.no_grad():
        outputs = depth_model["model"](**inputs)
    predicted_depth = outputs.predicted_depth
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    # Normalize to 0-255
    output = (output - output.min()) / (output.max() - output.min()) * 255.0
    return output.astype(np.float32)


def get_depth_at_bbox(depth_map: np.ndarray, bbox: list[float]) -> float:
    """Get average depth value in bbox region. bbox = [x, y, w, h]."""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    h_map, w_map = depth_map.shape[:2]
    x = max(0, min(x, w_map - 1))
    y = max(0, min(y, h_map - 1))
    x2 = min(x + w, w_map)
    y2 = min(y + h, h_map)
    if x2 <= x or y2 <= y:
        return 0.0
    region = depth_map[y:y2, x:x2]
    return float(region.mean())


def estimate_3d_bbox(bbox: list[float], depth_map: np.ndarray) -> dict:
    """Estimate 3D bounding box from 2D bbox and depth map."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    depth_val = get_depth_at_bbox(depth_map, bbox)
    # Normalize depth to pseudo-3D coordinate
    z = depth_val / 255.0 * 10.0  # Map to 0-10 unit range
    return {"x": float(cx), "y": float(cy), "z": float(z)}
