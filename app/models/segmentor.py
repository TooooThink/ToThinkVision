"""Segment Anything Model wrapper for instance segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_segmentor = None


def _get_segmentor():
    """Lazy-load SAM segmentor."""
    global _segmentor
    if _segmentor is not None:
        return _segmentor
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        model_type = "vit_h"
        checkpoint = Path(settings.model_cache_dir) / "sam_vit_h_4b8939.pth"
        if not checkpoint.exists():
            raise RuntimeError(f"SAM weights not found at {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=settings.device)
        predictor = SamPredictor(sam)
        _segmentor = predictor
        return predictor
    except ImportError:
        raise RuntimeError("segment-anything not installed. Install with: pip install segment-anything")


def segment_image(img: np.ndarray) -> list[dict]:
    """Run segmentation on a single image. Returns list of {mask, bbox, confidence}."""
    segmentor = _get_segmentor()

    segmentor.set_image(img)
    masks, scores, logits = segmentor.predict(
        multimask_output=True,
    )
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < settings.segmentation_threshold:
            continue
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        results.append({
            "mask": mask.astype(np.uint8),
            "bbox": bbox,
            "confidence": float(score),
        })
    return results


def get_contour_from_mask(mask: np.ndarray) -> list[dict[str, float]]:
    """Extract contour points from a binary mask."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    # Downsample contour for compactness
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
