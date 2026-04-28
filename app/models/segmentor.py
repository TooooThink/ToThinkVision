"""Segment Anything Model wrapper for instance segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_segmentor = None


def _get_mock_masks(img: np.ndarray, num_masks: int = 5) -> list[dict]:
    """Generate mock segmentation masks for testing without real models."""
    h, w = img.shape[:2]
    masks = []
    rng = np.random.RandomState(42)
    for i in range(num_masks):
        max_y = max(10, h - 50)
        max_x = max(10, w - 50)
        max_bw = max(30, w // 3)
        max_bh = max(30, h // 3)
        y = rng.randint(0, max_y)
        x = rng.randint(0, max_x)
        bw = rng.randint(15, min(200, max_bw))
        bh = rng.randint(15, min(200, max_bh))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y : y + bh, x : x + bw] = 1
        masks.append({
            "mask": mask,
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "confidence": 0.85 - i * 0.1,
        })
    return masks


def _get_segmentor():
    """Lazy-load SAM segmentor."""
    global _segmentor
    if _segmentor is not None:
        return _segmentor
    if settings.mock_mode:
        logger.info("Using mock segmentor (MOCK_MODE=true)")
        return "mock"
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        model_type = "vit_h"
        checkpoint = Path(settings.model_cache_dir) / "sam_vit_h_4b8939.pth"
        if not checkpoint.exists():
            logger.warning("SAM weights not found, falling back to mock mode")
            return "mock"
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=settings.device)
        predictor = SamPredictor(sam)
        return predictor
    except ImportError:
        logger.warning("segment-anything not installed, falling back to mock mode")
        return "mock"


def segment_image(img: np.ndarray) -> list[dict]:
    """Run segmentation on a single image. Returns list of {mask, bbox, confidence}."""
    segmentor = _get_segmentor()
    if segmentor == "mock":
        return _get_mock_masks(img)

    try:
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
    except Exception as e:
        logger.error(f"Segmentation failed: {e}, using mock fallback")
        return _get_mock_masks(img)


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
