"""Grounding DINO wrapper for open-vocabulary object detection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Default detection prompts for different modes
DETECTION_PROMPTS = {
    "ui": "button . text . input . icon . image . navigation . card . slider . toggle",
    "game": "character . NPC . item . weapon . door . wall . floor . prop . terrain . effect",
    "video": "person . object . text . vehicle . animal . building",
    "embodied": "table . chair . tool . object . obstacle . target . surface",
    "general": "person . car . animal . object . text . building . plant . furniture",
}

_detector = None


def _get_detector():
    """Lazy-load Grounding DINO detector."""
    global _detector
    if _detector is not None:
        return _detector
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        model_path = Path(settings.model_cache_dir) / "groundingdino_swint_ogc.pth"
        config_path = Path(settings.model_cache_dir) / "GroundingDINO_SwinT_OGC.py"
        if not model_path.exists():
            raise RuntimeError(f"Grounding DINO weights not found at {model_path}")
        model = load_model(str(config_path), str(model_path))
        model.to(device=settings.device)
        _detector = model
        return model
    except ImportError:
        raise RuntimeError("groundingdino-py not installed. Install with: pip install groundingdino-py")


def detect_objects(img: np.ndarray, mode: str = "general") -> list[dict]:
    """Run open-vocabulary detection. Returns list of {bbox, label, confidence}."""
    detector = _get_detector()

    from groundingdino.util.inference import load_image, predict
    prompt = DETECTION_PROMPTS.get(mode, DETECTION_PROMPTS["general"])
    image_transformed, _ = load_image(img)
    boxes, logits, phrases = predict(
        model=detector,
        image=image_transformed,
        caption=prompt,
        box_threshold=settings.detection_threshold,
        text_threshold=0.25,
    )
    h, w = img.shape[:2]
    detections = []
    for box, score, label in zip(boxes, logits, phrases):
        cx, cy, bw, bh = box.tolist()
        x = (cx - bw / 2) * w
        y = (cy - bh / 2) * h
        detections.append({
            "bbox": [float(x), float(y), float(bw * w), float(bh * h)],
            "label": label,
            "confidence": float(score),
        })
    return detections

