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


def _get_mock_detections(img: np.ndarray, mode: str = "general") -> list[dict]:
    """Generate mock detections for testing."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(99)
    prompts = DETECTION_PROMPTS.get(mode, DETECTION_PROMPTS["general"])
    labels = [l.strip() for l in prompts.split(" . ") if l.strip()]

    detections = []
    num_det = min(len(labels), 6)
    for i in range(num_det):
        max_bw = max(50, w // 3)
        max_bh = max(50, h // 3)
        bw = rng.randint(20, min(180, max_bw))
        bh = rng.randint(20, min(180, max_bh))
        x = rng.randint(0, w - bw - 10)
        y = rng.randint(0, h - bh - 10)
        detections.append({
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "label": labels[i % len(labels)],
            "confidence": 0.8 - i * 0.08,
        })
    return detections


def _get_detector():
    """Lazy-load Grounding DINO detector."""
    global _detector
    if _detector is not None:
        return _detector
    if settings.mock_mode:
        logger.info("Using mock detector (MOCK_MODE=true)")
        return "mock"
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        model_path = Path(settings.model_cache_dir) / "groundingdino_swint_ogc.pth"
        config_path = Path(settings.model_cache_dir) / "GroundingDINO_SwinT_OGC.py"
        if not model_path.exists():
            logger.warning("Grounding DINO weights not found, falling back to mock mode")
            return "mock"
        model = load_model(str(config_path), str(model_path))
        model.to(device=settings.device)
        return model
    except ImportError:
        logger.warning("groundingdino-py not installed, falling back to mock mode")
        return "mock"


def detect_objects(img: np.ndarray, mode: str = "general") -> list[dict]:
    """Run open-vocabulary detection. Returns list of {bbox, label, confidence}."""
    detector = _get_detector()
    if detector == "mock":
        return _get_mock_detections(img, mode)

    try:
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
    except Exception as e:
        logger.error(f"Detection failed: {e}, using mock fallback")
        return _get_mock_detections(img, mode)
