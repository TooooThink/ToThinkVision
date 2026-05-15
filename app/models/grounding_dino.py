"""Grounding DINO 1.6 — open-vocabulary object detection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

DETECTION_PROMPTS = {
    "ui": "button . text . input field . icon . image . navigation bar . card . slider . toggle",
    "game": "character . NPC . item . weapon . door . wall . floor . prop . terrain . effect",
    "video": "person . object . text . vehicle . animal . building . screen",
    "embodied": "table . chair . tool . object . obstacle . target . surface",
    "general": "person . car . animal . object . text . building . plant . furniture . device",
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
        x = rng.randint(0, max(10, w - bw - 10))
        y = rng.randint(0, max(10, h - bh - 10))
        detections.append({
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "label": labels[i % len(labels)],
            "confidence": 0.85 - i * 0.07,
        })
    return detections


class GroundingDINO:
    """Wrapper for Grounding DINO 1.6 open-vocabulary detection."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load Grounding DINO 1.6."""
        if settings.mock_mode:
            logger.info("Grounding DINO: using mock mode")
            return

        cache = Path(settings.model_cache_dir)
        try:
            # Try Grounding DINO 1.6
            model_path = cache / "groundingdino_1.6.pth"
            config_path = cache / "GroundingDINO_1.6.py"
            if not model_path.exists():
                logger.warning("Grounding DINO 1.6 weights not found, falling back to mock")
                return

            from groundingdino.util.inference import load_model
            self.model = load_model(str(config_path), str(model_path))
            self.model.to(device=self.device)
            logger.info("Grounding DINO 1.6 loaded")
        except ImportError:
            logger.warning("groundingdino-py not installed, falling back to mock")
        except Exception as e:
            logger.warning(f"Grounding DINO load failed: {e}")

    def detect(self, img: np.ndarray, mode: str = "general",
               custom_prompt: str | None = None) -> list[dict]:
        """Run open-vocabulary detection.

        Args:
            img: (H, W, 3) RGB image
            mode: detection mode preset
            custom_prompt: override default text prompt

        Returns:
            list of {bbox, label, confidence}
        """
        if self.model is None:
            return _get_mock_detections(img, mode)

        try:
            from groundingdino.util.inference import load_image, predict

            prompt = custom_prompt or DETECTION_PROMPTS.get(mode, DETECTION_PROMPTS["general"])
            image_transformed, _ = load_image(img)
            boxes, logits, phrases = predict(
                model=self.model,
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
            logger.error(f"Grounding DINO detection failed: {e}, using mock fallback")
            return _get_mock_detections(img, mode)


def get_detector() -> GroundingDINO:
    """Get or create detector instance."""
    global _detector
    if _detector is None:
        _detector = GroundingDINO()
    return _detector


def detect_objects(img: np.ndarray, mode: str = "general") -> list[dict]:
    """Convenience function for detection."""
    return get_detector().detect(img, mode)
