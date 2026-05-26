"""GroundingDINO — open-vocabulary object detection (IDEA Research).

Uses transformers-based GroundingDINO with automatic weight download from HuggingFace.
Official GroundingDINO repo weights used as fallback if HF is unavailable.
"""

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
    """Wrapper for GroundingDINO open-vocabulary detection.

    Tries two backends (in order):
    1. HuggingFace transformers — auto-downloads weights
    2. Grounding DINO official repo — requires .py config + .pth weights
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._backend = None  # "huggingface", "official"
        self._init_model()

    def _init_model(self):
        """Load GroundingDINO detector."""
        if settings.mock_mode:
            logger.info("GroundingDINO: using mock mode")
            return

        # Try HuggingFace transformers — uses HF_HOME env var for cache
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self.model.to(self.device)
            self._backend = "huggingface"
            logger.info("GroundingDINO loaded from HuggingFace (%s)", self.model_id)
            return
        except ImportError:
            logger.info("transformers not available, trying official repo")
        except Exception as e:
            logger.info(f"HF GroundingDINO load failed: {e}, trying official repo")

        # Try official IDEA-Research GroundingDINO repo
        cache = Path(settings.model_cache_dir)
        try:
            # Look for weights in multiple locations
            candidates = [
                cache / "GroundingDINO" / "groundingdino_swint_ogc.pth",
                cache / "groundingdino_swint_ogc.pth",
            ]
            model_path = None
            for c in candidates:
                if c.exists():
                    model_path = c
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"GroundingDINO weights not found. Searched: {candidates}. "
                    "Download: https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/groundingdino_swint_ogc.pth"
                )

            config_path = cache / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            if not config_path.exists():
                config_path = cache / "GroundingDINO_SwinT_OGC.py"

            from groundingdino.util.inference import load_model
            self.model = load_model(str(config_path), str(model_path), device=self.device)
            self._backend = "official"
            logger.info("GroundingDINO loaded from official repo (%s)", model_path)
            return
        except ImportError:
            logger.info("groundingdino package not installed")
        except Exception as e:
            raise RuntimeError(f"GroundingDINO failed to load via official repo: {e}")

        raise RuntimeError(
            "GroundingDINO is required but could not be loaded. Neither HuggingFace "
            "transformers nor the official repo weights are available."
        )

    def detect(self, img: np.ndarray, mode: str = "general",
               custom_prompt: str | None = None) -> list[dict]:
        """Run open-vocabulary detection.

        Args:
            img: (H, W, 3) RGB image
            mode: detection mode preset
            custom_prompt: override default text prompt

        Returns:
            list of {bbox, label, confidence} — bbox as [x, y, w, h]
        """
        if self.model is None:
            raise RuntimeError("GroundingDINO model not loaded. Set MOCK_MODE=true to allow mock detection.")

        prompt = custom_prompt or DETECTION_PROMPTS.get(mode, DETECTION_PROMPTS["general"])

        try:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(img)
            h, w = img.shape[:2]

            if self._backend == "huggingface":
                return self._detect_hf(pil_img, prompt, h, w)
            else:
                return self._detect_official(img, prompt)
        except Exception as e:
            raise RuntimeError(f"GroundingDINO detection failed: {e}")

    def _detect_hf(self, pil_img, caption: str, h: int, w: int) -> list[dict]:
        """Detect using HuggingFace transformers API."""
        import torch

        inputs = self.processor(images=pil_img, text=caption, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=settings.detection_threshold,
            text_threshold=0.25,
            target_sizes=[(h, w)],
        )

        detections = []
        if results and results[0]:
            r = results[0]
            boxes = r["boxes"].cpu().numpy()  # [x1, y1, x2, y2]
            scores = r["scores"].cpu().numpy()
            labels = r["labels"]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "label": label,
                    "confidence": float(score),
                })

        return detections

    def _detect_official(self, img: np.ndarray, caption: str) -> list[dict]:
        """Detect using official IDEA-Research GroundingDINO API."""
        from groundingdino.util.inference import load_image, predict

        _, image_tensor = load_image(img)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=settings.detection_threshold,
            text_threshold=0.25,
            device=self.device,
        )

        h, w = img.shape[:2]
        detections = []
        for box, score, label in zip(boxes, logits, phrases):
            cx, cy, bw_norm, bh_norm = box.tolist()
            x = (cx - bw_norm / 2) * w
            y = (cy - bh_norm / 2) * h
            detections.append({
                "bbox": [float(x), float(y), float(bw_norm * w), float(bh_norm * h)],
                "label": label,
                "confidence": float(score),
            })

        return detections


def get_detector() -> GroundingDINO:
    """Get or create detector instance."""
    global _detector
    if _detector is None:
        _detector = GroundingDINO()
    return _detector


def detect_objects(img: np.ndarray, mode: str = "general") -> list[dict]:
    """Convenience function for detection."""
    return get_detector().detect(img, mode)

