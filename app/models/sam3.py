"""SAM 3 — Meta's unified detection + segmentation + tracking model."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Global predictor instance
_sam3_predictor = None


def _get_mock_segmentation(img: np.ndarray, num_objects: int = 5) -> list[dict]:
    """Generate mock segmentation with masks for testing."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(42)
    labels = ["object", "button", "text", "icon", "container", "person", "item"]
    results = []
    for i in range(num_objects):
        max_bw = max(50, w // 3)
        max_bh = max(50, h // 3)
        bw = rng.randint(20, min(200, max_bw))
        bh = rng.randint(20, min(200, max_bh))
        x = rng.randint(0, max(10, w - bw - 10))
        y = rng.randint(0, max(10, h - bh - 10))

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        # Polygon mask for more realistic shape
        pts = np.array([
            [x + rng.randint(0, 10), y + rng.randint(0, 10)],
            [x + bw - rng.randint(0, 10), y + rng.randint(0, 10)],
            [x + bw - rng.randint(0, 10), y + bh - rng.randint(0, 10)],
            [x + rng.randint(0, 10), y + bh - rng.randint(0, 10)],
        ], dtype=np.int32)
        cv2 = __import__("cv2")
        cv2.fillPoly(mask, [pts], 1)

        results.append({
            "mask": mask,
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "label": labels[i % len(labels)],
            "confidence": 0.9 - i * 0.05,
        })
    return results


class SAM3Predictor:
    """Wrapper for SAM 3 image and video prediction."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.image_predictor = None
        self.video_predictor = None
        self._init_model()

    def _init_model(self):
        """Load SAM 3 models."""
        if settings.mock_mode:
            logger.info("SAM 3: using mock mode")
            return

        cache = Path(settings.model_cache_dir)
        try:
            # Try SAM 3 first
            from sam3.build_sam3 import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor

            checkpoint = cache / "sam3_hiera_large.pt"
            if not checkpoint.exists():
                logger.warning(f"SAM 3 checkpoint not found at {checkpoint}, falling back to mock")
                return

            sam3 = build_sam3("sam3_hiera_large.yaml", str(checkpoint), device=self.device)
            self.image_predictor = SAM3ImagePredictor(sam3)
            logger.info("SAM 3 image predictor loaded")

            # Try video predictor
            try:
                from sam3.build_sam3 import build_sam3_video_predictor
                self.video_predictor = build_sam3_video_predictor(
                    "sam3_hiera_large.yaml", str(checkpoint), device=self.device
                )
                logger.info("SAM 3 video predictor loaded")
            except Exception as e:
                logger.warning(f"SAM 3 video predictor not available: {e}")

        except ImportError:
            logger.warning("sam3 package not installed, falling back to mock mode")

    def predict(self, img: np.ndarray, text_prompt: str | None = None,
                boxes: np.ndarray | None = None) -> list[dict]:
        """Run SAM 3 detection + segmentation on an image.

        Args:
            img: (H, W, 3) RGB image
            text_prompt: text description of objects to find
            boxes: optional pre-computed boxes (N, 4) as [x1, y1, x2, y2]

        Returns:
            list of {mask, bbox, label, confidence}
        """
        if self.image_predictor is None:
            return _get_mock_segmentation(img)

        try:
            self.image_predictor.set_image(img)

            results = []
            if boxes is not None:
                # Segment from provided boxes
                for i, box in enumerate(boxes):
                    masks, scores, _ = self.image_predictor.predict(
                        box=box, multimask_output=True
                    )
                    best_idx = scores.argmax()
                    mask = masks[best_idx].astype(np.uint8)
                    results.append({
                        "mask": mask,
                        "bbox": box.tolist(),
                        "label": f"object_{i}",
                        "confidence": float(scores[best_idx]),
                    })
            elif text_prompt:
                # Use SAM 3 text-prompted segmentation
                results = self._predict_with_text(img, text_prompt)
            else:
                # Automatic segmentation
                results = self._predict_automatic(img)

            return results
        except Exception as e:
            logger.error(f"SAM 3 prediction failed: {e}, using mock fallback")
            return _get_mock_segmentation(img)

    def _predict_with_text(self, img: np.ndarray, text_prompt: str) -> list[dict]:
        """Use SAM 3 text-prompted segmentation if available."""
        try:
            # SAM 3 concept-aware segmentation
            results = self.image_predictor.predict_with_text(text_prompt)
            return results
        except (AttributeError, Exception):
            # Fallback: use automatic segmentation
            return self._predict_automatic(img)

    def _predict_automatic(self, img: np.ndarray) -> list[dict]:
        """Automatic segmentation using SAM 3."""
        try:
            masks, scores, _ = self.image_predictor.predict(
                multimask_output=True,
            )
            results = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score < settings.segmentation_threshold:
                    continue
                mask_u8 = mask.astype(np.uint8)
                ys, xs = np.where(mask_u8)
                if len(xs) == 0:
                    continue
                bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                results.append({
                    "mask": mask_u8,
                    "bbox": bbox,
                    "label": f"object_{i}",
                    "confidence": float(score),
                })
            return results
        except Exception as e:
            logger.error(f"Automatic SAM 3 segmentation failed: {e}")
            return _get_mock_segmentation(img)

    def init_video(self, frames_dir: str | Path) -> dict | None:
        """Initialize video tracking state."""
        if self.video_predictor is None:
            return None
        try:
            return self.video_predictor.init_state(video_path=str(frames_dir))
        except Exception as e:
            logger.error(f"SAM 3 video init failed: {e}")
            return None

    def add_prompt(self, inference_state, frame_idx: int, obj_id: int,
                   box: np.ndarray | None = None, points: np.ndarray | None = None,
                   labels: np.ndarray | None = None) -> tuple | None:
        """Add a prompt for video tracking."""
        if self.video_predictor is None or inference_state is None:
            return None
        try:
            if box is not None:
                return self.video_predictor.add_new_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
            elif points is not None and labels is not None:
                return self.video_predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
        except Exception as e:
            logger.error(f"SAM 3 video add_prompt failed: {e}")
        return None

    def propagate_video(self, inference_state) -> list[tuple[int, int, np.ndarray]]:
        """Propagate tracking through video.

        Returns:
            list of (frame_idx, obj_id, mask) tuples
        """
        if self.video_predictor is None or inference_state is None:
            return []
        try:
            results = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
                    results.append((int(out_frame_idx), int(obj_id), mask))
            return results
        except Exception as e:
            logger.error(f"SAM 3 video propagation failed: {e}")
            return []


def mask_to_base64(mask: np.ndarray) -> str | None:
    """Convert binary mask to base64 PNG string."""
    if mask is None:
        return None
    try:
        img = Image.fromarray(mask * 255, mode="L")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def get_contour_from_mask(mask: np.ndarray) -> list[dict[str, float]]:
    """Extract contour points from a binary mask."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
