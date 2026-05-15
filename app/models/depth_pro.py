"""Depth Pro — Apple's metric depth estimation (ICLR 2025)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_depth_model = None


def _get_mock_depth(img: np.ndarray) -> np.ndarray:
    """Generate mock metric depth map for testing."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    # Simulated perspective: bottom = closer (0.5m), top = farther (20m)
    depth = 0.5 + (y / h) * 19.5
    return depth.astype(np.float32)


class DepthPro:
    """Wrapper for Apple Depth Pro metric depth estimation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self._init_model()

    def _init_model(self):
        """Load Depth Pro model."""
        if settings.mock_mode:
            logger.info("Depth Pro: using mock mode")
            return

        try:
            import torch

            # Try Apple's Depth Pro
            depth_pro_dir = Path(settings.model_cache_dir) / "ml-depth-pro"
            if depth_pro_dir.exists():
                import sys
                sys.path.insert(0, str(depth_pro_dir))
                import depth_pro
                self.model, self.transform = depth_pro.create_model_and_transforms()
                self.model.eval().to(self.device)
                logger.info("Depth Pro loaded from local repo")
            else:
                # Try HuggingFace
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                processor = AutoImageProcessor.from_pretrained(
                    "apple/depth-pro", cache_dir=settings.model_cache_dir
                )
                model = AutoModelForDepthEstimation.from_pretrained(
                    "apple/depth-pro", cache_dir=settings.model_cache_dir
                )
                model.to(self.device)
                self.model = model
                self.transform = processor
                logger.info("Depth Pro loaded from HuggingFace")
        except ImportError:
            logger.warning("depth_pro not installed, falling back to mock")
        except Exception as e:
            logger.warning(f"Depth Pro load failed: {e}")

    def estimate(self, img: np.ndarray, f_px: float | None = None) -> np.ndarray:
        """Estimate metric depth map.

        Args:
            img: (H, W, 3) RGB image
            f_px: focal length in pixels (estimated if None)

        Returns:
            depth_map: (H, W) depth in meters
        """
        if self.model is None:
            return _get_mock_depth(img)

        try:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(img)
            h, w = img.shape[:2]

            if f_px is None:
                f_px = max(w, h)  # Default: focal length ≈ image diagonal

            if hasattr(self, "transform") and callable(self.transform):
                # Apple's Depth Pro API
                import torch
                input_tensor = self.transform(pil_img).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    prediction = self.model.infer(input_tensor, f_px=f_px)
                depth = prediction["depth"].squeeze().cpu().numpy()
            else:
                # HuggingFace transformers API
                import torch
                inputs = self.transform(images=pil_img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()

            # Interpolate to original size
            if depth.shape != (h, w):
                import cv2
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

            return depth.astype(np.float32)
        except Exception as e:
            logger.error(f"Depth Pro estimation failed: {e}, using mock fallback")
            return _get_mock_depth(img)

    def get_depth_at(self, depth_map: np.ndarray, bbox: list[float]) -> float:
        """Get average depth in bbox region."""
        x, y, w, h = bbox
        h_map, w_map = depth_map.shape[:2]
        x, y = max(0, int(x)), max(0, int(y))
        x2, y2 = min(int(x + w), w_map), min(int(y + h), h_map)
        if x2 <= x or y2 <= y:
            return 0.0
        return float(depth_map[y:y2, x:x2].mean())


def get_depth_model() -> DepthPro:
    """Get or create depth model instance."""
    global _depth_model
    if _depth_model is None:
        _depth_model = DepthPro()
    return _depth_model


def estimate_depth(img: np.ndarray) -> np.ndarray:
    """Convenience function for depth estimation."""
    return get_depth_model().estimate(img)
