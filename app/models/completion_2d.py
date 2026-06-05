"""2D mask completion via LaMa inpainting.

For objects flagged as partially visible, this module uses LaMa
(large mask inpainting) to generate the missing regions of the object.

LaMa is lightweight (~300MB), fast (<100ms on GPU), and runs locally
without API calls. When LaMa weights are unavailable,
returns the input unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_instance = None


class Completion2D:
    """2D mask completion using LaMa inpainting."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._backend = None
        self._init_model()

    def _init_model(self):
        """Load LaMa model."""
        from app.config import settings
        cache = Path(settings.model_cache_dir)

        # Try loading original LaMa from local cache
        lama_path = cache / "lama" / "big-lama.pt"
        if lama_path.exists():
            try:
                import torch
                self.model = torch.jit.load(str(lama_path))
                self.model.to(self.device)
                self.model.eval()
                self._backend = "lama_original"
                logger.info("Completion2D loaded: LaMa original weights from %s", lama_path)
                return
            except Exception as e:
                logger.info(f"LaMa original load failed ({e}), trying lama-cleaner...")

        # Try loading LaMa from lama-cleaner package
        try:
            from lama_cleaner.model_manager import ModelManager
            logger.info("lama-cleaner imported, initializing model...")
            self.lama_model = ModelManager(
                model_name="lama",
                device=self.device,
            )
            self._backend = "lama_direct"
            self.model = self.lama_model
            logger.info("Completion2D loaded: LaMa via lama-cleaner")
            return
        except Exception as e:
            logger.warning(f"LaMa via lama-cleaner failed: {e}", exc_info=True)

        # Try loading LaMa from HuggingFace transformers
        try:
            import torch
            from transformers import pipeline

            self.inpaint_pipe = pipeline(
                "image-inpainting",
                model="LamaMultimodal/Multimodal-Large-Image-Inpainting",
                device=0 if "cuda" in self.device else -1,
            )
            self._backend = "lama"
            self.model = self.inpaint_pipe
            logger.info("Completion2D loaded: LaMa via transformers")
            return
        except Exception as e:
            logger.info(f"LaMa HuggingFace load failed ({e})")

        logger.warning("LaMa not available, 2D completion disabled")

    def complete(
        self,
        image: np.ndarray,
        partial_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete an object image using LaMa inpainting.

        Args:
            image: (H, W, 3) RGB image of the object crop
            partial_mask: (H, W) binary mask of the observed region (1=observed)

        Returns:
            (completed_image, completed_mask):
                completed_image: (H, W, 3) with predicted missing regions filled
                completed_mask: (H, W) binary mask of the full object region
        """
        if self.model is None:
            logger.info("Completion2D: no model, returning input unchanged")
            return image, partial_mask

        if self._backend == "lama_direct":
            return self._complete_lama_direct(image, partial_mask)
        elif self._backend == "lama":
            return self._complete_lama_transformers(image, partial_mask)
        elif self._backend == "lama_original":
            return self._complete_lama_original(image, partial_mask)

        return image, partial_mask

    def _complete_lama_direct(
        self,
        image: np.ndarray,
        partial_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete using lama-cleaner direct API."""
        from lama_cleaner.schema import Config as LamaConfig

        h, w = image.shape[:2]

        # Inpainting mask: 1 where we want to fill (inverse of partial_mask)
        inpaint_mask = (partial_mask == 0).astype(np.uint8) * 255

        result = self.model(
            image=image,
            mask=inpaint_mask,
            config=LamaConfig(ldm_steps=1),
        )

        completed_image = result
        # Completed mask: original partial_mask + inpainted region
        completed_mask = np.ones((h, w), dtype=np.uint8)
        completed_mask[partial_mask == 1] = 1
        # The inpainted region is now part of the "object"
        completed_mask[inpaint_mask > 0] = 1

        return completed_image, completed_mask

    def _complete_lama_original(
        self,
        image: np.ndarray,
        partial_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete using original LaMa PyTorch JIT model."""
        import torch

        h, w = image.shape[:2]

        # Prepare image: (H, W, 3) -> (1, 3, H, W), normalized to [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Prepare mask: (H, W) -> (1, 1, H, W), 0=observed, 1=to_fill
        mask_tensor = torch.from_numpy((partial_mask == 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            result = self.model(img_tensor, mask_tensor)

        # Convert result back to numpy: (1, 3, H, W) -> (H, W, 3)
        result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        completed_image = (result_np * 255).clip(0, 255).astype(np.uint8)

        # Completed mask: all pixels are now "object"
        completed_mask = np.ones((h, w), dtype=np.uint8)

        return completed_image, completed_mask

    def _complete_lama_transformers(
        self,
        image: np.ndarray,
        partial_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete using HuggingFace transformers pipeline."""
        from PIL import Image

        h, w = image.shape[:2]

        # Convert to PIL
        pil_image = Image.fromarray(image)
        # Inpainting mask: 1 (white) where we want to fill
        pil_mask = Image.fromarray(
            ((partial_mask == 0).astype(np.uint8) * 255)
        )

        result = self.inpaint_pipe(pil_image, mask_image=pil_mask)

        if isinstance(result, list):
            completed_pil = result[0]["image"]
        elif isinstance(result, dict):
            completed_pil = result.get("image", pil_image)
        else:
            completed_pil = result

        completed_image = np.array(completed_pil.convert("RGB"))

        # Completed mask: all pixels are now "object"
        completed_mask = np.ones((h, w), dtype=np.uint8)

        return completed_image, completed_mask


def get_completion_2d(device: str = "cuda") -> Completion2D:
    """Get or create 2D completion instance."""
    global _instance
    if _instance is None:
        _instance = Completion2D(device)
    return _instance


def complete_object_2d(
    image: np.ndarray,
    partial_mask: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function for 2D completion."""
    return get_completion_2d(device).complete(image, partial_mask)
