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


class SAM3Predictor:
    """Wrapper for SAM 3 image and video prediction.

    Uses the official facebookresearch/sam3 API:
    - Image: build_sam3_image_model(enable_inst_interactivity=True)
    - Video: build_sam3_predictor(version="sam3")
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.image_predictor = None  # SAM3InteractiveImagePredictor
        self.video_predictor = None  # Sam3VideoPredictorMultiGPU
        self._init_model()

    def _init_model(self):
        """Load SAM 3 models."""
        try:
            # Official SAM 3 API — unified entry point
            from sam3.model_builder import build_sam3_image_model

            # Use SAM 3.1 weights from local cache (HF is gated + GPU nodes offline)
            cache = Path(settings.model_cache_dir)
            ckpt_v31 = cache / "sam3.1" / "sam3.1_multiplex.pt"
            ckpt_v3 = cache / "sam3.pt"
            ckpt = str(ckpt_v31 if ckpt_v31.exists() else ckpt_v3)
            has_local_ckpt = Path(ckpt).exists()

            self._model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=ckpt if has_local_ckpt else None,
                load_from_HF=not has_local_ckpt,
                enable_inst_interactivity=True,
                enable_segmentation=True,
            )

            # Verify the backbone actually loaded (not None from a bad checkpoint)
            if hasattr(self._model, "backbone") and self._model.backbone is None:
                raise RuntimeError(
                    f"SAM 3 backbone is None — checkpoint '{ckpt}' is incompatible or incomplete. "
                    "Try deleting the local checkpoint and re-downloading: "
                    f"rm -rf {Path(ckpt).parent}"
                )

            # Get the interactive predictor attached to the model
            if hasattr(self._model, "inst_interactive_predictor") and self._model.inst_interactive_predictor is not None:
                self.image_predictor = self._model.inst_interactive_predictor

                # Share the main model's backbone with the tracker predictor
                # (build_tracker defaults to with_backbone=False, so tracker.backbone is None)
                if self.image_predictor.model.backbone is None:
                    self.image_predictor.model.backbone = self._model.backbone
                    logger.info("SAM 3: shared main backbone with interactive predictor")

                # Verify the predictor has required methods for the API we use
                required_methods = ['set_image', 'predict']
                missing = [m for m in required_methods if not hasattr(self.image_predictor, m)]
                if missing:
                    raise RuntimeError(f"SAM 3 interactive predictor missing methods: {missing}")

                logger.info("SAM 3 image predictor loaded")
            else:
                raise RuntimeError("SAM 3 model loaded but interactive predictor not available")

            # Try video predictor
            try:
                from sam3.model_builder import build_sam3_video_predictor

                self.video_predictor = build_sam3_video_predictor(
                    checkpoint_path=ckpt if has_local_ckpt else None,
                )
                logger.info("SAM 3 video predictor loaded")
            except Exception as e:
                logger.warning(f"SAM 3 video predictor not available: {e}")

        except ImportError:
            raise RuntimeError(
                "SAM 3 is required but not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/sam3.git"
            )

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
        self.image_predictor.set_image(img)

        results = []
        if boxes is not None and len(boxes) > 0:
            # Segment from provided boxes
            for i, box in enumerate(boxes):
                # SAM 3 expects [x1, y1, x2, y2] format
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
            # Automatic segmentation (predict without prompts = auto mode)
            results = self._predict_automatic(img)

        return results

    def _predict_with_text(self, img: np.ndarray, text_prompt: str) -> list[dict]:
        """Use SAM 3 text-prompted segmentation."""
        # SAM 3 supports text prompts via its forward pass
        # The interactive predictor may have a text-based predict method
        if hasattr(self.image_predictor, "predict_with_text"):
            results = self.image_predictor.predict_with_text(text_prompt)
            return results
        # Fallback: use automatic segmentation
        return self._predict_automatic(img)

    def _predict_automatic(self, img: np.ndarray) -> list[dict]:
        """Automatic segmentation using SAM 3 without prompts."""
        # Call predict with no prompts — returns all detected masks
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

    def init_video(self, frames_dir: str | Path) -> dict | None:
        """Initialize video tracking state.

        SAM 3 video uses a request-based API. We start a session here.
        Returns session info dict or None.
        """
        if self.video_predictor is None:
            return None
        response = self.video_predictor.handle_request({
            "type": "start_session",
            "resource_path": str(frames_dir),
        })
        return {
            "session_id": response["session_id"],
            "predictor": self.video_predictor,
        }

    def add_prompt(self, inference_state, frame_idx: int, obj_id: int,
                   box: np.ndarray | None = None, points: np.ndarray | None = None,
                   labels: np.ndarray | None = None) -> dict | None:
        """Add a prompt for video tracking.

        Args:
            inference_state: session dict from init_video
            frame_idx: frame index
            obj_id: object ID
            box: [x1, y1, x2, y2] bounding box
            points: (N, 2) point coordinates
            labels: (N,) point labels (1=foreground, 0=background)
        """
        if inference_state is None or "predictor" not in inference_state:
            return None
        predictor = inference_state["predictor"]
        session_id = inference_state["session_id"]

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
        }

        if box is not None:
            request["bounding_boxes"] = [box.tolist()]
            request["bounding_box_labels"] = [1]  # 1 = foreground

        if points is not None and labels is not None:
            request["points"] = points.tolist()
            request["point_labels"] = labels.tolist()

        return predictor.handle_request(request)

    def propagate_video(self, inference_state) -> list[tuple[int, int, np.ndarray]]:
        """Propagate tracking through video.

        Returns:
            list of (frame_idx, obj_id, mask) tuples
        """
        if inference_state is None or "predictor" not in inference_state:
            return []
        predictor = inference_state["predictor"]
        session_id = inference_state["session_id"]

        results = []
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
        }):
            frame_idx = out["frame_index"]
            outputs = out.get("outputs", {})
            masks = outputs.get("out_binary_masks", [])
            obj_ids = outputs.get("out_obj_ids", [])
            for i in range(len(masks)):
                mask = masks[i].astype(np.uint8)
                results.append((int(frame_idx), int(obj_ids[i]), mask))

        # Clean up session
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception:
            pass

        return results


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
