"""World Model Adapter — detect and handle AI-generated video characteristics.

AI-generated videos (Sora, Runway, Kling, Gen-3) have unique properties:
- Inconsistent geometry (objects morph between frames)
- Non-physical motion (teleportation, impossible physics)
- Hallucinated objects (objects appear/disappear without reason)
- Temporal flickering (subtle per-frame noise)

This adapter detects these characteristics and adjusts pipeline parameters
to produce more robust 4D reconstructions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.schemas import PipelineConfig

logger = logging.getLogger(__name__)


class WorldModelAdapter:
    """Analyze video characteristics and adapt pipeline for AI-generated content."""

    def __init__(self):
        self.is_ai_generated = False
        self.characteristics: dict = {}

    def analyze_video(
        self,
        frame_paths: list[Path],
        depth_maps: list[np.ndarray] | None = None,
    ) -> dict:
        """Analyze video frames for AI-generated characteristics.

        Args:
            frame_paths: Paths to extracted video frames
            depth_maps: Optional pre-computed depth maps

        Returns:
            Dict with analysis results:
            - is_ai_generated: bool
            - temporal_consistency: float (0-1, higher = more consistent)
            - geometry_stability: float (0-1, higher = more stable)
            - flicker_score: float (0-1, higher = more flicker)
            - motion_coherence: float (0-1, higher = more coherent)
        """
        if len(frame_paths) < 3:
            return {"is_ai_generated": False, "reason": "too_few_frames"}

        characteristics = {}

        # ── 1. Temporal consistency (pixel-level difference between frames) ──
        consistency_scores = []
        for i in range(min(10, len(frame_paths) - 1)):
            try:
                img_a = self._load_frame_gray(frame_paths[i])
                img_b = self._load_frame_gray(frame_paths[i + 1])
                if img_a is None or img_b is None:
                    continue

                # Structural similarity approximation
                diff = np.abs(img_a.astype(float) - img_b.astype(float))
                consistency = 1.0 - (diff.mean() / 255.0)
                consistency_scores.append(consistency)
            except Exception:
                continue

        characteristics["temporal_consistency"] = float(np.mean(consistency_scores)) if consistency_scores else 0.8

        # ── 2. Geometry stability (depth map consistency) ──
        if depth_maps and len(depth_maps) >= 3:
            stability_scores = []
            for i in range(min(10, len(depth_maps) - 1)):
                d1 = depth_maps[i]
                d2 = depth_maps[i + 1]
                if d1.shape != d2.shape:
                    continue
                valid = (d1 > 0) & (d2 > 0) & (d1 < 100) & (d2 < 100)
                if np.sum(valid) < 100:
                    continue
                depth_diff = np.abs(d1[valid] - d2[valid])
                # Normalize by mean depth
                mean_depth = max(np.mean(d1[valid]), 0.01)
                relative_change = depth_diff.mean() / mean_depth
                stability_scores.append(1.0 - min(1.0, relative_change))
            characteristics["geometry_stability"] = float(np.mean(stability_scores)) if stability_scores else 0.8
        else:
            characteristics["geometry_stability"] = 0.8

        # ── 3. Flicker score (high-frequency temporal noise) ──
        if len(frame_paths) >= 5:
            flicker_scores = []
            for i in range(1, min(5, len(frame_paths) - 1)):
                try:
                    prev = self._load_frame_gray(frame_paths[i - 1])
                    curr = self._load_frame_gray(frame_paths[i])
                    next_f = self._load_frame_gray(frame_paths[i + 1])
                    if prev is None or curr is None or next_f is None:
                        continue

                    # Second derivative: high value = flicker
                    diff1 = curr.astype(float) - prev.astype(float)
                    diff2 = next_f.astype(float) - curr.astype(float)
                    second_deriv = np.abs(diff2 - diff1)
                    flicker = second_deriv.mean() / 255.0
                    flicker_scores.append(min(1.0, flicker * 5))  # Scale up
                except Exception:
                    continue
            characteristics["flicker_score"] = float(np.mean(flicker_scores)) if flicker_scores else 0.1
        else:
            characteristics["flicker_score"] = 0.1

        # ── 4. Determine if AI-generated ──
        # AI videos tend to have: lower geometry stability, higher flicker,
        # but often high temporal consistency (smooth transitions)
        geo_stable = characteristics["geometry_stability"]
        flicker = characteristics["flicker_score"]

        is_ai = geo_stable < 0.6 or flicker > 0.4
        characteristics["is_ai_generated"] = is_ai

        self.is_ai_generated = is_ai
        self.characteristics = characteristics

        if is_ai:
            logger.info(
                "World model adapter: AI-generated video detected "
                "(geo_stability=%.2f, flicker=%.2f, temporal=%.2f)",
                geo_stable, flicker, characteristics["temporal_consistency"],
            )
        else:
            logger.info(
                "World model adapter: Real video detected "
                "(geo_stability=%.2f, flicker=%.2f, temporal=%.2f)",
                geo_stable, flicker, characteristics["temporal_consistency"],
            )

        return characteristics

    def adjust_config(self, config: PipelineConfig) -> PipelineConfig:
        """Adjust pipeline config based on video analysis.

        For AI-generated video:
        - Increase ICP threshold (geometry is noisier)
        - Increase deformation threshold (more non-rigid motion expected)
        - Increase B-spline smoothing (clean up jitter)
        - Reduce completeness threshold (objects may not be fully observed)
        """
        if not self.is_ai_generated:
            return config

        # Create adjusted copy
        adjusted = config.model_copy()
        adjusted.is_world_model_video = True

        # Relax trajectory extraction thresholds
        adjusted.icp_distance_threshold = config.icp_distance_threshold * 2.0
        adjusted.deformation_threshold = config.deformation_threshold * 1.5
        adjusted.trajectory_smoothing = min(1.0, config.trajectory_smoothing + 0.3)

        # Relax completeness threshold (AI videos often have inconsistent object views)
        adjusted.completeness_threshold = max(0.3, config.completeness_threshold - 0.2)

        logger.info(
            "Config adjusted for AI video: icp_thresh=%.3f, deform_thresh=%.3f, smoothing=%.2f",
            adjusted.icp_distance_threshold,
            adjusted.deformation_threshold,
            adjusted.trajectory_smoothing,
        )

        return adjusted

    @staticmethod
    def _load_frame_gray(path: Path) -> np.ndarray | None:
        """Load a frame as grayscale numpy array."""
        try:
            from PIL import Image
            img = Image.open(path).convert("L")
            return np.array(img)
        except Exception:
            return None
