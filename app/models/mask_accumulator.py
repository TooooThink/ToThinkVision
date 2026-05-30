"""Multi-frame mask accumulation + completeness scoring.

Accumulates per-frame masks for each tracked object using binary OR,
then computes a completeness score to determine if the object is
fully observed or still partially visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Expected fill ratio per label type (mask_area / bbox_area).
# Irregular objects (chairs, people) naturally have lower ratios.
EXPECTED_FILL_RATIO = {
    "button": 0.9,
    "text": 0.85,
    "icon": 0.75,
    "image": 0.9,
    "screen": 0.9,
    "monitor": 0.85,
    "car": 0.7,
    "person": 0.6,
    "animal": 0.6,
    "chair": 0.6,
    "table": 0.7,
    "furniture": 0.65,
    "door": 0.85,
    "wall": 0.9,
    "floor": 0.9,
    "window": 0.8,
    "building": 0.7,
    "plant": 0.5,
    "vehicle": 0.7,
    "object": 0.7,
    "prop": 0.65,
    "terrain": 0.85,
    "effect": 0.5,
}


@dataclass
class CompletenessResult:
    """Result of completeness assessment for a tracked object."""
    score: float = 1.0
    is_complete: bool = True
    accumulated_mask: np.ndarray | None = None
    frames_contributed: int = 0
    growth_rate: float = 0.0
    area_history: list[float] = field(default_factory=list)


class MaskAccumulator:
    """Accumulates per-frame masks and computes completeness scores."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        growth_threshold: float = 0.02,
        window_size: int = 5,
    ):
        self.width = frame_width
        self.height = frame_height
        self.growth_threshold = growth_threshold
        self.window_size = window_size

        # Per track_id: accumulated binary mask (full-frame canvas)
        self._accumulated: dict[str, np.ndarray] = {}
        # Per track_id: list of (frame_idx, mask_area)
        self._area_history: dict[str, list[tuple[int, float]]] = {}
        # Per track_id: set of contributing frame indices
        self._frame_set: dict[str, set[int]] = {}

    def accumulate(
        self,
        track_id: str,
        mask: np.ndarray,
        bbox: list[float] | None = None,
        frame_idx: int = 0,
    ):
        """Add a per-frame mask to the accumulation buffer.

        Args:
            track_id: unique object identifier
            mask: binary mask (H, W) or full-frame mask. If bbox is provided,
                  mask is assumed to be cropped to bbox region.
            bbox: [x, y, w, h] if mask is cropped, None if mask is full-frame
            frame_idx: current frame index
        """
        # Ensure mask is binary
        mask_bin = (mask > 0).astype(np.uint8)

        # Place on full-frame canvas if bbox provided
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Skip if bbox is completely outside frame
            if x + w <= 0 or y + h <= 0 or x >= self.width or y >= self.height or w <= 0 or h <= 0:
                mask_bin = np.zeros((self.height, self.width), dtype=np.uint8)
            else:
                canvas = np.zeros((self.height, self.width), dtype=np.uint8)

                # Clip to frame bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, self.width)
                y2 = min(y + h, self.height)

                # Calculate mask region to copy
                mask_x1 = max(0, -x)  # offset in mask if x was negative
                mask_y1 = max(0, -y)  # offset in mask if y was negative
                mask_x2 = min(mask_x1 + (x2 - x1), mask_bin.shape[1])
                mask_y2 = min(mask_y1 + (y2 - y1), mask_bin.shape[0])

                # Only copy if both regions are valid
                if (mask_y2 > mask_y1 and mask_x2 > mask_x1 and
                    y2 > y1 and x2 > x1):
                    mask_region = mask_bin[mask_y1:mask_y2, mask_x1:mask_x2]
                    # Ensure shapes match before assignment
                    if (mask_region.shape[0] == y2 - y1 and
                        mask_region.shape[1] == x2 - x1):
                        canvas[y1:y2, x1:x2] = mask_region

                mask_bin = canvas

        if track_id not in self._accumulated:
            self._accumulated[track_id] = mask_bin.copy()
            self._area_history[track_id] = []
            self._frame_set[track_id] = set()

        # Binary OR with existing accumulation
        self._accumulated[track_id] = np.maximum(
            self._accumulated[track_id], mask_bin
        )

        # Track area and frames
        area = float(mask_bin.sum())
        self._area_history[track_id].append((frame_idx, area))
        self._frame_set[track_id].add(frame_idx)

    def get_completeness(
        self,
        track_id: str,
        label: str = "object",
        threshold: float = 0.6,
    ) -> CompletenessResult:
        """Compute completeness score for a tracked object.

        Combines three signals:
        1. Growth convergence (0.4) — is mask still growing?
        2. Boundary contact (0.3) — does mask fill its bbox edges?
        3. Fill ratio (0.3) — mask_area / bbox_area relative to expected

        Args:
            track_id: unique object identifier
            label: object label for expected fill ratio lookup
            threshold: score below which object is flagged incomplete

        Returns:
            CompletenessResult with score, is_complete, accumulated_mask
        """
        result = CompletenessResult()

        if track_id not in self._accumulated:
            result.score = 0.0
            result.is_complete = False
            return result

        acc_mask = self._accumulated[track_id]
        frames = self._frame_set[track_id]
        result.frames_contributed = len(frames)
        result.accumulated_mask = acc_mask.copy()

        # If only one frame contributed, likely incomplete
        if len(frames) <= 1:
            result.score = 0.2
            result.is_complete = False
            result.growth_rate = 1.0
            return result

        # Signal 1: Growth convergence
        area_history = self._area_history[track_id]
        result.area_history = [a for _, a in area_history]
        growth_rate = self._compute_growth_rate(area_history)
        result.growth_rate = growth_rate
        score_growth = max(0.0, 1.0 - growth_rate)

        # Signal 2: Boundary contact
        score_boundary = self._compute_boundary_score(acc_mask)

        # Signal 3: Fill ratio
        score_fill = self._compute_fill_score(acc_mask, label)

        # Penalty: small mask relative to frame
        frame_area = self.height * self.width
        mask_area_ratio = float(acc_mask.sum()) / frame_area
        size_penalty = min(1.0, mask_area_ratio * 3)  # <33% frame → penalized

        # Weighted combination with size penalty
        base_score = (
            0.4 * score_growth
            + 0.3 * score_boundary
            + 0.3 * score_fill
        )
        result.score = base_score * size_penalty
        result.is_complete = result.score >= threshold

        return result

    def get_accumulated_mask(self, track_id: str) -> np.ndarray | None:
        """Get the accumulated mask for a tracked object."""
        return self._accumulated.get(track_id)

    def reset(self):
        """Clear all accumulation state."""
        self._accumulated.clear()
        self._area_history.clear()
        self._frame_set.clear()

    # ─── Internal scoring methods ────────────────────────────

    def _compute_growth_rate(
        self, area_history: list[tuple[int, float]]
    ) -> float:
        """Compute normalized growth convergence score.

        Returns 0.0 = fully converged (no recent growth), 1.0 = still growing.

        Uses the ratio of recent area to maximum area. If the last frame's
        area is at least 95% of the max ever seen, we consider it converged.
        """
        if len(area_history) < 2:
            return 1.0

        areas = [a for _, a in area_history]
        max_area = max(areas)
        if max_area == 0:
            return 1.0

        # If last area >= 95% of max, growth has converged
        last_area = areas[-1]
        convergence = last_area / max_area

        # Convert to growth rate (0 = converged, 1 = still growing)
        growth_rate = max(0.0, 1.0 - convergence)

        # For sequences with > 3 frames, also check if recent frames are stable
        if len(areas) > 3:
            recent_std = np.std(areas[-self.window_size:])
            recent_mean = np.mean(areas[-self.window_size:])
            if recent_mean > 0:
                cv = recent_std / recent_mean  # coefficient of variation
                if cv < 0.05:  # very stable
                    growth_rate *= 0.5  # reduce growth rate if stable

        return growth_rate

    def _compute_boundary_score(self, acc_mask: np.ndarray) -> float:
        """Score based on how many bbox edges the mask touches.

        Returns 0.0 (touches no edges) to 1.0 (touches all 4 edges).
        Discounted if bbox touches image border.
        """
        ys, xs = np.where(acc_mask > 0)
        if len(ys) == 0:
            return 0.0

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        h, w = acc_mask.shape
        touched = 0

        # Top edge
        if y_min <= 1:
            touched += 1
        # Bottom edge
        if y_max >= h - 2:
            touched += 1
        # Left edge
        if x_min <= 1:
            touched += 1
        # Right edge
        if x_max >= w - 2:
            touched += 1

        score = touched / 4.0

        # Discount if object bbox touches image border (may be out-of-frame)
        at_border = (
            (y_min <= 1 and y_max >= h - 2)
            or (x_min <= 1 and x_max >= w - 2)
        )
        if at_border:
            # Reduce score: object may be cropped by frame, not occluded
            score *= 0.7

        return min(1.0, score)

    def _compute_fill_score(self, acc_mask: np.ndarray, label: str) -> float:
        """Score based on mask area relative to bbox area."""
        ys, xs = np.where(acc_mask > 0)
        if len(ys) == 0:
            return 0.0

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        mask_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        # Actual pixel count in the bounding box of the mask
        bbox_area = float((y_max - y_min + 1) * (x_max - x_min + 1))

        if bbox_area <= 0:
            return 0.0

        fill_ratio = mask_area / bbox_area

        # Normalize by expected fill for this label
        expected = EXPECTED_FILL_RATIO.get(label, 0.7)
        score = min(1.0, fill_ratio / expected)

        return score


# Module-level singleton (Pattern B)
_instance: MaskAccumulator | None = None


def get_accumulator(
    frame_width: int = 1,
    frame_height: int = 1,
    **kwargs,
) -> MaskAccumulator:
    """Get or create mask accumulator instance."""
    global _instance
    if _instance is None:
        _instance = MaskAccumulator(frame_width, frame_height, **kwargs)
    return _instance
