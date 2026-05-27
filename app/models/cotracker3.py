"""CoTracker3 — Meta's dense point tracking model for video.

Tracks up to 265x265 points jointly across video frames, producing
precise per-point 2D trajectories with visibility scores. Used to
extract accurate per-object motion trajectories (replaces ICP-based approach).

GitHub: https://github.com/facebookresearch/co-tracker
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_HAS_COTRACKER = False
try:
    import torch
    _HAS_COTRACKER = True
except ImportError:
    pass


class CoTracker3Predictor:
    """Wrapper for CoTracker3 dense point tracking in video."""

    def __init__(
        self,
        device: str = "cuda",
        mode: str = "offline",
        checkpoint_path: str | None = None,
    ):
        """Initialize CoTracker3.

        Args:
            device: torch device string
            mode: "offline" (full video) or "online" (streaming/chunked)
            checkpoint_path: optional local checkpoint path
        """
        self.device = device
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load CoTracker3 model."""
        if not _HAS_COTRACKER:
            logger.warning("PyTorch not available, CoTracker3 will use mock mode")
            return

        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not available, CoTracker3 will use mock mode")
            return

        try:
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                # Load from local checkpoint
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    f"cotracker3_{self.mode}",
                ).to(self.device)
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                # Load via torch.hub (auto-downloads weights)
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    f"cotracker3_{self.mode}",
                ).to(self.device)

            logger.info("CoTracker3 (%s mode) loaded successfully", self.mode)
        except Exception as e:
            logger.warning("CoTracker3 load failed: %s, using mock mode", e)
            self.model = None

    def track_video(
        self,
        frames: np.ndarray | list[np.ndarray],
        grid_size: int = 50,
        query_frame: int = 0,
    ) -> dict[str, Any]:
        """Track dense points across video frames.

        Args:
            frames: (T, H, W, 3) uint8 numpy array or list of frames
            grid_size: number of points per axis (grid_size^2 total points)
            query_frame: frame index to sample query points from

        Returns:
            dict with:
                - "tracks": (T, N, 2) float array of (x, y) positions per frame
                - "visibility": (T, N) bool array of point visibility
                - "query_points": (N, 2) float array of initial query positions
        """
        if self.model is None:
            return self._mock_track_video(frames, grid_size)

        import torch

        # Convert frames to tensor: (B, T, C, H, W)
        if isinstance(frames, list):
            frames = np.stack(frames)

        T, H, W, C = frames.shape
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)

        with torch.no_grad():
            if self.mode == "offline":
                pred_tracks, pred_visibility = self.model(
                    video, grid_size=grid_size, grid_query_frame=query_frame
                )
            else:
                # Online mode: process in chunks
                self.model(
                    video_chunk=video, is_first_step=True, grid_size=grid_size
                )
                pred_tracks, pred_visibility = self.model(
                    video_chunk=video
                )

        # Convert to numpy
        # pred_tracks: (1, T, N, 2) → (T, N, 2)
        tracks = pred_tracks[0].cpu().numpy()
        # pred_visibility: (1, T, N) or (1, T, N, 1) → (T, N)
        vis = pred_visibility[0].cpu().numpy()
        if vis.ndim == 3:
            vis = vis[:, :, 0]
        visibility = vis > 0.5

        # Extract query points from first frame
        query_points = tracks[query_frame]

        return {
            "tracks": tracks,
            "visibility": visibility,
            "query_points": query_points,
            "num_points": tracks.shape[1],
            "frame_width": W,
            "frame_height": H,
        }

    def track_object_points(
        self,
        frames: np.ndarray,
        object_masks: list[np.ndarray] | np.ndarray,
        points_per_object: int = 100,
    ) -> dict[str, np.ndarray]:
        """Track points specifically within object masks.

        Args:
            frames: (T, H, W, 3) video frames
            object_masks: (T, H, W) binary mask or list of per-frame masks
                        Can also be a dict {object_id: (T, H, W)} for multiple objects
            points_per_object: number of points to sample per object

        Returns:
            dict mapping object_id → {tracks, visibility}
        """
        if self.model is None:
            return self._mock_track_objects(frames, object_masks, points_per_object)

        import torch

        if isinstance(frames, list):
            frames = np.stack(frames)

        T, H, W, C = frames.shape
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)

        results = {}

        if isinstance(object_masks, dict):
            # Multiple objects
            for obj_id, masks in object_masks.items():
                if isinstance(masks, list):
                    masks = np.stack(masks)
                result = self._track_single_object(video, masks, points_per_object, T, H, W)
                results[obj_id] = result
        else:
            # Single object
            if isinstance(object_masks, list):
                object_masks = np.stack(object_masks)
            result = self._track_single_object(video, object_masks, points_per_object, T, H, W)
            results["object_0"] = result

        return results

    def _track_single_object(
        self, video, masks, points_per_object, T, H, W
    ) -> dict[str, Any]:
        """Track points within a single object mask."""
        import torch

        # Sample query points from first frame's mask
        first_mask = masks[0] if masks.ndim == 3 else masks
        ys, xs = np.where(first_mask > 0)

        if len(xs) == 0:
            return {"tracks": np.zeros((T, 0, 2)), "visibility": np.zeros((T, 0), dtype=bool)}

        # Sample random points within the mask
        n_pts = min(points_per_object, len(xs))
        indices = np.random.choice(len(xs), n_pts, replace=False)
        query_x = xs[indices].astype(np.float32)
        query_y = ys[indices].astype(np.float32)

        # CoTracker expects queries as (B, N, 3) tensor: (frame_idx, x, y)
        queries = torch.zeros(1, n_pts, 3, device=self.device)
        queries[0, :, 0] = 0  # query frame index
        queries[0, :, 1] = torch.from_numpy(query_x).to(self.device)
        queries[0, :, 2] = torch.from_numpy(query_y).to(self.device)

        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video, queries=queries)

        tracks = pred_tracks[0].cpu().numpy()
        vis = pred_visibility[0].cpu().numpy()
        if vis.ndim == 3:
            vis = vis[:, :, 0]
        visibility = vis > 0.5

        return {
            "tracks": tracks,
            "visibility": visibility,
            "query_points": np.stack([query_x, query_y], axis=-1),
        }

    def _mock_track_video(
        self, frames: np.ndarray | list, grid_size: int
    ) -> dict[str, Any]:
        """Generate mock tracking results."""
        if isinstance(frames, list):
            T = len(frames)
            H, W = frames[0].shape[:2] if hasattr(frames[0], "shape") else (480, 640)
        else:
            T, H, W = frames.shape[:3]

        N = grid_size * grid_size
        rng = np.random.RandomState(42)

        # Generate slowly drifting points
        base_x = rng.uniform(0, W, N)
        base_y = rng.uniform(0, H, N)
        tracks = np.zeros((T, N, 2))
        for t in range(T):
            tracks[t, :, 0] = base_x + rng.randn(N) * t * 0.5
            tracks[t, :, 1] = base_y + rng.randn(N) * t * 0.5

        visibility = np.ones((T, N), dtype=bool)
        query_points = tracks[0]

        return {
            "tracks": tracks,
            "visibility": visibility,
            "query_points": query_points,
            "num_points": N,
            "frame_width": W,
            "frame_height": H,
        }

    def _mock_track_objects(
        self, frames: np.ndarray, object_masks: Any, points_per_object: int
    ) -> dict[str, Any]:
        """Generate mock per-object tracking results."""
        T = frames.shape[0] if frames.ndim == 4 else len(frames)
        N = points_per_object
        rng = np.random.RandomState(42)

        tracks = rng.uniform(0, 100, (T, N, 2)).astype(np.float32)
        visibility = np.ones((T, N), dtype=bool)

        return {
            "tracks": tracks,
            "visibility": visibility,
            "query_points": tracks[0],
        }


# Global instance
_cotracker_predictor: CoTracker3Predictor | None = None


def get_cotracker(mode: str = "offline") -> CoTracker3Predictor:
    """Get or create a global CoTracker3 instance."""
    global _cotracker_predictor
    if _cotracker_predictor is None:
        _cotracker_predictor = CoTracker3Predictor(mode=mode)
    return _cotracker_predictor
