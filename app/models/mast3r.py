"""MASt3R — Meta's 3D point cloud reconstruction from video (2025)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings
from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

logger = logging.getLogger(__name__)

_reconstructor = None


def _get_mock_pointcloud(frames_dir: Path, num_frames: int = 10) -> tuple[dict, list[dict]]:
    """Generate mock point cloud and camera poses for testing."""
    rng = np.random.RandomState(77)
    h, w = 480, 640
    K = estimate_intrinsics(w, h)

    # Generate mock point cloud (random 3D scene)
    n_points = 5000
    points = rng.uniform(-5, 5, (n_points, 3))
    points[:, 2] = np.abs(points[:, 2]) + 1  # All points in front of camera
    colors = (rng.uniform(0, 1, (n_points, 3)) * 255).astype(np.uint8)

    # Generate mock camera poses (moving forward)
    poses = []
    for i in range(num_frames):
        z = -i * 0.5 - 2  # Camera moving backward
        position = (0.0, 0.0, z)
        # Simple look-at rotation
        R = np.eye(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position

        poses.append({
            "frame_idx": i,
            "intrinsics": K.tolist(),
            "extrinsics": T.tolist(),
            "position": position,
            "rotation": rt_matrix_to_quaternion(R),
        })

    pointcloud = {
        "points": points.tolist(),
        "colors": colors.tolist(),
    }
    return pointcloud, poses


class MASt3RReconstructor:
    """Wrapper for MASt3R 3D scene reconstruction from video frames."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load MASt3R model."""
        if settings.mock_mode:
            logger.info("MASt3R: using mock mode")
            return

        try:
            # Try MASt3R from local clone
            mast3r_dir = Path(settings.model_cache_dir) / "mast3r"
            if mast3r_dir.exists():
                import sys
                sys.path.insert(0, str(mast3r_dir))
                from mast3r.model import AsymmetricMASt3R
                self.model = AsymmetricMASt3R.from_pretrained(
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                ).to(self.device)
                logger.info("MASt3R loaded from local repo")
            else:
                # Try HuggingFace directly
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                    cache_dir=settings.model_cache_dir
                ).to(self.device)
                logger.info("MASt3R loaded from HuggingFace")
        except ImportError:
            logger.warning("mast3r not installed, using mock point cloud")
        except Exception as e:
            logger.warning(f"MASt3R load failed: {e}, using mock")

    def reconstruct(self, frames_dir: Path, sample_interval: int | None = None) -> tuple[dict, list[dict]]:
        """Reconstruct 3D point cloud from video frames.

        Args:
            frames_dir: directory containing frame images
            sample_interval: use every Nth frame (default from config)

        Returns:
            (pointcloud_dict, camera_poses_list)
            pointcloud_dict: {"points": [...], "colors": [...]}
            camera_poses_list: list of pose dicts
        """
        if self.model is None:
            if sample_interval is None:
                sample_interval = settings.mast3r_sample_interval
            return _get_mock_pointcloud(frames_dir,
                num_frames=len(list(frames_dir.glob("*.png"))) // sample_interval)

        try:
            from mast3r.utils.image import load_images
            from mast3r.image_pairs import make_pairs
            from mast3r.inference import inference

            sample_interval = sample_interval or settings.mast3r_sample_interval
            frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
            sampled_paths = frame_paths[::sample_interval]

            if len(sampled_paths) < 2:
                logger.warning("Not enough frames for MASt3R reconstruction")
                return _get_mock_pointcloud(frames_dir)

            # Load images
            images = load_images([str(p) for p in sampled_paths], size=512)

            # Create image pairs (sequential + complete for robustness)
            pairs = make_pairs(images, scene_graph="complete", prefilter=None)

            # Run inference
            output = inference(pairs, self.model, self.device, batch_size=settings.batch_size)

            # Extract point cloud and camera poses
            points_3d, colors = self._extract_pointcloud(output, images)
            camera_poses = self._extract_camera_poses(output, images)

            return {"points": points_3d.tolist(), "colors": colors.tolist()}, camera_poses
        except Exception as e:
            logger.error(f"MASt3R reconstruction failed: {e}, using mock")
            return _get_mock_pointcloud(frames_dir)

    def _extract_pointcloud(self, output: dict, images: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract merged point cloud from MASt3R output."""
        import torch

        all_points = []
        all_colors = []

        for view in output:
            if "pts3d" in view:
                pts = view["pts3d"].cpu().numpy()  # (H, W, 3)
                h, w = pts.shape[:2]

                # Flatten and filter invalid points
                pts_flat = pts.reshape(-1, 3)
                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
                pts_valid = pts_flat[valid]

                # Get corresponding colors
                img = view.get("img")
                if img is not None:
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    img_flat = img.reshape(-1, 3)
                    colors_valid = img_flat[valid]
                else:
                    colors_valid = np.ones((len(pts_valid), 3)) * 128

                all_points.append(pts_valid)
                all_colors.append(colors_valid)

        if not all_points:
            return np.zeros((0, 3)), np.zeros((0, 3))

        points = np.vstack(all_points)
        colors = np.vstack(all_colors).astype(np.uint8)

        # Downsample if too large
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        return points, colors

    def _extract_camera_poses(self, output: dict, images: list) -> list[dict]:
        """Extract camera poses from MASt3R output."""
        poses = []
        K = estimate_intrinsics(512, 512)  # Default for MASt3R 512 input

        for i, view in enumerate(output):
            # MASt3R outputs camera poses in view['camera_pose'] or similar
            # Extract or estimate from the model output
            if "camera_pose" in view:
                RT = view["camera_pose"].cpu().numpy()
            else:
                RT = np.eye(4)

            R = RT[:3, :3]
            position = rt_matrix_to_position(R, RT[:3, 3])
            rotation = rt_matrix_to_quaternion(R)

            poses.append({
                "frame_idx": i * settings.mast3r_sample_interval,
                "intrinsics": K.tolist(),
                "extrinsics": RT.tolist(),
                "position": tuple(float(x) for x in position),
                "rotation": tuple(float(x) for x in rotation),
            })

        return poses


def get_reconstructor() -> MASt3RReconstructor:
    """Get or create reconstructor instance."""
    global _reconstructor
    if _reconstructor is None:
        _reconstructor = MASt3RReconstructor()
    return _reconstructor
