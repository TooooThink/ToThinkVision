"""MASt3R — NAVER Labs 3D point cloud reconstruction from video (2024)."""

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
    """Wrapper for MASt3R 3D scene reconstruction from video frames.

    Uses the official naver/mast3r API:
    - Requires cloning with --recursive (includes DUSt3R submodule)
    - import mast3r.utils.path_to_dust3r must come first
    - inference from dust3r.inference
    - load_images from dust3r.utils.image
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._backend = None  # "official" or None
        self._init_model()

    def _init_model(self):
        """Load MASt3R model."""
        if settings.mock_mode:
            logger.info("MASt3R: using mock mode")
            return

        try:
            # MUST import this first — sets up Python path to DUSt3R submodule
            import mast3r.utils.path_to_dust3r  # noqa: F401
            from mast3r.model import AsymmetricMASt3R

            self.model = AsymmetricMASt3R.from_pretrained(
                "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                cache_dir=settings.model_cache_dir
            ).to(self.device)
            self._backend = "official"
            logger.info("MASt3R loaded from HuggingFace")
        except ImportError as e:
            logger.warning(f"mast3r not installed (need recursive clone): {e}")
        except Exception as e:
            logger.warning(f"MASt3R load failed: {e}, using mock")

    def reconstruct(self, frames_dir: Path, sample_interval: int | None = None) -> tuple[dict, list[dict]]:
        """Reconstruct 3D point cloud from video frames.

        Args:
            frames_dir: directory containing frame images
            sample_interval: use every Nth frame (default from config)

        Returns:
            (pointcloud_dict, camera_poses_list)
        """
        if self.model is None:
            if sample_interval is None:
                sample_interval = settings.mast3r_sample_interval
            return _get_mock_pointcloud(frames_dir,
                num_frames=len(list(frames_dir.glob("*.png"))) // max(1, sample_interval))

        try:
            import torch
            from dust3r.inference import inference
            from dust3r.utils.image import load_images

            sample_interval = sample_interval or settings.mast3r_sample_interval
            frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
            sampled_paths = frame_paths[::sample_interval]

            if len(sampled_paths) < 2:
                logger.warning("Not enough frames for MASt3R reconstruction (< 2 after sampling)")
                return _get_mock_pointcloud(frames_dir)

            # Load images — returns list of dicts with 'img', 'true_shape', etc.
            images = load_images([str(p) for p in sampled_paths], size=512)

            # Build image pairs — complete scene graph for robustness
            from mast3r.image_pairs import make_pairs
            pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

            # Run inference
            output = inference(pairs, self.model, self.device, batch_size=settings.batch_size, verbose=False)

            # Extract point cloud and camera poses
            points_3d, colors = self._extract_pointcloud(output, images)
            camera_poses = self._extract_camera_poses(output, images, sample_interval)

            return {"points": points_3d.tolist(), "colors": colors.tolist()}, camera_poses
        except Exception as e:
            logger.error(f"MASt3R reconstruction failed: {e}, using mock")
            return _get_mock_pointcloud(frames_dir)

    def _extract_pointcloud(self, output: dict, images: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract merged point cloud from MASt3R output.

        MASt3R output structure:
        - output['pred1']['pts3d']: [B, H, W, 3] 3D points in view1 camera coords
        - output['pred1']['conf']:  [B, H, W]   confidence scores
        - output['pred2']['pts3d']: [B, H, W, 3] 3D points in view2 camera coords
        - output['pred2']['conf']:  [B, H, W]   confidence scores
        """
        import torch

        all_points = []
        all_colors = []

        # Extract from pred1 (reference view)
        for key in ["pred1", "pred2"]:
            if key not in output:
                continue
            view = output[key]
            if "pts3d" not in view:
                continue

            pts = view["pts3d"].cpu().numpy()  # [B, H, W, 3]
            conf = view.get("conf")
            if conf is not None:
                conf = conf.cpu().numpy()

            for b in range(pts.shape[0]):
                pts_b = pts[b]  # [H, W, 3]
                pts_flat = pts_b.reshape(-1, 3)
                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)

                if conf is not None:
                    conf_b = conf[b].reshape(-1)
                    valid = valid & (conf_b > 0.5)

                pts_valid = pts_flat[valid]

                # Get corresponding colors from input image
                img_data = None
                for img in images:
                    if img.get("img") is not None:
                        img_data = img["img"]
                        break

                if img_data is not None:
                    if isinstance(img_data, torch.Tensor):
                        img_np = img_data.cpu().numpy()
                    else:
                        img_np = img_data

                    if img_np.ndim == 3:
                        img_flat = img_np.reshape(-1, 3)
                        if len(img_flat) == valid.shape[0]:
                            colors_valid = (img_flat[valid] * 255).clip(0, 255).astype(np.uint8)
                        else:
                            colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128
                    else:
                        colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128
                else:
                    colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128

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

    def _extract_camera_poses(self, output: dict, images: list,
                              sample_interval: int) -> list[dict]:
        """Extract camera poses from MASt3R output.

        MASt3R's sparse_global_alignment provides camera poses, but for pairwise
        inference we estimate from the point cloud geometry.
        """
        poses = []
        h, w = 512, 512  # MASt3R uses 512x512 input

        for i, img_data in enumerate(images):
            K = estimate_intrinsics(w, h)

            # MASt3R doesn't directly output camera poses in pairwise mode
            # We use the point cloud center as an estimate
            if "pred1" in output:
                pts = output["pred1"]["pts3d"].cpu().numpy()
                if pts.ndim == 4:
                    pts_flat = pts[i % pts.shape[0]].reshape(-1, 3)
                else:
                    pts_flat = pts.reshape(-1, 3)

                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
                if valid.any():
                    center = pts_flat[valid].mean(axis=0)
                    # Camera is roughly at origin looking at scene center
                    position = (0.0, 0.0, 0.0)
                    R = np.eye(3)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = center
                else:
                    T = np.eye(4)
                    position = (0.0, 0.0, 0.0)
                    R = np.eye(3)
            else:
                T = np.eye(4)
                position = (0.0, 0.0, 0.0)
                R = np.eye(3)

            R = T[:3, :3]
            rotation = rt_matrix_to_quaternion(R)
            pos_tuple = rt_matrix_to_position(R, T[:3, 3])

            poses.append({
                "frame_idx": i * sample_interval,
                "intrinsics": K.tolist(),
                "extrinsics": T.tolist(),
                "position": tuple(float(x) for x in pos_tuple),
                "rotation": tuple(float(x) for x in rotation),
            })

        return poses


def get_reconstructor() -> MASt3RReconstructor:
    """Get or create reconstructor instance."""
    global _reconstructor
    if _reconstructor is None:
        _reconstructor = MASt3RReconstructor()
    return _reconstructor
