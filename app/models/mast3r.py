"""VGGT — Meta Visual Geometry Grounded Transformer for 3D reconstruction (CVPR 2025).

Upgraded from MASt3R: 45x faster (0.2s vs 9s), better quality, superior camera pose estimation.
MASt3R kept as fallback if VGGT is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings
from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

logger = logging.getLogger(__name__)

_reconstructor = None


class VGGTReconstructor:
    """Wrapper for VGGT 3D scene reconstruction from video frames.

    VGGT (Visual Geometry Grounded Transformer, Meta, CVPR 2025):
    - Feed-forward transformer: ~0.2s per reconstruction (vs ~9s for MASt3R)
    - Superior camera pose estimation and point cloud quality
    - HuggingFace: meta/VGGT

    Falls back to MASt3R if VGGT is not available.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._backend = None  # "vggt", "mast3r", or None
        self._mast3r_model = None
        self._init_model()

    def _init_model(self):
        """Load VGGT model, fall back to MASt3R."""
        # Try VGGT first (official facebookresearch/vggt API)
        try:
            import torch
            from vggt.models.vggt import VGGT

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
            self._dtype = dtype
            self._backend = "vggt"
            logger.info("VGGT loaded from facebookresearch/vggt")
            return
        except ImportError as e:
            logger.info(f"VGGT not installed: {e}, trying MASt3R fallback")
        except Exception as e:
            logger.info(f"VGGT load failed: {e}, trying MASt3R fallback")

        # Fall back to MASt3R
        try:
            import mast3r.utils.path_to_dust3r  # noqa: F401
            from mast3r.model import AsymmetricMASt3R

            self._mast3r_model = AsymmetricMASt3R.from_pretrained(
                "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                cache_dir=settings.model_cache_dir,
            ).to(self.device)
            self._backend = "mast3r"
            logger.info("VGGT: falling back to MASt3R")
        except ImportError as e:
            logger.warning(f"MASt3R not installed either (need recursive clone): {e}")
        except Exception as e:
            logger.warning(f"MASt3R load failed: {e}")

    def reconstruct(self, frames_dir: Path, sample_interval: int | None = None) -> tuple[dict, list[dict]]:
        """Reconstruct 3D point cloud from video frames.

        Args:
            frames_dir: directory containing frame images
            sample_interval: use every Nth frame (default from config)

        Returns:
            (pointcloud_dict, camera_poses_list)
        """
        if self.model is None and self._mast3r_model is None:
            raise RuntimeError(
                "Neither VGGT nor MASt3R is available. "
                "Install VGGT: pip install transformers (with meta/VGGT weights) "
                "or MASt3R: git clone --recursive https://github.com/naver/mast3r.git"
            )

        if self._backend == "vggt":
            return self._reconstruct_vggt(frames_dir, sample_interval)
        elif self._backend == "mast3r":
            return self._reconstruct_mast3r(frames_dir, sample_interval)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _reconstruct_vggt(self, frames_dir: Path, sample_interval: int | None) -> tuple[dict, list[dict]]:
        """VGGT 3D reconstruction — feed-forward transformer, ~0.2s."""
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        sample_interval = sample_interval or settings.mast3r_sample_interval
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        sampled_paths = frame_paths[::sample_interval]

        if len(sampled_paths) < 2:
            raise RuntimeError("Not enough frames for VGGT reconstruction (< 2 after sampling)")

        try:
            # Load and preprocess images using official VGGT API
            image_paths = [str(p) for p in sampled_paths]
            images = load_and_preprocess_images(image_paths).to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    predictions = self.model(images)

            # Extract point cloud and camera poses from VGGT outputs
            points_3d, colors = self._extract_vggt_pointcloud(predictions, image_paths)
            camera_poses = self._extract_vggt_poses(predictions, sampled_paths, sample_interval)

            return {"points": points_3d.tolist(), "colors": colors.tolist()}, camera_poses
        except Exception as e:
            logger.error(f"VGGT reconstruction failed: {e}, falling back to MASt3R")
            # If VGGT fails at runtime, try MASt3R
            if self._mast3r_model is not None:
                old_backend = self._backend
                self._backend = "mast3r"
                try:
                    return self._reconstruct_mast3r(frames_dir, sample_interval)
                except Exception:
                    self._backend = old_backend
            raise RuntimeError(f"VGGT reconstruction failed: {e}")

    def _extract_vggt_pointcloud(self, predictions, image_paths: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract point cloud from VGGT predictions."""
        from PIL import Image as PILImage
        import torch

        all_points = []
        all_colors = []

        # VGGT outputs point maps per view
        pointmaps = predictions.get('point') if isinstance(predictions, dict) else getattr(predictions, 'point', None)
        if pointmaps is None:
            pointmaps = predictions.get('point_map') if isinstance(predictions, dict) else getattr(predictions, 'point_map', None)

        if pointmaps is not None:
            if isinstance(pointmaps, torch.Tensor):
                pointmaps = pointmaps.cpu().numpy()

            for i in range(pointmaps.shape[0]):
                pts = pointmaps[i]  # [H, W, 3] or [3, H, W]
                if pts.shape[0] == 3:
                    pts = pts.transpose(1, 2, 0)  # [H, W, 3]

                pts_flat = pts.reshape(-1, 3)
                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
                pts_valid = pts_flat[valid]

                # Get colors from source image
                if i < len(image_paths):
                    img = PILImage.open(image_paths[i]).convert("RGB")
                    img_np = np.array(img.resize((pts.shape[1], pts.shape[0])))
                    img_flat = img_np.reshape(-1, 3)
                    colors_valid = img_flat[valid]
                else:
                    colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128

                all_points.append(pts_valid)
                all_colors.append(colors_valid)

        if not all_points:
            return np.zeros((0, 3)), np.zeros((0, 3))

        points = np.vstack(all_points)
        colors = np.vstack(all_colors).astype(np.uint8)

        # Downsample
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        return points, colors

    def _extract_vggt_poses(self, predictions, frame_paths: list,
                            sample_interval: int) -> list[dict]:
        """Extract camera poses from VGGT predictions."""
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        poses = []

        # Extract camera parameters from predictions
        camera_params = predictions.get('camera') if isinstance(predictions, dict) else getattr(predictions, 'camera', None)

        if camera_params is not None:
            # Convert pose encoding to extrinsic/intrinsic matrices
            with torch.no_grad():
                extrinsic, intrinsic = pose_encoding_to_extri_intri(camera_params, (512, 512))

            if isinstance(extrinsic, torch.Tensor):
                extrinsic = extrinsic.cpu().numpy()
            if isinstance(intrinsic, torch.Tensor):
                intrinsic = intrinsic.cpu().numpy()

            for i in range(len(frame_paths)):
                if i < extrinsic.shape[0]:
                    RT = extrinsic[i]
                    K = intrinsic[i] if intrinsic.ndim == 3 else intrinsic
                else:
                    RT = np.eye(4)
                    K = estimate_intrinsics(512, 512)

                R = RT[:3, :3]
                position = rt_matrix_to_position(R, RT[:3, 3])
                rotation = rt_matrix_to_quaternion(R)

                poses.append({
                    "frame_idx": i * sample_interval,
                    "intrinsics": K.tolist() if isinstance(K, np.ndarray) else K,
                    "extrinsics": RT.tolist(),
                    "position": tuple(float(x) for x in position),
                    "rotation": tuple(float(x) for x in rotation),
                })
        else:
            # Fallback: identity poses
            for i in range(len(frame_paths)):
                K = estimate_intrinsics(512, 512)
                RT = np.eye(4)
                position = rt_matrix_to_position(RT[:3, :3], RT[:3, 3])
                rotation = rt_matrix_to_quaternion(RT[:3, :3])

                poses.append({
                    "frame_idx": i * sample_interval,
                    "intrinsics": K.tolist(),
                    "extrinsics": RT.tolist(),
                    "position": tuple(float(x) for x in position),
                    "rotation": tuple(float(x) for x in rotation),
                })

        return poses

    # ── MASt3R fallback methods ──────────────────────────────────

    def _reconstruct_mast3r(self, frames_dir: Path, sample_interval: int | None) -> tuple[dict, list[dict]]:
        """MASt3R fallback reconstruction."""
        import torch
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        sample_interval = sample_interval or settings.mast3r_sample_interval
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        sampled_paths = frame_paths[::sample_interval]

        if len(sampled_paths) < 2:
            raise RuntimeError("Not enough frames for MASt3R reconstruction (< 2 after sampling)")

        # Load images
        images = load_images([str(p) for p in sampled_paths], size=512)

        # Try sparse global alignment (MASt3R-SfM)
        try:
            return self._reconstruct_mast3r_sfm(images, sampled_paths, sample_interval)
        except Exception as e:
            logger.warning(f"MASt3R sparse_global_alignment failed: {e}, falling back to pairwise")

        # Fallback: pairwise inference
        from mast3r.image_pairs import make_pairs
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._mast3r_model, self.device, batch_size=settings.batch_size, verbose=False)

        points_3d, colors = self._extract_pointcloud(output, images)
        camera_poses = self._extract_camera_poses(output, images, sample_interval)

        return {"points": points_3d.tolist(), "colors": colors.tolist()}, camera_poses

    def _reconstruct_mast3r_sfm(self, images: list, frame_paths: list,
                                sample_interval: int) -> tuple[dict, list[dict]]:
        """Full MASt3R-SfM pipeline with sparse global alignment."""
        import tempfile
        from mast3r.image_pairs import make_pairs
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            scene = sparse_global_alignment(
                imgs=images, pairs_in=pairs, output_dir=tmpdir,
                model=self._mast3r_model, device=self.device,
            )
            points_3d, colors = self._extract_scene_pointcloud(scene, images)
            camera_poses = self._extract_scene_poses(scene, images, sample_interval)

        return {"points": points_3d.tolist(), "colors": colors.tolist()}, camera_poses

    def _extract_scene_pointcloud(self, scene, images: list) -> tuple[np.ndarray, np.ndarray]:
        import torch
        all_points, all_colors = [], []

        for i, view in enumerate(scene.get("pts3d", [])):
            pts = view if isinstance(view, np.ndarray) else view.cpu().numpy()
            pts_flat = pts.reshape(-1, 3)
            valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
            pts_valid = pts_flat[valid]

            if i < len(images):
                img_data = images[i].get("img")
                if img_data is not None:
                    img_np = img_data.cpu().numpy() if isinstance(img_data, torch.Tensor) else img_data
                    if img_np.ndim == 3:
                        img_flat = img_np.reshape(-1, 3)
                        colors_valid = (img_flat[valid] * 255).clip(0, 255).astype(np.uint8) if len(img_flat) >= valid.shape[0] else np.ones((len(pts_valid), 3), dtype=np.uint8) * 128
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

        points, colors = np.vstack(all_points), np.vstack(all_colors).astype(np.uint8)
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points, colors = points[indices], colors[indices]

        return points, colors

    def _extract_scene_poses(self, scene, images: list, sample_interval: int) -> list[dict]:
        poses = []
        h, w = 512, 512
        K = estimate_intrinsics(w, h)

        cameras = scene.get("cameras", scene.get("views", []))
        for i, cam in enumerate(cameras):
            if hasattr(cam, "camera_pose"):
                RT = cam.camera_pose
            elif isinstance(cam, dict) and "camera_pose" in cam:
                RT = np.array(cam["camera_pose"])
            else:
                RT = np.eye(4)

            R = RT[:3, :3]
            position = rt_matrix_to_position(R, RT[:3, 3])
            rotation = rt_matrix_to_quaternion(R)

            poses.append({
                "frame_idx": i * sample_interval,
                "intrinsics": K.tolist(),
                "extrinsics": RT.tolist(),
                "position": tuple(float(x) for x in position),
                "rotation": tuple(float(x) for x in rotation),
            })

        return poses

    def _extract_pointcloud(self, output: dict, images: list) -> tuple[np.ndarray, np.ndarray]:
        import torch
        all_points, all_colors = [], []

        for key in ["pred1", "pred2"]:
            if key not in output:
                continue
            view = output[key]
            if "pts3d" not in view:
                continue

            pts = view["pts3d"].cpu().numpy()
            conf = view.get("conf")
            if conf is not None:
                conf = conf.cpu().numpy()

            for b in range(pts.shape[0]):
                pts_b = pts[b]
                pts_flat = pts_b.reshape(-1, 3)
                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)

                if conf is not None:
                    conf_b = conf[b].reshape(-1)
                    valid = valid & (conf_b > 0.5)

                pts_valid = pts_flat[valid]

                img_data = None
                for img in images:
                    if img.get("img") is not None:
                        img_data = img["img"]
                        break

                if img_data is not None:
                    img_np = img_data.cpu().numpy() if isinstance(img_data, torch.Tensor) else img_data
                    if img_np.ndim == 3:
                        img_flat = img_np.reshape(-1, 3)
                        colors_valid = (img_flat[valid] * 255).clip(0, 255).astype(np.uint8) if len(img_flat) == valid.shape[0] else np.ones((len(pts_valid), 3), dtype=np.uint8) * 128
                    else:
                        colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128
                else:
                    colors_valid = np.ones((len(pts_valid), 3), dtype=np.uint8) * 128

                all_points.append(pts_valid)
                all_colors.append(colors_valid)

        if not all_points:
            return np.zeros((0, 3)), np.zeros((0, 3))

        points, colors = np.vstack(all_points), np.vstack(all_colors).astype(np.uint8)
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points, colors = points[indices], colors[indices]

        return points, colors

    def _extract_camera_poses(self, output: dict, images: list, sample_interval: int) -> list[dict]:
        poses = []
        h, w = 512, 512

        for i, img_data in enumerate(images):
            K = estimate_intrinsics(w, h)

            if "pred1" in output:
                pts = output["pred1"]["pts3d"].cpu().numpy()
                if pts.ndim == 4:
                    pts_flat = pts[i % pts.shape[0]].reshape(-1, 3)
                else:
                    pts_flat = pts.reshape(-1, 3)

                valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
                if valid.any():
                    center = pts_flat[valid].mean(axis=0)
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


# Backward-compatible alias
MASt3RReconstructor = VGGTReconstructor


def get_reconstructor() -> VGGTReconstructor:
    """Get or create reconstructor instance."""
    global _reconstructor
    if _reconstructor is None:
        _reconstructor = VGGTReconstructor()
    return _reconstructor
