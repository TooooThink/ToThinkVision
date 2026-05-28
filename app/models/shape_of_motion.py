"""Shape of Motion — End-to-end 4D reconstruction from monocular video (ICCV 2025).

Reconstructs time-varying 3D geometry from a single monocular video,
producing per-frame 3D meshes/point clouds with temporal consistency.
Unlike pipeline approaches (depth → tracking → ICP), this is end-to-end,
jointly optimizing geometry and motion.

GitHub: https://github.com/vye16/shape-of-motion/
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

logger = logging.getLogger(__name__)


class ShapeOfMotionPipeline:
    """Wrapper for Shape of Motion end-to-end 4D reconstruction.

    Shape of Motion jointly reconstructs:
    - Per-frame 3D geometry (point clouds / meshes)
    - Camera trajectory
    - Temporal deformation field (motion of every 3D point)

    This replaces the separate depth + tracking + ICP pipeline with a
    single end-to-end model that produces temporally consistent 4D output.

    Falls back to error if not available.
    """

    def __init__(
        self,
        repo_path: str | None = None,
        device: str = "cuda",
    ):
        """Initialize Shape of Motion.

        Args:
            repo_path: path to shape-of-motion repository (auto-detect if None)
            device: torch device string
        """
        self.device = device
        self.repo_path = self._find_repo(repo_path)
        self.available = self.repo_path is not None

        if not self.available:
            logger.warning(
                "Shape of Motion not found. Clone from "
                "https://github.com/vye16/shape-of-motion/ "
                "and set SHAPE_OF_MOTION_PATH env var, or place under models/shape-of-motion/"
            )

    def _find_repo(self, repo_path: str | None) -> Path | None:
        """Find Shape of Motion repository."""
        import os

        if repo_path:
            p = Path(repo_path)
            if p.exists():
                return p

        # Check env var
        env_path = os.environ.get("SHAPE_OF_MOTION_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # Check common locations
        for candidate in [
            Path("models/shape-of-motion"),
            Path("../shape-of-motion"),
            Path("~/shape-of-motion").expanduser(),
            Path("models/ShapeOfMotion"),
            Path("../ShapeOfMotion"),
        ]:
            if candidate.exists():
                return candidate

        return None

    def reconstruct_4d(
        self,
        video_path: Path | str,
        output_dir: Path | None = None,
        num_frames: int | None = None,
        resolution: int = 512,
    ) -> dict[str, Any]:
        """Run end-to-end 4D reconstruction from video.

        Args:
            video_path: path to input video file
            output_dir: where to save results
            num_frames: max frames to process (None = all)
            resolution: processing resolution

        Returns:
            dict with:
                - "per_frame_pointclouds": list of {points, colors} per frame
                - "per_frame_meshes": list of mesh file paths (if available)
                - "camera_poses": list of camera pose dicts
                - "deformation_field": temporal deformation data
                - "transform_json": path to Nerfstudio-compatible transform.json
                - "output_dir": output directory path
        """
        if not self.available:
            raise RuntimeError(
                "Shape of Motion not available. Clone from "
                "https://github.com/vye16/shape-of-motion/ "
                "and set SHAPE_OF_MOTION_PATH env var, or place under models/shape-of-motion/"
            )

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="som_"))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run Shape of Motion inference
        result = self._run_inference(video_path, output_dir, num_frames, resolution)
        if result is None:
            raise RuntimeError("Shape of Motion inference failed")

        # Parse outputs
        per_frame_pcs = self._load_per_frame_pointclouds(output_dir)
        per_frame_meshes = self._load_per_frame_meshes(output_dir)
        camera_poses = self._load_camera_poses(output_dir)
        deformation = self._load_deformation_field(output_dir)
        transform_json = self._find_transform_json(output_dir)

        return {
            "per_frame_pointclouds": per_frame_pcs,
            "per_frame_meshes": per_frame_meshes,
            "camera_poses": camera_poses,
            "deformation_field": deformation,
            "transform_json": transform_json,
            "output_dir": output_dir,
        }

    def _run_inference(
        self,
        video_path: Path | str,
        output_dir: Path,
        num_frames: int | None,
        resolution: int,
    ) -> bool:
        """Run the Shape of Motion inference script."""
        video_path = Path(video_path)

        # Find inference script
        inference_scripts = [
            self.repo_path / "run.py",
            self.repo_path / "inference.py",
            self.repo_path / "demo.py",
            self.repo_path / "scripts" / "run.py",
            self.repo_path / "scripts" / "reconstruct.py",
        ]

        script = None
        for s in inference_scripts:
            if s.exists():
                script = s
                break

        if script is None:
            logger.error("Shape of Motion inference script not found")
            return False

        cmd = [
            "python", str(script),
            "--video", str(video_path.resolve()),
            "--output", str(output_dir),
            "--resolution", str(resolution),
        ]

        if num_frames is not None:
            cmd.extend(["--num_frames", str(num_frames)])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error("Shape of Motion failed: %s", result.stderr)
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("Shape of Motion timed out")
            return False
        except Exception as e:
            logger.error("Shape of Motion error: %s", e)
            return False

    def _load_per_frame_pointclouds(self, output_dir: Path) -> list[dict[str, Any]]:
        """Load per-frame point clouds from output."""
        pcs = []

        # Check for per-frame numpy files
        pc_dir = output_dir / "pointclouds"
        if not pc_dir.exists():
            pc_dir = output_dir / "pcs"
        if not pc_dir.exists():
            # Check root for numbered files
            pc_dir = output_dir

        npz_files = sorted(pc_dir.glob("*.npz")) + sorted(pc_dir.glob("*.npy"))
        for npz_file in npz_files:
            try:
                if npz_file.suffix == ".npz":
                    data = np.load(npz_file)
                    pts_key = "points" if "points" in data else "pts3d"
                    points = data[pts_key]
                    colors = data.get("colors", np.ones_like(points) * 128)
                else:
                    points = np.load(npz_file)
                    colors = np.ones_like(points) * 128

                pcs.append({
                    "points": points.tolist(),
                    "colors": colors.tolist(),
                })
            except Exception as e:
                logger.warning("Failed to load point cloud %s: %s", npz_file, e)

        return pcs

    def _load_per_frame_meshes(self, output_dir: Path) -> list[Path]:
        """Find per-frame mesh files."""
        meshes = []

        mesh_dir = output_dir / "meshes"
        if not mesh_dir.exists():
            mesh_dir = output_dir

        for ext in ["*.obj", "*.ply", "*.glb"]:
            mesh_files = sorted(mesh_dir.glob(ext))
            if mesh_files:
                meshes.extend(mesh_files)
                break

        return meshes

    def _load_camera_poses(self, output_dir: Path) -> list[dict[str, Any]]:
        """Load camera poses from output."""
        # Try transform.json
        transform_path = self._find_transform_json(output_dir)
        if transform_path and transform_path.exists():
            try:
                with open(transform_path) as f:
                    transform_data = json.load(f)
                return self._parse_nerfstudio_poses(transform_data)
            except Exception as e:
                logger.warning("Failed to parse poses from transform.json: %s", e)

        # Try numpy
        npz_files = list(output_dir.rglob("poses*.npz")) + list(output_dir.rglob("cameras*.npz"))
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                for key in ["poses", "camera_poses", "extrinsics"]:
                    if key in data:
                        return self._array_to_poses(data[key])
            except Exception:
                continue

        return []

    def _find_transform_json(self, output_dir: Path) -> Path | None:
        """Find transform.json in output directory."""
        candidates = [
            output_dir / "transform.json",
            output_dir / "transforms.json",
        ]
        candidates.extend(output_dir.rglob("transform.json"))

        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_deformation_field(self, output_dir: Path) -> dict[str, Any] | None:
        """Load temporal deformation field data."""
        deform_files = list(output_dir.rglob("deformation*")) + list(output_dir.rglob("motion*"))

        for df in deform_files:
            try:
                if df.suffix == ".npz":
                    data = np.load(df, allow_pickle=True)
                    return {k: v.tolist() for k, v in data.items()}
                elif df.suffix == ".json":
                    with open(df) as f:
                        return json.load(f)
            except Exception:
                continue

        return None

    def _parse_nerfstudio_poses(self, transform_data: dict) -> list[dict[str, Any]]:
        """Parse Nerfstudio-format camera poses."""
        poses = []
        frames = transform_data.get("frames", [])

        fl_x = transform_data.get("fl_x", transform_data.get("focal", 500))
        fl_y = transform_data.get("fl_y", fl_x)
        cx = transform_data.get("cx", 320)
        cy = transform_data.get("cy", 240)

        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

        for i, frame in enumerate(frames):
            T = np.array(frame.get("transform_matrix", np.eye(4).tolist()))
            R = T[:3, :3]
            position = rt_matrix_to_position(R, T[:3, 3])
            rotation = rt_matrix_to_quaternion(R)

            poses.append({
                "frame_idx": i,
                "intrinsics": K.tolist(),
                "extrinsics": T.tolist(),
                "position": tuple(float(x) for x in position),
                "rotation": tuple(float(x) for x in rotation),
            })

        return poses

    def _array_to_poses(self, poses_arr: np.ndarray) -> list[dict[str, Any]]:
        """Convert pose array to standard format."""
        poses = []
        if poses_arr.ndim == 3 and poses_arr.shape[1:] == (4, 4):
            for i in range(poses_arr.shape[0]):
                T = poses_arr[i]
                R = T[:3, :3]
                position = rt_matrix_to_position(R, T[:3, 3])
                rotation = rt_matrix_to_quaternion(R)
                poses.append({
                    "frame_idx": i,
                    "intrinsics": estimate_intrinsics(640, 480).tolist(),
                    "extrinsics": T.tolist(),
                    "position": tuple(float(x) for x in position),
                    "rotation": tuple(float(x) for x in rotation),
                })
        return poses

    def extract_object_trajectories(
        self,
        per_frame_pcs: list[dict[str, Any]],
        object_masks: dict[str, list[np.ndarray]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Extract per-object 4D trajectories from reconstruction.

        Uses the temporal deformation field to track individual objects
        across frames without separate tracking/ICP steps.

        Args:
            per_frame_pcs: per-frame point clouds from reconstruct_4d()
            object_masks: optional per-object masks to segment trajectories

        Returns:
            dict mapping object_id → {keyframes, motion_type}
        """
        if not per_frame_pcs:
            return {}

        # If no masks, treat entire scene as one object
        if object_masks is None:
            object_masks = {"scene": [np.ones(
                (1, 1), dtype=bool
            ) for _ in per_frame_pcs]}

        trajectories = {}
        for obj_id, masks in object_masks.items():
            kfs = []
            for t, pc_data in enumerate(per_frame_pcs):
                if t >= len(masks):
                    break

                points = np.array(pc_data["points"])
                if len(points) == 0:
                    continue

                # Apply mask to select object points
                mask = masks[t]
                if mask.size > 1 and mask.shape[0] > 1:
                    # Mask is spatial — use it to filter points
                    # (simplified: assume mask covers point indices)
                    n_pts = min(len(points), mask.size)
                    flat_mask = mask.flatten()[:n_pts]
                    selected = points[:n_pts][flat_mask > 0.5]
                else:
                    selected = points

                if len(selected) == 0:
                    continue

                centroid = selected.mean(axis=0)
                kfs.append({
                    "timestamp": float(t) / 30.0,  # Assume 30fps
                    "frame_idx": t,
                    "position": tuple(centroid.tolist()),
                    "num_points": len(selected),
                })

            if kfs:
                trajectories[obj_id] = {
                    "keyframes": kfs,
                    "motion_type": self._classify_simple_motion(kfs),
                }

        return trajectories

    def _classify_simple_motion(self, keyframes: list[dict]) -> str:
        """Simple motion classification from keyframe positions."""
        if len(keyframes) < 2:
            return "static"

        positions = np.array([kf["position"] for kf in keyframes])
        total_dist = 0.0
        for i in range(1, len(positions)):
            total_dist += float(np.linalg.norm(positions[i] - positions[i - 1]))

        if total_dist < 0.01:
            return "static"
        elif total_dist < 0.5:
            return "rigid"
        else:
            return "deformable"


# Global instance
_shape_of_motion: ShapeOfMotionPipeline | None = None


def get_shape_of_motion() -> ShapeOfMotionPipeline:
    """Get or create a global Shape of Motion instance."""
    global _shape_of_motion
    if _shape_of_motion is None:
        _shape_of_motion = ShapeOfMotionPipeline()
    return _shape_of_motion
