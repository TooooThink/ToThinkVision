"""Spann3R — 3D Reconstruction with Spatial Memory (3DV 2025).

Built on DUSt3R, Spann3R adds a spatial memory mechanism that enables
incremental 3D reconstruction from long image sequences. It maintains
a memory of previously seen 3D structures and uses them to guide
alignment of new views, reducing drift in long sequences.

GitHub: https://github.com/HengyiWang/Spann3R
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

logger = logging.getLogger(__name__)


class Spann3RReconstructor:
    """Wrapper for Spann3R 3D reconstruction with spatial memory.

    Spann3R extends DUSt3R with:
    - Spatial memory: stores 3D structures from processed views
    - Incremental alignment: new views aligned against memory, reducing drift
    - Better camera pose estimation for long sequences
    - NeRF/3DGS-compatible output (transform.json)

    Falls back to error if not available.
    """

    def __init__(
        self,
        repo_path: str | None = None,
        device: str = "cuda",
    ):
        """Initialize Spann3R.

        Args:
            repo_path: path to Spann3R repository (auto-detect if None)
            device: torch device string
        """
        self.device = device
        self.repo_path = self._find_repo(repo_path)
        self.available = self.repo_path is not None

        if not self.available:
            logger.warning(
                "Spann3R not found. Clone from https://github.com/HengyiWang/Spann3R "
                "and set SPANN3R_PATH env var, or place under models/Spann3R/"
            )

    def _find_repo(self, repo_path: str | None) -> Path | None:
        """Find Spann3R repository."""
        import os

        if repo_path:
            p = Path(repo_path)
            if p.exists():
                return p

        # Check env var
        env_path = os.environ.get("SPANN3R_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # Check common locations
        for candidate in [
            Path("models/Spann3R"),
            Path("../Spann3R"),
            Path("~/Spann3R").expanduser(),
        ]:
            if candidate.exists():
                return candidate

        return None

    def reconstruct(
        self,
        frames_dir: Path,
        sample_interval: int = 10,
        output_dir: Path | None = None,
        use_spatial_memory: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Reconstruct 3D scene from image frames.

        Args:
            frames_dir: directory containing frame images
            sample_interval: use every Nth frame
            output_dir: where to save results
            use_spatial_memory: enable spatial memory for better long-sequence alignment

        Returns:
            (pointcloud_dict, camera_poses_list)
        """
        if not self.available:
            raise RuntimeError(
                "Spann3R not available. Clone from https://github.com/HengyiWang/Spann3R "
                "and set SPANN3R_PATH env var, or place under models/Spann3R/"
            )

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="spann3r_"))

        output_dir = Path(output_dir).resolve()  # Convert to absolute path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare input directory with sampled frames
        input_dir = output_dir / "input"
        input_dir.mkdir(exist_ok=True)

        frame_paths = sorted(
            list(Path(frames_dir).glob("*.jpg"))
            + list(Path(frames_dir).glob("*.png"))
        )
        sampled_paths = frame_paths[::sample_interval]

        if len(sampled_paths) < 2:
            raise RuntimeError("Not enough frames for Spann3R (< 2 after sampling)")

        # Symlink sampled frames
        for i, fp in enumerate(sampled_paths):
            dst = input_dir / f"{i:06d}.jpg"
            if not dst.exists():
                dst.symlink_to(fp.resolve())

        # Run Spann3R demo script
        demo_script = self.repo_path / "demo.py"
        if not demo_script.exists():
            # Try alternative entry point
            demo_script = self.repo_path / "run.py"
            if not demo_script.exists():
                raise RuntimeError("Spann3R demo script not found")

        cmd = [
            "python", str(demo_script),
            "--demo_path", str(input_dir),
            "--save_path", str(output_dir),
            "--kf_every", str(1),  # Already sampled, use all provided frames
            "--save_ori",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Spann3R failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Spann3R timed out")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Spann3R error: {e}")

        # Spann3R saves output to workspace/demo_name/, where demo_name is the last part of demo_path
        # We passed --demo_path as input_dir, so output is in output_dir/input/
        demo_name = input_dir.name
        actual_output_dir = output_dir / demo_name

        if actual_output_dir.exists():
            logger.info("Spann3R output found in subdirectory: %s", actual_output_dir)
            output_dir = actual_output_dir
        else:
            logger.warning("Expected Spann3R output in %s, checking parent directory", actual_output_dir)

        # Log what files were generated
        logger.info("Spann3R output directory: %s", output_dir)
        all_files = list(output_dir.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]  # Only files
        logger.info("Spann3R generated %d files:", len(all_files))
        for f in all_files:
            logger.info("  - %s (%.2f MB)", f.relative_to(output_dir), f.stat().st_size / 1024 / 1024)

        # Parse outputs
        pointcloud = self._load_pointcloud(output_dir)
        camera_poses = self._load_camera_poses(output_dir, sample_interval)

        if pointcloud is None or camera_poses is None:
            raise RuntimeError(f"Spann3R produced no valid output (pointcloud={pointcloud is not None}, camera_poses={camera_poses is not None})")

        return pointcloud, camera_poses

    def _load_pointcloud(self, output_dir: Path) -> dict[str, Any] | None:
        """Load point cloud from Spann3R output."""
        # Try transform.json first (Nerfstudio-compatible format)
        transform_path = output_dir / "transform.json"
        if not transform_path.exists():
            # Look in subdirectories
            candidates = list(output_dir.rglob("transform.json"))
            if candidates:
                transform_path = candidates[0]
                logger.info("Found transform.json in subdirectory: %s", transform_path)

        if transform_path.exists():
            try:
                with open(transform_path) as f:
                    transform_data = json.load(f)
                logger.info("Successfully loaded transform.json")
                return self._parse_nerfstudio_pointcloud(transform_data, output_dir)
            except Exception as e:
                logger.warning("Failed to parse transform.json: %s", e)
        else:
            logger.warning("No transform.json found in %s", output_dir)

        # Try .ply file
        ply_files = list(output_dir.rglob("*.ply"))
        if ply_files:
            logger.info("Found %d PLY files, trying first: %s", len(ply_files), ply_files[0])
            try:
                result = self._load_ply(ply_files[0])
                if result:
                    logger.info("Successfully loaded PLY with %d points", len(result["points"]))
                return result
            except Exception as e:
                logger.warning("Failed to load PLY: %s", e)
        else:
            logger.warning("No PLY files found in %s", output_dir)

        # Try numpy point cloud
        npz_files = list(output_dir.rglob("*.npz"))
        logger.info("Found %d NPZ files", len(npz_files))
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                if "points" in data or "pts3d" in data:
                    pts_key = "points" if "points" in data else "pts3d"
                    points = data[pts_key]
                    colors = data.get("colors", np.ones_like(points) * 128)
                    logger.info("Loaded NPZ with %d points", len(points))
                    return {"points": points.tolist(), "colors": colors.tolist()}
            except Exception:
                continue

        logger.error("No valid point cloud found in %s", output_dir)
        return None

    def _parse_nerfstudio_pointcloud(
        self, transform_data: dict, output_dir: Path
    ) -> dict[str, Any] | None:
        """Parse Nerfstudio-format transform.json for point cloud data."""
        # transform.json primarily contains camera poses; point cloud may be separate
        # Check for accompanying point cloud files
        for ext in [".ply", ".npy", ".npz"]:
            candidates = list(output_dir.rglob(f"*{ext}"))
            if candidates:
                try:
                    if ext == ".ply":
                        return self._load_ply(candidates[0])
                    elif ext == ".npy":
                        points = np.load(candidates[0])
                        return {"points": points.tolist(), "colors": (np.ones_like(points) * 128).tolist()}
                except Exception:
                    continue

        # If only transform.json, we have poses but no point cloud
        return None

    def _load_ply(self, ply_path: Path) -> dict[str, Any] | None:
        """Load point cloud from PLY file."""
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        # PlyElement supports direct field access: vertex["x"], vertex["y"], etc.
        # Also get the underlying numpy structured array for dtype inspection
        vertex_array = vertex.data

        x = np.asarray(vertex_array["x"])
        y = np.asarray(vertex_array["y"])
        z = np.asarray(vertex_array["z"])
        points = np.stack([x, y, z], axis=-1)

        colors = np.ones_like(points) * 128
        field_names = vertex_array.dtype.names if vertex_array.dtype.names else []
        if "red" in field_names:
            colors[:, 0] = vertex_array["red"]
            colors[:, 1] = vertex_array["green"]
            colors[:, 2] = vertex_array["blue"]

        # Downsample if too many points
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        return {"points": points.tolist(), "colors": colors.astype(np.uint8).tolist()}

    def _load_ply_simple(self, ply_path: Path) -> dict[str, Any] | None:
        """Simple PLY file parser (ASCII and binary_little_endian)."""
        with open(ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            # Parse header
            num_vertices = 0
            is_binary = False
            for hl in header_lines:
                if hl.startswith("element vertex"):
                    num_vertices = int(hl.split()[-1])
                if "binary_little_endian" in hl:
                    is_binary = True

            if is_binary:
                # Read binary: assume float32 x,y,z + uchar r,g,b
                raw = f.read()
                points = []
                colors = []
                offset = 0
                for _ in range(num_vertices):
                    if offset + 15 > len(raw):
                        break
                    x = np.frombuffer(raw, dtype=np.float32, count=1, offset=offset)[0]
                    y = np.frombuffer(raw, dtype=np.float32, count=1, offset=offset + 4)[0]
                    z = np.frombuffer(raw, dtype=np.float32, count=1, offset=offset + 8)[0]
                    r = raw[offset + 12]
                    g = raw[offset + 13]
                    b = raw[offset + 14]
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    offset += 15  # 3*float32 + 3*uchar
            else:
                # ASCII
                points = []
                colors = []
                for _ in range(num_vertices):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    parts = line.split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
                        else:
                            colors.append([128, 128, 128])

        if not points:
            return None

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)

        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        return {"points": points.tolist(), "colors": colors.tolist()}

    def _load_camera_poses(
        self, output_dir: Path, sample_interval: int
    ) -> list[dict[str, Any]] | None:
        """Load camera poses from Spann3R output."""
        # Try transforms.json (Nerfstudio format) - note the 's' at the end
        transform_path = output_dir / "transforms.json"
        if not transform_path.exists():
            candidates = list(output_dir.rglob("transforms.json"))
            if candidates:
                transform_path = candidates[0]
                logger.info("Found transforms.json in subdirectory: %s", transform_path)

        if transform_path.exists():
            try:
                with open(transform_path) as f:
                    transform_data = json.load(f)
                poses = self._parse_nerfstudio_poses(transform_data, sample_interval)
                if poses:
                    logger.info("Successfully loaded %d camera poses from transforms.json", len(poses))
                return poses
            except Exception as e:
                logger.warning("Failed to parse camera poses from transforms.json: %s", e)
        else:
            logger.warning("No transforms.json found for camera poses in %s", output_dir)

        # Try numpy camera data
        npz_files = list(output_dir.rglob("*.npz"))
        logger.info("Found %d NPZ files for camera poses", len(npz_files))
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                if "poses" in data or "camera_poses" in data:
                    poses_key = "poses" if "poses" in data else "camera_poses"
                    poses_arr = data[poses_key]
                    logger.info("Loading camera poses from NPZ key: %s", poses_key)
                    return self._array_to_poses(poses_arr, sample_interval)
            except Exception:
                continue

        logger.error("No valid camera poses found in %s", output_dir)
        return None

    def _parse_nerfstudio_poses(
        self, transform_data: dict, sample_interval: int
    ) -> list[dict[str, Any]]:
        """Parse Nerfstudio transform.json camera poses."""
        poses = []
        frames = transform_data.get("frames", [])

        # Get intrinsics from transform
        fl_x = transform_data.get("fl_x", transform_data.get("focal", 500))
        fl_y = transform_data.get("fl_y", fl_x)
        cx = transform_data.get("cx", 320)
        cy = transform_data.get("cy", 240)
        w = transform_data.get("w", 640)
        h = transform_data.get("h", 480)

        K = np.array([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1],
        ])

        for i, frame in enumerate(frames):
            transform_matrix = frame.get("transform_matrix", np.eye(4).tolist())
            if isinstance(transform_matrix, list):
                T = np.array(transform_matrix)
            else:
                T = np.eye(4)

            R = T[:3, :3]
            position = rt_matrix_to_position(R, T[:3, 3])
            rotation = rt_matrix_to_quaternion(R)

            poses.append({
                "frame_idx": i * sample_interval,
                "intrinsics": K.tolist(),
                "extrinsics": T.tolist(),
                "position": tuple(float(x) for x in position),
                "rotation": tuple(float(x) for x in rotation),
            })

        return poses

    def _array_to_poses(
        self, poses_arr: np.ndarray, sample_interval: int
    ) -> list[dict[str, Any]]:
        """Convert numpy pose array to standard pose dicts."""
        poses = []
        if poses_arr.ndim == 2:
            # (N*4, 4) or (N, 12) etc
            n = poses_arr.shape[0] // 4 if poses_arr.shape[1] == 4 else poses_arr.shape[0]
            for i in range(n):
                T = np.eye(4)
                if poses_arr.shape[1] == 4:
                    T = poses_arr[i * 4: (i + 1) * 4]
                elif poses_arr.shape[1] >= 12:
                    T[:3, :] = poses_arr[i, :12].reshape(3, 4)
                R = T[:3, :3]
                position = rt_matrix_to_position(R, T[:3, 3])
                rotation = rt_matrix_to_quaternion(R)
                poses.append({
                    "frame_idx": i * sample_interval,
                    "intrinsics": estimate_intrinsics(640, 480).tolist(),
                    "extrinsics": T.tolist(),
                    "position": tuple(float(x) for x in position),
                    "rotation": tuple(float(x) for x in rotation),
                })
        elif poses_arr.ndim == 3:
            # (N, 4, 4)
            for i in range(poses_arr.shape[0]):
                T = poses_arr[i]
                R = T[:3, :3]
                position = rt_matrix_to_position(R, T[:3, 3])
                rotation = rt_matrix_to_quaternion(R)
                poses.append({
                    "frame_idx": i * sample_interval,
                    "intrinsics": estimate_intrinsics(640, 480).tolist(),
                    "extrinsics": T.tolist(),
                    "position": tuple(float(x) for x in position),
                    "rotation": tuple(float(x) for x in rotation),
                })
        return poses


# Global instance
_spann3r: Spann3RReconstructor | None = None


def get_spann3r() -> Spann3RReconstructor:
    """Get or create a global Spann3R instance."""
    global _spann3r
    if _spann3r is None:
        _spann3r = Spann3RReconstructor()
    return _spann3r
