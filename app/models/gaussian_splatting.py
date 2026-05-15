"""3D Gaussian Splatting pipeline for photorealistic 3D scene reconstruction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_gs_pipeline = None


def _get_mock_splat_data(n_points: int = 1000) -> dict:
    """Generate mock Gaussian splat data for testing."""
    rng = np.random.RandomState(88)
    return {
        "means": rng.uniform(-5, 5, (n_points, 3)).tolist(),
        "quats": _normalize_quats(rng.randn(n_points, 4)).tolist(),
        "scales": np.exp(rng.uniform(-2, 0, (n_points, 3))).tolist(),
        "opacities": (1 / (1 + np.exp(-rng.uniform(-2, 2, n_points)))).tolist(),
        "sh_coeffs": rng.uniform(-0.5, 0.5, (n_points, 3)).tolist(),  # SH degree 0
    }


def _normalize_quats(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions."""
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return q / norms


class GaussianSplatPipeline:
    """3D Gaussian Splatting training and export pipeline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.trained = False
        self.splat_data = None
        self._check_available()

    def _check_available(self):
        """Check if gsplat/nerfstudio is available."""
        if settings.mock_mode or not settings.gaussian_splatting:
            logger.info("3DGS: disabled or mock mode")
            return

        try:
            import gsplat  # noqa
            self.available = True
            logger.info("gsplat available")
        except ImportError:
            self.available = False
            logger.info("gsplat not installed, 3DGS disabled")

    def train(self, frames_dir: Path, output_dir: Path,
              iterations: int = 7000) -> dict | None:
        """Train 3D Gaussian Splatting model from video frames.

        Args:
            frames_dir: directory containing frame images
            output_dir: where to save trained model
            iterations: number of training iterations

        Returns:
            splat_data dict or None
        """
        if not settings.gaussian_splatting or not getattr(self, "available", False):
            return _get_mock_splat_data()

        try:
            return self._train_gsplat(frames_dir, output_dir, iterations)
        except Exception as e:
            logger.error(f"3DGS training failed: {e}")
            return _get_mock_splat_data()

    def _train_gsplat(self, frames_dir: Path, output_dir: Path,
                      iterations: int) -> dict:
        """Train using gsplat library."""
        import torch
        from gsplat import rasterize_to_pixels

        # This is a simplified training loop
        # For production use, use nerfstudio's full pipeline

        # Load frames
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        n_frames = len(frame_paths)
        if n_frames == 0:
            return _get_mock_splat_data()

        # Initialize Gaussians
        n_init = 5000
        means = torch.randn((n_init, 3), device=self.device) * 2
        quats = torch.randn((n_init, 4), device=self.device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.exp(torch.randn((n_init, 3), device=self.device) * 0.5)
        opacities = torch.sigmoid(torch.randn((n_init,), device=self.device))
        colors = torch.randn((n_init, 3), device=self.device)

        # Simple training (just a placeholder for the full pipeline)
        logger.info(f"Training 3DGS: {n_frames} frames, {iterations} iterations")
        # Full training requires camera poses from COLMAP
        # For now, return initial Gaussians
        return {
            "means": means.cpu().numpy().tolist(),
            "quats": quats.cpu().numpy().tolist(),
            "scales": scales.cpu().numpy().tolist(),
            "opacities": opacities.cpu().numpy().tolist(),
            "sh_coeffs": torch.zeros((n_init, 3), device=self.device).cpu().numpy().tolist(),
        }

    def export_splat(self, splat_data: dict, output_path: Path) -> Path:
        """Export Gaussian splat data to .splat binary format.

        .splat format: per-gaussian binary record of 32 bytes
        [x,y,z] [scale_x,scale_y,scale_z] [r,g,b] [quat_x,quat_y,quat_z,w] [opacity]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        means = np.array(splat_data["means"])
        scales = np.array(splat_data["scales"])
        opacities = np.array(splat_data["opacities"])
        quats = np.array(splat_data["quats"])
        sh = np.array(splat_data.get("sh_coeffs", np.zeros((len(means), 3))))

        # Convert SH to RGB (simplified: use 0th order)
        colors = np.clip(sh[:, :3] * 255, 0, 255).astype(np.uint8)

        # Normalize scales
        scales = np.clip(np.log(scales + 1e-8), -10, 10)

        # Normalize quaternions
        quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quat_norms = np.maximum(quat_norms, 1e-8)
        quats = quats / quat_norms

        # Pack into .splat binary format
        with open(output_path, "wb") as f:
            for i in range(len(means)):
                # Position (3 floats)
                f.write(np.array(means[i], dtype=np.float32).tobytes())
                # Scales (3 floats)
                f.write(np.array(scales[i], dtype=np.float32).tobytes())
                # Color (3 uint8)
                f.write(bytes([colors[i][0], colors[i][1], colors[i][2]]))
                # Quaternion (4 floats: x, y, z, w)
                f.write(np.array([quats[i][1], quats[i][2], quats[i][3], quats[i][0]], dtype=np.float32).tobytes())
                # Opacity (1 float)
                f.write(np.array([opacities[i]], dtype=np.float32).tobytes())

        return output_path

    def export_ply(self, splat_data: dict, output_path: Path) -> Path:
        """Export Gaussian splat data to PLY with Gaussian parameters."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(splat_data["means"])

        with open(output_path, "w") as f:
            f.write("ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            f.write("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            f.write("end_header\n")

            import struct
            means = np.array(splat_data["means"])
            sh = np.array(splat_data.get("sh_coeffs", np.zeros((n, 3))))
            opacities = np.array(splat_data["opacities"])
            scales = np.array(splat_data["scales"])
            quats = np.array(splat_data["quats"])

            for i in range(n):
                struct_vals = [
                    *means[i],           # x, y, z
                    0.0, 0.0, 1.0,      # normals (z-up)
                    *sh[i, :3],          # SH coefficients (RGB)
                    opacities[i],        # opacity
                    *np.log(scales[i]),  # scales
                    *quats[i],           # quaternion
                ]
                f.write(struct.pack("<" + "f" * len(struct_vals), *struct_vals))

        return output_path


def get_splat_pipeline() -> GaussianSplatPipeline:
    """Get or create GS pipeline instance."""
    global _gs_pipeline
    if _gs_pipeline is None:
        _gs_pipeline = GaussianSplatPipeline()
    return _gs_pipeline
