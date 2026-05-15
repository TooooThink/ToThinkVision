"""Splat Exporter — 3D Gaussian Splatting format for Unity/UE real-time rendering."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from app.config import settings
from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat

logger = logging.getLogger(__name__)


class SplatExporter(BaseExporter):
    """Exports 3D Gaussian splat data to .splat and .ply formats.

    .splat — compact binary format used by UnityGaussianSplatting and UnrealSplat
    .ply — Gaussian parameters with spherical harmonics (standard format)
    """

    def __init__(self, fmt: ExportFormat = ExportFormat.UNITY_SPLAT):
        self.fmt = fmt
        if fmt in (ExportFormat.UNITY_SPLAT, ExportFormat.UE_SPLAT):
            self.format_name = fmt.value
            self.file_extension = ".splat"
            self.mime_type = "application/octet-stream"
        else:
            self.format_name = "splat_ply"
            self.file_extension = ".ply"
            self.mime_type = "application/x-ply"

    def export(self, data: StructuredOutput) -> Path:
        if data.gaussian_splats is not None:
            return self._export_from_splat_data(data)
        elif data.point_cloud is not None:
            return self._export_from_pointcloud(data)
        else:
            raise ValueError("No 3D data available for splat export. Process video with 3DGS or MASt3R first.")

    def _export_from_splat_data(self, data: StructuredOutput) -> Path:
        """Export from Gaussian Splat data."""
        splat = data.gaussian_splats.model_dump()
        if self.file_extension == ".splat":
            return self._write_splat_binary(splat, data)
        else:
            return self._write_ply_with_gaussians(splat, data)

    def _export_from_pointcloud(self, data: StructuredOutput) -> Path:
        """Export from MASt3R point cloud (no Gaussians, just points)."""
        pc = data.point_cloud
        points = np.array(pc.points)
        colors = np.array(pc.colors) if pc.colors else None

        if self.file_extension == ".splat":
            # Convert point cloud to simple splats
            n = len(points)
            splat_data = {
                "means": points.tolist(),
                "quats": _identity_quats(n),
                "scales": np.full((n, 3), 0.01).tolist(),
                "opacities": np.full(n, 0.8).tolist(),
                "sh_coeffs": colors.tolist() if colors is not None else np.zeros((n, 3)).tolist(),
            }
            return self._write_splat_binary(splat_data, data)
        else:
            return self._write_ply_from_pointcloud(pc, data)

    def _write_splat_binary(self, splat_data: dict, data: StructuredOutput) -> Path:
        """Write .splat binary format.

        Each splat record (32 bytes):
        [x,y,z] float32   - position
        [s0,s1,s2] float32 - log-scales
        [r,g,b] uint8     - color
        [qx,qy,qz,qw] float32 - rotation
        [opacity] float32
        """
        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        means = np.array(splat_data["means"], dtype=np.float32)
        quats = np.array(splat_data["quats"], dtype=np.float32)
        scales = np.array(splat_data["scales"], dtype=np.float32)
        opacities = np.array(splat_data["opacities"], dtype=np.float32)
        sh = np.array(splat_data.get("sh_coeffs", []), dtype=np.float32)

        n = len(means)

        # Convert SH coeffs to RGB if needed
        if sh.ndim == 2 and sh.shape[1] >= 3:
            # Use 0th-order SH for base color
            colors = (sh[:, :3] * 255).clip(0, 255).astype(np.uint8)
        elif sh.ndim == 1:
            colors = np.full((n, 3), 128, dtype=np.uint8)
        else:
            colors = np.full((n, 3), 128, dtype=np.uint8)

        # Clamp scales
        scales = np.clip(np.log(np.abs(scales) + 1e-8), -10, 2)

        # Normalize quaternions
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        quats = quats / norms

        with open(out_path, "wb") as f:
            for i in range(n):
                # Position (3 x float32 = 12 bytes)
                f.write(means[i].tobytes())
                # Scales (3 x float32 = 12 bytes)
                f.write(scales[i].tobytes())
                # Color (3 x uint8 = 3 bytes)
                f.write(bytes(colors[i].tolist()))
                # Quaternion (4 x float32 = 16 bytes) — order: x, y, z, w
                f.write(np.array([quats[i, 1], quats[i, 2], quats[i, 3], quats[i, 0]], dtype=np.float32).tobytes())
                # Opacity (1 x float32 = 4 bytes)
                f.write(np.array([opacities[i]], dtype=np.float32).tobytes())

        logger.info(f"Splat binary written: {out_path} ({n} splats, {out_path.stat().st_size / 1024:.1f} KB)")
        return out_path

    def _write_ply_with_gaussians(self, splat_data: dict, data: StructuredOutput) -> Path:
        """Write PLY with full Gaussian parameters (for 3DGS import)."""
        out_path = self._output_path(data.source_file, suffix=".gauss.ply")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        means = np.array(splat_data["means"])
        quats = np.array(splat_data["quats"])
        scales = np.array(splat_data["scales"])
        opacities = np.array(splat_data["opacities"])
        sh = np.array(splat_data.get("sh_coeffs", []))

        n = len(means)
        sh_degree = max(0, int((sh.shape[1] if sh.ndim > 1 else 0) / 3) - 1) if n > 0 else 0
        n_sh_coeffs = (sh_degree + 1) ** 2 * 3 if sh_degree > 0 else 3

        with open(out_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
            for i in range(min(3, n_sh_coeffs)):
                f.write(f"property float f_dc_{i}\n".encode())
            for i in range(min(45, n_sh_coeffs)):
                f.write(f"property float f_rest_{i}\n".encode())
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")

            for i in range(n):
                vals = [
                    *means[i],           # x, y, z
                    0.0, 0.0, 1.0,      # normals (z-up)
                    *(sh[i, :3] if len(sh) > i else [0.0, 0.0, 0.0]),  # SH DC
                    *([0.0] * min(45, n_sh_coeffs - 3)),  # SH rest
                    float(opacities[i]),
                    *np.log(np.abs(scales[i]) + 1e-8),
                    *quats[i],
                ]
                f.write(struct.pack("<" + "f" * len(vals), *vals))

        return out_path

    def _write_ply_from_pointcloud(self, pc, data: StructuredOutput) -> Path:
        """Write simple PLY from point cloud (no Gaussian params)."""
        from app.utils.pointcloud import save_ply

        out_path = self._output_path(data.source_file, suffix=".pointcloud.ply")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        points = np.array(pc.points)
        colors = np.array(pc.colors) if pc.colors else None

        save_ply(out_path, points, colors)
        return out_path


def _identity_quats(n: int) -> list:
    """Generate identity quaternions."""
    return [[1.0, 0.0, 0.0, 0.0]] * n
