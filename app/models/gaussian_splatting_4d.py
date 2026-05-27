"""4D Gaussian Splatting with HexPlane time-space decomposition.

Extends 3D Gaussian Splatting to handle temporal dynamics:
- Per-object 4D Gaussians with time-dependent deformation
- HexPlane encoding: decomposes 4D into 6 2D plane grids (xy, xz, xt, yz, yt, zt)
- Scene = background_3dgs + sum(per_object_4dgs)

Based on: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" (Wu et al., 2023)

Requirements: torch, gsplat (optional), nerfstudio (optional)
This module is heavy and disabled by default (enable_4dgs=False).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas import GaussianSplat4D, PipelineConfig

logger = logging.getLogger(__name__)

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    pass

_HAS_GSPLAT = False
try:
    import gsplat
    _HAS_GSPLAT = True
except ImportError:
    pass


class HexPlaneGrid:
    """HexPlane: 6-plane decomposition of 4D space.

    Decomposes 4D (x,y,z,t) into six 2D planes:
    (x,y), (x,z), (x,t), (y,z), (y,t), (z,t)

    Each plane has a learnable feature grid. Features are combined
    via MLP to produce Gaussian deformation parameters.
    """

    def __init__(
        self,
        grid_resolution: int = 128,
        feature_dim: int = 8,
        bbox_min: tuple = (-5, -5, -5, 0),
        bbox_max: tuple = (5, 5, 5, 10),
        device: str = "cuda",
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for HexPlaneGrid")

        self.resolution = grid_resolution
        self.feature_dim = feature_dim
        self.bbox_min = np.array(bbox_min, dtype=np.float32)
        self.bbox_max = np.array(bbox_max, dtype=np.float32)
        self.device = device

        # 6 plane grids: xy, xz, xt, yz, yt, zt
        # Each is (feature_dim, resolution, resolution)
        self.planes = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, grid_resolution, grid_resolution) * 0.1)
            for _ in range(6)
        ]).to(device)

        # Plane axis pairs
        self.plane_axes = [
            (0, 1),  # xy
            (0, 2),  # xz
            (0, 3),  # xt
            (1, 2),  # yz
            (1, 3),  # yt
            (2, 3),  # zt
        ]

        # Feature decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 3 for position offset, 3 for scale offset
        ).to(device)

    def encode(self, xyz: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Encode 4D coordinates into HexPlane features.

        Args:
            xyz: (N, 3) spatial coordinates
            t: (N, 1) temporal coordinates

        Returns:
            features: (N, feature_dim * 6) concatenated plane features
        """
        # Normalize to [0, 1]
        xytz = torch.cat([xyz, t], dim=-1)  # (N, 4)
        xytz_norm = (xytz - torch.tensor(self.bbox_min[:4], device=self.device)) / \
                     (torch.tensor(self.bbox_max[:4], device=self.device) - torch.tensor(self.bbox_min[:4], device=self.device))
        xytz_norm = xytz_norm.clamp(0, 1) * 2 - 1  # [-1, 1] for grid_sample

        plane_features = []
        for i, (ax0, ax1) in enumerate(self.plane_axes):
            # Sample from plane grid
            coords = torch.stack([
                xytz_norm[:, ax1],
                xytz_norm[:, ax0],
            ], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

            feat = F.grid_sample(
                self.planes[i].unsqueeze(0),  # (1, C, R, R)
                coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, C, 1, N)
            plane_features.append(feat.squeeze(0).squeeze(1).T)  # (N, C)

        return torch.cat(plane_features, dim=-1)  # (N, C*6)

    def get_deformation(self, xyz: "torch.Tensor", t: "torch.Tensor") -> tuple:
        """Get position and scale deformation at given 4D coordinates.

        Returns:
            pos_offset: (N, 3) position deformation
            scale_offset: (N, 3) scale deformation
        """
        features = self.encode(xyz, t)
        output = self.decoder(features)  # (N, 6)
        pos_offset = output[:, :3]
        scale_offset = output[:, 3:]
        return pos_offset, scale_offset

    def parameters(self):
        """Return all learnable parameters."""
        params = list(self.planes.parameters())
        params += list(self.decoder.parameters())
        return params


class GaussianSplat4DPipeline:
    """4D Gaussian Splatting training and inference pipeline."""

    def __init__(self, device: str = "cuda", config: PipelineConfig | None = None):
        self.device = device
        self.config = config or PipelineConfig()

        if not _HAS_TORCH:
            logger.warning("PyTorch not available, 4DGS will use mock mode")
            self.available = False
        elif not torch.cuda.is_available():
            logger.warning("CUDA not available, 4DGS will use mock mode")
            self.available = False
        else:
            self.available = True

    def train(
        self,
        frame_dir: Path,
        camera_poses: list[dict[str, Any]],
        per_object_masks: dict[str, list[np.ndarray]] | None = None,
        output_dir: Path | None = None,
        num_iterations: int = 5000,
    ) -> dict[str, GaussianSplat4D]:
        """Train per-object 4D Gaussians.

        Args:
            frame_dir: Directory with extracted frames
            camera_poses: Camera poses per frame
            per_object_masks: Per-object segmentation masks per frame
            output_dir: Where to save results
            num_iterations: Training iterations

        Returns:
            Dict mapping object_id → GaussianSplat4D
        """
        if not self.available:
            logger.info("4DGS not available, returning mock data")
            return self._mock_result()

        from PIL import Image

        # Load frames
        frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if not frame_paths:
            logger.warning("No frames found in %s", frame_dir)
            return {}

        # Load first frame to get dimensions
        first_frame = np.array(Image.open(frame_paths[0]).convert("RGB"))
        H, W = first_frame.shape[:2]

        # Initialize per-object Gaussians from point clouds
        # For now, use a simplified approach: train a single 4DGS scene
        # and decompose by object masks during rendering

        # Initialize HexPlane
        bbox_min = (-5, -5, -5, 0)
        bbox_max = (5, 5, 5, len(frame_paths) / 30.0)  # time range based on frames/fps
        hexplane = HexPlaneGrid(
            grid_resolution=128,
            feature_dim=8,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            device=self.device,
        )

        # Initialize Gaussians from first frame depth (simplified)
        # In a full implementation, this would use COLMAP sparse point cloud
        n_init = 10000
        means = torch.randn(n_init, 3, device=self.device) * 2
        quats = torch.randn(n_init, 4, device=self.device)
        quats = F.normalize(quats, dim=-1)
        scales = torch.ones(n_init, 3, device=self.device) * 0.01
        opacities = torch.ones(n_init, 1, device=self.device) * 0.9
        sh_coeffs = torch.randn(n_init, 3, device=self.device) * 0.1 + 0.5

        # Optimizable parameters
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        sh_coeffs.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [means, quats, scales, opacities, sh_coeffs] + hexplane.parameters(),
            lr=0.001,
        )

        # Load all frames as tensors
        frames = []
        for fp in frame_paths[:min(50, len(frame_paths))]:  # Limit for memory
            img = np.array(Image.open(fp).convert("RGB")).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).to(self.device))

        logger.info("Training 4DGS: %d Gaussians, %d frames, %d iterations",
                     n_init, len(frames), num_iterations)

        # Simplified training loop (full version would use differentiable rasterization)
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Sample random time
            t_val = torch.rand(1, device=self.device) * bbox_max[3]
            t = t_val.expand(n_init, 1)

            # Get deformation from HexPlane
            pos_offset, scale_offset = hexplane.get_deformation(means, t)

            # Deformed Gaussians
            deformed_means = means + pos_offset * 0.1
            deformed_scales = torch.exp(scales + scale_offset * 0.1)

            # Simple photometric loss (placeholder for full rasterization)
            frame_idx = min(int(t_val.item() / bbox_max[3] * len(frames)), len(frames) - 1)
            target = frames[frame_idx]

            # Mock rendering: project Gaussians and compare
            loss = self._simple_render_loss(deformed_means, deformed_scales, sh_coeffs, opacities, target, H, W)

            loss.backward()
            optimizer.step()

            if iteration % 500 == 0:
                logger.info("4DGS iteration %d/%d, loss=%.4f", iteration, num_iterations, loss.item())

        # Export results
        with torch.no_grad():
            result = GaussianSplat4D(
                means=means.cpu().numpy().tolist(),
                quats=quats.cpu().numpy().tolist(),
                scales=scales.cpu().numpy().tolist(),
                opacities=opacities.squeeze().cpu().numpy().tolist(),
                sh_coeffs=sh_coeffs.cpu().numpy().tolist(),
                temporal_coeffs=[],  # HexPlane features exported separately
                time_range=(0.0, bbox_max[3]),
                num_gaussians=n_init,
            )

        # Save checkpoint
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "means": means.cpu().numpy(),
                "quats": quats.cpu().numpy(),
                "scales": scales.cpu().numpy(),
                "opacities": opacities.cpu().numpy(),
                "sh_coeffs": sh_coeffs.cpu().numpy(),
                "hexplane_state": {k: v.cpu().numpy() for k, v in hexplane.state_dict().items()},
            }
            np.savez_compressed(output_dir / "4dgs_checkpoint.npz", **checkpoint)
            logger.info("4DGS checkpoint saved to %s", output_dir)

        return {"scene": result}

    def _simple_render_loss(self, means, scales, sh_coeffs, opacities, target, H, W):
        """Simplified rendering loss (placeholder for full rasterization).

        In a full implementation, this would use gsplat or 3dgs for
        differentiable rasterization.
        """
        # Project Gaussians to image plane (simplified)
        # Just use a simple projection + splatting approximation
        n = means.shape[0]

        # Project to 2D (perspective)
        z = means[:, 2].clamp(min=0.1)
        fx = fy = W / 2.0
        cx, cy = W / 2.0, H / 2.0
        x2d = means[:, 0] * fx / z + cx
        y2d = means[:, 1] * fy / z + cy

        # Bin to image grid (very simplified)
        x2d_int = x2d.long().clamp(0, W - 1)
        y2d_int = y2d.long().clamp(0, H - 1)

        # Create rough image
        rendered = torch.zeros(H, W, 3, device=self.device)
        colors = sh_coeffs.clamp(0, 1)

        # Scatter colors (very rough approximation)
        for c in range(3):
            rendered.index_put_(
                (y2d_int, x2d_int),
                colors[:, c] * opacities.squeeze(),
                accumulate=True,
            )

        # Normalize
        rendered = rendered / rendered.max().clamp(min=1e-6)

        # L1 loss
        loss = F.l1_loss(rendered, target)
        return loss

    def _mock_result(self) -> dict[str, GaussianSplat4D]:
        """Return mock 4DGS result when GPU/PyTorch not available."""
        n = 1000
        return {
            "scene": GaussianSplat4D(
                means=np.random.randn(n, 3).tolist(),
                quats=np.random.randn(n, 4).tolist(),
                scales=np.random.rand(n, 3).tolist(),
                opacities=np.random.rand(n).tolist(),
                sh_coeffs=np.random.rand(n, 3).tolist(),
                temporal_coeffs=[],
                time_range=(0.0, 1.0),
                num_gaussians=n,
            ),
        }

    def export(self, gaussians_4d: dict[str, GaussianSplat4D], output_path: Path):
        """Export 4D Gaussian parameters."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        export_data = {}
        for name, gs in gaussians_4d.items():
            export_data[name] = gs.model_dump()

        with open(output_path, "w") as f:
            json.dump(export_data, f)

        logger.info("Exported 4DGS parameters to %s", output_path)
