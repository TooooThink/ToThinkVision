"""3D Gaussian Splatting pipeline for photorealistic 3D scene reconstruction."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Headless environment for COLMAP on HPC/SSH nodes (no X11/Qt display)
_COLMAP_ENV = os.environ.copy()
_COLMAP_ENV.setdefault("QT_QPA_PLATFORM", "offscreen")

_gs_pipeline = None


class GaussianSplatPipeline:
    """3D Gaussian Splatting training and export pipeline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.trained = False
        self.splat_data = None
        self._check_available()

    def _check_available(self):
        """Check if gsplat/nerfstudio is available."""
        if not settings.gaussian_splatting:
            logger.info("3DGS: disabled")
            self.available = False
            return

        try:
            import gsplat  # noqa
            import torch  # noqa
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
            return None

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._train_gsplat(frames_dir, output_dir, iterations)
        except Exception as e:
            logger.error(f"3DGS training failed: {e}")
            return None

    def _train_gsplat(self, frames_dir: Path, output_dir: Path,
                      iterations: int) -> dict:
        """Train using gsplat library with proper rendering loop.

        Full training requires camera poses. We try COLMAP first,
        then fall back to MASt3R poses if available.
        """
        import torch
        from gsplat import rasterize_to_pixels

        # ─── Load frames ─────────────────────────────────────
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        n_frames = len(frame_paths)
        if n_frames == 0:
            return None

        # Load first frame to get dimensions
        from PIL import Image as PILImage
        pil_img = PILImage.open(frame_paths[0])
        img_w, img_h = pil_img.size
        pil_img.close()

        # ─── Camera poses ────────────────────────────────────
        # Try COLMAP first
        camera_poses = self._run_colmap(frames_dir, output_dir / "colmap")

        if camera_poses is None:
            # Fallback: use MASt3R poses if available, or estimate simple poses
            logger.warning("COLMAP failed, using estimated camera poses")
            camera_poses = self._estimate_camera_poses(frame_paths, img_w, img_h)

        # ─── Initialize Gaussians ────────────────────────────
        # Initialize from point cloud center or scene bounds
        all_poses = np.array([p["extrinsics"] for p in camera_poses])
        camera_centers = all_poses[:, :3, 3]
        scene_center = camera_centers.mean(axis=0)
        scene_scale = max(1.0, np.linalg.norm(camera_centers - scene_center, axis=1).max())

        n_init = 50000  # Start with 50K Gaussians
        means = torch.randn((n_init, 3), device=self.device) * scene_scale * 0.3 + torch.tensor(scene_center, device=self.device)
        quats = torch.randn((n_init, 4), device=self.device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.log(torch.ones((n_init, 3), device=self.device) * scene_scale * 0.01)
        opacities = torch.inverse(torch.sigmoid)(torch.ones((n_init,), device=self.device) * 0.1)
        sh_dc = torch.zeros((n_init, 3), device=self.device)  # RGB base color

        # Make parameters trainable
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        sh_dc.requires_grad_(True)

        # ─── Optimizer ───────────────────────────────────────
        optimizers = {
            "means": torch.optim.Adam([means], lr=1.6e-4),
            "quats": torch.optim.Adam([quats], lr=1e-3),
            "scales": torch.optim.Adam([scales], lr=5e-3),
            "opacities": torch.optim.Adam([opacities], lr=5e-2),
            "sh_dc": torch.optim.Adam([sh_dc], lr=2.5e-3),
        }

        # ─── Load target images ──────────────────────────────
        target_images = []
        for fp in frame_paths:
            img = PILImage.open(fp).convert("RGB")
            img = img.resize((img_w, img_h))
            target_images.append(torch.tensor(np.array(img), dtype=torch.float32, device=self.device) / 255.0)

        # ─── Training loop ───────────────────────────────────
        logger.info(f"Training 3DGS: {n_frames} frames, {iterations} iterations, {n_init} initial Gaussians")

        # Track Gaussians that need densification
        radii_accum = torch.zeros(n_init, device=self.device)
        grad_accum = torch.zeros(n_init, device=self.device)
        gaussian_counts = torch.zeros(n_init, device=self.device)

        n_current = n_init  # Track current number of Gaussians

        for step in range(iterations):
            # Select random frame and camera
            frame_idx = step % n_frames
            pose = camera_poses[frame_idx]
            target_img = target_images[frame_idx]

            K = np.array(pose["intrinsics"])
            RT = np.array(pose["extrinsics"])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Convert to tensors
            viewmatrix = torch.tensor(RT, dtype=torch.float32, device=self.device)
            projmatrix = torch.tensor(K, dtype=torch.float32, device=self.device)

            # ─── Rasterize ─────────────────────────────────
            render_colors, render_alphas, info = rasterize_to_pixels(
                means[:n_current],                    # (N, 3)
                quats[:n_current],                    # (N, 4)
                scales[:n_current],                   # (N, 3)
                torch.sigmoid(opacities[:n_current]), # (N,)
                sh_dc[:n_current],                    # (N, 3)
                viewmatrix,
                projmatrix,
                img_w, img_h,
                bg_color=torch.zeros(3, device=self.device),
                packed=False,
            )

            # ─── Compute loss ──────────────────────────────
            loss = torch.nn.functional.l1_loss(render_colors, target_img)

            # Add SSIM-like structural loss for sharper details
            loss += 0.2 * (1 - self._ssim_approx(render_colors, target_img))

            # ─── Backprop ──────────────────────────────────
            loss.backward()

            # ─── Optimizer step ────────────────────────────
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad()

            # ─── Adaptive density control ──────────────────
            if step > 500 and step % 100 == 0:
                # Get radii from rasterization info
                if "radii" in info:
                    radii = info["radii"]
                    valid_mask = radii > 0
                    grad_accum[:n_current][valid_mask] += torch.abs(torch.cat([
                        means.grad[:n_current][valid_mask, :3].abs().max(dim=1).values,
                        scales.grad[:n_current][valid_mask, :3].abs().max(dim=1).values,
                    ]))
                    gaussian_counts[:n_current][valid_mask] += 1

                n_current = self._densify(
                    means, quats, scales, opacities, sh_dc,
                    optimizers, grad_accum, gaussian_counts,
                    n_current, step, iterations
                )

            # Log progress
            if step % 500 == 0:
                logger.info(f"3DGS step {step}/{iterations}, loss={loss.item():.4f}, Gaussians={n_current}")

        # ─── Return final parameters ────────────────────────
        logger.info(f"3DGS training complete: {n_current} Gaussians after {iterations} steps")

        with torch.no_grad():
            return {
                "means": means[:n_current].cpu().numpy().tolist(),
                "quats": quats[:n_current].cpu().numpy().tolist(),
                "scales": torch.exp(scales[:n_current]).cpu().numpy().tolist(),
                "opacities": torch.sigmoid(opacities[:n_current]).cpu().numpy().tolist(),
                "sh_coeffs": sh_dc[:n_current].cpu().numpy().tolist(),
            }

    def _densify(self, means, quats, scales, opacities, sh_dc,
                 optimizers, grad_accum, gaussian_counts,
                 n_current, step, total_steps):
        """Adaptive density control: clone and split Gaussians."""
        with torch.no_grad():
            valid = gaussian_counts[:n_current] > 0
            avg_grad = grad_accum[:n_current][valid] / gaussian_counts[:n_current][valid].clamp(min=1)

            # Clone under-reconstructed Gaussians (high gradient, small scale)
            small_scales = torch.exp(scales[:n_current][valid]).max(dim=-1).values < 0.01
            to_clone = valid.clone()
            to_clone[valid] = small_scales & (avg_grad > 0.0002)

            # Split over-reconstructed Gaussians (high gradient, large scale)
            large_scales = torch.exp(scales[:n_current][valid]).max(dim=-1).values > 0.01
            to_split = valid.clone()
            to_split[valid] = large_scales & (avg_grad > 0.0002)

            n_clone = to_clone.sum().item()
            n_split = to_split.sum().item()

            if n_clone + n_split == 0:
                # Reset accumulators
                grad_accum[:n_current] = 0
                gaussian_counts[:n_current] = 0
                return n_current

            # Clone Gaussians
            clone_ids = torch.where(to_clone)[0]
            if n_clone > 0:
                new_means = means[clone_ids].clone()
                new_quats = quats[clone_ids].clone()
                new_scales = scales[clone_ids].clone()
                new_opacities = opacities[clone_ids].clone()
                new_sh_dc = sh_dc[clone_ids].clone()

                means.data = torch.cat([means.data, new_means])
                quats.data = torch.cat([quats.data, new_quats])
                scales.data = torch.cat([scales.data, new_scales])
                opacities.data = torch.cat([opacities.data, new_opacities])
                sh_dc.data = torch.cat([sh_dc.data, new_sh_dc])

            # Split Gaussians (replace with two smaller ones)
            split_ids = torch.where(to_split)[0]
            if n_split > 0:
                old_scales = torch.exp(scales[split_ids])
                new_s1 = old_scales / 1.6
                new_s2 = old_scales / 1.6

                split_means1 = means[split_ids] + torch.randn_like(means[split_ids]) * new_s1
                split_means2 = means[split_ids] + torch.randn_like(means[split_ids]) * new_s2

                means.data[split_ids] = split_means1
                scales.data[split_ids] = torch.log(new_s1)

                new_split_means = split_means2
                new_split_scales = torch.log(new_s2)
                new_split_quats = quats[split_ids].clone()
                new_split_opacities = opacities[split_ids].clone()
                new_split_sh_dc = sh_dc[split_ids].clone()

                means.data = torch.cat([means.data, new_split_means])
                quats.data = torch.cat([quats.data, new_split_quats])
                scales.data = torch.cat([scales.data, new_split_scales])
                opacities.data = torch.cat([opacities.data, new_split_opacities])
                sh_dc.data = torch.cat([sh_dc.data, new_split_sh_dc])

            # Prune low-opacity Gaussians
            opacity_mask = torch.sigmoid(opacities[:n_current + n_clone + n_split * 2]) > 0.005
            keep = torch.where(opacity_mask)[0]

            if len(keep) < n_current + n_clone + n_split * 2:
                means.data = means.data[keep].clone()
                quats.data = quats.data[keep].clone()
                scales.data = scales.data[keep].clone()
                opacities.data = opacities.data[keep].clone()
                sh_dc.data = sh_dc.data[keep].clone()

                # Reset accumulators
                new_n = len(keep)
                grad_accum.resize_(new_n).zero_()
                gaussian_counts.resize_(new_n).zero_()

                return new_n

            return n_current + n_clone + n_split * 2

    @staticmethod
    def _ssim_approx(pred, target, window_size=11):
        """Approximate SSIM loss for perceptual quality."""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_pred = pred.mean(dim=(0, 1), keepdim=True)
        mu_target = target.mean(dim=(0, 1), keepdim=True)

        sigma_pred = ((pred - mu_pred) ** 2).mean(dim=(0, 1), keepdim=True)
        sigma_target = ((target - mu_target) ** 2).mean(dim=(0, 1), keepdim=True)
        sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean(dim=(0, 1), keepdim=True)

        ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
                   ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        return ssim_map.mean()

    def _run_colmap(self, frames_dir: Path, output_dir: Path) -> list[dict] | None:
        """Run COLMAP to estimate camera poses from frames.

        Returns list of {intrinsics, extrinsics, position, rotation} or None if failed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        database = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        try:
            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--ImageReader.camera_model", "PINHOLE",
                "--ImageReader.single_camera", "1",
            ], check=True, capture_output=True, text=True, timeout=120, env=_COLMAP_ENV)

            # Feature matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database),
            ], check=True, capture_output=True, text=True, timeout=300, env=_COLMAP_ENV)

            # Sparse reconstruction
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--output_path", str(sparse_dir),
            ], check=True, capture_output=True, text=True, timeout=600, env=_COLMAP_ENV)

            # Export to text format
            model_path = sparse_dir / "0"
            if not model_path.exists():
                return None

            return self._parse_colmap_model(model_path, frames_dir)

        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning(f"COLMAP failed: {e}")
            return None

    def _parse_colmap_model(self, model_path: Path, frames_dir: Path) -> list[dict] | None:
        """Parse COLMAP sparse model into camera poses."""
        try:
            import pycolmap
        except ImportError:
            # Fallback: parse text format manually
            return self._parse_colmap_text(model_path)

        try:
            reconstruction = pycolmap.Reconstruction(str(model_path))
            poses = []
            for image_id, image in reconstruction.images.items():
                cam = reconstruction.cameras[image.camera_id]
                K = cam.calibration_matrix()
                R = image.rotmat()
                T = image.tvec
                extrinsics = np.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T

                from app.utils.camera import rt_matrix_to_position, rt_matrix_to_quaternion

                poses.append({
                    "frame_idx": image_id,
                    "intrinsics": K.tolist(),
                    "extrinsics": extrinsics.tolist(),
                    "position": tuple(float(x) for x in rt_matrix_to_position(R, T)),
                    "rotation": tuple(float(x) for x in rt_matrix_to_quaternion(R)),
                })
            return poses
        except Exception as e:
            logger.warning(f"pycolmap parse failed: {e}")
            return self._parse_colmap_text(model_path)

    def _parse_colmap_text(self, model_path: Path) -> list[dict] | None:
        """Parse COLMAP text format (cameras.txt, images.txt)."""
        cameras_file = model_path / "cameras.txt"
        images_file = model_path / "images.txt"

        if not cameras_file.exists() or not images_file.exists():
            return None

        # Parse cameras
        cameras = {}
        with open(cameras_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                fx = float(parts[4])
                fy = float(parts[5]) if len(parts) > 5 else fx
                cx = float(parts[6]) if len(parts) > 6 else fx
                cy = float(parts[7]) if len(parts) > 7 else fy
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                cameras[cam_id] = K

        # Parse images
        poses = []
        from app.utils.camera import rt_matrix_to_position, rt_matrix_to_quaternion

        with open(images_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                cam_id = int(parts[8])
                image_name = parts[9]

                # Convert quaternion to rotation matrix
                q = np.array([qw, qx, qy, qz])
                q = q / np.linalg.norm(q)
                R = self._quat_to_rotmat(q)

                T = np.array([tx, ty, tz])
                extrinsics = np.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T

                K = cameras.get(cam_id, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                poses.append({
                    "frame_idx": image_id,
                    "intrinsics": K,
                    "extrinsics": extrinsics.tolist(),
                    "position": tuple(float(x) for x in rt_matrix_to_position(R, T)),
                    "rotation": tuple(float(x) for x in rt_matrix_to_quaternion(R)),
                })

        return poses if poses else None

    @staticmethod
    def _quat_to_rotmat(q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])

    def _estimate_camera_poses(self, frame_paths: list[Path], w: int, h: int) -> list[dict]:
        """Estimate simple camera poses when COLMAP is unavailable.

        Uses a circular arc assumption: camera orbits around scene center.
        """
        from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

        K = estimate_intrinsics(w, h)
        n = len(frame_paths)
        poses = []

        for i in range(n):
            t = i / max(n - 1, 1)
            angle = t * np.pi * 0.5  # 90 degree arc
            radius = 3.0

            # Camera position on arc
            cx = radius * np.sin(angle)
            cy = 0.5
            cz = radius * np.cos(angle)

            # Look-at rotation
            target = np.array([0, 0, 0])
            camera_pos = np.array([cx, cy, cz])
            forward = (target - camera_pos)
            forward /= np.linalg.norm(forward)

            right = np.cross(forward, np.array([0, 1, 0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)

            R = np.column_stack([right, up, -forward])
            T = -R.T @ camera_pos

            extrinsics = np.eye(4)
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = T

            poses.append({
                "frame_idx": i,
                "intrinsics": K.tolist(),
                "extrinsics": extrinsics.tolist(),
                "position": tuple(float(x) for x in rt_matrix_to_position(R, T)),
                "rotation": tuple(float(x) for x in rt_matrix_to_quaternion(R)),
            })

        return poses

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

        import struct

        with open(output_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
            f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")

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
                    *np.log(scales[i] + 1e-8),  # log scales
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
