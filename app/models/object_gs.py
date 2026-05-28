"""ObjectGS — Object-aware 3D Gaussian Splatting (ICCV 2025).

Per-object 3D reconstruction using Gaussian Splatting with semantic
constraints. Each object gets its own set of Gaussians, enabling
independent editing, moving, and deletion.

GitHub: https://github.com/RuijieZhu94/ObjectGS
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ObjectGSPipeline:
    """Wrapper for ObjectGS per-object 3D Gaussian Splatting."""

    def __init__(
        self,
        repo_path: str | None = None,
        device: str = "cuda",
    ):
        """Initialize ObjectGS.

        Args:
            repo_path: path to ObjectGS repository (auto-detect if None)
            device: torch device string
        """
        self.device = device
        self.repo_path = self._find_repo(repo_path)
        self.available = self.repo_path is not None

        if not self.available:
            logger.warning(
                "ObjectGS not found. Clone from https://github.com/RuijieZhu94/ObjectGS "
                "and set OBJECT_GS_PATH env var, or place under models/ObjectGS/"
            )

    def _find_repo(self, repo_path: str | None) -> Path | None:
        """Find ObjectGS repository."""
        import os

        if repo_path:
            p = Path(repo_path)
            if p.exists():
                return p

        # Check env var
        env_path = os.environ.get("OBJECT_GS_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # Check common locations
        for candidate in [
            Path("models/ObjectGS"),
            Path("../ObjectGS"),
            Path("~/ObjectGS").expanduser(),
        ]:
            if candidate.exists():
                return candidate

        return None

    def train(
        self,
        frame_dir: Path,
        masks_dir: Path | None = None,
        output_dir: Path | None = None,
        num_iterations: int = 5000,
        use_2dgs: bool = False,
    ) -> dict[str, Any]:
        """Train per-object 3D Gaussians.

        Args:
            frame_dir: directory with input images
            masks_dir: directory with per-object segmentation masks
            output_dir: where to save results
            num_iterations: training iterations
            use_2dgs: use 2D Gaussian Splatting variant

        Returns:
            dict with:
                - "scene_mesh": path to combined scene mesh
                - "object_meshes": dict {object_id: path}
                - "gaussians": dict {object_id: GaussianSplatData}
        """
        if not self.available:
            raise RuntimeError(
                "ObjectGS not available. Clone from https://github.com/RuijieZhu94/ObjectGS "
                "and set OBJECT_GS_PATH env var, or place under models/ObjectGS/"
            )

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="objectgs_"))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data directory structure
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Copy/link frames
        scene_dir = data_dir / "scene"
        scene_dir.mkdir(exist_ok=True)

        frame_paths = sorted(
            list(Path(frame_dir).glob("*.jpg"))
            + list(Path(frame_dir).glob("*.png"))
        )

        for i, fp in enumerate(frame_paths):
            dst = scene_dir / f"{i:06d}.jpg"
            if not dst.exists():
                dst.symlink_to(fp.resolve())

        # Run training
        script = "train_2d.sh" if use_2dgs else "train_3d.sh"
        script_path = self.repo_path / script

        if not script_path.exists():
            raise RuntimeError(f"ObjectGS training script not found: {script_path}")

        try:
            result = subprocess.run(
                ["bash", str(script_path), str(data_dir)],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"ObjectGS training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("ObjectGS training timed out")
        except RuntimeError:
            raise

        # Export per-object meshes
        object_meshes = self._export_object_meshes(output_dir)

        # Export scene mesh
        scene_mesh = self._export_scene_mesh(output_dir)

        return {
            "scene_mesh": scene_mesh,
            "object_meshes": object_meshes,
            "output_dir": output_dir,
        }

    def _export_object_meshes(self, output_dir: Path) -> dict[str, Path]:
        """Export per-object meshes from trained model."""
        export_script = self.repo_path / "export_object_mesh.py"

        if not export_script.exists():
            logger.warning("ObjectGS export script not found")
            return {}

        try:
            # Export all objects (label_id = -1)
            result = subprocess.run(
                [
                    "python",
                    str(export_script),
                    "-m", str(output_dir),
                    "--query_label_id", "-1",
                ],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.warning("Object mesh export failed: %s", result.stderr)
                return {}

        except Exception as e:
            logger.warning("Object mesh export error: %s", e)
            return {}

        # Find exported meshes
        meshes = {}
        mesh_dir = output_dir / "object_meshes"
        if mesh_dir.exists():
            for mesh_file in mesh_dir.glob("*.obj"):
                obj_id = mesh_file.stem
                meshes[obj_id] = mesh_file

        return meshes

    def _export_scene_mesh(self, output_dir: Path) -> Path | None:
        """Export combined scene mesh."""
        export_script = self.repo_path / "export_mesh.py"

        if not export_script.exists():
            return None

        try:
            result = subprocess.run(
                [
                    "python",
                    str(export_script),
                    "-m", str(output_dir),
                ],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                return None

            scene_mesh = output_dir / "scene_mesh.obj"
            return scene_mesh if scene_mesh.exists() else None

        except Exception:
            return None


# Global instance
_objectgs_pipeline: ObjectGSPipeline | None = None


def get_objectgs_pipeline() -> ObjectGSPipeline:
    """Get or create a global ObjectGS instance."""
    global _objectgs_pipeline
    if _objectgs_pipeline is None:
        _objectgs_pipeline = ObjectGSPipeline()
    return _objectgs_pipeline
