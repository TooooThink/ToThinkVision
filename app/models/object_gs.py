"""ObjectGS — Object-aware 3D Gaussian Splatting (ICCV 2025).

Per-object 3D reconstruction using Gaussian Splatting with semantic
constraints. Each object gets its own set of Gaussians, enabling
independent editing, moving, and deletion.

GitHub: https://github.com/RuijieZhu94/ObjectGS
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _isolated_gpu_env() -> dict:
    """Build env dict for ObjectGS subprocesses.

    - Sets TORCH_CUDA_ARCH_LIST to A100 (sm_80) to avoid JIT compilation issues.
    - Assigns export subprocesses to GPU 1 to isolate CUDA context.
      If mesh export segfaults, GPU 0 (main pipeline) stays clean.
    """
    env = os.environ.copy()
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    import torch as _torch
    if _torch.cuda.is_available() and _torch.cuda.device_count() > 1:
        env["CUDA_VISIBLE_DEVICES"] = "1"
    return env


def _find_colmap() -> str:
    """Find the colmap binary.

    Checks (in order):
    1. ``COLMAP_BIN`` environment variable (project-level config in run script)
    2. ``colmap`` on PATH
    """
    import shutil

    # 1. Explicit env var (set in run_test_real.sh or similar)
    env_bin = os.environ.get("COLMAP_BIN")
    if env_bin:
        logger.info("COLMAP_BIN=%s, exists=%s, executable=%s",
                        env_bin, os.path.isfile(env_bin), os.access(env_bin, os.X_OK))
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin

    # 2. Standard PATH lookup
    colmap_path = shutil.which("colmap")
    if colmap_path:
        logger.info("colmap found on PATH: %s", colmap_path)
        return colmap_path

    logger.info("colmap NOT FOUND (COLMAP_BIN=%s, PATH search failed)", env_bin)
    return "colmap"  # fallback to bare name, will raise FileNotFoundError


# Headless environment for COLMAP on HPC/SSH nodes (no X11/Qt display)
_COLMAP_ENV = os.environ.copy()
_COLMAP_ENV.setdefault("QT_QPA_PLATFORM", "offscreen")
# Fix QStandardPaths error on SLURM nodes where /run/user/<uid> is not writable
_xdg_dir = _COLMAP_ENV.get("XDG_RUNTIME_DIR", "")
if not _xdg_dir or not os.path.isdir(_xdg_dir) or not os.access(_xdg_dir, os.W_OK):
    _xdg_dir = f"/tmp/runtime-{os.getuid()}"
    os.makedirs(_xdg_dir, mode=0o700, exist_ok=True)
_COLMAP_ENV["XDG_RUNTIME_DIR"] = _xdg_dir


def _is_colmap_opengl_error(stderr) -> bool:
    """Return True if the COLMAP stderr indicates an OpenGL/context failure.

    On headless HPC nodes without a GL context (no Xvfb, no EGL), COLMAP's
    GPU SIFT extraction aborts with ``Check failed: context_.create()``.

    Args:
        stderr: bytes or str — COLMAP may emit non-UTF-8 bytes (progress bars,
            binary control chars) so we always decode with errors='replace'.
    """
    if not stderr:
        return False
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    markers = ("context_.create()", "OpenGLContextManager", "opengl_utils",
               "Failed to create OpenGL", "QStandardPaths")
    return any(m in stderr for m in markers)


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

    def _run_colmap_for_scene(
        self, scene_dir: Path, images_dir: Path, frame_paths: list[Path]
    ) -> None:
        """Run COLMAP on scene images to produce sparse reconstruction.

        ObjectGS requires pre-computed COLMAP output (cameras, images, points3D)
        in ``scene_dir/sparse/0/``.  This method runs the full COLMAP SfM pipeline
        (feature extraction → matching → sparse reconstruction) and produces that
        output automatically.

        Args:
            scene_dir: directory containing symlinked images (e.g. 000000.jpg …)
            frame_paths: original image file paths (used to derive absolute symlinks
                so COLMAP can read images regardless of working directory)
        """
        sparse_dir = scene_dir / "sparse"
        # Clean old COLMAP output to avoid mixing data from different runs
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True)

        colmap_bin = _find_colmap()
        logger.info("Using COLMAP binary: %s", colmap_bin)

        # COLMAP reads images from images_dir so extr.name = "000000.jpg" (no prefix).
        # ObjectGS later joins with source_path/images/ to find the files.
        db_path = sparse_dir / "database.db"

        logger.info("Running COLMAP feature extraction on %d images…", len(frame_paths))
        try:
            subprocess.run(
                [
                    colmap_bin, "feature_extractor",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--ImageReader.camera_model", "PINHOLE",
                    "--ImageReader.single_camera", "1",
                ],
                check=True,
                capture_output=True,
                timeout=600,
                env=_COLMAP_ENV,
            )
        except subprocess.CalledProcessError as e:
            if _is_colmap_opengl_error(e.stderr):
                logger.warning(
                    "COLMAP GPU feature extraction failed (no OpenGL context). "
                    "Falling back to CPU SIFT extraction."
                )
                try:
                    subprocess.run(
                        [
                            colmap_bin, "feature_extractor",
                            "--database_path", str(db_path),
                            "--image_path", str(images_dir),
                            "--ImageReader.camera_model", "PINHOLE",
                            "--ImageReader.single_camera", "1",
                            "--SiftExtraction.use_gpu", "0",
                        ],
                        check=True,
                        capture_output=True,
                        timeout=1200,   # CPU is slower
                        env=_COLMAP_ENV,
                    )
                except subprocess.CalledProcessError as e2:
                    raise RuntimeError(
                        f"COLMAP feature_extractor (CPU) failed (exit {e2.returncode}):\n"
                        f"{e2.stderr[-2000:].decode('utf-8', errors='replace')}"
                    ) from e2
            else:
                raise RuntimeError(
                    f"COLMAP feature_extractor failed (exit {e.returncode}):\n"
                    f"{e.stderr[-2000:].decode('utf-8', errors='replace')}"
                ) from e

        logger.info("Running COLMAP exhaustive matching…")
        try:
            subprocess.run(
                [
                    colmap_bin, "exhaustive_matcher",
                    "--database_path", str(db_path),
                ],
                check=True,
                capture_output=True,
                timeout=600,
                env=_COLMAP_ENV,
            )
        except subprocess.CalledProcessError as e:
            if _is_colmap_opengl_error(e.stderr):
                logger.warning(
                    "COLMAP GPU matching failed (no OpenGL context). "
                    "Falling back to CPU matching."
                )
                try:
                    subprocess.run(
                        [
                            colmap_bin, "exhaustive_matcher",
                            "--database_path", str(db_path),
                            "--SiftMatching.use_gpu", "0",
                        ],
                        check=True,
                        capture_output=True,
                        timeout=1200,
                        env=_COLMAP_ENV,
                    )
                except subprocess.CalledProcessError as e2:
                    raise RuntimeError(
                        f"COLMAP exhaustive_matcher (CPU) failed (exit {e2.returncode}):\n"
                        f"{e2.stderr[-2000:].decode('utf-8', errors='replace')}"
                    ) from e2
            else:
                raise RuntimeError(
                    f"COLMAP exhaustive_matcher failed (exit {e.returncode}):\n"
                    f"{e.stderr[-2000:].decode('utf-8', errors='replace')}"
                ) from e

        logger.info("Running COLMAP sparse reconstruction…")
        try:
            subprocess.run(
                [
                    colmap_bin, "mapper",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--output_path", str(sparse_dir),
                ],
                check=True,
                capture_output=True,
                timeout=1200,
                env=_COLMAP_ENV,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"COLMAP mapper failed (exit {e.returncode}):\n"
                f"{e.stderr[-2000:].decode('utf-8', errors='replace')}"
            ) from e

        # COLMAP mapper creates sparse/0/, sparse/1/, etc.
        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            # Fallback: some COLMAP versions output directly to sparse/
            if (sparse_dir / "cameras.txt").exists() or (sparse_dir / "cameras.bin").exists():
                model_dir.mkdir(exist_ok=True)
                for f in sparse_dir.iterdir():
                    if f.is_file() and f.suffix in (".txt", ".bin") and f.name != "database.db":
                        shutil.move(str(f), str(model_dir / f.name))
            else:
                raise RuntimeError(
                    "COLMAP produced no sparse reconstruction. "
                    "Ensure images have sufficient overlap and texture."
                )

        logger.info("COLMAP sparse reconstruction saved to %s", model_dir)

        # Convert points3D.bin → points3D.ply (ObjectGS requires PLY format)
        self._convert_points_to_ply(model_dir)

    def _convert_points_to_ply(self, model_dir: Path) -> None:
        """Convert COLMAP points3D.bin to points3D.ply for ObjectGS."""
        import struct

        bin_path = model_dir / "points3D.bin"
        ply_path = model_dir / "points3D.ply"

        if not bin_path.exists():
            logger.warning("points3D.bin not found, skipping PLY conversion")
            return

        if ply_path.exists():
            return  # already converted

        with open(bin_path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]

            xyz = np.zeros((num_points, 3), dtype=np.float32)
            rgb = np.zeros((num_points, 3), dtype=np.uint8)

            for i in range(num_points):
                struct.unpack("<Q", f.read(8))  # point3d_id
                xyz[i] = struct.unpack("<ddd", f.read(24))
                rgb[i] = struct.unpack("<BBB", f.read(3))
                struct.unpack("<d", f.read(8))  # error
                # track: list of (image_id, point2D_idx) pairs
                track_length = struct.unpack("<Q", f.read(8))[0]
                f.read(track_length * 8)  # skip track data

        # Write PLY
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(num_points):
                f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} "
                        f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")

        logger.info("Converted points3D.bin → points3D.ply (%d points)", num_points)

    def _patch_training_scripts(self, scene_dir: Path, num_iterations: int = 5000) -> None:
        """Patch hardcoded dataset paths in ObjectGS source files.

        The ObjectGS repo has ``datasets/replica`` hardcoded in YAML configs.
        ``train.py`` dynamically appends the scene name at runtime via
        ``os.path.join(source_path, args.scene_name)``.  So we replace
        the source_path value with the absolute path to ``data_dir``
        (= ``scene_dir.parent``), and ObjectGS will append ``/scene``.

        Handles both the original ``datasets/replica`` and paths left over
        from previous runs (which are absolute paths under outputs/).
        """
        import re

        # data_dir is scene_dir's parent (data_dir/scene/ → data_dir/)
        abs_data_dir = str(scene_dir.parent.resolve())
        repo = self.repo_path

        # Pattern: matches any source_path value in YAML (original or previously patched)
        # e.g. "source_path: datasets/replica" or "source_path: /some/old/path"
        source_path_pattern = re.compile(
            r'(source_path\s*:\s*).*',
        )

        patched_files = []
        for fpath in repo.rglob("*.yaml"):
            if not fpath.is_file():
                continue
            if any(p in fpath.parts for p in ('.git', '__pycache__', 'outputs')):
                continue

            try:
                content = fpath.read_text(errors='ignore')
                if 'source_path' not in content:
                    continue

                patched = source_path_pattern.sub(
                    f'source_path: {abs_data_dir}',
                    content,
                )
                # Override training iterations for faster runs
                iterations_pattern = re.compile(r'(iterations\s*:\s*)[\d_]+')
                patched = iterations_pattern.sub(
                    f'\\g<1>{num_iterations:_}',
                    patched,
                )
                if patched != content:
                    fpath.write_text(patched)
                    patched_files.append(str(fpath.relative_to(repo)))
            except Exception:
                continue

        if patched_files:
            logger.warning(
                ">>> Patched source_path → '%s' in: %s",
                abs_data_dir, ', '.join(patched_files),
            )
        else:
            logger.warning(
                ">>> No YAML files with source_path found to patch"
            )

        # ------------------------------------------------------------------
        # Patch train.py: fix hardcoded camera index [10] that crashes when
        # fewer than 11 cameras are reconstructed by COLMAP.
        # Replace `.copy()[10]` → `.copy()[0]` (always use first camera).
        # ------------------------------------------------------------------
        train_py = repo / "train.py"
        if train_py.exists():
            try:
                content = train_py.read_text(errors="ignore")
                # Match patterns like:  .copy()[10]  or  .copy()[123]
                fixed = re.sub(
                    r'(\.copy\(\))\[\d+\]',
                    r'\1[0]',
                    content,
                )
                if fixed != content:
                    train_py.write_text(fixed)
                    logger.info("Patched train.py: hardcoded camera index → [0]")
            except Exception as e:
                logger.warning("Could not patch train.py camera index: %s", e)

        # Copy patched files to ObjectGS repo
        patches_dir = Path(__file__).parent / "objectgs_patches"
        import filecmp
        for patch_name, target_rel in [
            ("render.py", "gaussian_renderer/render.py"),
            ("cameras.py", "scene/cameras.py"),
        ]:
            patched = patches_dir / patch_name
            target = repo / target_rel
            if patched.exists() and target.exists():
                if not filecmp.cmp(str(patched), str(target), shallow=False):
                    shutil.copy2(str(patched), str(target))
                    logger.info("Copied patched %s → %s", patch_name, target)

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

        # ObjectGS expects images in source_path/scene/images_all/ (per YAML config).
        # Only place images there — NOT at root level (which would confuse COLMAP).
        images_dir = scene_dir / "images_all"
        images_dir.mkdir(exist_ok=True)
        for i, fp in enumerate(frame_paths):
            dst = images_dir / f"{i:06d}.jpg"
            if not dst.exists():
                dst.symlink_to(fp.resolve())

        # ObjectGS requires per-object masks at object_mask_all/.
        # Each pixel value = object label (1, 2, 3...), 0 = background.
        mask_dir = scene_dir / "object_mask_all"
        mask_dir.mkdir(exist_ok=True)

        has_real_masks = False
        if masks_dir and Path(masks_dir).exists():
            # Copy real per-frame label masks from SAM3
            mask_files = sorted(Path(masks_dir).glob("*.png"))
            if mask_files:
                for mf in mask_files:
                    dst = mask_dir / mf.name
                    if not dst.exists():
                        shutil.copy2(str(mf), str(dst))
                has_real_masks = True
                logger.info("Using %d real per-object masks from SAM3", len(mask_files))

        if not has_real_masks:
            # Fallback: dummy all-white masks (whole image = one object)
            for i, fp in enumerate(frame_paths):
                mask_dst = mask_dir / f"{i:06d}.png"
                if not mask_dst.exists():
                    from PIL import Image as PILImage
                    img = PILImage.open(fp)
                    w, h = img.size
                    dummy_mask = PILImage.new("L", (w, h), 255)
                    dummy_mask.save(str(mask_dst))
            logger.warning("No real masks available, using dummy all-white masks")
        else:
            # Fill gaps: frames without real masks get dummy masks
            from PIL import Image as PILImage
            gaps_filled = 0
            for i, fp in enumerate(frame_paths):
                mask_dst = mask_dir / f"{i:06d}.png"
                if not mask_dst.exists():
                    img = PILImage.open(fp)
                    w, h = img.size
                    dummy_mask = PILImage.new("L", (w, h), 255)
                    dummy_mask.save(str(mask_dst))
                    gaps_filled += 1
            if gaps_filled:
                logger.warning("Filled %d frames with dummy masks (no SAM3 mask available)", gaps_filled)

        # Run COLMAP on the images/ subdirectory so extr.name = "000000.jpg".
        # ObjectGS then does os.path.join(source_path/images/, extr.name).
        self._run_colmap_for_scene(scene_dir, images_dir, frame_paths)

        # Patch hardcoded dataset paths in training scripts and configs
        self._patch_training_scripts(scene_dir, num_iterations=num_iterations)

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
                timeout=7200,  # 2 hour timeout
                env=_isolated_gpu_env(),
            )

            if result.returncode != 0:
                raise RuntimeError(f"ObjectGS training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("ObjectGS training timed out")
        except RuntimeError:
            raise

        # Skip mesh export: ObjectGS "reconstruct radiance fields" segfaults
        # and crashes the entire pipeline (GPU memory corruption propagates
        # to parent process). Training data is preserved for later mesh export.
        logger.info("ObjectGS training complete. Mesh export skipped (known segfault).")

        return {
            "scene_mesh": None,
            "object_meshes": {},
            "output_dir": output_dir,
        }

    def _find_model_path(self, output_dir: Path) -> Path | None:
        """Find the actual ObjectGS training output directory.

        ObjectGS saves to {repo}/outputs/replica/objectgs/scene/TIMESTAMP/.
        Returns the most recent timestamped directory, or None if not found.
        """
        # Search in ObjectGS repo's outputs directory
        outputs_base = self.repo_path / "outputs"
        for scene_dir in outputs_base.rglob("scene"):
            if scene_dir.is_dir():
                # Find timestamped subdirectories
                timestamped = sorted(
                    [d for d in scene_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True,
                )
                for ts_dir in timestamped:
                    if (ts_dir / "config.yaml").exists():
                        return ts_dir
        # Fallback: check if config.yaml is directly in output_dir
        if (output_dir / "config.yaml").exists():
            return output_dir
        return None

    def _export_object_meshes(self, output_dir: Path) -> dict[str, Path]:
        """Export per-object meshes from trained model."""
        export_script = self.repo_path / "export_object_mesh.py"

        if not export_script.exists():
            logger.warning("ObjectGS export script not found")
            return {}

        # Find the actual model directory (ObjectGS saves to repo/outputs/...)
        model_path = self._find_model_path(output_dir)
        if model_path is None:
            logger.warning("ObjectGS model output not found (no config.yaml)")
            return {}

        logger.info("ObjectGS model path: %s", model_path)

        try:
            # Export all objects (label_id = -1)
            result = subprocess.run(
                [
                    "python",
                    str(export_script),
                    "-m", str(model_path),
                    "--query_label_id", "-1",
                ],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600,
                env=_isolated_gpu_env(),
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

        model_path = self._find_model_path(output_dir)
        if model_path is None:
            return None

        try:
            result = subprocess.run(
                [
                    "python",
                    str(export_script),
                    "-m", str(model_path),
                ],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600,
                env=_isolated_gpu_env(),
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
