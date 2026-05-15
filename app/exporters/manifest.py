"""Export Manifest — generates a README.txt listing all output files and their usage."""

from __future__ import annotations

import textwrap
from pathlib import Path

from app.config import settings

UNITY_SCRIPTS = [
    "TTVSceneImporter.cs",
    "TTVRuntimeLoader.cs",
]


class ExportManifest:
    """Generates a manifest file describing all exported outputs."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.files: list[dict] = []

    def add(self, file_path: str | Path, description: str, target_software: str = ""):
        """Register an exported file."""
        self.files.append({
            "path": str(file_path),
            "name": Path(file_path).name,
            "description": description,
            "target": target_software,
        })

    def generate(self, output_dir: Path | None = None) -> Path:
        """Write manifest as README.txt in the output directory."""
        if output_dir is None:
            stem = Path(self.source_name).stem
            output_dir = settings.output_dir / stem

        manifest_path = output_dir / "README_EXPORT.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"ToThinkVision Export — {Path(self.source_name).stem}",
            "=" * 60,
            "",
            "This directory contains AI-analyzed results from your video/image.",
            "Each file serves a different purpose in your downstream workflow.",
            "",
        ]

        # Group by target software
        groups: dict[str, list] = {}
        for f in self.files:
            target = f["target"] or "Other"
            if target not in groups:
                groups[target] = []
            groups[target].append(f)

        for target, files in groups.items():
            lines.append(f"─── {target} ───")
            lines.append("")
            for f in files:
                lines.append(f"  {f['name']}")
                lines.append(f"    {f['description']}")
                lines.append("")

        lines.extend([
            "",
            "─── Unity Setup Instructions ───",
            "",
            "To import into Unity:",
            "",
            "  1. Copy the following scripts to your Unity project under Assets/Editor/:",
            f"     - {UNITY_SCRIPTS[0]}",
            f"     - {UNITY_SCRIPTS[1]}",
            "     (Find them in the ToThinkVision repo: unity/)",
            "",
            "  2. In Unity, go to: GameObject → ToThinkVision → Import Scene from JSON",
            "  3. Select the *_unity_json.json file in this directory",
            "  4. The scene is built automatically with GameObjects, Colliders, and Materials",
            "",
            "  For .splat files (3D Gaussian Splatting):",
            "  1. Install UnityGaussianSplatting plugin:",
            "     https://github.com/aras-p/UnityGaussianSplatting",
            "  2. In Unity: GameObject → ToThinkVision → Import Splat File",
            "  3. Select the *.splat file in this directory",
            "",
            "─── Photoshop Instructions ───",
            "",
            "  1. Open the *.psd file in Photoshop",
            "  2. Each detected object is a separate layer with transparent background",
            "  3. Layers are positioned at their original locations",
            "  4. For video: each frame is a Group containing object layers",
            "",
            "─── After Effects Instructions ───",
            "",
            "  1. Open After Effects",
            "  2. File → Scripts → Run Script File",
            "  3. Select the *.jsx file in this directory",
            "  4. The script creates a composition with:",
            "     - Your original video as a semi-transparent reference",
            "     - Each detected object as an image layer with correct position",
            "     - Position keyframes from motion tracking",
            "     - Opacity keyframes for appear/disappear timing",
            "",
            "─── Data Files ───",
            "",
            "  *_full_json.json     Complete structured data (all objects, 3D, camera poses)",
            "  *_depth.png          Depth map visualization (colored heatmap)",
            "  *_detection.png      Detection overlay (bounding boxes + labels)",
            "  *_pointcloud.png     Point cloud top-down preview",
            "",
            "  Per-object files (in subdirectory):",
            "    <obj_id>_crop.png     Object cropped from original image",
            "    <obj_id>_mask.png     Binary segmentation mask (white = object)",
            "    <obj_id>_masked.png   Object with transparent background",
            "",
        ])

        manifest_path.write_text("\n".join(lines), encoding="utf-8")
        return manifest_path


def build_manifest_for_export(
    source_name: str,
    export_format: str,
    output_path: Path,
    data,
) -> list[Path]:
    """Build a manifest for a single export operation. Returns list of created files."""
    manifest = ExportManifest(source_name)

    # Add the main export file
    format_descriptions = {
        "unity_splat": "Unity 3D Gaussian Splatting file — drag into scene with plugin",
        "ue_splat": "UE5 3D Gaussian Splatting file — import with UnrealSplat plugin",
        "unity_json": "Unity scene description — import via GameObject → ToThinkVision menu",
        "ue_json": "UE5 scene description — import via Blueprint script",
        "collision_json": "Pure collision box data for game physics",
        "psd_static": "Photoshop PSD — each object is a transparent layer",
        "psd_animated": "Photoshop PSD — each frame is a Group with object layers",
        "ae_project": "After Effects ExtendScript — run in AE to create animated composition",
        "full_json": "Complete structured data with 2D + 3D + camera + Gaussian info",
    }

    target_map = {
        "unity_splat": "Unity (with plugin)",
        "ue_splat": "Unreal Engine (with plugin)",
        "unity_json": "Unity",
        "ue_json": "Unreal Engine",
        "collision_json": "Game Engine",
        "psd_static": "Photoshop",
        "psd_animated": "Photoshop",
        "ae_project": "After Effects",
        "full_json": "Universal",
    }

    desc = format_descriptions.get(export_format, export_format)
    target = target_map.get(export_format, "")
    manifest.add(output_path, desc, target)

    # Add auxiliary files
    if hasattr(data, "detection_overlay_png_path") and data.detection_overlay_png_path:
        manifest.add(data.detection_overlay_png_path, "Detection visualization overlay", "All")

    if hasattr(data, "depth_map_png_path") and data.depth_map_png_path:
        manifest.add(data.depth_map_png_path, "Depth map visualization (heatmap)", "All")

    if hasattr(data, "point_cloud_preview_png_path") and data.point_cloud_preview_png_path:
        manifest.add(data.point_cloud_preview_png_path, "Point cloud top-down preview", "All")

    # Write manifest
    manifest_path = manifest.generate()
    return [output_path, manifest_path]
