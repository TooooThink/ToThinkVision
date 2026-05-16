"""OBJ 3D exporter — generates Wavefront OBJ from point cloud / 3D data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class ObjExporter(BaseExporter):
    """Exports 3D point cloud / Gaussian splat data to Wavefront OBJ format.

    OBJ is a widely supported 3D format for import into Blender, Maya, Unity,
    and most 3D modeling software.
    """

    format_name = "obj"
    file_extension = ".obj"
    mime_type = "model/obj"

    def export(self, data: StructuredOutput) -> Path:
        if data.gaussian_splats is not None:
            return self._export_from_splats(data)
        elif data.point_cloud and data.point_cloud.points:
            return self._export_from_pointcloud(data)
        else:
            raise ValueError("No 3D data available for OBJ export.")

    def _export_from_splats(self, data: StructuredOutput) -> Path:
        """Export Gaussian splats as OBJ vertices."""
        splats = data.gaussian_splats
        points = np.array(splats.means)
        colors = self._sh_to_rgb(splats.sh_coeffs) if splats.sh_coeffs else None

        return self._write_obj(
            data, points, colors,
            f"ToThinkVision_{Path(data.source_file).stem}_3DGS",
        )

    def _export_from_pointcloud(self, data: StructuredOutput) -> Path:
        """Export point cloud as OBJ vertices."""
        pc = data.point_cloud
        points = np.array(pc.points)
        colors = np.array(pc.colors) if pc.colors else None

        return self._write_obj(
            data, points, colors,
            f"ToThinkVision_{Path(data.source_file).stem}_PointCloud",
        )

    def _write_obj(
        self,
        data: StructuredOutput,
        points: np.ndarray,
        colors: np.ndarray | None,
        object_name: str,
    ) -> Path:
        """Write Wavefront OBJ file with vertices and optional colors."""
        n = len(points)
        if n == 0:
            raise ValueError("No points to export")

        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        has_colors = colors is not None and len(colors) == n

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# ToThinkVision OBJ export\n")
            f.write(f"# Source: {data.source_file}\n")
            f.write(f"# Points: {n}\n")
            f.write(f"o {object_name}\n\n")

            # Write vertices with optional colors
            for i in range(n):
                if has_colors:
                    c = colors[i].astype(np.float64) / 255.0
                    f.write(f"v {points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                else:
                    f.write(f"v {points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}\n")

            # Write normals if available
            if data.point_cloud and data.point_cloud.normals:
                normals = np.array(data.point_cloud.normals)
                for i in range(min(len(normals), n)):
                    f.write(f"vn {normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f}\n")

            # Write camera poses as reference points
            if data.camera_poses:
                f.write(f"\n# Camera poses ({len(data.camera_poses)} frames)\n")
                for pose in data.camera_poses:
                    frame_idx = pose.frame_idx if hasattr(pose, "frame_idx") else pose["frame_idx"]
                    position = pose.position if hasattr(pose, "position") else pose["position"]
                    f.write(f"# camera_frame_{frame_idx}: {position}\n")

            # Write bounding boxes for detected objects
            if data.objects:
                f.write(f"\n# Object bounding boxes ({len(data.objects)} objects)\n")
                for obj in data.objects:
                    label = obj.label_custom or obj.label.value
                    if obj.bbox_3d:
                        f.write(f"# {obj.id} ({label}): center=({obj.bbox_3d.x:.2f}, {obj.bbox_3d.y:.2f}, {obj.bbox_3d.z:.2f})\n")
                    else:
                        f.write(f"# {obj.id} ({label}): 2D box=({obj.bbox.x:.0f}, {obj.bbox.y:.0f}, {obj.bbox.w:.0f}x{obj.bbox.h:.0f})\n")

        return out_path

    @staticmethod
    def _sh_to_rgb(sh_coeffs) -> np.ndarray | None:
        """Convert spherical harmonic coefficients to RGB colors."""
        if sh_coeffs is None:
            return None
        sh = np.array(sh_coeffs)
        if sh.ndim == 2 and sh.shape[1] >= 3:
            return (sh[:, :3] * 255).clip(0, 255).astype(np.uint8)
        return None
