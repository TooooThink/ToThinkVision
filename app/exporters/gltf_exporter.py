"""glTF 2.0 exporter — generates mesh + scene from point cloud / 3D data."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class GltfExporter(BaseExporter):
    """Exports 3D point cloud / Gaussian splat data to glTF 2.0 format.

    glTF is the universal 3D interchange format supported by Unity, Blender,
    web viewers (three.js, Babylon.js), and most 3D engines.
    """

    format_name = "gltf"
    file_extension = ".gltf"
    mime_type = "model/gltf+json"

    def export(self, data: StructuredOutput) -> Path:
        if data.gaussian_splats is not None:
            return self._export_from_splats(data)
        elif data.point_cloud and data.point_cloud.points:
            return self._export_from_pointcloud(data)
        else:
            raise ValueError("No 3D data available for glTF export.")

    def _export_from_splats(self, data: StructuredOutput) -> Path:
        """Export Gaussian splats as point-based glTF."""
        splats = data.gaussian_splats
        means = np.array(splats.means)
        colors = self._sh_to_rgb(splats.sh_coeffs) if splats.sh_coeffs else None

        return self._build_gltf(
            data, means, colors,
            f"ToThinkVision_{Path(data.source_file).stem}_3DGS",
        )

    def _export_from_pointcloud(self, data: StructuredOutput) -> Path:
        """Export point cloud as glTF points."""
        pc = data.point_cloud
        means = np.array(pc.points)
        colors = np.array(pc.colors) if pc.colors else None

        return self._build_gltf(
            data, means, colors,
            f"ToThinkVision_{Path(data.source_file).stem}_PointCloud",
        )

    def _build_gltf(
        self,
        data: StructuredOutput,
        points: np.ndarray,
        colors: np.ndarray | None,
        scene_name: str,
    ) -> Path:
        """Build glTF 2.0 scene with binary buffer."""
        n = len(points)
        if n == 0:
            raise ValueError("No points to export")

        # ─── Build binary buffer ────────────────────────────
        buf = bytearray()
        byte_offset = 0

        # Positions accessor
        pos_buf = points.astype(np.float32).tobytes()
        pos_offset = byte_offset
        pos_len = len(pos_buf)
        buf += pos_buf
        byte_offset += pos_len

        # Colors accessor (optional)
        col_offset = None
        col_len = 0
        if colors is not None and len(colors) == n:
            col_arr = (colors.astype(np.float32) / 255.0).clip(0, 1)
            col_buf = col_arr.tobytes()
            col_offset = byte_offset
            col_len = len(col_buf)
            buf += col_buf
            byte_offset += col_len

        # Pad buffer to 4-byte boundary
        padding = (4 - (byte_offset % 4)) % 4
        buf += b"\x00" * padding
        byte_offset += padding

        # ─── Build glTF JSON structure ──────────────────────
        buffer_view_idx = 0
        buffer_view_count = 1
        if col_offset is not None:
            buffer_view_count = 2

        accessors = [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
                "max": points.max(axis=0).tolist(),
                "min": points.min(axis=0).tolist(),
            }
        ]

        if col_offset is not None:
            accessors.append({
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
                "max": [1.0, 1.0, 1.0],
                "min": [0.0, 0.0, 0.0],
            })

        attributes = {"POSITION": 0}
        if col_offset is not None:
            attributes["COLOR_0"] = 1

        gltf = {
            "asset": {"version": "2.0", "generator": "ToThinkVision"},
            "scene": 0,
            "scenes": [{"name": scene_name, "nodes": [0]}],
            "nodes": [{"name": "point_cloud", "mesh": 0}],
            "meshes": [{
                "name": "point_cloud_mesh",
                "primitives": [{
                    "attributes": attributes,
                    "mode": 0,  # POINTS
                }]
            }],
            "buffers": [{"byteLength": byte_offset, "uri": f"{Path(data.source_file).stem}.bin"}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len},
            ],
            "accessors": accessors,
        }

        if col_offset is not None:
            gltf["bufferViews"].append({
                "buffer": 0, "byteOffset": col_offset, "byteLength": col_len,
            })

        # Add camera poses as named nodes
        if data.camera_poses:
            for pose in data.camera_poses:
                frame_idx = pose.frame_idx if hasattr(pose, "frame_idx") else pose["frame_idx"]
                position = pose.position if hasattr(pose, "position") else pose["position"]
                rotation = pose.rotation if hasattr(pose, "rotation") else pose["rotation"]

                cam_node = {
                    "name": f"camera_frame_{frame_idx}",
                    "translation": list(position),
                    "rotation": list(rotation),
                }
                gltf["nodes"].append(cam_node)

        # ─── Write files ────────────────────────────────────
        out_dir = self._output_path(data.source_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = self._output_path(data.source_file)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, indent=2)

        bin_path = out_dir / f"{Path(data.source_file).stem}.bin"
        with open(bin_path, "wb") as f:
            f.write(buf)

        return json_path

    @staticmethod
    def _sh_to_rgb(sh_coeffs) -> np.ndarray | None:
        """Convert spherical harmonic coefficients to RGB colors."""
        if sh_coeffs is None:
            return None
        sh = np.array(sh_coeffs)
        if sh.ndim == 2 and sh.shape[1] >= 3:
            return (sh[:, :3] * 255).clip(0, 255).astype(np.uint8)
        return None
