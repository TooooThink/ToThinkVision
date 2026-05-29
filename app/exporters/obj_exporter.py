"""OBJ 3D exporter — generates Wavefront OBJ with mesh + MTL material + UV + texture."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from app.config import settings
from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class ObjExporter(BaseExporter):
    """Exports 3D mesh data to Wavefront OBJ format with materials, UV coordinates, and textures.

    OBJ is a widely-supported 3D format for import into Blender, Maya, Unity,
    and most 3D modeling software.
    """

    format_name = "obj"
    file_extension = ".obj"
    mime_type = "model/obj"

    def export(self, data: StructuredOutput) -> Path | None:
        objects_with_mesh = [obj for obj in data.objects
                            if obj.mesh_3d is not None and obj.mesh_3d.vertices]
        if objects_with_mesh:
            return self._export_meshes(data, objects_with_mesh)
        elif data.gaussian_splats is not None:
            return self._export_from_splats(data)
        elif data.point_cloud and data.point_cloud.points:
            return self._export_from_pointcloud(data)
        else:
            import logging
            logging.getLogger(__name__).warning("No 3D data available for OBJ export — skipping.")
            return None

    def _export_meshes(self, data: StructuredOutput,
                       objects_with_mesh: list) -> Path:
        """Export per-object meshes as OBJ with MTL material file, UV coordinates, and textures."""
        out_dir = self._output_path(data.source_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(data.source_file).stem
        obj_path = self._output_path(data.source_file)
        mtl_path = obj_path.with_suffix(".mtl")

        vertex_offset = 0
        uv_offset = 0
        normal_offset = 0

        with open(obj_path, "w", encoding="utf-8") as f:
            f.write("# ToThinkVision OBJ export\n")
            f.write(f"# Source: {data.source_file}\n")
            f.write(f"# Objects: {len(objects_with_mesh)}\n")
            f.write(f"mtllib {mtl_path.name}\n\n")

        # Write MTL file and collect texture copies
        with open(mtl_path, "w", encoding="utf-8") as mf:
            mf.write("# ToThinkVision Materials\n\n")

            for obj in objects_with_mesh:
                mesh = obj.mesh_3d
                label = obj.label_custom or obj.id
                mat_name = f"mat_{label.replace(' ', '_').replace('.', '_')}"

                # MTL material definition
                mf.write(f"newmtl {mat_name}\n")
                mf.write("Ka 0.2 0.2 0.2\n")   # Ambient
                mf.write("Kd 1.0 1.0 1.0\n")   # Diffuse
                mf.write("Ks 0.5 0.5 0.5\n")   # Specular
                mf.write("Ns 96.078431\n")     # Shininess
                mf.write("d 1.0\n")            # Opacity
                mf.write("illum 2\n")          # Lighting model

                # Add texture if available
                if mesh.texture_path and Path(mesh.texture_path).exists():
                    tex_path = Path(mesh.texture_path)
                    tex_name = tex_path.name
                    mf.write(f"map_Kd {tex_name}\n")

                    # Copy texture to output dir
                    dest_tex = out_dir / tex_name
                    if not dest_tex.exists():
                        shutil.copy2(tex_path, dest_tex)

                mf.write("\n")

        # Write OBJ geometry
        with open(obj_path, "a", encoding="utf-8") as f:
            for obj in objects_with_mesh:
                mesh = obj.mesh_3d
                label = obj.label_custom or obj.id
                mat_name = f"mat_{label.replace(' ', '_').replace('.', '_')}"

                vertices = mesh.vertices
                faces = mesh.faces
                normals = mesh.normals
                uv_coords = mesh.uv_coords
                uv_face_map = mesh.uv_face_map

                # Use material
                f.write(f"usemtl {mat_name}\n")
                f.write(f"o {label}\n\n")

                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Write UV coordinates
                if uv_coords:
                    f.write("\n")
                    for uv in uv_coords:
                        f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

                # Write normals
                if normals:
                    f.write("\n")
                    for n in normals:
                        f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

                # Write faces
                if faces:
                    f.write("\n")
                    has_uv = uv_coords is not None and uv_face_map is not None and len(uv_face_map) == len(faces)
                    has_normal = normals is not None

                    for fi, face in enumerate(faces):
                        if has_uv and has_normal:
                            f.write(f"f {face[0]+vertex_offset+1}/{fi+uv_offset+1}/{face[0]+normal_offset+1} "
                                    f"{face[1]+vertex_offset+1}/{fi+uv_offset+1}/{face[1]+normal_offset+1} "
                                    f"{face[2]+vertex_offset+1}/{fi+uv_offset+1}/{face[2]+normal_offset+1}\n")
                        elif has_uv:
                            f.write(f"f {face[0]+vertex_offset+1}/{fi+uv_offset+1} "
                                    f"{face[1]+vertex_offset+1}/{fi+uv_offset+1} "
                                    f"{face[2]+vertex_offset+1}/{fi+uv_offset+1}\n")
                        elif has_normal:
                            f.write(f"f {face[0]+vertex_offset+1}//{face[0]+normal_offset+1} "
                                    f"{face[1]+vertex_offset+1}//{face[1]+normal_offset+1} "
                                    f"{face[2]+vertex_offset+1}//{face[2]+normal_offset+1}\n")
                        else:
                            f.write(f"f {face[0]+vertex_offset+1} "
                                    f"{face[1]+vertex_offset+1} "
                                    f"{face[2]+vertex_offset+1}\n")

                vertex_offset += len(vertices)
                if uv_coords:
                    uv_offset += len(uv_coords)
                if normals:
                    normal_offset += len(normals)

                f.write("\n")

        # Write camera poses as comments
        if data.camera_poses:
            with open(obj_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Camera poses ({len(data.camera_poses)} frames)\n")
                for pose in data.camera_poses:
                    frame_idx = pose.frame_idx if hasattr(pose, "frame_idx") else pose["frame_idx"]
                    position = pose.position if hasattr(pose, "position") else pose["position"]
                    f.write(f"# camera_frame_{frame_idx}: {position}\n")

        # Write object bounding boxes
        with open(obj_path, "a", encoding="utf-8") as f:
            f.write(f"\n# Object bounding boxes\n")
            for obj in objects_with_mesh:
                mesh = obj.mesh_3d
                label = obj.label_custom or obj.id
                if mesh.bounds:
                    f.write(f"# {obj.id} ({label}): "
                            f"bounds=({mesh.bounds['min'][0]:.2f}, {mesh.bounds['min'][1]:.2f}, {mesh.bounds['min'][2]:.2f}) "
                            f"to ({mesh.bounds['max'][0]:.2f}, {mesh.bounds['max'][1]:.2f}, {mesh.bounds['max'][2]:.2f})\n")

        return obj_path

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

            for i in range(n):
                if has_colors:
                    c = colors[i].astype(np.float64) / 255.0
                    f.write(f"v {points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                else:
                    f.write(f"v {points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}\n")

            if data.point_cloud and data.point_cloud.normals:
                normals = np.array(data.point_cloud.normals)
                for i in range(min(len(normals), n)):
                    f.write(f"vn {normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f}\n")

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
