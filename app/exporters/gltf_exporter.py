"""glTF 2.0 exporter — generates mesh + scene with full UV, textures, PBR materials."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class GltfExporter(BaseExporter):
    """Exports 3D data to glTF 2.0 format with full mesh + UV + textures + PBR materials.

    glTF is the universal 3D interchange format supported by Unity, Blender,
    web viewers (three.js, Babylon.js), and most 3D engines.
    """

    format_name = "gltf"
    file_extension = ".gltf"
    mime_type = "model/gltf+json"

    def export(self, data: StructuredOutput) -> Path | None:
        # Prefer per-object meshes with textures
        objects_with_mesh = [obj for obj in data.objects
                            if obj.mesh_3d is not None and obj.mesh_3d.vertices]
        if objects_with_mesh:
            return self._export_from_meshes(data, objects_with_mesh)

        if data.gaussian_splats is not None:
            return self._export_from_splats(data)
        elif data.point_cloud and data.point_cloud.points:
            return self._export_from_pointcloud(data)
        else:
            import logging
            logging.getLogger(__name__).warning("No 3D data available for glTF export — skipping.")
            return None

    def _export_from_meshes(self, data: StructuredOutput,
                            objects_with_mesh: list) -> Path:
        """Export per-object meshes as a full glTF scene with materials, UV, and textures."""
        out_dir = self._output_path(data.source_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # ─── Build binary buffer ────────────────────────────
        buf = bytearray()
        byte_offset = 0

        accessors = []
        buffer_views = []
        meshes = []
        nodes = []
        materials = []
        textures = []
        samplers = [{
            "magFilter": 9729,
            "minFilter": 9987,
            "wrapS": 33648,
            "wrapT": 33648,
        }]
        images = []

        for obj in objects_with_mesh:
            mesh = obj.mesh_3d
            if not mesh or not mesh.vertices:
                continue

            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32) if mesh.faces else None
            normals = np.array(mesh.normals, dtype=np.float32) if mesh.normals else None
            uv_coords = np.array(mesh.uv_coords, dtype=np.float32) if mesh.uv_coords else None

            n_verts = len(vertices)
            n_faces = len(faces) if faces is not None else 0

            # ── Vertex positions ──
            pos_buf = vertices.tobytes()
            pos_offset = byte_offset
            buf += pos_buf
            byte_offset += len(pos_buf)
            pos_accessor_idx = len(accessors)
            accessors.append({
                "bufferView": len(buffer_views),
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "max": vertices.max(axis=0).tolist(),
                "min": vertices.min(axis=0).tolist(),
            })
            buffer_views.append({
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(pos_buf),
                "target": 34962,
            })

            # ── Normals ──
            normal_accessor_idx = None
            if normals is not None and len(normals) == n_verts:
                norm_buf = normals.tobytes()
                norm_offset = byte_offset
                buf += norm_buf
                byte_offset += len(norm_buf)
                normal_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": len(buffer_views),
                    "componentType": 5126,
                    "count": n_verts,
                    "type": "VEC3",
                })
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": norm_offset,
                    "byteLength": len(norm_buf),
                    "target": 34962,
                })

            # ── UV coordinates ──
            uv_accessor_idx = None
            if uv_coords is not None and len(uv_coords) == n_verts:
                uv_buf = uv_coords.tobytes()
                uv_offset = byte_offset
                buf += uv_buf
                byte_offset += len(uv_buf)
                uv_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": len(buffer_views),
                    "componentType": 5126,
                    "count": n_verts,
                    "type": "VEC2",
                })
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": uv_offset,
                    "byteLength": len(uv_buf),
                    "target": 34962,
                })

            # ── Index buffer (faces) ──
            index_accessor_idx = None
            if faces is not None and n_faces > 0:
                idx_buf = faces.tobytes()
                idx_offset = byte_offset
                buf += idx_buf
                byte_offset += len(idx_buf)
                index_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": len(buffer_views),
                    "componentType": 5123,
                    "count": n_faces * 3,
                    "type": "SCALAR",
                    "max": [n_verts - 1],
                    "min": [0],
                })
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": idx_offset,
                    "byteLength": len(idx_buf),
                    "target": 34963,
                })

            # ── Material + Texture ──
            mat_idx = len(materials)
            material = {
                "name": obj.label_custom or obj.id,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }

            # Add texture if available
            if mesh.texture_path and Path(mesh.texture_path).exists():
                tex_idx = self._add_texture_to_buffer(
                    mesh.texture_path, buf, byte_offset, buffer_views, images, textures
                )
                byte_offset = buf.__len__()
                if tex_idx is not None:
                    material["pbrMetallicRoughness"]["baseColorTexture"] = {
                        "index": tex_idx,
                    }

            materials.append(material)

            # ── Mesh primitive ──
            primitive = {
                "attributes": {"POSITION": pos_accessor_idx},
                "mode": 4,  # TRIANGLES
                "material": mat_idx,
            }

            if normal_accessor_idx is not None:
                primitive["attributes"]["NORMAL"] = normal_accessor_idx

            if uv_accessor_idx is not None:
                primitive["attributes"]["TEXCOORD_0"] = uv_accessor_idx

            if index_accessor_idx is not None:
                primitive["indices"] = index_accessor_idx

            meshes.append({
                "name": obj.id,
                "primitives": [primitive],
            })

            # ── Node ──
            node = {
                "name": obj.id,
                "mesh": len(meshes) - 1,
            }
            if obj.bbox_3d:
                node["translation"] = [obj.bbox_3d.x, obj.bbox_3d.y, obj.bbox_3d.z]
            nodes.append(node)

        # ── Pad buffer ──
        padding = (4 - (byte_offset % 4)) % 4
        buf += b"\x00" * padding

        # ── Build glTF JSON ──
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "ToThinkVision",
            },
            "scene": 0,
            "scenes": [{
                "name": f"ToThinkVision_{Path(data.source_file).stem}",
                "nodes": list(range(len(nodes))),
            }],
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "textures": textures,
            "images": images,
            "samplers": samplers,
            "buffers": [{"byteLength": byte_offset, "uri": f"{Path(data.source_file).stem}.bin"}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }

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
                gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

        # ── Write files ──
        json_path = self._output_path(data.source_file)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, indent=2)

        bin_path = json_path.parent / f"{Path(data.source_file).stem}.bin"
        with open(bin_path, "wb") as f:
            f.write(buf)

        return json_path

    def _add_texture_to_buffer(
        self,
        texture_path: str,
        buf: bytearray,
        byte_offset: int,
        buffer_views: list,
        images: list,
        textures: list,
    ) -> int | None:
        """Add a texture image to the glTF binary buffer and return texture index."""
        try:
            img = Image.open(texture_path).convert("RGBA")
            img_bytes = img.tobytes()
            img_w, img_h = img.size

            tex_offset = byte_offset
            buf += img_bytes
            byte_offset += len(img_bytes)

            tex_image_idx = len(images)
            images.append({
                "bufferView": len(buffer_views),
                "mimeType": "image/png",
                "width": img_w,
                "height": img_h,
            })
            buffer_views.append({
                "buffer": 0,
                "byteOffset": tex_offset,
                "byteLength": len(img_bytes),
            })

            tex_sampler_idx = len(textures)
            textures.append({
                "sampler": 0,
                "source": tex_image_idx,
            })

            return tex_sampler_idx
        except Exception as e:
            return None

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
        buf += pos_buf
        byte_offset += len(pos_buf)

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

        # Pad buffer
        padding = (4 - (byte_offset % 4)) % 4
        buf += b"\x00" * padding
        byte_offset += padding

        # ─── Build glTF JSON structure ───
        accessors = [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": n,
                "type": "VEC3",
                "max": points.max(axis=0).tolist(),
                "min": points.min(axis=0).tolist(),
            }
        ]

        if col_offset is not None:
            accessors.append({
                "bufferView": 1,
                "componentType": 5126,
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
                {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_buf)},
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
                gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

        # ─── Write files ───
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
