"""Animated glTF 2.0 exporter — meshes + per-object keyframe animation.

Exports a complete animated scene where each object has its own mesh
and keyframe animation (translation + rotation channels), plus an
animated camera path from recovered camera poses.

Output: .glb (binary glTF, single self-contained file)
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas import CameraPose, Mesh3D, ObjectTrajectory4D, StructuredObject

logger = logging.getLogger(__name__)


class AnimatedGLTFExporter:
    """Export animated glTF/GLB with per-object keyframe animation."""

    def export(
        self,
        objects: list[StructuredObject],
        trajectories: dict[str, ObjectTrajectory4D],
        camera_poses: list[CameraPose] | None = None,
        output_dir: Path | None = None,
        filename: str = "scene_animated.glb",
    ) -> Path:
        """Export animated glTF scene.

        Args:
            objects: Structured objects with mesh data
            trajectories: Per-object 6DoF trajectories
            camera_poses: Camera animation path
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to exported .glb file
        """
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter objects with both mesh and trajectory data
        animated_objects = [
            obj for obj in objects
            if obj.mesh_3d is not None and obj.id in trajectories
        ]

        if not animated_objects:
            # Fallback: export static objects without animation
            animated_objects = [
                obj for obj in objects if obj.mesh_3d is not None
            ]
            if not animated_objects:
                raise ValueError("No objects with mesh data for animated glTF export")

        # Build binary buffer
        buf = bytearray()
        accessors = []
        buffer_views = []
        meshes = []
        nodes = []
        materials = []
        textures = []
        images = []
        animations = []
        samplers = [{
            "magFilter": 9729,  # LINEAR
            "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
            "wrapS": 33648,     # REPEAT
            "wrapT": 33648,
        }]

        # ── Build mesh + animation for each object ──
        for obj_idx, obj in enumerate(animated_objects):
            mesh = obj.mesh_3d
            trajectory = trajectories.get(obj.id)

            # ── Mesh primitive ──
            mesh_idx = len(meshes)
            node_idx = len(nodes)

            # Vertex positions
            vertices = np.array(mesh.vertices, dtype=np.float32)
            pos_accessor = self._add_accessor(
                buf, accessors, buffer_views, vertices, "VEC3", 5126,
                target=34962,  # ARRAY_BUFFER
            )

            # Normals
            normal_accessor = None
            if mesh.normals and len(mesh.normals) == len(vertices):
                normals = np.array(mesh.normals, dtype=np.float32)
                normal_accessor = self._add_accessor(
                    buf, accessors, buffer_views, normals, "VEC3", 5126,
                    target=34962,
                )

            # UV coordinates
            uv_accessor = None
            if mesh.uv_coords and len(mesh.uv_coords) == len(vertices):
                uv_coords = np.array(mesh.uv_coords, dtype=np.float32)
                uv_accessor = self._add_accessor(
                    buf, accessors, buffer_views, uv_coords, "VEC2", 5126,
                    target=34962,
                )

            # Indices
            index_accessor = None
            if mesh.faces:
                faces = np.array(mesh.faces, dtype=np.uint16).flatten()
                index_accessor = self._add_accessor(
                    buf, accessors, buffer_views, faces, "SCALAR", 5123,
                    target=34963,  # ELEMENT_ARRAY_BUFFER
                )

            # Material
            mat_idx = len(materials)
            material = {
                "name": obj.label_custom or obj.id,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.7,
                },
            }

            # Add texture if available
            if mesh.texture_path and Path(mesh.texture_path).exists():
                tex_idx = self._add_texture(buf, buffer_views, images, textures, mesh.texture_path)
                if tex_idx is not None:
                    material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}

            materials.append(material)

            # Build primitive
            primitive = {
                "attributes": {"POSITION": pos_accessor},
                "mode": 4,  # TRIANGLES
                "material": mat_idx,
            }
            if normal_accessor is not None:
                primitive["attributes"]["NORMAL"] = normal_accessor
            if uv_accessor is not None:
                primitive["attributes"]["TEXCOORD_0"] = uv_accessor
            if index_accessor is not None:
                primitive["indices"] = index_accessor

            meshes.append({"name": obj.id, "primitives": [primitive]})

            # Node
            node = {
                "name": f"{obj.id}_{obj.label_custom or 'object'}",
                "mesh": mesh_idx,
            }
            nodes.append(node)

            # ── Animation ──
            if trajectory and trajectory.keyframes:
                anim = self._build_animation(
                    node_idx, trajectory, buf, accessors, buffer_views
                )
                if anim:
                    animations.append(anim)

        # ── Camera animation ──
        if camera_poses:
            cam_node_idx = len(nodes)
            cam_node = {
                "name": "AnimatedCamera",
                "camera": 0,
            }
            nodes.append(cam_node)

            # Add camera
            cameras = [{
                "type": "perspective",
                "perspective": {
                    "yfov": 1.047,  # 60 degrees in radians
                    "znear": 0.01,
                    "zfar": 1000.0,
                    "aspectRatio": 1.778,
                },
            }]

            # Camera animation
            if len(camera_poses) > 1:
                cam_anim = self._build_camera_animation(
                    cam_node_idx, camera_poses, buf, accessors, buffer_views
                )
                if cam_anim:
                    animations.append(cam_anim)
        else:
            cameras = []

        # ── Pad buffer to 4-byte alignment ──
        padding = (4 - (len(buf) % 4)) % 4
        buf += b"\x00" * padding

        # ── Build glTF JSON ──
        scene_node_indices = list(range(len(animated_objects)))
        if camera_poses:
            scene_node_indices.append(len(nodes) - 1)

        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "ToThinkVision 4D",
                "copyright": "Generated by ToThinkVision",
            },
            "scene": 0,
            "scenes": [{
                "name": "AnimatedScene",
                "nodes": scene_node_indices,
            }],
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "textures": textures,
            "images": images,
            "samplers": samplers,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buf)}],
        }

        if animations:
            gltf["animations"] = animations

        if cameras:
            gltf["cameras"] = cameras

        # ── Write GLB (binary glTF) ──
        output_path = output_dir / filename
        self._write_glb(output_path, gltf, bytes(buf))

        logger.info(
            "Exported animated glTF: %d objects, %d animations, %d keyframes total",
            len(animated_objects), len(animations),
            sum(len(a.get("channels", [])) for a in animations),
        )

        return output_path

    def _add_accessor(
        self,
        buf: bytearray,
        accessors: list,
        buffer_views: list,
        data: np.ndarray,
        type_str: str,
        component_type: int,
        target: int | None = None,
    ) -> int:
        """Add data to buffer and create accessor + buffer view."""
        data_bytes = data.tobytes()
        offset = len(buf)
        buf.extend(data_bytes)

        bv_idx = len(buffer_views)
        bv = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data_bytes),
        }
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)

        acc_idx = len(accessors)
        acc = {
            "bufferView": bv_idx,
            "componentType": component_type,
            "count": len(data) if data.ndim == 1 else data.shape[0],
            "type": type_str,
        }

        # Add min/max for position data
        if data.ndim == 2 and data.shape[1] <= 4:
            acc["min"] = data.min(axis=0).tolist()
            acc["max"] = data.max(axis=0).tolist()
        elif data.ndim == 1 and type_str == "SCALAR":
            acc["min"] = [float(data.min())]
            acc["max"] = [float(data.max())]

        accessors.append(acc)
        return acc_idx

    def _build_animation(
        self,
        node_idx: int,
        trajectory: ObjectTrajectory4D,
        buf: bytearray,
        accessors: list,
        buffer_views: list,
    ) -> dict | None:
        """Build glTF animation from ObjectTrajectory4D.

        Creates two channels per object:
        1. Translation (position over time)
        2. Rotation (orientation over time)
        """
        keyframes = trajectory.keyframes
        if len(keyframes) < 2:
            return None

        # Time input accessor
        times = np.array([kf.timestamp for kf in keyframes], dtype=np.float32)
        time_accessor = self._add_accessor(
            buf, accessors, buffer_views, times, "SCALAR", 5126,
        )

        channels = []
        anim_samplers = []

        # Translation channel
        translations = np.array([kf.position for kf in keyframes], dtype=np.float32)
        trans_accessor = self._add_accessor(
            buf, accessors, buffer_views, translations, "VEC3", 5126,
        )

        anim_samplers.append({
            "input": time_accessor,
            "output": trans_accessor,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": 0,
            "target": {
                "node": node_idx,
                "path": "translation",
            },
        })

        # Rotation channel
        # glTF uses (x, y, z, w) quaternion order
        rotations = np.array(
            [[kf.rotation[1], kf.rotation[2], kf.rotation[3], kf.rotation[0]]
             for kf in keyframes],
            dtype=np.float32,
        )
        rot_accessor = self._add_accessor(
            buf, accessors, buffer_views, rotations, "VEC4", 5126,
        )

        anim_samplers.append({
            "input": time_accessor,
            "output": rot_accessor,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": 1,
            "target": {
                "node": node_idx,
                "path": "rotation",
            },
        })

        # Scale channel (if non-uniform scale)
        scales = np.array([kf.scale for kf in keyframes], dtype=np.float32)
        scale_variation = np.any(np.abs(scales - 1.0) > 0.01)
        if scale_variation:
            scale_accessor = self._add_accessor(
                buf, accessors, buffer_views, scales, "VEC3", 5126,
            )
            anim_samplers.append({
                "input": time_accessor,
                "output": scale_accessor,
                "interpolation": "LINEAR",
            })
            channels.append({
                "sampler": 2,
                "target": {
                    "node": node_idx,
                    "path": "scale",
                },
            })

        return {
            "name": f"anim_{trajectory.object_id}",
            "channels": channels,
            "samplers": anim_samplers,
        }

    def _build_camera_animation(
        self,
        cam_node_idx: int,
        camera_poses: list[CameraPose],
        buf: bytearray,
        accessors: list,
        buffer_views: list,
    ) -> dict | None:
        """Build camera animation from recovered camera poses."""
        if len(camera_poses) < 2:
            return None

        # Subsample if too many poses
        max_poses = 100
        if len(camera_poses) > max_poses:
            step = len(camera_poses) // max_poses
            camera_poses = camera_poses[::step]

        # Time input (assume uniform spacing at 30fps)
        times = np.array([i / 30.0 for i in range(len(camera_poses))], dtype=np.float32)
        time_accessor = self._add_accessor(
            buf, accessors, buffer_views, times, "SCALAR", 5126,
        )

        # Positions
        positions = np.array(
            [list(p.position) if isinstance(p.position, tuple) else p.position
             for p in camera_poses],
            dtype=np.float32,
        )
        pos_accessor = self._add_accessor(
            buf, accessors, buffer_views, positions, "VEC3", 5126,
        )

        # Rotations (glTF order: x, y, z, w)
        rotations = np.array(
            [[p.rotation[1], p.rotation[2], p.rotation[3], p.rotation[0]]
             if isinstance(p.rotation, tuple) else
             [p.rotation[1], p.rotation[2], p.rotation[3], p.rotation[0]]
             for p in camera_poses],
            dtype=np.float32,
        )
        rot_accessor = self._add_accessor(
            buf, accessors, buffer_views, rotations, "VEC4", 5126,
        )

        return {
            "name": "camera_animation",
            "channels": [
                {"sampler": 0, "target": {"node": cam_node_idx, "path": "translation"}},
                {"sampler": 1, "target": {"node": cam_node_idx, "path": "rotation"}},
            ],
            "samplers": [
                {"input": time_accessor, "output": pos_accessor, "interpolation": "LINEAR"},
                {"input": time_accessor, "output": rot_accessor, "interpolation": "LINEAR"},
            ],
        }

    def _add_texture(
        self,
        buf: bytearray,
        buffer_views: list,
        images: list,
        textures: list,
        texture_path: str,
    ) -> int | None:
        """Add texture image to buffer, return texture index."""
        try:
            img = Image.open(texture_path).convert("RGBA")
            img_bytes = img.tobytes()
            img_w, img_h = img.size

            offset = len(buf)
            buf.extend(img_bytes)

            bv_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(img_bytes),
            })

            img_idx = len(images)
            images.append({
                "bufferView": bv_idx,
                "mimeType": "image/png",
                "width": img_w,
                "height": img_h,
            })

            tex_idx = len(textures)
            textures.append({"sampler": 0, "source": img_idx})

            return tex_idx
        except Exception:
            return None

    def _write_glb(self, path: Path, gltf_json: dict, binary_data: bytes):
        """Write binary glTF (.glb) file.

        GLB format:
        - Header (12 bytes): magic + version + length
        - JSON chunk: chunk_length + chunk_type + padded_json
        - Binary chunk: chunk_length + chunk_type + binary_data
        """
        # JSON chunk
        json_str = json.dumps(gltf_json, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")
        # Pad to 4-byte alignment with spaces
        json_padding = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * json_padding

        json_chunk_length = len(json_bytes)
        json_chunk_type = 0x4E4F534A  # "JSON" in little-endian

        # Binary chunk
        bin_padding = (4 - (len(binary_data) % 4)) % 4
        binary_data_padded = binary_data + b"\x00" * bin_padding

        bin_chunk_length = len(binary_data_padded)
        bin_chunk_type = 0x004E4942  # "BIN\x00" in little-endian

        # Total length
        total_length = 12 + 8 + json_chunk_length + 8 + bin_chunk_length

        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<III", 0x46546C67, 2, total_length))  # "glTF", version 2

            # JSON chunk
            f.write(struct.pack("<II", json_chunk_length, json_chunk_type))
            f.write(json_bytes)

            # Binary chunk
            f.write(struct.pack("<II", bin_chunk_length, bin_chunk_type))
            f.write(binary_data_padded)

        logger.info("Wrote GLB: %s (%.1f KB)", path, total_length / 1024)
