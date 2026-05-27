"""Blender scene exporter.

Generates:
1. Per-object OBJ mesh files
2. A Python script that runs inside Blender to:
   - Import all OBJ meshes
   - Set up materials and textures
   - Add keyframe animations from 4D trajectories
   - Create camera animation
   - Set up lighting and scene

Usage in Blender: File → Open → select the .py script, or
    blender --python scene_import.py

This approach avoids the heavy `bpy` dependency — the script runs
inside Blender's Python environment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.schemas import CameraPose, Mesh3D, ObjectTrajectory4D, StructuredObject

logger = logging.getLogger(__name__)


class BlenderExporter:
    """Export scene as Blender import script + OBJ meshes."""

    def export(
        self,
        objects: list[StructuredObject],
        trajectories: dict[str, ObjectTrajectory4D],
        camera_poses: list[CameraPose] | None = None,
        output_dir: Path | None = None,
        filename: str = "scene",
    ) -> Path:
        """Export Blender scene.

        Args:
            objects: Structured objects with mesh data
            trajectories: Per-object 6DoF trajectories
            camera_poses: Camera animation path
            output_dir: Output directory
            filename: Base filename (no extension)

        Returns:
            Path to the generated Python import script
        """
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_dir = output_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True)

        # ── Export per-object OBJ meshes ──
        obj_files = {}
        for obj in objects:
            if obj.mesh_3d is None:
                continue
            obj_path = mesh_dir / f"{self._sanitize(obj.id)}.obj"
            self._write_obj(obj.mesh_3d, obj_path, obj.id)
            obj_files[obj.id] = obj_path.name

        # ── Generate trajectory JSON ──
        traj_data = {}
        for oid, traj in trajectories.items():
            traj_data[oid] = {
                "object_id": traj.object_id,
                "motion_type": traj.motion_type,
                "keyframes": [
                    {
                        "timestamp": kf.timestamp,
                        "frame_idx": kf.frame_idx,
                        "position": list(kf.position),
                        "rotation": list(kf.rotation),  # (w, x, y, z)
                        "scale": list(kf.scale),
                    }
                    for kf in traj.keyframes
                ],
            }

        traj_path = output_dir / f"{filename}_trajectories.json"
        with open(traj_path, "w") as f:
            json.dump(traj_data, f, indent=2)

        # ── Generate camera path JSON ──
        cam_path_file = None
        if camera_poses:
            cam_data = [
                {
                    "frame_idx": p.frame_idx,
                    "position": list(p.position),
                    "rotation": list(p.rotation),  # (w, x, y, z)
                }
                for p in camera_poses
            ]
            cam_path_file = output_dir / f"{filename}_camera.json"
            with open(cam_path_file, "w") as f:
                json.dump(cam_data, f, indent=2)

        # ── Generate Blender import script ──
        script = self._generate_import_script(
            objects=objects,
            obj_files=obj_files,
            trajectories=trajectories,
            camera_poses=camera_poses,
            scene_name=filename,
        )

        script_path = output_dir / f"{filename}_import.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        logger.info(
            "Exported Blender scene: %d meshes, %d trajectories, script=%s",
            len(obj_files), len(trajectories), script_path.name,
        )

        return script_path

    def _write_obj(self, mesh: Mesh3D, path: Path, name: str):
        """Write OBJ file with optional MTL for textures."""
        vertices = np.array(mesh.vertices)
        normals = np.array(mesh.normals) if mesh.normals else None
        uv_coords = mesh.uv_coords
        faces = mesh.faces

        with open(path, "w") as f:
            f.write(f"# ToThinkVision OBJ Export - {name}\n")

            # MTL reference
            if mesh.texture_path and Path(mesh.texture_path).exists():
                mtl_name = Path(mesh.texture_path).stem
                mtl_path = path.with_suffix(".mtl")
                f.write(f"mtllib {mtl_path.name}\n")
                self._write_mtl(mtl_path, mesh.texture_path, mtl_name)

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Normals
            if normals is not None:
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # UV coordinates
            if uv_coords:
                for uv in uv_coords:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Faces
            if mesh.texture_path and Path(mesh.texture_path).exists():
                f.write(f"usemtl material_{self._sanitize(name)}\n")

            if faces:
                for face in faces:
                    if uv_coords and normals is not None:
                        # v/vt/vn format
                        parts = []
                        for idx in face:
                            parts.append(f"{idx+1}/{idx+1}/{idx+1}")
                        f.write(f"f {' '.join(parts)}\n")
                    elif normals is not None:
                        # v//vn format
                        parts = [f"{idx+1}//{idx+1}" for idx in face]
                        f.write(f"f {' '.join(parts)}\n")
                    else:
                        # v format
                        parts = [str(idx+1) for idx in face]
                        f.write(f"f {' '.join(parts)}\n")

    def _write_mtl(self, mtl_path: Path, texture_path: str, name: str):
        """Write MTL material file."""
        with open(mtl_path, "w") as f:
            f.write(f"# ToThinkVision MTL - {name}\n")
            f.write(f"newmtl material_{name}\n")
            f.write("Ka 0.1 0.1 0.1\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 50.0\n")
            f.write("d 1.0\n")
            f.write(f"map_Kd {Path(texture_path).name}\n")

        # Copy texture to same directory
        import shutil
        tex_src = Path(texture_path)
        tex_dst = mtl_path.parent / tex_src.name
        if tex_src.exists() and not tex_dst.exists():
            shutil.copy2(tex_src, tex_dst)

    def _generate_import_script(
        self,
        objects: list[StructuredObject],
        obj_files: dict[str, str],
        trajectories: dict[str, ObjectTrajectory4D],
        camera_poses: list[CameraPose] | None,
        scene_name: str,
    ) -> str:
        """Generate Blender Python import script."""
        return f'''#!/usr/bin/env python3
"""ToThinkVision 4D Scene Import Script for Blender.

Usage:
  1. Open Blender
  2. File → Open → select this .py file
  3. Or run from command line: blender --python {scene_name}_import.py

This script imports the 4D scene with animated objects.
"""

import json
import math
import os
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector, Euler

# ─── Configuration ───────────────────────────────────────────
SCRIPT_DIR = Path(bpy.data.filepath).parent if bpy.data.filepath else Path(__file__).parent
MESH_DIR = SCRIPT_DIR / "meshes"
TRAJ_FILE = SCRIPT_DIR / "{scene_name}_trajectories.json"
CAM_FILE = SCRIPT_DIR / "{scene_name}_camera.json"
FPS = 30


def clear_scene():
    """Clear existing objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def import_meshes():
    """Import all OBJ mesh files."""
    imported = {{}}
    mesh_dir = str(MESH_DIR)

    for obj_file in MESH_DIR.glob("*.obj"):
        bpy.ops.wm.obj_import(filepath=str(obj_file), directory=mesh_dir)
        obj_name = obj_file.stem
        imported[obj_name] = bpy.context.selected_objects[-1] if bpy.context.selected_objects else None
        if imported[obj_name]:
            imported[obj_name].name = obj_name

    return imported


def load_trajectories():
    """Load trajectory data from JSON."""
    if not TRAJ_FILE.exists():
        return {{}}
    with open(TRAJ_FILE) as f:
        return json.load(f)


def apply_animation(objects, trajectories):
    """Apply keyframe animation to objects."""
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = 1

    for obj_name, obj in objects.items():
        if obj is None:
            continue

        # Find matching trajectory
        traj = None
        for tid, tdata in trajectories.items():
            sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in tid).strip("_")
            if sanitized == obj_name or tid == obj_name:
                traj = tdata
                break

        if traj is None or not traj.get("keyframes"):
            continue

        keyframes = traj["keyframes"]

        # Update frame range
        max_frame = int(keyframes[-1]["timestamp"] * FPS)
        scene.frame_end = max(scene.frame_end, max_frame)

        for kf in keyframes:
            frame = int(kf["timestamp"] * FPS)

            # Position
            pos = kf["position"]
            obj.location = Vector((pos[0], pos[1], pos[2]))
            obj.keyframe_insert(data_path="location", frame=frame)

            # Rotation (quaternion w,x,y,z → Blender quaternion)
            rot = kf["rotation"]
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = Quaternion((rot[0], rot[1], rot[2], rot[3]))
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            # Scale
            scale = kf["scale"]
            obj.scale = Vector((scale[0], scale[1], scale[2]))
            obj.keyframe_insert(data_path="scale", frame=frame)


def setup_camera(camera_poses=None):
    """Create and animate camera."""
    # Create camera if none exists
    if not bpy.data.cameras:
        cam_data = bpy.data.cameras.new("MainCamera")
        cam_obj = bpy.data.objects.new("MainCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
    else:
        cam_obj = [o for o in bpy.data.objects if o.type == "CAMERA"][0]

    cam_obj.data.lens = 35

    # Load camera path
    if CAM_FILE.exists():
        with open(CAM_FILE) as f:
            cam_poses = json.load(f)

    if not cam_poses:
        # Default camera position
        cam_obj.location = Vector((0, -5, 2))
        cam_obj.rotation_euler = Euler((math.radians(75), 0, 0))
        return

    # Animate camera
    for i, pose in enumerate(cam_poses):
        frame = i
        pos = pose["position"]
        rot = pose["rotation"]

        cam_obj.location = Vector((pos[0], pos[1], pos[2]))
        cam_obj.keyframe_insert(data_path="location", frame=frame)

        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = Quaternion((rot[0], rot[1], rot[2], rot[3]))
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Set as active camera
    bpy.context.scene.camera = cam_obj


def setup_lighting():
    """Add basic 3-point lighting."""
    # Key light
    key_data = bpy.data.lights.new("KeyLight", type="AREA")
    key_data.energy = 500
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = Vector((3, -3, 5))
    key_obj.rotation_euler = Euler((math.radians(45), 0, math.radians(45)))
    bpy.context.collection.objects.link(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new("FillLight", type="AREA")
    fill_data.energy = 200
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = Vector((-3, -2, 3))
    fill_obj.rotation_euler = Euler((math.radians(55), 0, math.radians(-30)))
    bpy.context.collection.objects.link(fill_obj)

    # Rim light
    rim_data = bpy.data.lights.new("RimLight", type="AREA")
    rim_data.energy = 300
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = Vector((0, 3, 4))
    rim_obj.rotation_euler = Euler((math.radians(-45), 0, 0))
    bpy.context.collection.objects.link(rim_obj)


def setup_render():
    """Configure render settings."""
    scene = bpy.context.scene

    # Use Cycles for quality
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128

    # Output settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = FPS

    # Set interpolation to linear for all fcurves
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = "LINEAR"


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ToThinkVision 4D Scene Import")
    print("=" * 40)

    clear_scene()
    objects = import_meshes()
    print(f"Imported {{len(objects)}} meshes")

    trajectories = load_trajectories()
    print(f"Loaded {{len(trajectories)}} trajectories")

    apply_animation(objects, trajectories)
    setup_camera()
    setup_lighting()
    setup_render()

    # Set timeline
    scene = bpy.context.scene
    scene.frame_set(0)

    print(f"Scene ready! Frames: {{scene.frame_start}}-{{scene.frame_end}}")
    print(f"Press Space to play animation.")
'''

    @staticmethod
    def _sanitize(name: str) -> str:
        """Make a valid identifier."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_") or "object"
