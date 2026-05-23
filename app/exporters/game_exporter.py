"""Module 2: Game Scene Exporter — v2 with 3D point cloud and mesh support."""

from __future__ import annotations

from pathlib import Path

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class GameExporter(BaseExporter):
    """Exports game scene structured data to Unity, UE, and collision formats."""

    def __init__(self, fmt: ExportFormat = ExportFormat.UNITY_JSON):
        self.fmt = fmt
        if fmt == ExportFormat.UNITY_JSON:
            self.format_name = "unity"
        elif fmt == ExportFormat.UE_JSON:
            self.format_name = "ue"
        else:
            self.format_name = "collision"

    def export(self, data: StructuredOutput) -> Path:
        if self.fmt == ExportFormat.UNITY_JSON:
            return self._export_unity(data)
        elif self.fmt == ExportFormat.UE_JSON:
            return self._export_ue(data)
        else:
            return self._export_collision(data)

    def _export_unity(self, data: StructuredOutput) -> Path:
        """Export as Unity-importable JSON with full 3D mesh support."""
        game_objects = []
        for obj in data.objects:
            unity_obj = {
                "name": obj.label_custom or obj.label.value,
                "id": obj.id,
                "tag": self._unity_tag(obj),
                "layer": self._unity_layer(obj),
                "transform": {
                    "position": {
                        "x": self._to_unity_coord(obj.bbox.x + obj.bbox.w / 2),
                        "y": self._to_unity_coord(obj.bbox.y + obj.bbox.h / 2),
                        "z": obj.bbox_3d.z if obj.bbox_3d else 0.0,
                    },
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "scale": {
                        "x": self._to_unity_coord(obj.bbox.w),
                        "y": self._to_unity_coord(obj.bbox.h),
                        "z": 1.0,
                    },
                },
                "components": [],
            }

            # ─── 3D mesh data if available ────
            if obj.mesh_3d and obj.mesh_3d.vertices:
                mesh = obj.mesh_3d
                unity_obj["mesh_3d"] = {
                    "has_mesh": True,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces) if mesh.faces else 0,
                    "bounds": mesh.bounds,
                    "texture_path": mesh.texture_path,
                    "mesh_file": obj.mesh_obj_file,
                    "import_as": "MeshFilter + MeshRenderer",
                }
                # Update transform with 3D position from mesh bounds
                if mesh.bounds:
                    center = mesh.bounds.get("min", [0, 0, 0])
                    extents = mesh.bounds.get("max", [0, 0, 0])
                    unity_obj["transform"]["position"] = {
                        "x": self._to_unity_coord((center[0] + extents[0]) / 2),
                        "y": self._to_unity_coord((center[1] + extents[1]) / 2),
                        "z": self._to_unity_coord((center[2] + extents[2]) / 2),
                    }
                    size = [extents[i] - center[i] for i in range(3)]
                    unity_obj["transform"]["scale"] = {
                        "x": self._to_unity_coord(size[0]),
                        "y": self._to_unity_coord(size[1]),
                        "z": self._to_unity_coord(size[2]),
                    }
                # Add mesh renderer component
                unity_obj["components"].append({
                    "type": "MeshFilter",
                    "source": "external" if mesh.texture_path else "procedural",
                })
                unity_obj["components"].append({
                    "type": "MeshRenderer",
                    "material": {
                        "metallic": 0.0,
                        "smoothness": 0.5,
                        "texture": mesh.texture_path,
                    },
                })
                unity_obj["components"].append({
                    "type": "MeshCollider",
                    "convex": len(mesh.vertices) < 256,
                })
            else:
                # Fallback: 2D collider
                if obj.relations.collision_with or obj.label.value.startswith("game_"):
                    unity_obj["components"].append({
                        "type": "BoxCollider2D",
                        "isTrigger": obj.label.value in ("game_door", "game_effect"),
                        "size": {"x": self._to_unity_coord(obj.bbox.w), "y": self._to_unity_coord(obj.bbox.h)},
                        "offset": {"x": 0, "y": 0},
                    })

            if obj.dominant_color and "mesh_3d" not in unity_obj:
                unity_obj["components"].append({
                    "type": "SpriteRenderer",
                    "color": obj.dominant_color,
                })

            if obj.label.value in ("game_npc", "game_item", "game_prop"):
                unity_obj["components"].append({
                    "type": "Rigidbody2D",
                    "bodyType": "Dynamic" if obj.label.value == "game_npc" else "Static",
                })

            game_objects.append(unity_obj)

        # Add 3D scene data
        scene_3d = None
        if data.point_cloud:
            scene_3d = {
                "point_cloud": {
                    "count": len(data.point_cloud.points),
                    "bounds": self._compute_bounds(data.point_cloud.points),
                },
                "camera_poses": len(data.camera_poses),
                "scene_mesh": data.scene_mesh_path,
            }

        output = {
            "format": "unity_json",
            "version": "2022.3",
            "scene_name": Path(data.source_file).stem,
            "dimensions": {
                "width": data.metadata.width if data.metadata else 1920,
                "height": data.metadata.height if data.metadata else 1080,
            },
            "game_objects": game_objects,
            "scene_3d": scene_3d,
            "physics_settings": {
                "gravity": {"x": 0, "y": -9.81},
                "pixels_per_unit": 100,
            },
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_ue(self, data: StructuredOutput) -> Path:
        """Export as Unreal Engine compatible JSON with 3D mesh support."""
        actors = []
        for obj in data.objects:
            actor = {
                "ActorLabel": obj.label_custom or obj.label.value,
                "ActorId": obj.id,
                "RootComponent": {
                    "Mobility": "Static" if obj.label.value in ("game_floor", "game_wall", "game_terrain") else "Movable",
                    "RelativeLocation": {
                        "X": obj.bbox_3d.x if obj.bbox_3d else self._to_unity_coord(obj.bbox.x + obj.bbox.w / 2),
                        "Y": obj.bbox_3d.y if obj.bbox_3d else self._to_unity_coord(obj.bbox.y + obj.bbox.h / 2),
                        "Z": obj.bbox_3d.z if obj.bbox_3d else 0.0,
                    },
                    "RelativeScale3D": {
                        "X": self._to_unity_coord(obj.bbox.w) / 100,
                        "Y": self._to_unity_coord(obj.bbox.h) / 100,
                        "Z": 1.0,
                    },
                },
                "Components": [],
            }

            # ─── 3D mesh data if available ────
            if obj.mesh_3d and obj.mesh_3d.vertices:
                mesh = obj.mesh_3d
                actor["StaticMesh"] = {
                    "has_mesh": True,
                    "vertices": len(mesh.vertices),
                    "texture_path": mesh.texture_path,
                    "mesh_file": obj.mesh_obj_file,
                }
                if mesh.bounds:
                    actor["RootComponent"]["RelativeLocation"] = {
                        "X": self._to_unity_coord((mesh.bounds["min"][0] + mesh.bounds["max"][0]) / 2),
                        "Y": self._to_unity_coord((mesh.bounds["min"][1] + mesh.bounds["max"][1]) / 2),
                        "Z": self._to_unity_coord((mesh.bounds["min"][2] + mesh.bounds["max"][2]) / 2),
                    }
                    size = [mesh.bounds["max"][i] - mesh.bounds["min"][i] for i in range(3)]
                    actor["RootComponent"]["RelativeScale3D"] = {
                        "X": self._to_unity_coord(size[0]),
                        "Y": self._to_unity_coord(size[1]),
                        "Z": self._to_unity_coord(size[2]),
                    }
                actor["Components"].append({
                    "Type": "StaticMeshComponent",
                    "CollisionEnabled": "QueryAndPhysics",
                })
            elif obj.label.value in ("game_floor", "game_wall", "game_terrain"):
                actor["Components"].append({
                    "Type": "StaticMeshComponent",
                    "CollisionEnabled": "QueryAndPhysics",
                })
            elif obj.label.value in ("game_npc", "game_item", "game_prop"):
                actor["Components"].append({
                    "Type": "SkeletalMeshComponent",
                    "CollisionEnabled": "QueryOnly",
                })

            if obj.dominant_color and "StaticMesh" not in actor:
                actor["Components"].append({
                    "Type": "MaterialInterface",
                    "ParameterValues": [{"Name": "BaseColor", "Value": obj.dominant_color}],
                })

            actors.append(actor)

        output = {
            "format": "ue_json",
            "version": "5.3",
            "level_name": Path(data.source_file).stem,
            "actors": actors,
            "world_settings": {
                "WorldGravityZ": -980.0,
                "DefaultFloorZ": 0.0,
            },
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_collision(self, data: StructuredOutput) -> Path:
        """Export as pure collision box data."""
        colliders = []
        for obj in data.objects:
            if not obj.label.value.startswith(("game_", "embodied_")):
                continue
            collider = {
                "id": obj.id,
                "type": "box",
                "center": {
                    "x": obj.bbox.x + obj.bbox.w / 2,
                    "y": obj.bbox.y + obj.bbox.h / 2,
                },
                "half_extents": {
                    "x": obj.bbox.w / 2,
                    "y": obj.bbox.h / 2,
                },
                "is_trigger": obj.label.value in ("game_door", "game_effect"),
                "layer": obj.label.value,
                "contour": obj.contour if obj.contour else None,
            }
            colliders.append(collider)

        output = {
            "format": "collision_json",
            "colliders": colliders,
            "collider_count": len(colliders),
        }
        return self.save_json(output, self._output_path(data.source_file))

    @staticmethod
    def _compute_bounds(points: list) -> dict:
        if not points:
            return {"min": [0, 0, 0], "max": [0, 0, 0]}
        import numpy as np
        pts = np.array(points)
        return {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
        }

    @staticmethod
    def _unity_tag(obj) -> str:
        tag_map = {
            "game_floor": "Ground", "game_wall": "Wall", "game_door": "Door",
            "game_npc": "NPC", "game_item": "Item", "game_prop": "Prop",
            "game_terrain": "Terrain", "game_effect": "Effect",
        }
        return tag_map.get(obj.label.value, "Untagged")

    @staticmethod
    def _unity_layer(obj) -> str:
        layer_map = {
            "game_floor": "Ground", "game_wall": "Wall",
            "game_npc": "Characters", "game_item": "Items",
            "game_prop": "Props", "game_terrain": "Terrain",
        }
        return layer_map.get(obj.label.value, "Default")

    @staticmethod
    def _to_unity_coord(val: float) -> float:
        return val / 100.0
