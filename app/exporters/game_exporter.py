"""Module 2: Game Scene Exporter — converts structured game objects to Unity/UE formats."""

from __future__ import annotations

from pathlib import Path

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class GameExporter(BaseExporter):
    """Exports game scene structured data to Unity, UE, and collision box formats."""

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
        """Export as Unity-importable JSON (prefab-like structure)."""
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

            # Collider component
            if obj.relations.collision_with or obj.label.value.startswith("game_"):
                unity_obj["components"].append({
                    "type": "BoxCollider2D" if obj.bbox.w > 0 and obj.bbox.h > 0 else "PolygonCollider2D",
                    "isTrigger": obj.label.value in ("game_door", "game_effect"),
                    "size": {"x": self._to_unity_coord(obj.bbox.w), "y": self._to_unity_coord(obj.bbox.h)},
                    "offset": {"x": 0, "y": 0},
                })

            # Renderer component
            if obj.dominant_color:
                unity_obj["components"].append({
                    "type": "SpriteRenderer",
                    "color": obj.dominant_color,
                })

            # Rigidbody for dynamic objects
            if obj.label.value in ("game_npc", "game_item", "game_prop"):
                unity_obj["components"].append({
                    "type": "Rigidbody2D",
                    "bodyType": "Dynamic" if obj.label.value == "game_npc" else "Static",
                })

            game_objects.append(unity_obj)

        output = {
            "format": "unity_json",
            "version": "2022.3",
            "scene_name": Path(data.source_file).stem,
            "dimensions": {
                "width": data.metadata.width if data.metadata else 1920,
                "height": data.metadata.height if data.metadata else 1080,
            },
            "game_objects": game_objects,
            "physics_settings": {
                "gravity": {"x": 0, "y": -9.81},
                "pixels_per_unit": 100,
            },
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_ue(self, data: StructuredOutput) -> Path:
        """Export as Unreal Engine compatible JSON (actor-like structure)."""
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

            if obj.label.value in ("game_floor", "game_wall", "game_terrain"):
                actor["Components"].append({
                    "Type": "StaticMeshComponent",
                    "CollisionEnabled": "QueryAndPhysics",
                })
            elif obj.label.value in ("game_npc", "game_item", "game_prop"):
                actor["Components"].append({
                    "Type": "SkeletalMeshComponent",
                    "CollisionEnabled": "QueryOnly",
                })

            if obj.dominant_color:
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
    def _unity_tag(obj) -> str:
        tag_map = {
            "game_floor": "Ground",
            "game_wall": "Wall",
            "game_door": "Door",
            "game_npc": "NPC",
            "game_item": "Item",
            "game_prop": "Prop",
            "game_terrain": "Terrain",
            "game_effect": "Effect",
        }
        return tag_map.get(obj.label.value, "Untagged")

    @staticmethod
    def _unity_layer(obj) -> str:
        layer_map = {
            "game_floor": "Ground",
            "game_wall": "Wall",
            "game_npc": "Characters",
            "game_item": "Items",
            "game_prop": "Props",
            "game_terrain": "Terrain",
        }
        return layer_map.get(obj.label.value, "Default")

    @staticmethod
    def _to_unity_coord(val: float) -> float:
        """Convert pixel coordinate to Unity world units (assume 100 px = 1 unit)."""
        return val / 100.0
