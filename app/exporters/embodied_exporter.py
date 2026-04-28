"""Module 4: Embodied AI Exporter — converts structured scene data to robot action sequences and pose data."""

from __future__ import annotations

import csv
from pathlib import Path

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class EmbodiedExporter(BaseExporter):
    """Exports embodied AI structured data to robot action sequences and pose CSV."""

    def __init__(self, fmt: ExportFormat = ExportFormat.EMBODIED_JSON):
        self.fmt = fmt
        if fmt == ExportFormat.ROBOT_ACTION:
            self.format_name = "robot_action"
        elif fmt == ExportFormat.POSE_CSV:
            self.format_name = "pose_csv"
            self.file_extension = ".csv"
            self.mime_type = "text/csv"
        else:
            self.format_name = "embodied_json"

    def export(self, data: StructuredOutput) -> Path:
        if self.fmt == ExportFormat.ROBOT_ACTION:
            return self._export_robot_action(data)
        elif self.fmt == ExportFormat.POSE_CSV:
            return self._export_pose_csv(data)
        else:
            return self._export_embodied_json(data)

    def _export_embodied_json(self, data: StructuredOutput) -> Path:
        """Export as comprehensive embodied AI scene description."""
        scene_objects = []
        for obj in data.objects:
            scene_obj = {
                "id": obj.id,
                "type": obj.label.value,
                "label": obj.label_custom,
                "pose_3d": {
                    "position": {
                        "x": obj.bbox_3d.x if obj.bbox_3d else 0.0,
                        "y": obj.bbox_3d.y if obj.bbox_3d else 0.0,
                        "z": obj.bbox_3d.z if obj.bbox_3d else 0.0,
                    },
                    "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                },
                "bounding_box_3d": {
                    "length": obj.bbox.w / 100.0,
                    "width": obj.bbox.h / 100.0,
                    "height": 0.1,
                },
                "depth": obj.depth_value,
                "physical_properties": {
                    "is_obstacle": obj.label.value in ("embodied_obstacle", "game_wall", "game_terrain"),
                    "is_graspable": obj.label.value in ("embodied_tool", "game_item", "game_prop"),
                    "is_surface": obj.label.value in ("embodied_surface", "game_floor"),
                    "friction": 0.5,
                },
                "trajectory": obj.temporal.trajectory,
                "confidence": obj.confidence,
            }
            scene_objects.append(scene_obj)

        # Generate interaction sequence
        interaction_sequence = self._generate_interaction_sequence(data)

        output = {
            "format": "embodied_json",
            "scene_name": Path(data.source_file).stem,
            "objects": scene_objects,
            "object_count": len(scene_objects),
            "interaction_sequence": interaction_sequence,
            "metadata": {
                "fps": data.metadata.fps if data.metadata else 30.0,
                "total_frames": data.frame_count,
            },
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_robot_action(self, data: StructuredOutput) -> Path:
        """Export as robot action sequence for training."""
        actions = []
        fps = data.metadata.fps if data.metadata else 30.0

        # Sort objects by appearance frame
        sorted_objects = sorted(data.objects, key=lambda o: o.temporal.appear_frame)

        action_id = 0
        for obj in sorted_objects:
            if not obj.bbox_3d:
                continue

            # Generate approach action
            actions.append({
                "action_id": f"act_{action_id:04d}",
                "action_type": "approach",
                "target_id": obj.id,
                "target_label": obj.label_custom or obj.label.value,
                "timestamp": obj.temporal.appear_frame / fps if fps > 0 else 0,
                "frame": obj.temporal.appear_frame,
                "end_effector_pose": {
                    "position": {
                        "x": obj.bbox_3d.x / 100.0,
                        "y": obj.bbox_3d.y / 100.0,
                        "z": obj.bbox_3d.z / 100.0 + 0.1,
                    },
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "gripper_open": True,
                "force_limit": 10.0,
                "speed": 0.1,
            })
            action_id += 1

            # If object is graspable, add grasp action
            if obj.label.value in ("embodied_tool", "game_item", "game_prop"):
                actions.append({
                    "action_id": f"act_{action_id:04d}",
                    "action_type": "grasp",
                    "target_id": obj.id,
                    "target_label": obj.label_custom or obj.label.value,
                    "timestamp": (obj.temporal.appear_frame + 1) / fps if fps > 0 else 0,
                    "frame": obj.temporal.appear_frame + 1,
                    "end_effector_pose": {
                        "position": {
                            "x": obj.bbox_3d.x / 100.0,
                            "y": obj.bbox_3d.y / 100.0,
                            "z": obj.bbox_3d.z / 100.0,
                        },
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "gripper_open": False,
                    "force_limit": 20.0,
                    "speed": 0.05,
                })
                action_id += 1

        output = {
            "format": "robot_action",
            "source": data.source_file,
            "actions": actions,
            "action_count": len(actions),
            "total_duration": data.metadata.duration_seconds if data.metadata else 0,
            "frame_rate": fps,
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_pose_csv(self, data: StructuredOutput) -> Path:
        """Export as CSV pose file."""
        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "object_id", "label",
                "pos_x", "pos_y", "pos_z",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "depth", "is_moving", "velocity_x", "velocity_y",
            ])
            for obj in data.objects:
                is_moving = len(obj.temporal.trajectory) > 1
                writer.writerow([
                    obj.temporal.frame_index,
                    obj.id,
                    obj.label.value,
                    round(obj.bbox_3d.x, 4) if obj.bbox_3d else 0,
                    round(obj.bbox_3d.y, 4) if obj.bbox_3d else 0,
                    round(obj.bbox_3d.z, 4) if obj.bbox_3d else 0,
                    round(obj.bbox.x, 2),
                    round(obj.bbox.y, 2),
                    round(obj.bbox.w, 2),
                    round(obj.bbox.h, 2),
                    round(obj.depth_value or 0, 4),
                    is_moving,
                    round(obj.temporal.velocity.get("vx", 0), 2) if obj.temporal.velocity else 0,
                    round(obj.temporal.velocity.get("vy", 0), 2) if obj.temporal.velocity else 0,
                ])

        return out_path

    @staticmethod
    def _generate_interaction_sequence(data: StructuredOutput) -> list[dict]:
        """Generate a high-level interaction sequence from detected objects."""
        sequence = []
        step = 0

        # Find targets, obstacles, and tools
        targets = [o for o in data.objects if o.label.value in ("embodied_target", "game_npc", "game_item")]
        obstacles = [o for o in data.objects if o.label.value in ("embodied_obstacle", "game_wall")]
        tools = [o for o in data.objects if o.label.value in ("embodied_tool", "game_prop")]

        for target in targets:
            path = {"step": step, "action": "navigate_to", "target": target.id}
            if target.bbox_3d:
                path["goal_position"] = {
                    "x": target.bbox_3d.x / 100.0,
                    "y": target.bbox_3d.y / 100.0,
                }
            sequence.append(path)
            step += 1

            # Check if tool needed
            for tool in tools:
                sequence.append({
                    "step": step,
                    "action": "pick_up",
                    "target": tool.id,
                    "tool_type": tool.label.value,
                })
                step += 1

            sequence.append({"step": step, "action": "interact", "target": target.id})
            step += 1

        for obstacle in obstacles:
            sequence.append({
                "step": step,
                "action": "avoid",
                "obstacle": obstacle.id,
                "obstacle_bounds": {
                    "x": obstacle.bbox.x,
                    "y": obstacle.bbox.y,
                    "w": obstacle.bbox.w,
                    "h": obstacle.bbox.h,
                },
            })
            step += 1

        return sequence
