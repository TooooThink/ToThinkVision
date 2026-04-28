"""Module 3: Video Motion Exporter — converts structured video data to AE keyframes, trajectory, and PR markers."""

from __future__ import annotations

import csv
from pathlib import Path

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class VideoExporter(BaseExporter):
    """Exports video motion structured data to AE keyframes, trajectory CSV, and PR markers."""

    def __init__(self, fmt: ExportFormat = ExportFormat.AE_KEYFRAMES):
        self.fmt = fmt
        if fmt == ExportFormat.AE_KEYFRAMES:
            self.format_name = "ae_keyframes"
        elif fmt == ExportFormat.VIDEO_TRAJECTORY:
            self.format_name = "video_trajectory"
            self.file_extension = ".csv"
            self.mime_type = "text/csv"
        else:
            self.format_name = "pr_markers"

    def export(self, data: StructuredOutput) -> Path:
        if self.fmt == ExportFormat.AE_KEYFRAMES:
            return self._export_ae_keyframes(data)
        elif self.fmt == ExportFormat.VIDEO_TRAJECTORY:
            return self._export_trajectory(data)
        else:
            return self._export_pr_markers(data)

    def _export_ae_keyframes(self, data: StructuredOutput) -> Path:
        """Export as After Effects-compatible keyframe JSON."""
        fps = data.metadata.fps if data.metadata else 30.0
        compositions = []

        # Group objects by track
        object_tracks: dict[str, list[dict]] = {}
        for obj in data.objects:
            if obj.id not in object_tracks:
                object_tracks[obj.id] = []
            if obj.temporal.trajectory:
                for point in obj.temporal.trajectory:
                    object_tracks[obj.id].append(point)
            else:
                # Single frame object
                object_tracks[obj.id].append({
                    "x": obj.bbox.x + obj.bbox.w / 2,
                    "y": obj.bbox.y + obj.bbox.h / 2,
                    "t": obj.temporal.frame_index,
                })

        for obj in data.objects:
            track_points = object_tracks.get(obj.id, [])
            if not track_points:
                continue

            keyframes = []
            for pt in track_points:
                t_seconds = pt["t"] / fps if fps > 0 else 0
                kf = {
                    "time": round(t_seconds, 4),
                    "frame": pt["t"],
                    "position": {
                        "x": round(pt["x"], 2),
                        "y": round(pt["y"], 2),
                    },
                    "scale": {"x": 100, "y": 100},
                    "rotation": 0,
                    "opacity": 100,
                }
                keyframes.append(kf)

            if not keyframes:
                continue

            composition_layer = {
                "layer_name": obj.label_custom or obj.label.value,
                "layer_id": obj.id,
                "type": "shape",
                "parent": obj.relations.parent_id,
                "in_point": obj.temporal.appear_frame / fps if fps > 0 else 0,
                "out_point": (obj.temporal.disappear_frame / fps if obj.temporal.disappear_frame > 0 else data.frame_count / fps) if fps > 0 else 0,
                "keyframes": {
                    "position": keyframes,
                },
            }

            if obj.dominant_color:
                composition_layer["fill_color"] = obj.dominant_color

            if obj.text_content:
                composition_layer["text"] = obj.text_content

            compositions.append(composition_layer)

        output = {
            "format": "ae_keyframes",
            "composition_name": Path(data.source_file).stem,
            "width": data.metadata.width if data.metadata else 1920,
            "height": data.metadata.height if data.metadata else 1080,
            "frame_rate": fps,
            "duration_frames": data.frame_count,
            "duration_seconds": data.metadata.duration_seconds if data.metadata else 0,
            "layers": compositions,
            "layer_count": len(compositions),
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_trajectory(self, data: StructuredOutput) -> Path:
        """Export as CSV trajectory file."""
        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "object_id", "label", "frame", "center_x", "center_y",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "velocity_x", "velocity_y",
            ])
            for obj in data.objects:
                if obj.temporal.trajectory:
                    for pt in obj.temporal.trajectory:
                        writer.writerow([
                            obj.id,
                            obj.label.value,
                            pt["t"],
                            round(pt["x"], 2),
                            round(pt["y"], 2),
                            round(obj.bbox.x, 2),
                            round(obj.bbox.y, 2),
                            round(obj.bbox.w, 2),
                            round(obj.bbox.h, 2),
                            round(obj.temporal.velocity.get("vx", 0), 2) if obj.temporal.velocity else 0,
                            round(obj.temporal.velocity.get("vy", 0), 2) if obj.temporal.velocity else 0,
                        ])
                else:
                    cx = obj.bbox.x + obj.bbox.w / 2
                    cy = obj.bbox.y + obj.bbox.h / 2
                    writer.writerow([
                        obj.id,
                        obj.label.value,
                        obj.temporal.frame_index,
                        round(cx, 2),
                        round(cy, 2),
                        round(obj.bbox.x, 2),
                        round(obj.bbox.y, 2),
                        round(obj.bbox.w, 2),
                        round(obj.bbox.h, 2),
                        0, 0,
                    ])

        return out_path

    def _export_pr_markers(self, data: StructuredOutput) -> Path:
        """Export as Premiere Pro marker JSON."""
        fps = data.metadata.fps if data.metadata else 30.0
        markers = []

        # Group by frame to create key scene markers
        frame_objects: dict[int, list] = {}
        for obj in data.objects:
            fi = obj.temporal.frame_index
            if fi not in frame_objects:
                frame_objects[fi] = []
            frame_objects[fi].append(obj)

        # Create markers for frames with significant changes
        for frame_idx in sorted(frame_objects.keys()):
            objs = frame_objects[frame_idx]
            new_objs = [o for o in objs if o.temporal.appear_frame == frame_idx]
            if new_objs:
                time_sec = frame_idx / fps if fps > 0 else 0
                labels = ", ".join(set(o.label_custom or o.label.value for o in new_objs[:5]))
                markers.append({
                    "name": f"Frame {frame_idx}: {labels}",
                    "timecode": self._frames_to_timecode(frame_idx, fps),
                    "time_seconds": round(time_sec, 3),
                    "duration_seconds": round(1.0 / fps, 3),
                    "comment": f"Objects appeared: {labels}",
                    "object_ids": [o.id for o in new_objs],
                })

        output = {
            "format": "pr_markers",
            "source": data.source_file,
            "frame_rate": fps,
            "total_frames": data.frame_count,
            "markers": markers,
            "marker_count": len(markers),
        }
        return self.save_json(output, self._output_path(data.source_file))

    @staticmethod
    def _frames_to_timecode(frame: int, fps: float) -> str:
        """Convert frame number to HH:MM:SS:FF timecode."""
        total_seconds = frame / fps if fps > 0 else 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = frame % int(fps) if fps > 0 else 0
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
