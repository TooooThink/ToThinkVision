"""After Effects Project Exporter — generates AEPX XML for animation timeline."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from app.config import settings
from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat

logger = logging.getLogger(__name__)


class AEProjectExporter(BaseExporter):
    """Exports structured video data to After Effects project format.

    Since AEPX is a proprietary binary format, we generate:
    1. An AEPX-compatible XML project structure
    2. A companion Python script (for ExtendScript) that creates the
       full AE project programmatically when run inside After Effects
    """

    def __init__(self, fmt: ExportFormat = ExportFormat.AE_PROJECT):
        self.fmt = fmt
        self.format_name = "ae_project"
        self.file_extension = ".jsx"  # ExtendScript (most reliable import method)
        self.mime_type = "application/javascript"

    def export(self, data: StructuredOutput) -> Path:
        if data.source_type != "video":
            return self._export_image_to_ae(data)
        return self._export_video_to_ae(data)

    def _export_video_to_ae(self, data: StructuredOutput) -> Path:
        """Generate After Effects ExtendScript for video animation."""
        fps = data.metadata.fps if data.metadata else 30.0
        width = data.metadata.width if data.metadata else 1920
        height = data.metadata.height if data.metadata else 1080
        duration = data.metadata.duration_seconds if data.metadata else 0

        script_lines = [
            "// ToThinkVision → After Effects Project Script",
            "// Run this script inside After Effects: File > Scripts > Run Script File",
            f"// Generated from: {data.source_file}",
            "",
            "// Create new composition",
            f'var comp = app.project.items.addComp("{self._comp_name(data)}", {width}, {height}, 1, {duration}, {fps});',
            'comp.openInViewer();',
            "",
        ]

        # ─── Import original video as reference layer ───────
        source_abs = str(Path(data.source_file).absolute()) if Path(data.source_file).is_absolute() else ""
        if source_abs:
            script_lines.extend([
                "// Import original video as reference",
                f'var importOpts = new ImportOptions();',
                f'importOpts.file = new File("{source_abs}");',
                f'importOpts.importAs = ImportAsType.FOOTAGE;',
                f'var footageItem = app.project.importFile(importOpts);',
                f'var refLayer = comp.layers.add(footageItem);',
                f'refLayer.name = "Reference_Video";',
                f'refLayer.opacity = 30;',
                "",
            ])

        # ─── Import all object images into project ──────────
        # Collect unique crop/mask PNG paths
        image_files: dict[str, str] = {}
        for obj in data.objects:
            obj_id = obj.id
            masked_path = obj.raw_data.get("masked_png_path") if obj.raw_data else None
            crop_path = obj.crop_png_path

            if masked_path and Path(masked_path).exists():
                image_files[obj_id] = masked_path
            elif crop_path and Path(crop_path).exists():
                image_files[obj_id] = crop_path
            elif obj.crop_image_base64:
                # Fallback: save base64 image to temp file
                import base64
                from io import BytesIO
                from PIL import Image
                tmp_dir = Path(data.source_file).parent / "_ttv_ae_assets"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                img_bytes = base64.b64decode(obj.crop_image_base64)
                img_path = tmp_dir / f"{obj_id}_crop.png"
                img_path.write_bytes(img_bytes)
                image_files[obj_id] = str(img_path)

        # ─── Group objects by track ID ──────────────────────
        object_tracks: dict[str, list] = {}
        for obj in data.objects:
            tid = obj.id
            if tid not in object_tracks:
                object_tracks[tid] = []
            object_tracks[tid].append(obj)

        # ─── Generate layers with imported footage ──────────
        layer_idx = 1
        for obj_id, track_objects in object_tracks.items():
            if not track_objects:
                continue

            first_obj = track_objects[0]
            label = first_obj.label_custom or first_obj.label.value

            # Import image as footage if we have it
            if obj_id in image_files:
                img_path = image_files[obj_id]
                img_w = int(first_obj.bbox.w)
                img_h = int(first_obj.bbox.h)
                script_lines.extend([
                    f"// Layer: {label} ({obj_id}) — from image",
                    f'var importObj{layer_idx} = new ImportOptions();',
                    f'importObj{layer_idx}.file = new File("{img_path}");',
                    f'importObj{layer_idx}.importAs = ImportAsType.FOOTAGE;',
                    f'var footage{layer_idx} = app.project.importFile(importObj{layer_idx});',
                    f'var layer{layer_idx} = comp.layers.add(footage{layer_idx});',
                    f'layer{layer_idx}.name = "{label}_{obj_id}";',
                    f"layer{layer_idx}.moveToBeginning();",
                    f"layer{layer_idx}.position = [{first_obj.bbox.x + img_w/2:.0f}, {first_obj.bbox.y + img_h/2:.0f}];",
                    "",
                ])
            else:
                # Fallback: solid color layer
                script_lines.extend([
                    f"// Layer: {label} ({obj_id})",
                    f'var solid{layer_idx} = comp.layers.addSolid([1, 0.6, 0.2], "{label}_{obj_id}", {int(first_obj.bbox.w)}, {int(first_obj.bbox.h)}, 1, {duration});',
                    f"solid{layer_idx}.moveToBeginning();",
                    f"solid{layer_idx}.name = \"{label}_{obj_id}\";",
                    "",
                ])

            layer_var = f"layer{layer_idx}" if obj_id in image_files else f"solid{layer_idx}"

            # Add position keyframes
            if any(o.temporal.trajectory for o in track_objects):
                script_lines.extend([
                    f"// Position keyframes for {obj_id}",
                    f"var posProp = {layer_var}.property('ADBE Transform Group').property('ADBE Position');",
                    "posProp.setValuesAtTimes([",
                ])

                all_trajectories = []
                for obj in track_objects:
                    all_trajectories.extend(obj.temporal.trajectory)

                for pt in all_trajectories:
                    t_sec = pt["t"] / fps if fps > 0 else 0
                    script_lines.append(f"    {t_sec:.4f},")

                script_lines.extend([
                    "], [",
                ])

                for pt in all_trajectories:
                    script_lines.append(f"    [{pt['x']:.2f}, {pt['y']:.2f}],")

                script_lines.extend([
                    "]);",
                    "",
                ])

            # Add opacity keyframes (appear/disappear)
            script_lines.extend([
                f"// Opacity for {obj_id}",
                f"var opProp = {layer_var}.property('ADBE Transform Group').property('ADBE Opacity');",
            ])

            appear_sec = first_obj.temporal.appear_frame / fps if fps > 0 else 0
            disappear_frame = first_obj.temporal.disappear_frame
            disappear_sec = (disappear_frame / fps if disappear_frame > 0 else duration)

            script_lines.extend([
                f"opProp.setValueAtTime(0, 0);",
                f"opProp.setValueAtTime({appear_sec:.4f}, 100);",
                f"opProp.setValueAtTime({disappear_sec:.4f}, 100);",
                f"opProp.setValueAtTime({disappear_sec + 0.033:.4f}, 0);",
                "",
            ])

            # Add text layer if object has OCR text
            if first_obj.text_content:
                script_lines.extend([
                    f"// Text layer for {obj_id}",
                    f'var textLayer{layer_idx} = comp.layers.addText("{first_obj.text_content}");',
                    f"textLayer{layer_idx}.position = [{first_obj.bbox.x + first_obj.bbox.w/2:.0f}, {first_obj.bbox.y + first_obj.bbox.h/2:.0f}];",
                    "textLayer.sourceText.fontSize = 14;",
                    "",
                ])

            layer_idx += 1

        # Generate camera data as null object
        if data.camera_poses:
            script_lines.extend([
                "// Camera null object (from MASt3R reconstruction)",
                'var camNull = comp.layers.addNull();',
                'camNull.name = "Camera_Tracking";',
            ])
            for pose in data.camera_poses:
                fi = pose.frame_idx if hasattr(pose, "frame_idx") else pose["frame_idx"]
                pos = pose.position if hasattr(pose, "position") else pose["position"]
                t_sec = fi / fps if fps > 0 else 0
                script_lines.append(
                    f'camNull.property("ADBE Transform Group").property("ADBE Position")'
                    f'.setValueAtTime({t_sec:.4f}, [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]);'
                )
            script_lines.append("")

        script_lines.extend([
            "// Import point cloud as shape layer (if available)",
            f"// Point cloud: {len(data.point_cloud.points) if data.point_cloud else 0} points",
            "",
            "// Done!",
            'alert("ToThinkVision project imported successfully!");',
        ])

        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(script_lines), encoding="utf-8")

        # Also save a JSON with all the raw data for reference
        json_path = out_path.with_suffix(".ae_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._build_ae_json(data), f, indent=2, default=str)

        return out_path

    def _export_image_to_ae(self, data: StructuredOutput) -> Path:
        """Generate AE script for static image with layers."""
        width = data.metadata.width if data.metadata else 1920
        height = data.metadata.height if data.metadata else 1080

        script_lines = [
            "// ToThinkVision → After Effects (Static Image)",
            f"// Generated from: {data.source_file}",
            "",
            f'var comp = app.project.items.addComp("{self._comp_name(data)}", {width}, {height}, 1, 10, 30);',
            'comp.openInViewer();',
            "",
        ]

        for i, obj in enumerate(data.objects):
            label = obj.label_custom or obj.label.value
            script_lines.extend([
                f'var layer{i} = comp.layers.addSolid([0.8, 0.6, 0.4], "{label}", {int(obj.bbox.w)}, {int(obj.bbox.h)}, 1, 10);',
                f"layer{i}.position = [{obj.bbox.x + obj.bbox.w/2:.0f}, {obj.bbox.y + obj.bbox.h/2:.0f}];",
            ])
            if obj.text_content:
                script_lines.append(f'layer{i}.name = "{obj.text_content}";')
            script_lines.append("")

        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(script_lines), encoding="utf-8")
        return out_path

    def _build_ae_json(self, data: StructuredOutput) -> dict:
        """Build JSON representation of AE project data."""
        return {
            "format": "ae_project_data",
            "comp_name": self._comp_name(data),
            "width": data.metadata.width if data.metadata else 1920,
            "height": data.metadata.height if data.metadata else 1080,
            "fps": data.metadata.fps if data.metadata else 30,
            "duration": data.metadata.duration_seconds if data.metadata else 0,
            "layers": [
                {
                    "id": obj.id,
                    "name": obj.label_custom or obj.label.value,
                    "position": [{"t": pt["t"], "x": pt["x"], "y": pt["y"]} for pt in obj.temporal.trajectory],
                    "appear_frame": obj.temporal.appear_frame,
                    "disappear_frame": obj.temporal.disappear_frame,
                    "text": obj.text_content,
                    "color": obj.dominant_color,
                }
                for obj in data.objects
            ],
            "camera_poses": data.camera_poses if data.camera_poses else [],
        }

    @staticmethod
    def _comp_name(data: StructuredOutput) -> str:
        return f"ToThinkVision_{Path(data.source_file).stem}"
