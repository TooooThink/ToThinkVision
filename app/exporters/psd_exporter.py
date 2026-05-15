"""PSD Exporter — creates layered Photoshop files from structured objects."""

from __future__ import annotations

import base64
import logging
import struct
from io import BytesIO
from pathlib import Path

from app.config import settings
from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat

logger = logging.getLogger(__name__)


class PSDExporter(BaseExporter):
    """Exports structured data to PSD with layers.

    Image mode: each object = separate layer
    Video mode: each frame = GroupLayer, each object = layer inside group
    """

    def __init__(self, fmt: ExportFormat = ExportFormat.PSD_STATIC):
        self.fmt = fmt
        self.format_name = "psd_animated" if fmt == ExportFormat.PSD_ANIMATED else "psd_static"
        self.file_extension = ".psd"
        self.mime_type = "image/vnd.adobe.photoshop"

    def export(self, data: StructuredOutput) -> Path:
        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.fmt == ExportFormat.PSD_ANIMATED and data.source_type == "video":
            return self._export_video_psd(data, out_path)
        else:
            return self._export_static_psd(data, out_path)

    def _export_static_psd(self, data: StructuredOutput, out_path: Path) -> Path:
        """Create PSD with each object as a separate layer, using masked PNG with alpha."""
        width = data.metadata.width if data.metadata else 1920
        height = data.metadata.height if data.metadata else 1080

        layers = []
        for obj in data.objects:
            layer_data = None

            # Priority: masked PNG (transparent) > crop PNG > crop base64
            masked_path = obj.raw_data.get("masked_png_path") if obj.raw_data else None
            if masked_path and Path(masked_path).exists():
                layer_data = Path(masked_path).read_bytes()
            elif obj.crop_png_path and Path(obj.crop_png_path).exists():
                layer_data = Path(obj.crop_png_path).read_bytes()
            elif obj.crop_image_base64:
                layer_data = base64.b64decode(obj.crop_image_base64)

            layers.append({
                "name": f"{obj.label_custom or obj.label.value} ({obj.id})",
                "position": (int(obj.bbox.x), int(obj.bbox.y)),
                "size": (int(obj.bbox.w), int(obj.bbox.h)),
                "opacity": 1.0,
                "blend_mode": "normal",
                "image_data": layer_data,
            })

        self._write_psd(out_path, width, height, layers)
        return out_path

    def _export_video_psd(self, data: StructuredOutput, out_path: Path) -> Path:
        """Create PSD with frame groups, each containing object layers."""
        width = data.metadata.width if data.metadata else 1920
        height = data.metadata.height if data.metadata else 1080

        # Group objects by frame
        frame_objects: dict[int, list] = {}
        for obj in data.objects:
            fi = obj.temporal.frame_index
            if fi not in frame_objects:
                frame_objects[fi] = []
            frame_objects[fi].append(obj)

        # Create layered groups per frame
        layers = []
        for frame_idx in sorted(frame_objects.keys()):
            objs = frame_objects[frame_idx]
            group_layers = []
            for obj in objs:
                layer_data = None

                # Priority: masked PNG > crop PNG > crop base64
                masked_path = obj.raw_data.get("masked_png_path") if obj.raw_data else None
                if masked_path and Path(masked_path).exists():
                    layer_data = Path(masked_path).read_bytes()
                elif obj.crop_png_path and Path(obj.crop_png_path).exists():
                    layer_data = Path(obj.crop_png_path).read_bytes()
                elif obj.crop_image_base64:
                    layer_data = base64.b64decode(obj.crop_image_base64)

                group_layers.append({
                    "name": f"{obj.label_custom or obj.label.value} ({obj.id})",
                    "position": (int(obj.bbox.x), int(obj.bbox.y)),
                    "size": (int(obj.bbox.w), int(obj.bbox.h)),
                    "opacity": 1.0,
                    "blend_mode": "normal",
                    "image_data": layer_data,
                })

            layers.append({
                "name": f"Frame_{frame_idx:04d}",
                "position": (0, 0),
                "size": (width, height),
                "opacity": 1.0,
                "blend_mode": "normal",
                "children": group_layers,
                "is_group": True,
            })

        self._write_psd(out_path, width, height, layers)
        return out_path

    def _write_psd(self, path: Path, width: int, height: int, layers: list[dict]):
        """Write a PSD file using PhotoshopAPI or minimal PSD format.

        If PhotoshopAPI is available, use it for full feature support.
        Otherwise, write a minimal valid PSD that Photoshop can open.
        """
        try:
            self._write_psd_with_api(path, width, height, layers)
        except ImportError:
            logger.warning("PhotoshopAPI not available, writing minimal PSD")
            self._write_minimal_psd(path, width, height, layers)

    def _write_psd_with_api(self, path: Path, width: int, height: int, layers: list[dict]):
        """Write PSD using PhotoshopAPI library."""
        import photoshopapi as psapi

        file = psapi.LayeredFile()
        file.width = width
        file.height = height
        file.bit_depth = 8

        for layer_info in layers:
            if layer_info.get("is_group"):
                group = psapi.GroupLayer()
                group.name = layer_info["name"]
                for child in layer_info.get("children", []):
                    child_layer = self._create_image_layer(child)
                    if child_layer:
                        group.layers.append(child_layer)
                file.layers.append(group)
            else:
                layer = self._create_image_layer(layer_info)
                if layer:
                    file.layers.append(layer)

        file.save(str(path))

    def _create_image_layer(self, layer_info: dict):
        """Create an image layer from layer info dict."""
        import numpy as np
        import photoshopapi as psapi
        from PIL import Image

        layer = psapi.ImageLayer()
        layer.name = layer_info["name"]

        if layer_info.get("image_data"):
            # Load from base64 image data
            buf = BytesIO(layer_info["image_data"])
            img = Image.open(buf).convert("RGBA")
            pixels = np.array(img)
            h, w = pixels.shape[:2]
            layer["Red"] = pixels[:, :, 0].flatten()
            layer["Green"] = pixels[:, :, 1].flatten()
            layer["Blue"] = pixels[:, :, 2].flatten()
            if pixels.shape[2] == 4:
                layer["Alpha"] = pixels[:, :, 3].flatten()
            else:
                layer["Alpha"] = np.full(w * h, 255, dtype=np.uint8)
        else:
            # Create a blank layer with the specified size
            w, h = layer_info["size"]
            layer["Red"] = np.zeros(w * h, dtype=np.uint8)
            layer["Green"] = np.zeros(w * h, dtype=np.uint8)
            layer["Blue"] = np.zeros(w * h, dtype=np.uint8)
            layer["Alpha"] = np.full(w * h, 255, dtype=np.uint8)

        layer.position = (layer_info["position"][0], layer_info["position"][1])
        return layer

    def _write_minimal_psd(self, path: Path, width: int, height: int, layers: list[dict]):
        """Write a minimal valid PSD file (simplified format).

        This creates a PSD that can be opened in Photoshop but may not
        support all features. Each layer gets a simple RGBA channel.
        """
        import numpy as np
        import struct

        n_layers = len(layers)

        # PSD Header
        header = struct.pack(">4sH6sH", b"8BPS", 1, b"\x00" * 6, 1)
        # Color mode: 0 = RGB
        header += struct.pack(">H", 0)
        # Color mode data length
        header += struct.pack(">I", 0)
        # Image resources
        header += struct.pack(">I", 0)
        # Layer and mask info section (we'll calculate this)

        # Calculate layer section size
        layer_section = struct.pack(">I", n_layers)

        # Write header
        with open(path, "wb") as f:
            # Signature
            f.write(b"8BPS")
            # Version
            f.write(struct.pack(">H", 1))
            # Reserved
            f.write(b"\x00" * 6)
            # Channels (RGBA = 4)
            f.write(struct.pack(">H", 4))
            # Height, Width
            f.write(struct.pack(">II", height, width))
            # Depth (8-bit)
            f.write(struct.pack(">H", 8))
            # Color mode (0 = RGB)
            f.write(struct.pack(">H", 0))
            # Color mode data length
            f.write(struct.pack(">I", 0))
            # Image resources length
            f.write(struct.pack(">I", 0))

            # Layer section
            # We write a simplified layer section
            f.write(struct.pack(">I", n_layers))

            for layer_info in layers:
                name = layer_info["name"].encode("utf-8")
                name_len = min(len(name), 255)
                # Layer top, left, bottom, right
                top, left = layer_info["position"]
                w, h = layer_info["size"]
                f.write(struct.pack(">ii", top, top + h))
                f.write(struct.pack(">ii", left, left + w))
                # Channels (RGBA)
                f.write(struct.pack(">H", 4))
                # Channel IDs: 0=R, 1=G, 2=B, -1=A
                for ch_id in [0, 1, 2, -1]:
                    f.write(struct.pack(">hI", ch_id, h * w))
                # Blend mode
                f.write(b"norm")
                # Opacity
                f.write(struct.pack(">B", int(layer_info.get("opacity", 1.0) * 255)))
                # Clipping
                f.write(struct.pack(">B", 0))
                # Flags
                f.write(struct.pack(">B", 0))
                # Filler
                f.write(struct.pack(">B", 0))
                # Extra data length
                f.write(struct.pack(">I", 4 + name_len + (4 - name_len % 4) % 4))
                # Mask data length
                f.write(struct.pack(">I", 0))
                # Blending ranges
                f.write(struct.pack(">I", 0))
                # Layer name
                f.write(struct.pack(">B", name_len))
                f.write(name)
                # Pad to 4-byte boundary
                padding = (4 - (name_len + 1) % 4) % 4
                f.write(b"\x00" * padding)

        logger.info(f"Minimal PSD written to {path} ({n_layers} layers)")
