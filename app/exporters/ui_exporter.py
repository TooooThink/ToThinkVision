"""Module 1: UI Exporter — converts structured UI objects to Figma JSON, HTML/CSS, and UI JSON."""

from __future__ import annotations

from pathlib import Path

from app.exporters.base import BaseExporter
from app.schemas import StructuredOutput, ExportFormat


class UIExporter(BaseExporter):
    """Exports UI structured data to multiple frontend formats."""

    def __init__(self, fmt: ExportFormat = ExportFormat.UI_JSON):
        self.fmt = fmt
        if fmt == ExportFormat.FIGMA_JSON:
            self.format_name = "figma"
            self.file_extension = ".json"
        elif fmt == ExportFormat.HTML_CSS:
            self.format_name = "html_css"
            self.file_extension = ".html"
            self.mime_type = "text/html"
        else:
            self.format_name = "ui_json"
            self.file_extension = ".json"

    def export(self, data: StructuredOutput) -> Path:
        if self.fmt == ExportFormat.FIGMA_JSON:
            return self._export_figma_json(data)
        elif self.fmt == ExportFormat.HTML_CSS:
            return self._export_html_css(data)
        else:
            return self._export_ui_json(data)

    def _export_ui_json(self, data: StructuredOutput) -> Path:
        """Export as simplified UI JSON with component hierarchy."""
        components = []
        for obj in data.objects:
            comp = {
                "id": obj.id,
                "type": obj.label.value,
                "custom_label": obj.label_custom,
                "bbox": {"x": obj.bbox.x, "y": obj.bbox.y, "width": obj.bbox.w, "height": obj.bbox.h},
                "z_index": obj.z_index,
                "color": obj.dominant_color,
                "text": obj.text_content,
                "interaction": obj.interaction.type.value,
                "parent_id": obj.relations.parent_id,
                "confidence": obj.confidence,
            }
            components.append(comp)

        output = {
            "source": data.source_file,
            "format": "ui_json",
            "dimensions": {"width": data.metadata.width if data.metadata else 0, "height": data.metadata.height if data.metadata else 0},
            "components": components,
            "component_count": len(components),
        }
        return self.save_json(output, self._output_path(data.source_file))

    def _export_figma_json(self, data: StructuredOutput) -> Path:
        """Export as Figma-compatible JSON structure."""
        nodes = []
        for obj in data.objects:
            node = {
                "id": obj.id,
                "name": obj.label_custom or obj.label.value,
                "type": self._figma_type(obj),
                "absoluteBoundingBox": {
                    "x": obj.bbox.x,
                    "y": obj.bbox.y,
                    "width": obj.bbox.w,
                    "height": obj.bbox.h,
                },
                "fills": [{"type": "SOLID", "color": self._hex_to_rgba(obj.dominant_color)}] if obj.dominant_color else [],
                "visible": True,
                "constraints": {"vertical": "TOP", "horizontal": "LEFT"},
            }
            if obj.text_content:
                node["characters"] = obj.text_content
                node["style"] = {"fontFamily": "Inter", "fontSize": 14, "textAlignHorizontal": "LEFT"}
            nodes.append(node)

        figma_doc = {
            "document": {
                "id": "0:0",
                "name": Path(data.source_file).stem,
                "type": "DOCUMENT",
                "children": [{
                    "id": "1:0",
                    "name": "Page 1",
                    "type": "CANVAS",
                    "children": nodes,
                    "background": {"r": 1, "g": 1, "b": 1, "a": 1},
                    "prototypeDevice": {"type": "AUTOMATIC"},
                }],
            },
            "name": Path(data.source_file).stem,
            "version": "1.0",
        }
        return self.save_json(figma_doc, self._output_path(data.source_file))

    def _export_html_css(self, data: StructuredOutput) -> Path:
        """Export as a self-contained HTML file with inline CSS."""
        elements_html = ""
        for obj in data.objects:
            style = self._build_css(obj)
            tag = self._html_tag(obj)
            content = obj.text_content or ""
            attrs = f'id="{obj.id}"'
            if obj.interaction.clickable:
                attrs += ' role="button" tabindex="0"'
            elements_html += f"    <{tag} {attrs} style=\"{style}\">{content}</{tag}>\n"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{Path(data.source_file).stem} — UI Reconstruction</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ position: relative; width: {data.metadata.width if data.metadata else 1920}px; min-height: {data.metadata.height if data.metadata else 1080}px; overflow: auto; background: #f0f0f0; }}
</style>
</head>
<body>
{elements_html}
</body>
</html>"""
        out_path = self._output_path(data.source_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def _html_tag(self, obj) -> str:
        tag_map = {
            "ui_button": "button",
            "ui_text": "p",
            "ui_input": "input",
            "ui_icon": "span",
            "ui_image": "img",
            "ui_container": "div",
            "ui_nav": "nav",
            "ui_card": "div",
            "ui_slider": "input",
            "ui_toggle": "input",
        }
        return tag_map.get(obj.label.value, "div")

    def _build_css(self, obj) -> str:
        parts = [
            f"position:absolute",
            f"left:{obj.bbox.x}px",
            f"top:{obj.bbox.y}px",
            f"width:{obj.bbox.w}px",
            f"height:{obj.bbox.h}px",
            f"z-index:{obj.z_index}",
        ]
        if obj.dominant_color:
            parts.append(f"background-color:{obj.dominant_color}")
        if obj.label.value == "ui_input":
            parts.append("border:1px solid #ccc")
            parts.append("border-radius:4px")
        elif obj.label.value == "ui_button":
            parts.append("border:none")
            parts.append("border-radius:6px")
            parts.append("cursor:pointer")
        elif obj.label.value == "ui_text":
            parts.append("display:flex")
            parts.append("align-items:center")
            parts.append("color:#333")
        return ";".join(parts)

    def _figma_type(self, obj) -> str:
        type_map = {
            "ui_button": "RECTANGLE",
            "ui_text": "TEXT",
            "ui_input": "RECTANGLE",
            "ui_icon": "RECTANGLE",
            "ui_image": "RECTANGLE",
            "ui_container": "FRAME",
            "ui_nav": "FRAME",
            "ui_card": "FRAME",
        }
        return type_map.get(obj.label.value, "RECTANGLE")

    @staticmethod
    def _hex_to_rgba(hex_color: str | None) -> dict:
        if not hex_color:
            return {"r": 0.8, "g": 0.8, "b": 0.8, "a": 1.0}
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        else:
            r, g, b = 200, 200, 200
        return {"r": r / 255, "g": g / 255, "b": b / 255, "a": 1.0}
