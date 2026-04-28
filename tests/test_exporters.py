"""Tests for all exporters."""

import json
import os
from pathlib import Path

import pytest

from app.exporters.ui_exporter import UIExporter
from app.exporters.game_exporter import GameExporter
from app.exporters.video_exporter import VideoExporter
from app.exporters.embodied_exporter import EmbodiedExporter
from app.schemas import (
    BBox2D,
    BBox3D,
    ExportFormat,
    Interaction,
    InteractionType,
    ObjectType,
    StructuredObject,
    StructuredOutput,
    TemporalInfo,
    VideoMetadata,
)


@pytest.fixture
def sample_image_output():
    return StructuredOutput(
        source_file="test_ui.png",
        source_type="image",
        metadata=None,
        objects=[
            StructuredObject(
                id="obj_0000",
                label=ObjectType.UI_BUTTON,
                label_custom="Submit Button",
                confidence=0.92,
                bbox=BBox2D(x=100, y=200, w=120, h=40),
                dominant_color="#3b82f6",
                text_content="Submit",
                interaction=Interaction(type=InteractionType.CLICKABLE, clickable=True),
                z_index=5,
            ),
            StructuredObject(
                id="obj_0001",
                label=ObjectType.UI_TEXT,
                label_custom="Title",
                confidence=0.88,
                bbox=BBox2D(x=50, y=30, w=300, h=45),
                dominant_color="#1f2937",
                text_content="Welcome Page",
                z_index=10,
            ),
        ],
        frame_count=1,
        processing_time_seconds=1.5,
    )


@pytest.fixture
def sample_video_output():
    return StructuredOutput(
        source_file="test_video.mp4",
        source_type="video",
        metadata=VideoMetadata(fps=30.0, total_frames=90, width=1920, height=1080, duration_seconds=3.0),
        objects=[
            StructuredObject(
                id="obj_0000",
                label=ObjectType.VIDEO_OBJECT,
                label_custom="Player",
                confidence=0.95,
                bbox=BBox2D(x=100, y=200, w=80, h=120),
                bbox_3d=BBox3D(x=140, y=260, z=5.0),
                temporal=TemporalInfo(
                    frame_index=0,
                    appear_frame=0,
                    disappear_frame=89,
                    trajectory=[{"x": 140 + i * 5, "y": 260 + i * 2, "t": i} for i in range(10)],
                    velocity={"vx": 5.0, "vy": 2.0},
                ),
                z_index=3,
            ),
        ],
        frame_count=10,
        processing_time_seconds=5.0,
    )


@pytest.fixture
def sample_game_output():
    return StructuredOutput(
        source_file="test_game.png",
        source_type="image",
        objects=[
            StructuredObject(
                id="obj_0000",
                label=ObjectType.GAME_FLOOR,
                confidence=0.90,
                bbox=BBox2D(x=0, y=800, w=1920, h=280),
                bbox_3d=BBox3D(x=960, y=940, z=1.0),
                z_index=0,
            ),
            StructuredObject(
                id="obj_0001",
                label=ObjectType.GAME_NPC,
                label_custom="Merchant",
                confidence=0.85,
                bbox=BBox2D(x=400, y=300, w=60, h=100),
                bbox_3d=BBox3D(x=430, y=350, z=3.5),
                z_index=5,
            ),
        ],
        frame_count=1,
        processing_time_seconds=2.0,
    )


def test_ui_json_export(sample_image_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = UIExporter(ExportFormat.UI_JSON)
    path = exporter.export(sample_image_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "ui_json"
    assert data["component_count"] == 2


def test_figma_json_export(sample_image_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = UIExporter(ExportFormat.FIGMA_JSON)
    path = exporter.export(sample_image_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "document" in data
    assert data["document"]["children"][0]["type"] == "CANVAS"


def test_html_css_export(sample_image_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = UIExporter(ExportFormat.HTML_CSS)
    path = exporter.export(sample_image_output)
    assert path.exists()
    assert path.suffix == ".html"
    html = path.read_text()
    assert "<!DOCTYPE html>" in html
    assert "Submit" in html


def test_unity_json_export(sample_game_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = GameExporter(ExportFormat.UNITY_JSON)
    path = exporter.export(sample_game_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "unity_json"
    assert len(data["game_objects"]) == 2


def test_ue_json_export(sample_game_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = GameExporter(ExportFormat.UE_JSON)
    path = exporter.export(sample_game_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "ue_json"
    assert len(data["actors"]) == 2


def test_collision_json_export(sample_game_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = GameExporter(ExportFormat.COLLISION_JSON)
    path = exporter.export(sample_game_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "collision_json"


def test_ae_keyframes_export(sample_video_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = VideoExporter(ExportFormat.AE_KEYFRAMES)
    path = exporter.export(sample_video_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "ae_keyframes"
    assert data["layer_count"] == 1


def test_video_trajectory_export(sample_video_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = VideoExporter(ExportFormat.VIDEO_TRAJECTORY)
    path = exporter.export(sample_video_output)
    assert path.exists()
    assert path.suffix == ".csv"
    content = path.read_text()
    assert "object_id" in content
    assert "obj_0000" in content


def test_pr_markers_export(sample_video_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = VideoExporter(ExportFormat.PR_MARKERS)
    path = exporter.export(sample_video_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "pr_markers"


def test_embodied_json_export(sample_game_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = EmbodiedExporter(ExportFormat.EMBODIED_JSON)
    path = exporter.export(sample_game_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "embodied_json"
    assert "interaction_sequence" in data


def test_robot_action_export(sample_game_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = EmbodiedExporter(ExportFormat.ROBOT_ACTION)
    path = exporter.export(sample_game_output)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "robot_action"


def test_pose_csv_export(sample_video_output, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    exporter = EmbodiedExporter(ExportFormat.POSE_CSV)
    path = exporter.export(sample_video_output)
    assert path.exists()
    assert path.suffix == ".csv"
    content = path.read_text()
    assert "object_id" in content
