"""Tests for unified schema models."""

import pytest

from app.schemas import (
    BBox2D,
    BBox3D,
    Interaction,
    InteractionType,
    ObjectRelation,
    ObjectType,
    StructuredObject,
    StructuredOutput,
    TemporalInfo,
    VideoMetadata,
)


def test_bbox_2d():
    bbox = BBox2D(x=10, y=20, w=100, h=50)
    assert bbox.x == 10
    assert bbox.w == 100


def test_bbox_3d():
    bbox = BBox3D(x=1.5, y=2.5, z=3.0)
    assert bbox.z == 3.0


def test_structured_object_defaults():
    obj = StructuredObject(
        id="obj_0001",
        bbox=BBox2D(x=0, y=0, w=100, h=100),
    )
    assert obj.id == "obj_0001"
    assert obj.label == ObjectType.GENERIC
    assert obj.confidence == 0.0
    assert obj.text_content is None
    assert obj.temporal.frame_index == 0
    assert obj.interaction.type == InteractionType.NONE


def test_structured_object_full():
    obj = StructuredObject(
        id="obj_0001",
        label=ObjectType.UI_BUTTON,
        label_custom="Submit Button",
        confidence=0.92,
        bbox=BBox2D(x=50, y=100, w=120, h=40),
        contour=[{"x": 50, "y": 100}, {"x": 170, "y": 100}, {"x": 170, "y": 140}, {"x": 50, "y": 140}],
        bbox_3d=BBox3D(x=110, y=120, z=2.5),
        depth_value=180.0,
        dominant_color="#3b82f6",
        color_palette=["#3b82f6", "#1d4ed8"],
        z_index=5,
        text_content="Submit",
        text_confidence=0.95,
        temporal=TemporalInfo(
            frame_index=10,
            appear_frame=0,
            disappear_frame=-1,
            trajectory=[{"x": 110, "y": 120, "t": i} for i in range(11)],
            velocity={"vx": 0.5, "vy": 0.2},
        ),
        relations=ObjectRelation(
            parent_id=None,
            collision_with=["obj_0002"],
            relative_positions=[{"target_id": "obj_0002", "relation": "above"}],
        ),
        interaction=Interaction(type=InteractionType.CLICKABLE, clickable=True),
    )
    assert obj.label == ObjectType.UI_BUTTON
    assert obj.interaction.clickable is True
    assert len(obj.temporal.trajectory) == 11
    assert "obj_0002" in obj.relations.collision_with


def test_structured_output():
    obj = StructuredObject(id="obj_0001", bbox=BBox2D(x=0, y=0, w=100, h=100))
    metadata = VideoMetadata(fps=30.0, total_frames=90, width=1920, height=1080, duration_seconds=3.0)
    output = StructuredOutput(
        source_file="test.png",
        source_type="image",
        metadata=metadata,
        objects=[obj],
        frame_count=1,
        processing_time_seconds=1.5,
        model_versions={"detector": "dino"},
    )
    assert output.source_type == "image"
    assert len(output.objects) == 1
    assert output.metadata.fps == 30.0


def test_object_type_enum():
    assert ObjectType.UI_BUTTON.value == "ui_button"
    assert ObjectType.GAME_NPC.value == "game_npc"
    assert ObjectType.EMBODIED_TARGET.value == "embodied_target"


def test_export_format_enum():
    from app.schemas import ExportFormat
    assert ExportFormat.FIGMA_JSON.value == "figma_json"
    assert ExportFormat.UNITY_JSON.value == "unity_json"
    assert ExportFormat.AE_KEYFRAMES.value == "ae_keyframes"
    assert ExportFormat.EMBODIED_JSON.value == "embodied_json"
    assert ExportFormat.FULL_JSON.value == "full_json"


def test_model_dump_json():
    obj = StructuredObject(id="test", bbox=BBox2D(x=1, y=2, w=3, h=4))
    output = StructuredOutput(source_file="test.png", source_type="image", objects=[obj])
    json_str = output.model_dump_json()
    assert "test" in json_str
    assert "image" in json_str
