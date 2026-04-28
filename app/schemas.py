"""Unified intermediate JSON schema for all vision structured data."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    UI_BUTTON = "ui_button"
    UI_TEXT = "ui_text"
    UI_INPUT = "ui_input"
    UI_ICON = "ui_icon"
    UI_IMAGE = "ui_image"
    UI_CONTAINER = "ui_container"
    UI_NAV = "ui_nav"
    UI_CARD = "ui_card"
    UI_SLIDER = "ui_slider"
    UI_TOGGLE = "ui_toggle"
    GAME_FLOOR = "game_floor"
    GAME_WALL = "game_wall"
    GAME_DOOR = "game_door"
    GAME_NPC = "game_npc"
    GAME_ITEM = "game_item"
    GAME_PROP = "game_prop"
    GAME_TERRAIN = "game_terrain"
    GAME_EFFECT = "game_effect"
    VIDEO_OBJECT = "video_object"
    VIDEO_PARTICLE = "video_particle"
    VIDEO_TEXT = "video_text"
    VIDEO_FX = "video_fx"
    EMBODIED_OBSTACLE = "embodied_obstacle"
    EMBODIED_TARGET = "embodied_target"
    EMBODIED_TOOL = "embodied_tool"
    EMBODIED_SURFACE = "embodied_surface"
    GENERIC = "generic"


class RelativePosition(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    OVERLAP = "overlap"


class InteractionType(str, Enum):
    CLICKABLE = "clickable"
    SCROLLABLE = "scrollable"
    TOGGLE = "toggle"
    DRAGGABLE = "draggable"
    NONE = "none"


class BBox2D(BaseModel):
    x: float = Field(..., description="Top-left x")
    y: float = Field(..., description="Top-left y")
    w: float = Field(..., description="Width")
    h: float = Field(..., description="Height")


class BBox3D(BaseModel):
    x: float = Field(..., description="3D x coordinate")
    y: float = Field(..., description="3D y coordinate")
    z: float = Field(..., description="3D z coordinate (depth)")


class TemporalInfo(BaseModel):
    frame_index: int = Field(0, description="Frame number where object appears")
    appear_frame: int = Field(0, description="First frame where object appears")
    disappear_frame: int = Field(-1, description="Last frame; -1 means still visible")
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {x, y, t} trajectory points",
    )
    velocity: dict[str, float] | None = Field(
        None, description="Average velocity {vx, vy}"
    )


class ObjectRelation(BaseModel):
    parent_id: str | None = Field(None, description="Parent object ID (hierarchy)")
    collision_with: list[str] = Field(
        default_factory=list, description="IDs of objects this collides with"
    )
    relative_positions: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {target_id, relation: above|below|left_of|...}",
    )


class Interaction(BaseModel):
    type: InteractionType = InteractionType.NONE
    clickable: bool = False
    scrollable: bool = False
    toggle_state: bool | None = Field(None, description="Current toggle state if applicable")
    direction: str | None = Field(None, description="Scroll direction: vertical|horizontal")


class StructuredObject(BaseModel):
    """Single detected/segmented object in the unified intermediate format."""

    id: str = Field(..., description="Unique object identifier")
    label: ObjectType = Field(ObjectType.GENERIC, description="Object type classification")
    label_custom: str | None = Field(None, description="Custom label from detection model")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence")

    # 2D geometry
    bbox: BBox2D = Field(..., description="2D bounding box")
    contour: list[dict[str, float]] = Field(
        default_factory=list,
        description="Contour points as [{x, y}, ...]",
    )

    # 3D / depth
    bbox_3d: BBox3D | None = Field(None, description="3D bounding box (estimated from depth)")
    depth_value: float | None = Field(None, description="Average depth value in bbox region")

    # Appearance
    dominant_color: str | None = Field(None, description="Dominant color as hex string")
    color_palette: list[str] = Field(default_factory=list, description="Top N dominant colors")
    z_index: int = Field(0, description="Layer index (higher = visually on top)")

    # Text (OCR)
    text_content: str | None = Field(None, description="OCR recognized text")
    text_confidence: float | None = Field(None, description="OCR confidence")

    # Temporal (for video)
    temporal: TemporalInfo = Field(default_factory=TemporalInfo)

    # Relations
    relations: ObjectRelation = Field(default_factory=ObjectRelation)

    # Interaction metadata
    interaction: Interaction = Field(default_factory=Interaction)

    # Raw model output for debugging
    raw_data: dict[str, Any] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0


class StructuredOutput(BaseModel):
    """Complete structured output for an image or video."""

    source_file: str
    source_type: str = Field(..., description="image or video")
    metadata: VideoMetadata | None = None
    objects: list[StructuredObject] = Field(default_factory=list)
    frame_count: int = 0
    processing_time_seconds: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)


class ExportFormat(str, Enum):
    # UI exports
    FIGMA_JSON = "figma_json"
    HTML_CSS = "html_css"
    UI_JSON = "ui_json"

    # Game exports
    UNITY_JSON = "unity_json"
    UE_JSON = "ue_json"
    COLLISION_JSON = "collision_json"

    # Video exports
    AE_KEYFRAMES = "ae_keyframes"
    VIDEO_TRAJECTORY = "video_trajectory"
    PR_MARKERS = "pr_markers"

    # Embodied exports
    EMBODIED_JSON = "embodied_json"
    ROBOT_ACTION = "robot_action"
    POSE_CSV = "pose_csv"

    # Universal
    FULL_JSON = "full_json"
