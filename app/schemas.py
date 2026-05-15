"""Unified intermediate JSON schema — v2 with 3D, point cloud, camera pose support."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    # UI types
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
    # Game types
    GAME_FLOOR = "game_floor"
    GAME_WALL = "game_wall"
    GAME_DOOR = "game_door"
    GAME_NPC = "game_npc"
    GAME_ITEM = "game_item"
    GAME_PROP = "game_prop"
    GAME_TERRAIN = "game_terrain"
    GAME_EFFECT = "game_effect"
    # Video types
    VIDEO_OBJECT = "video_object"
    VIDEO_PARTICLE = "video_particle"
    VIDEO_TEXT = "video_text"
    VIDEO_FX = "video_fx"
    # Embodied types
    EMBODIED_OBSTACLE = "embodied_obstacle"
    EMBODIED_TARGET = "embodied_target"
    EMBODIED_TOOL = "embodied_tool"
    EMBODIED_SURFACE = "embodied_surface"
    # Generic
    GENERIC = "generic"


class InteractionType(str, Enum):
    CLICKABLE = "clickable"
    SCROLLABLE = "scrollable"
    TOGGLE = "toggle"
    DRAGGABLE = "draggable"
    NONE = "none"


# ─── Geometry ───────────────────────────────────────────────

class BBox2D(BaseModel):
    x: float = Field(..., description="Top-left x")
    y: float = Field(..., description="Top-left y")
    w: float = Field(..., description="Width")
    h: float = Field(..., description="Height")


class BBox3D(BaseModel):
    x: float = Field(..., description="3D x coordinate (meters)")
    y: float = Field(..., description="3D y coordinate (meters)")
    z: float = Field(..., description="3D z coordinate / depth (meters)")


# ─── 3D Data Structures ─────────────────────────────────────

class PointCloud(BaseModel):
    """3D point cloud from MASt3R or depth back-projection."""
    points: list[tuple[float, float, float]] = Field(default_factory=list, description="(N,3) xyz in meters")
    colors: list[tuple[int, int, int]] = Field(default_factory=list, description="(N,3) RGB 0-255")
    normals: list[tuple[float, float, float]] | None = Field(None, description="(N,3) normals")
    confidence: list[float] | None = Field(None, description="(N,) per-point confidence")


class CameraPose(BaseModel):
    """Camera intrinsics + extrinsics for a frame."""
    frame_idx: int = 0
    intrinsics: list[list[float]] = Field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], description="3x3 K matrix")
    extrinsics: list[list[float]] = Field(default_factory=lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], description="4x4 RT matrix")
    position: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Camera center in world coords")
    rotation: tuple[float, float, float, float] = Field(default=(1.0, 0.0, 0.0, 0.0), description="Quaternion (w,x,y,z)")


class GaussianSplatData(BaseModel):
    """3D Gaussian Splatting parameters."""
    means: list[tuple[float, float, float]] = Field(default_factory=list, description="Gaussian centers (N,3)")
    quats: list[tuple[float, float, float, float]] = Field(default_factory=list, description="Rotations (N,4)")
    scales: list[tuple[float, float, float]] = Field(default_factory=list, description="Scales (N,3)")
    opacities: list[float] = Field(default_factory=list, description="Opacities (N,)")
    sh_coeffs: list[list[float]] = Field(default_factory=list, description="Spherical harmonic coeffs (N, C)")


class PSDLayer(BaseModel):
    """PSD layer specification."""
    name: str = ""
    position: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)
    opacity: float = 1.0
    blend_mode: str = "normal"
    image_data: bytes | None = None
    children: list[PSDLayer] = Field(default_factory=list, description="Nested layers for groups")


# ─── Core Object Model ──────────────────────────────────────

class TemporalInfo(BaseModel):
    frame_index: int = 0
    appear_frame: int = 0
    disappear_frame: int = -1  # -1 = still visible
    trajectory: list[dict[str, Any]] = Field(default_factory=list, description="[{x, y, t}] per frame")
    velocity: dict[str, float] | None = None
    depth_per_frame: list[float] = Field(default_factory=list, description="Depth (meters) per frame")


class ObjectRelation(BaseModel):
    parent_id: str | None = None
    collision_with: list[str] = Field(default_factory=list)
    relative_positions: list[dict[str, str]] = Field(default_factory=list)


class Interaction(BaseModel):
    type: InteractionType = InteractionType.NONE
    clickable: bool = False
    scrollable: bool = False
    toggle_state: bool | None = None
    direction: str | None = None


class StructuredObject(BaseModel):
    """Single detected/segmented object."""
    id: str = Field(..., description="Unique object ID")
    label: ObjectType = ObjectType.GENERIC
    label_custom: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # 2D
    bbox: BBox2D
    contour: list[dict[str, float]] = Field(default_factory=list, description="Contour [{x,y}]")
    mask_base64: str | None = Field(None, description="Segmentation mask as base64 PNG")
    mask_png_path: str | None = Field(None, description="Path to exported mask PNG file")

    # 3D
    bbox_3d: BBox3D | None = None
    depth_value: float | None = Field(None, description="Depth in meters")
    point_cloud_indices: list[int] = Field(default_factory=list, description="Indices into StructuredOutput.point_cloud")

    # Appearance
    dominant_color: str | None = None
    color_palette: list[str] = Field(default_factory=list)
    z_index: int = 0

    # OCR
    text_content: str | None = None
    text_confidence: float | None = None

    # Temporal
    temporal: TemporalInfo = Field(default_factory=TemporalInfo)

    # Relations
    relations: ObjectRelation = Field(default_factory=ObjectRelation)

    # Interaction
    interaction: Interaction = Field(default_factory=Interaction)

    # Crop image for PSD layers
    crop_image_base64: str | None = Field(None, description="Cropped object image as base64")
    crop_png_path: str | None = Field(None, description="Path to exported crop PNG file (for PS/AE/Unity)")

    # Raw model data
    raw_data: dict[str, Any] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0


class StructuredOutput(BaseModel):
    """Complete structured output."""
    source_file: str
    source_type: str  # "image" or "video"
    metadata: VideoMetadata | None = None
    objects: list[StructuredObject] = Field(default_factory=list)
    frame_count: int = 0
    processing_time_seconds: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)

    # 3D data
    point_cloud: PointCloud | None = None
    camera_poses: list[CameraPose] = Field(default_factory=list)
    gaussian_splats: GaussianSplatData | None = None
    ply_file_path: str | None = None
    splat_file_path: str | None = None

    # Exported image files (for PS/AE/Unity consumption)
    depth_map_png_path: str | None = Field(None, description="Path to depth map colored visualization PNG")
    detection_overlay_png_path: str | None = Field(None, description="Path to image with detection bboxes and labels drawn")
    point_cloud_preview_png_path: str | None = Field(None, description="Path to point cloud preview render PNG")


class ExportFormat(str, Enum):
    # UI exports
    FIGMA_JSON = "figma_json"
    HTML_CSS = "html_css"
    UI_JSON = "ui_json"

    # Game 3D exports
    UNITY_SPLAT = "unity_splat"
    UE_SPLAT = "ue_splat"
    GLTF = "gltf"
    OBJ_3D = "obj_3d"
    UNITY_JSON = "unity_json"
    UE_JSON = "ue_json"
    COLLISION_JSON = "collision_json"

    # Video exports
    AE_KEYFRAMES = "ae_keyframes"
    VIDEO_TRAJECTORY = "video_trajectory"
    PR_MARKERS = "pr_markers"
    AE_PROJECT = "ae_project"

    # PSD exports
    PSD_STATIC = "psd_static"
    PSD_ANIMATED = "psd_animated"

    # Embodied exports
    EMBODIED_JSON = "embodied_json"
    ROBOT_ACTION = "robot_action"
    POSE_CSV = "pose_csv"

    # Universal
    FULL_JSON = "full_json"


class PipelineConfig(BaseModel):
    """Per-request pipeline configuration: which models to run and which formats to export."""

    # Model toggles (True = run, False = skip)
    enable_sam3: bool = True
    enable_omniparser: bool = True
    enable_grounding_dino: bool = True
    enable_strongsort: bool = True
    enable_depth_pro: bool = True
    enable_mast3r: bool = True
    enable_gaussian_splatting: bool = False

    # Detection mode
    mode: str = "general"  # general, ui, game, video, embodied

    # Video parameters
    max_video_frames: int = 300
    frame_sample_interval: float = 0.0
    mast3r_sample_interval: int = 5

    # Detection thresholds
    detection_threshold: float = 0.35
    segmentation_threshold: float = 0.5
    ocr_threshold: float = 0.4
