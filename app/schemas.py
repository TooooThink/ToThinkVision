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


class Mesh3D(BaseModel):
    """Triangle mesh with optional UV and texture for a single object."""
    vertices: list[tuple[float, float, float]] = Field(default_factory=list, description="(V,3) vertex positions in meters")
    faces: list[tuple[int, int, int]] = Field(default_factory=list, description="(F,3) triangle face indices")
    normals: list[tuple[float, float, float]] | None = Field(None, description="(V,3) per-vertex normals")
    # UV coordinates
    uv_coords: list[tuple[float, float]] | None = Field(None, description="(UV,2) UV coordinates [0,1]")
    uv_face_map: list[tuple[int, int, int]] | None = Field(None, description="(F,3) mapping from faces to UV coords")
    # Texture
    texture_path: str | None = Field(None, description="Path to texture PNG file")
    texture_base64: str | None = Field(None, description="Texture as base64 PNG")
    # Bounding box
    bounds: dict[str, list[float]] | None = Field(None, description="min/max xyz of the mesh")
    # Point cloud this mesh was generated from
    point_count: int = 0

    # Completion tracking (partial object view)
    completion_applied: bool = Field(False, description="Whether generative completion was used")
    completion_method: str | None = Field(None, description="2d_lama, 3d_heuristic, 3d_pvd, or None")
    original_point_count: int = Field(0, description="Point count before completion")


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

    # Completeness tracking (partial object view detection)
    completeness_score: float = Field(1.0, description="0.0=fully partial, 1.0=fully observed")
    is_complete: bool = Field(True, description="Whether object appears fully observed in video")
    accumulated_mask_frames: int = Field(0, description="Number of frames contributing to mask accumulation")


# ─── 4D Trajectory Data Structures ──────────────────────────

class TrajectoryKeyframe(BaseModel):
    """Single keyframe in a 6DoF object trajectory."""
    timestamp: float = Field(..., description="Time in seconds")
    frame_idx: int = Field(..., description="Frame index")
    position: tuple[float, float, float] = Field(..., description="3D position in world coords (meters)")
    rotation: tuple[float, float, float, float] = Field((1.0, 0.0, 0.0, 0.0), description="Quaternion (w,x,y,z)")
    scale: tuple[float, float, float] = Field((1.0, 1.0, 1.0), description="Scale factors")
    velocity: tuple[float, float, float] | None = Field(None, description="Linear velocity (m/s)")
    angular_velocity: tuple[float, float, float] | None = Field(None, description="Angular velocity (rad/s)")
    is_rigid: bool = Field(True, description="Whether motion is rigid at this keyframe")
    deformation_score: float = Field(0.0, ge=0.0, le=1.0, description="0=rigid, 1=fully deforming")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Tracking confidence")


class ObjectTrajectory4D(BaseModel):
    """Complete 6DoF trajectory for a single object over time."""
    object_id: str
    keyframes: list[TrajectoryKeyframe] = Field(default_factory=list)
    motion_type: str = Field("static", description="static, rigid, deformable, disappearing")
    total_distance: float = Field(0.0, description="Total path length in meters")
    max_speed: float = Field(0.0, description="Maximum speed in m/s")
    avg_speed: float = Field(0.0, description="Average speed in m/s")
    bspline_coeffs: list[list[float]] | None = Field(None, description="B-spline coefficients for smoothed trajectory")
    duration: float = Field(0.0, description="Trajectory duration in seconds")


class GaussianSplat4D(BaseModel):
    """4D Gaussian Splatting parameters with temporal deformation."""
    means: list[tuple[float, float, float]] = Field(default_factory=list, description="Gaussian centers (N,3)")
    quats: list[tuple[float, float, float, float]] = Field(default_factory=list, description="Rotations (N,4)")
    scales: list[tuple[float, float, float]] = Field(default_factory=list, description="Scales (N,3)")
    opacities: list[float] = Field(default_factory=list, description="Opacities (N,)")
    sh_coeffs: list[list[float]] = Field(default_factory=list, description="Spherical harmonic coeffs")
    # Temporal (HexPlane decomposition)
    temporal_coeffs: list[list[float]] = Field(default_factory=list, description="HexPlane time coefficients")
    deformation_field: list[list[float]] | None = Field(None, description="Per-gaussian deformation field")
    time_range: tuple[float, float] = Field((0.0, 1.0), description="Valid time range")
    num_gaussians: int = Field(0, description="Total number of 4D Gaussians")


class SceneGraphNode(BaseModel):
    """Node in the dynamic 4D scene graph."""
    object_id: str
    label: str = ""
    trajectory: ObjectTrajectory4D | None = None
    mesh_3d: Mesh3D | None = None
    is_static: bool = Field(False, description="Whether object is static (ground, wall, etc.)")
    category: str = Field("object", description="Semantic category")
    bounds_3d: dict[str, list[float]] | None = Field(None, description="3D bounding box over time")


class SceneGraphEdge(BaseModel):
    """Time-varying relationship between two objects."""
    source_id: str
    target_id: str
    relation: str = Field(..., description="above, below, left_of, right_of, in_front, behind, near, in_contact, inside, parent")
    time_range: tuple[float, float] = Field(..., description="When this relation holds (start_sec, end_sec)")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    distance: float | None = Field(None, description="Distance between objects (meters)")


class InteractionEvent(BaseModel):
    """Discrete interaction event in the 4D scene."""
    timestamp: float
    frame_idx: int
    event_type: str = Field(..., description="collision, contact_start, contact_end, pick_up, put_down, enter, exit")
    object_ids: list[str] = Field(default_factory=list, description="Objects involved")
    description: str = ""
    position: tuple[float, float, float] | None = None


class SceneGraph4D(BaseModel):
    """Dynamic 4D scene graph with time-varying relationships."""
    nodes: list[SceneGraphNode] = Field(default_factory=list)
    edges: list[SceneGraphEdge] = Field(default_factory=list)
    interaction_events: list[InteractionEvent] = Field(default_factory=list)
    time_range: tuple[float, float] = Field((0.0, 0.0), description="Scene time range")
    num_static_objects: int = 0
    num_dynamic_objects: int = 0


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
    mesh_3d: Mesh3D | None = Field(None, description="3D triangle mesh reconstructed from depth maps")
    mesh_obj_file: str | None = Field(None, description="Path to exported mesh file (OBJ/glTF) for this object")

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

    # 4D trajectory (6DoF motion over time)
    trajectory_4d: ObjectTrajectory4D | None = Field(None, description="6DoF trajectory over time")

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
    scene_mesh_path: str | None = Field(None, description="Path to combined scene mesh (OBJ/glTF)")

    # 4D data (trajectory + scene graph + animated export)
    scene_graph_4d: SceneGraph4D | None = Field(None, description="Dynamic 4D scene graph")
    gaussian_splats_4d: GaussianSplat4D | None = Field(None, description="4D Gaussian Splatting data")
    animated_gltf_path: str | None = Field(None, description="Path to animated glTF/GLB scene")
    usd_path: str | None = Field(None, description="Path to USD scene file")
    blend_path: str | None = Field(None, description="Path to Blender scene/script")
    scene_graph_json_path: str | None = Field(None, description="Path to scene graph JSON")

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

    # 4D Scene exports
    ANIMATED_GLTF = "animated_gltf"
    USD_SCENE = "usd_scene"
    BLENDER_SCENE = "blender_scene"
    SCENE_GRAPH_JSON = "scene_graph_json"


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
    enable_3d_reconstruction: bool = True  # Per-object 3D mesh from depth back-projection

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

    # Partial object view completion
    enable_completion_2d: bool = True  # LaMa inpainting for 2D mask completion
    enable_completion_3d: bool = True  # 3D point cloud completion
    completeness_threshold: float = 0.6  # Below this, flag as incomplete
    detection_frequency: int = 0  # 0 = keyframe-only, N = detect every N frames

    # Advanced 3D/4D models
    enable_cotracker3: bool = True  # Dense point tracking for accurate trajectories
    enable_objectgs: bool = False  # Per-object 3D Gaussian Splatting (needs repo)
    enable_spann3r: bool = False  # Spatial memory 3D reconstruction (alternative to MASt3R)
    enable_shape_of_motion: bool = False  # End-to-end 4D reconstruction from monocular video

    # 4D Scene Decomposition
    enable_4d_trajectory: bool = True  # Extract per-object 6DoF trajectories
    enable_4dgs: bool = False  # 4D Gaussian Splatting (heavy, needs multi-GPU)
    enable_scene_graph: bool = True  # Build dynamic scene graph
    enable_animated_export: bool = True  # Export animated scenes (glTF/USD/Blender)

    # 4D Trajectory parameters
    trajectory_smoothing: float = Field(0.5, ge=0.0, le=1.0, description="B-spline smoothing factor (0=none, 1=max)")
    icp_distance_threshold: float = Field(0.05, gt=0.0, description="ICP inlier threshold (meters)")
    deformation_threshold: float = Field(0.3, gt=0.0, description="Above this ICP residual = non-rigid motion")
    is_world_model_video: bool = Field(False, description="Relax thresholds for AI-generated video")
