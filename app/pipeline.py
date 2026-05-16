"""Main pipeline v2: SAM 3 + OmniParser + GroundingDINO + Depth Pro + StrongSORT + MASt3R + 3DGS."""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.exporters.image_exporter import (
    export_crop_image,
    export_depth_visualization,
    export_detection_overlay,
    export_mask_image,
    export_mask_with_alpha,
    export_point_cloud_preview,
)
from app.models.depth_pro import DepthPro, get_depth_model
from app.models.gaussian_splatting import GaussianSplatPipeline, get_splat_pipeline
from app.models.grounding_dino import GroundingDINO, get_detector
from app.models.mast3r import MASt3RReconstructor, get_reconstructor
from app.models.omniparser import OmniParser, get_omniparser
from app.models.sam3 import SAM3Predictor, get_contour_from_mask, mask_to_base64
from app.models.strongsort_wrapper import StrongSORTTracker, get_tracker
from app.preprocessor import extract_frames, is_image, is_video, preprocess_image
from app.schemas import (
    BBox2D,
    BBox3D,
    CameraPose,
    ExportFormat,
    GaussianSplatData,
    Interaction,
    InteractionType,
    ObjectRelation,
    ObjectType,
    PipelineConfig,
    PointCloud,
    StructuredObject,
    StructuredOutput,
    TemporalInfo,
    VideoMetadata,
)
from app.utils.camera import estimate_intrinsics
from app.utils.color import extract_dominant_color
from app.utils.geometry import check_collision, compute_relative_position, compute_z_index
from app.utils.pointcloud import backproject_depth, compute_normals, filter_pointcloud

logger = logging.getLogger(__name__)

# Label classification
UI_LABELS = {"button", "text", "input", "icon", "image", "navigation", "card", "slider", "toggle"}
GAME_LABELS = {"floor", "wall", "door", "npc", "item", "prop", "terrain", "effect", "character", "weapon"}


def classify_object_type(label: str) -> ObjectType:
    """Classify a detected label into ObjectType enum."""
    label_lower = label.lower()
    for kw, obj_type in [
        ("button", ObjectType.UI_BUTTON), ("text", ObjectType.UI_TEXT),
        ("input", ObjectType.UI_INPUT), ("icon", ObjectType.UI_ICON),
        ("navigation", ObjectType.UI_NAV), ("card", ObjectType.UI_CARD),
        ("slider", ObjectType.UI_SLIDER), ("toggle", ObjectType.UI_TOGGLE),
        ("floor", ObjectType.GAME_FLOOR), ("wall", ObjectType.GAME_WALL),
        ("door", ObjectType.GAME_DOOR), ("npc", ObjectType.GAME_NPC),
        ("character", ObjectType.GAME_NPC), ("item", ObjectType.GAME_ITEM),
        ("prop", ObjectType.GAME_PROP), ("terrain", ObjectType.GAME_TERRAIN),
        ("effect", ObjectType.GAME_EFFECT), ("weapon", ObjectType.GAME_ITEM),
        ("obstacle", ObjectType.EMBODIED_OBSTACLE), ("target", ObjectType.EMBODIED_TARGET),
        ("tool", ObjectType.EMBODIED_TOOL), ("table", ObjectType.EMBODIED_SURFACE),
        ("chair", ObjectType.EMBODIED_SURFACE), ("surface", ObjectType.EMBODIED_SURFACE),
    ]:
        if kw in label_lower:
            return obj_type
    return ObjectType.GENERIC


def process_file(file_path: Path, mode: str = "general", config: PipelineConfig | None = None) -> StructuredOutput:
    """Run the full v2 vision pipeline on a file."""
    if config is None:
        config = PipelineConfig(mode=mode)
    if is_video(file_path):
        return _process_video(file_path, config)
    elif is_image(file_path):
        return _process_image(file_path, config)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _process_image(file_path: Path, config: PipelineConfig) -> StructuredOutput:
    """Process a single image through the full v2 pipeline."""
    start_time = time.time()
    img, img_info = preprocess_image(file_path)

    # ─── Detection ───────────────────────────────────────────
    detections = []

    # OmniParser for UI mode
    if config.enable_omniparser and config.mode == "ui":
        omni = get_omniparser()
        omni_results = omni.parse_to_boxes(img)
        for r in omni_results:
            detections.append({
                "bbox": r["bbox_pixel"],
                "label": r["type"],
                "confidence": r["confidence"],
                "interactivity": r["interactivity"],
                "text": r["content"],
            })

    # Grounding DINO for general modes (or combined with OmniParser)
    if config.enable_grounding_dino and (config.mode != "ui" or not detections):
        detector = get_detector()
        det_results = detector.detect(img, config.mode)
        detections.extend(det_results)

    # ─── SAM 3 Segmentation ─────────────────────────────────
    boxes = np.array([d["bbox"] for d in detections]) if detections else None
    if config.enable_sam3:
        sam3 = SAM3Predictor()
        seg_results = sam3.predict(img, boxes=boxes)
    else:
        seg_results = []

    # Merge detection labels with segmentation masks
    detections = _merge_detection_with_segmentation(detections, seg_results)

    # ─── Depth Pro ──────────────────────────────────────────
    if config.enable_depth_pro:
        depth_model = get_depth_model()
        depth_map = depth_model.estimate(img)
    else:
        depth_model = DepthPro()
        depth_map = None

    # ─── OCR (from OmniParser or PaddleOCR) ─────────────────
    ocr_texts = {d.get("text") for d in detections if d.get("text")}

    # ─── Build StructuredObjects ────────────────────────────
    objects = []
    K = estimate_intrinsics(img_info["width"], img_info["height"])
    for det in detections:
        bbox = det["bbox"]
        seg = det.get("segmentation", {})
        mask = seg.get("mask")
        dominant_color, color_palette = extract_dominant_color(img, bbox)

        if depth_map is not None:
            depth_val = depth_model.get_depth_at(depth_map, bbox)
            points_3d = backproject_depth(np.array([[depth_val]]), K)
        else:
            depth_val = 0.0

        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

        contour = get_contour_from_mask(mask) if mask is not None else []
        mask_b64 = mask_to_base64(mask) if mask is not None else None

        # Crop object image
        crop_img = _crop_object(img, bbox)
        crop_b64 = _image_to_base64(crop_img) if crop_img is not None else None

        obj = StructuredObject(
            id=det.get("id", f"obj_{len(objects):04d}"),
            label=classify_object_type(det.get("label", "object")),
            label_custom=det.get("label"),
            confidence=det.get("confidence", 0.5),
            bbox=BBox2D(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3]),
            contour=contour,
            mask_base64=mask_b64,
            bbox_3d=BBox3D(x=cx, y=cy, z=depth_val),
            depth_value=depth_val,
            dominant_color=dominant_color,
            color_palette=color_palette,
            text_content=det.get("text"),
            crop_image_base64=crop_b64,
            raw_data=det,
        )
        objects.append(obj)

    objects = _compute_relations(objects)

    # ─── 3D Point Cloud (single image: back-project depth) ──
    point_cloud = None
    if depth_map is not None:
        points = backproject_depth(depth_map, K)
        colors = img.reshape(-1, 3)
        points, colors = filter_pointcloud(points, colors, voxel_size=0.02)
        normals = compute_normals(points[:50000], k=10)
        point_cloud = PointCloud(
            points=points.tolist(),
            colors=colors.tolist() if colors is not None else [],
            normals=normals.tolist() if len(normals) > 0 else None,
        )

    # ─── Export visual outputs ──────────────────────────────
    img_dir = _ensure_export_dir(file_path)

    # Detection overlay
    detection_png = export_detection_overlay(img, detections, img_dir, file_path.name)

    # Depth visualization
    depth_png = export_depth_visualization(depth_map, img_dir, file_path.name) if depth_map is not None else None

    # Per-object: crop, mask, masked (transparent) PNGs
    _export_object_images(img, objects, img_dir)

    # Point cloud preview
    pc_preview = None
    if point_cloud and point_cloud.points:
        pc_preview = export_point_cloud_preview(
            np.array(point_cloud.points), np.array(point_cloud.colors), img_dir, file_path.name
        )

    elapsed = time.time() - start_time
    model_versions = {}
    if config.enable_sam3:
        model_versions["segmentation"] = "sam3"
    if config.enable_omniparser and config.mode == "ui":
        model_versions["detection"] = "omniparser"
    elif config.enable_grounding_dino:
        model_versions["detection"] = "grounding_dino"
    if config.enable_depth_pro:
        model_versions["depth"] = "depth_pro"

    return StructuredOutput(
        source_file=file_path.name,
        source_type="image",
        metadata=None,
        objects=objects,
        frame_count=1,
        processing_time_seconds=round(elapsed, 2),
        model_versions=model_versions,
        point_cloud=point_cloud,
        depth_map_png_path=str(depth_png) if depth_png else None,
        detection_overlay_png_path=str(detection_png) if detection_png else None,
        point_cloud_preview_png_path=str(pc_preview) if pc_preview else None,
    )


def _process_video(file_path: Path, config: PipelineConfig) -> StructuredOutput:
    """Process a video through the full v2 pipeline with 3D reconstruction."""
    start_time = time.time()

    # ─── Extract frames (persist to output dir so MASt3R/3DGS can access) ──
    frame_dir = settings.output_dir / f"{file_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths, video_meta = extract_frames(file_path, frame_dir)

    metadata = VideoMetadata(
        fps=video_meta["fps"],
        total_frames=video_meta["total_frames"],
        width=video_meta["width"],
        height=video_meta["height"],
        duration_seconds=video_meta["duration_seconds"],
    )
    fps = video_meta["fps"]

    # ─── SAM 3 Video Tracking ───────────────────────────────
    sam3 = SAM3Predictor() if config.enable_sam3 else None
    inference_state = sam3.init_video(str(frame_dir)) if sam3 else None
    sam3_video_initialized = inference_state is not None

    # ─── Per-frame processing ───────────────────────────────
    tracker = get_tracker(fps=fps) if config.enable_strongsort else None
    depth_model = get_depth_model() if config.enable_depth_pro else DepthPro()
    all_frame_objects: list[StructuredObject] = {}  # keyed by track_id
    frame_idx = 0
    keyframe_objects: dict[str, dict] = {}  # track_id → segmentation for SAM 3 prompts

    for frame_path in frame_paths:
        img, img_info = preprocess_image(frame_path)

        # Detection on first frame (or every N frames)
        if frame_idx % max(1, int(fps)) == 0:
            detections = []
            # OmniParser for UI mode
            if config.enable_omniparser and config.mode == "ui":
                omni = get_omniparser()
                omni_results = omni.parse_to_boxes(img)
                for r in omni_results:
                    detections.append({
                        "bbox": r["bbox_pixel"],
                        "label": r["type"],
                        "confidence": r["confidence"],
                    })

            if config.enable_grounding_dino and (config.mode != "ui" or not detections):
                detector = get_detector()
                detections.extend(detector.detect(img, config.mode))

            # SAM 3 segmentation
            boxes = np.array([d["bbox"] for d in detections]) if detections else None
            if sam3:
                seg_results = sam3.predict(img, boxes=boxes)
            else:
                seg_results = []
            detections = _merge_detection_with_segmentation(detections, seg_results)

            # Feed keyframe detections to SAM 3 video predictor for propagation
            if sam3_video_initialized and detections:
                for i, det in enumerate(detections):
                    bbox = det["bbox"]
                    box_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    sam3.add_prompt(inference_state, frame_idx, i, box=box_xyxy)

        else:
            detections = []  # Use tracking for subsequent frames

        # ─── Depth Pro per frame ────────────────────────────
        depth_map = depth_model.estimate(img)

        # ─── StrongSORT tracking ───────────────────────────
        if tracker:
            tracked = tracker.update(detections, frame=img, frame_idx=frame_idx)
        else:
            tracked = [{"bbox": d["bbox"], "id": f"obj_{i:04d}", "label": d.get("label"), "confidence": d.get("confidence", 0.5)} for i, d in enumerate(detections)]

        # ─── Build objects ─────────────────────────────────
        for track in tracked:
            bbox = track["bbox"]
            tid = track["id"]
            dominant_color, color_palette = extract_dominant_color(img, bbox)
            depth_val = depth_model.get_depth_at(depth_map, bbox)

            if tid in all_frame_objects:
                # Update existing track
                existing = all_frame_objects[tid]
                existing.temporal.trajectory.extend(track.get("history", []))
                existing.temporal.depth_per_frame.append(depth_val)
                if track.get("disappear_frame", -1) >= 0:
                    existing.temporal.disappear_frame = track["disappear_frame"]
            else:
                obj = StructuredObject(
                    id=tid,
                    label=classify_object_type(track.get("label", "object")),
                    label_custom=track.get("label"),
                    confidence=track.get("confidence", 0.5),
                    bbox=BBox2D(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3]),
                    depth_value=depth_val,
                    dominant_color=dominant_color,
                    color_palette=color_palette,
                    temporal=TemporalInfo(
                        frame_index=frame_idx,
                        appear_frame=track.get("appear_frame", frame_idx),
                        disappear_frame=track.get("disappear_frame", -1),
                        trajectory=track.get("history", []),
                        depth_per_frame=[depth_val],
                    ),
                )
                all_frame_objects[tid] = obj

        frame_idx += 1

    # ─── Propagate SAM 3 masks through video ────────────────
    if inference_state is not None:
        propagation_results = sam3.propagate_video(inference_state)
        # Merge propagated masks into tracked objects
        _merge_propagated_masks(all_frame_objects, propagation_results)

    # ─── 3D Reconstruction (MASt3R) ─────────────────────────
    pc_data = {"points": [], "colors": []}
    camera_poses = []
    if config.enable_mast3r:
        reconstructor = get_reconstructor()
        pc_data, camera_poses = reconstructor.reconstruct(frame_dir)

    # ─── 3D Gaussian Splatting (optional) ───────────────────
    gs_data = None
    if config.enable_gaussian_splatting:
        gs_pipe = get_splat_pipeline()
        gs_data = gs_pipe.train(frame_dir, settings.output_dir / "gs_training")

    # ─── Merge objects (dictionary → list) ──────────────────
    final_objects = list(all_frame_objects.values())

    # Compute velocity for tracked objects
    if tracker:
        for obj in final_objects:
            vel = tracker.compute_velocity(obj.id)
            if vel:
                obj.temporal.velocity = vel

    final_objects = _compute_relations(final_objects)

    # ─── Export visual outputs ──────────────────────────────
    img_dir = _ensure_export_dir(file_path)

    # Detection overlay (last processed frame)
    detection_png = export_detection_overlay(img, detections, img_dir, file_path.name)

    # Point cloud preview
    pc_preview = None
    pc_points = pc_data.get("points", [])
    if pc_points:
        pts = np.array(pc_points)
        pc_colors = pc_data.get("colors", [])
        clr = np.array(pc_colors) if pc_colors and len(pc_colors) == len(pts) else None
        pc_preview = export_point_cloud_preview(pts, clr, img_dir, file_path.name)

    # Export per-object crops and masks from the last frame
    _export_object_images(img, final_objects, img_dir)

    # ─── Build output ───────────────────────────────────────
    elapsed = time.time() - start_time
    model_versions = {}
    if config.enable_sam3:
        model_versions["segmentation"] = "sam3"
    if config.enable_omniparser and config.mode == "ui":
        model_versions["detection"] = "omniparser"
    elif config.enable_grounding_dino:
        model_versions["detection"] = "grounding_dino"
    if config.enable_strongsort:
        model_versions["tracking"] = "strongsort"
    if config.enable_depth_pro:
        model_versions["depth"] = "depth_pro"
    if config.enable_mast3r:
        model_versions["reconstruction"] = "mast3r"
    if config.enable_gaussian_splatting and gs_data:
        model_versions["gaussian_splatting"] = "gsplat"

    # Save PLY/splat file paths if 3D data is available
    ply_path = None
    splat_path = None
    if config.enable_gaussian_splatting and gs_data:
        gs_pipe = get_splat_pipeline()
        ply_save = settings.output_dir / f"{file_path.stem}_3dgs.ply"
        splat_save = settings.output_dir / f"{file_path.stem}_3dgs.splat"
        gs_pipe.export_ply(gs_data, ply_save)
        gs_pipe.export_splat(gs_data, splat_save)
        ply_path = str(ply_save)
        splat_path = str(splat_save)
    elif pc_points:
        # Fallback: export point cloud as PLY
        from app.utils.pointcloud import save_ply
        ply_save = settings.output_dir / f"{file_path.stem}_pointcloud.ply"
        pts_arr = np.array(pc_points)
        clr_arr = np.array(pc_colors) if pc_colors and len(pc_colors) == len(pts_arr) else None
        save_ply(ply_save, pts_arr, clr_arr)
        ply_path = str(ply_save)

    output = StructuredOutput(
        source_file=file_path.name,
        source_type="video",
        metadata=metadata,
        objects=final_objects,
        frame_count=frame_idx,
        processing_time_seconds=round(elapsed, 2),
        model_versions=model_versions,
        point_cloud=PointCloud(
            points=pc_data.get("points", []),
            colors=pc_data.get("colors", []),
        ),
        camera_poses=[CameraPose(**pose) for pose in camera_poses],
        detection_overlay_png_path=str(detection_png) if detection_png else None,
        point_cloud_preview_png_path=str(pc_preview) if pc_preview else None,
        ply_file_path=ply_path,
        splat_file_path=splat_path,
    )

    if gs_data:
        output.gaussian_splats = GaussianSplatData(
            means=gs_data.get("means", []),
            quats=gs_data.get("quats", []),
            scales=gs_data.get("scales", []),
            opacities=gs_data.get("opacities", []),
            sh_coeffs=gs_data.get("sh_coeffs", []),
        )

    return output


def _merge_detection_with_segmentation(
    detections: list[dict], seg_results: list[dict]
) -> list[dict]:
    """Merge detection results with segmentation masks using IoU matching."""
    if not seg_results:
        return detections

    merged = []
    used_seg = set()

    for det in detections:
        det_bbox = det["bbox"]
        best_seg = None
        best_iou = 0

        for i, seg in enumerate(seg_results):
            seg_bbox = seg["bbox"]
            iou = _calc_iou(det_bbox, seg_bbox)
            if iou > best_iou:
                best_iou = iou
                best_seg = seg

        if best_seg is not None and best_iou > 0.2:
            det["segmentation"] = best_seg
            det["confidence"] = max(det.get("confidence", 0), best_seg.get("confidence", 0))
            merged.append(det)
        else:
            merged.append(det)

    # Add unused segmentation results
    for i, seg in enumerate(seg_results):
        if i not in used_seg:
            merged.append({
                "bbox": seg["bbox"],
                "label": seg.get("label", "object"),
                "confidence": seg.get("confidence", 0.5),
                "segmentation": seg,
            })

    return merged


def _calc_iou(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _crop_object(img: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Crop object region from image."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_img, w_img = img.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    if x2 <= x or y2 <= y:
        return None
    return img[y:y2, x:x2]


def _image_to_base64(img: np.ndarray) -> str | None:
    """Convert image to base64 PNG string."""
    try:
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _compute_relations(objects: list[StructuredObject]) -> list[StructuredObject]:
    """Compute spatial relationships between objects."""
    if not objects:
        return objects

    bboxes = [(obj.id, [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h]) for obj in objects]
    z_indices = compute_z_index(bboxes)

    for i, obj in enumerate(objects):
        obj.z_index = z_indices.get(obj.id, i)
        obj_bbox = [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h]

        for j, other in enumerate(objects):
            if i == j:
                continue
            other_bbox = [other.bbox.x, other.bbox.y, other.bbox.w, other.bbox.h]

            if check_collision(obj_bbox, other_bbox):
                if other.id not in obj.relations.collision_with:
                    obj.relations.collision_with.append(other.id)

            rel = compute_relative_position(obj_bbox, other_bbox)
            obj.relations.relative_positions.append({
                "target_id": other.id,
                "relation": rel,
            })

        if obj.label in (ObjectType.UI_BUTTON, ObjectType.UI_ICON):
            obj.interaction = Interaction(type=InteractionType.CLICKABLE, clickable=True)
        elif obj.label == ObjectType.UI_SLIDER:
            obj.interaction = Interaction(type=InteractionType.SCROLLABLE, scrollable=True, direction="horizontal")
        elif obj.label == ObjectType.UI_TOGGLE:
            obj.interaction = Interaction(type=InteractionType.TOGGLE, toggle_state=False)

    return objects


def _ensure_export_dir(file_path: Path) -> Path:
    """Create and return export subdirectory for a source file."""
    stem = Path(file_path).stem
    d = settings.output_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _export_object_images(img: np.ndarray, objects: list[StructuredObject], output_dir: Path):
    """Export crop, mask, and masked PNG for each object."""
    for obj in objects:
        bbox = [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h]
        obj_id = obj.id

        # Crop PNG
        crop_img = _crop_object(img, bbox)
        if crop_img is not None:
            crop_path = export_crop_image(crop_img, [0, 0, bbox[2], bbox[3]], output_dir, obj_id)
            if crop_path:
                obj.crop_png_path = str(crop_path)

        # Mask PNG
        if obj.mask_base64:
            import base64
            from io import BytesIO
            mask_bytes = base64.b64decode(obj.mask_base64)
            mask_pil = __import__("PIL", fromlist=["Image"]).Image.open(BytesIO(mask_bytes))
            mask = np.array(mask_pil) > 0
            mask_path = export_mask_image(mask.astype(np.uint8), output_dir, obj_id)
            if mask_path:
                obj.mask_png_path = str(mask_path)

            # Masked PNG (transparent background for PS/AE/Unity)
            crop_img_obj = _crop_object(img, bbox)
            if crop_img_obj is not None and mask is not None:
                import cv2
                # Resize mask to match crop
                mask_cropped = cv2.resize(
                    mask.astype(np.uint8) * 255, (crop_img_obj.shape[1], crop_img_obj.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                masked_path = export_mask_with_alpha(crop_img_obj, mask_cropped > 0, output_dir, obj_id)
                if masked_path:
                    obj.raw_data["masked_png_path"] = str(masked_path)


def _merge_propagated_masks(
    objects: dict[str, StructuredObject],
    propagation_results: list[tuple[int, int, np.ndarray]],
):
    """Merge SAM 3 propagated video masks into tracked objects.

    Args:
        objects: dict of track_id → StructuredObject (updated in place)
        propagation_results: list of (frame_idx, obj_id, mask) from SAM 3
    """
    for frame_idx, obj_id, mask in propagation_results:
        # Map SAM 3 obj_id to our track ID
        track_id = f"obj_{obj_id:04d}"
        if track_id not in objects:
            continue

        obj = objects[track_id]
        mask_b64 = mask_to_base64(mask)
        if mask_b64:
            obj.mask_base64 = mask_b64

        # Update bbox from mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            obj.bbox.x = float(xs.min())
            obj.bbox.y = float(ys.min())
            obj.bbox.w = float(xs.max() - xs.min())
            obj.bbox.h = float(ys.max() - ys.min())

        # Update contour
        obj.contour = get_contour_from_mask(mask) if mask is not None else []
