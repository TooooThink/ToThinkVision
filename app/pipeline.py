"""Main pipeline orchestration: preprocess → detect → segment → OCR → depth → track → export."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from app.config import settings
from app.models.depth_estimator import estimate_depth, estimate_3d_bbox, get_depth_at_bbox
from app.models.detector import detect_objects
from app.models.ocr_engine import run_ocr
from app.models.segmentor import get_contour_from_mask, segment_image
from app.models.tracker import ObjectTracker
from app.preprocessor import extract_frames, is_image, is_video, preprocess_image
from app.schemas import (
    BBox2D,
    BBox3D,
    ExportFormat,
    Interaction,
    InteractionType,
    ObjectRelation,
    ObjectType,
    StructuredObject,
    StructuredOutput,
    TemporalInfo,
    VideoMetadata,
)
from app.utils.color import extract_dominant_color
from app.utils.geometry import check_collision, compute_relative_position, compute_z_index

logger = logging.getLogger(__name__)


# Label classification mapping
UI_LABELS = {"button", "text", "input", "icon", "image", "navigation", "card", "slider", "toggle"}
GAME_LABELS = {"floor", "wall", "door", "npc", "item", "prop", "terrain", "effect", "character", "weapon"}
EMBODIED_LABELS = {"table", "chair", "tool", "object", "obstacle", "target", "surface"}


def classify_object_type(label: str) -> ObjectType:
    """Classify a detected label into ObjectType enum."""
    label_lower = label.lower()
    for kw, obj_type in [
        ("button", ObjectType.UI_BUTTON),
        ("text", ObjectType.UI_TEXT),
        ("input", ObjectType.UI_INPUT),
        ("icon", ObjectType.UI_ICON),
        ("navigation", ObjectType.UI_NAV),
        ("card", ObjectType.UI_CARD),
        ("slider", ObjectType.UI_SLIDER),
        ("toggle", ObjectType.UI_TOGGLE),
        ("floor", ObjectType.GAME_FLOOR),
        ("wall", ObjectType.GAME_WALL),
        ("door", ObjectType.GAME_DOOR),
        ("npc", ObjectType.GAME_NPC),
        ("character", ObjectType.GAME_NPC),
        ("item", ObjectType.GAME_ITEM),
        ("prop", ObjectType.GAME_PROP),
        ("terrain", ObjectType.GAME_TERRAIN),
        ("effect", ObjectType.GAME_EFFECT),
        ("weapon", ObjectType.GAME_ITEM),
        ("obstacle", ObjectType.EMBODIED_OBSTACLE),
        ("target", ObjectType.EMBODIED_TARGET),
        ("tool", ObjectType.EMBODIED_TOOL),
        ("table", ObjectType.EMBODIED_SURFACE),
        ("chair", ObjectType.EMBODIED_SURFACE),
        ("surface", ObjectType.EMBODIED_SURFACE),
    ]:
        if kw in label_lower:
            return obj_type
    return ObjectType.GENERIC


def process_file(file_path: Path, mode: str = "general") -> StructuredOutput:
    """Run the full vision pipeline on a file.

    Args:
        file_path: Path to input image or video
        mode: Detection mode hint (ui, game, video, embodied, general)
    """
    start_time = time.time()
    logger.info(f"Processing {file_path} in mode={mode}")

    # Step 1: Determine file type
    if is_video(file_path):
        return _process_video(file_path, mode)
    elif is_image(file_path):
        return _process_image(file_path, mode)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _process_image(file_path: Path, mode: str) -> StructuredOutput:
    """Process a single image through the full pipeline."""
    start_time = time.time()
    # Step 2: Preprocess
    img, img_info = preprocess_image(file_path)

    # Step 3: Run models
    # Depth estimation
    depth_map = estimate_depth(img)

    # Segmentation
    seg_results = segment_image(img)

    # Detection
    det_results = detect_objects(img, mode)

    # OCR
    ocr_results = run_ocr(img)

    # Step 4: Merge results into StructuredObjects
    objects = _merge_detections(img, depth_map, det_results, seg_results, ocr_results, frame_idx=0)

    # Step 5: Compute relations
    objects = _compute_relations(objects)

    elapsed = time.time() - start_time

    return StructuredOutput(
        source_file=file_path.name,
        source_type="image",
        metadata=None,
        objects=objects,
        frame_count=1,
        processing_time_seconds=round(elapsed, 2),
        model_versions={"segmentor": "sam", "detector": "dino", "ocr": "paddleocr", "depth": "depth-anything"},
    )


def _process_video(file_path: Path, mode: str) -> StructuredOutput:
    """Process a video through the full pipeline."""
    import tempfile

    # Step 2: Extract frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        frame_paths, video_meta = extract_frames(file_path, frame_dir / "frames")

    metadata = VideoMetadata(
        fps=video_meta["fps"],
        total_frames=video_meta["total_frames"],
        width=video_meta["width"],
        height=video_meta["height"],
        duration_seconds=video_meta["duration_seconds"],
    )

    # Step 3-5: Process each frame
    tracker = ObjectTracker()
    all_objects: list[StructuredObject] = []
    frame_idx = 0

    for frame_path in frame_paths:
        img, img_info = preprocess_image(frame_path)
        depth_map = estimate_depth(img)
        seg_results = segment_image(img)
        det_results = detect_objects(img, mode)
        ocr_results = run_ocr(img)

        # Merge detections
        frame_objects = _merge_detections(img, depth_map, det_results, seg_results, ocr_results, frame_idx=frame_idx)

        # Track across frames
        det_for_tracking = [
            {"bbox": [o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h], "label": o.label_custom or o.label.value, "confidence": o.confidence}
            for o in frame_objects
        ]
        tracks = tracker.update(det_for_tracking, frame_idx)

        # Update objects with tracking info
        track_map = {t["id"]: t for t in tracks}
        for obj in frame_objects:
            # Find matching track
            best_track = None
            best_iou = 0
            obj_bbox = [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h]
            for tid, track in track_map.items():
                iou = ObjectTracker.iou(obj_bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track:
                obj.id = best_track["id"]
                obj.temporal = TemporalInfo(
                    frame_index=frame_idx,
                    appear_frame=best_track["appear_frame"],
                    disappear_frame=best_track["disappear_frame"],
                    trajectory=best_track["history"],
                    velocity=tracker.compute_velocity(best_track["id"]),
                )

        all_objects.extend(frame_objects)
        frame_idx += 1

    # Step 6: Compute relations on final frame objects
    if all_objects:
        # Use last occurrence of each object
        last_objects: dict[str, StructuredObject] = {}
        for obj in all_objects:
            last_objects[obj.id] = obj
        final_objects = list(last_objects.values())
        final_objects = _compute_relations(final_objects)
    else:
        final_objects = []

    elapsed = time.time() - start_time

    return StructuredOutput(
        source_file=file_path.name,
        source_type="video",
        metadata=metadata,
        objects=final_objects,
        frame_count=frame_idx,
        processing_time_seconds=round(elapsed, 2),
        model_versions={"segmentor": "sam", "detector": "dino", "ocr": "paddleocr", "depth": "depth-anything", "tracker": "aot"},
    )


def _merge_detections(
    img: np.ndarray,
    depth_map: np.ndarray,
    det_results: list[dict],
    seg_results: list[dict],
    ocr_results: list[dict],
    frame_idx: int,
) -> list[StructuredObject]:
    """Merge segmentation, detection, and OCR results into StructuredObjects."""
    objects: list[StructuredObject] = []
    obj_counter = 0

    # Merge detections with segmentation masks
    for det in det_results:
        det_bbox = det["bbox"]
        # Find best matching segmentation mask
        best_seg = None
        best_iou = 0
        for seg in seg_results:
            seg_bbox = seg["bbox"]
            iou = ObjectTracker.iou(det_bbox, seg_bbox)
            if iou > best_iou:
                best_iou = iou
                best_seg = seg

        # Extract color
        dominant_color, color_palette = extract_dominant_color(img, det_bbox)

        # Get depth
        depth_val = get_depth_at_bbox(depth_map, det_bbox)
        bbox_3d = estimate_3d_bbox(det_bbox, depth_map)

        # Find matching OCR text
        matched_text = None
        matched_text_conf = None
        for ocr in ocr_results:
            ocr_bbox = ocr["bbox"]
            if check_collision(det_bbox, ocr_bbox):
                matched_text = ocr["text"]
                matched_text_conf = ocr["confidence"]
                break

        # Get contour from segmentation mask
        contour = []
        if best_seg is not None and "mask" in best_seg:
            mask = best_seg["mask"]
            if isinstance(mask, np.ndarray):
                contour = get_contour_from_mask(mask)

        obj_id = f"obj_{obj_counter:04d}"
        obj_counter += 1

        obj = StructuredObject(
            id=obj_id,
            label=classify_object_type(det["label"]),
            label_custom=det["label"],
            confidence=det["confidence"],
            bbox=BBox2D(x=det_bbox[0], y=det_bbox[1], w=det_bbox[2], h=det_bbox[3]),
            contour=contour,
            bbox_3d=BBox3D(x=bbox_3d["x"], y=bbox_3d["y"], z=bbox_3d["z"]),
            depth_value=depth_val,
            dominant_color=dominant_color,
            color_palette=color_palette,
            z_index=0,
            text_content=matched_text,
            text_confidence=matched_text_conf,
            temporal=TemporalInfo(frame_index=frame_idx),
            raw_data={"detection": det, "segmentation": bool(best_seg)},
        )
        objects.append(obj)

    # Add OCR-only detections (text regions not covered by detection)
    for ocr in ocr_results:
        ocr_bbox = ocr["bbox"]
        # Check if already covered
        covered = False
        for obj in objects:
            obj_bbox = [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h]
            if check_collision(ocr_bbox, obj_bbox):
                covered = True
                break
        if not covered:
            dominant_color, _ = extract_dominant_color(img, ocr_bbox)
            depth_val = get_depth_at_bbox(depth_map, ocr_bbox)
            bbox_3d = estimate_3d_bbox(ocr_bbox, depth_map)

            obj_id = f"obj_{obj_counter:04d}"
            obj_counter += 1
            objects.append(StructuredObject(
                id=obj_id,
                label=ObjectType.UI_TEXT,
                confidence=ocr["confidence"],
                bbox=BBox2D(x=ocr_bbox[0], y=ocr_bbox[1], w=ocr_bbox[2], h=ocr_bbox[3]),
                bbox_3d=BBox3D(x=bbox_3d["x"], y=bbox_3d["y"], z=bbox_3d["z"]),
                depth_value=depth_val,
                dominant_color=dominant_color,
                text_content=ocr["text"],
                text_confidence=ocr["confidence"],
                temporal=TemporalInfo(frame_index=frame_idx),
            ))

    return objects


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

            # Check collision
            if check_collision(obj_bbox, other_bbox):
                if other.id not in obj.relations.collision_with:
                    obj.relations.collision_with.append(other.id)

            # Compute relative position
            rel = compute_relative_position(obj_bbox, other_bbox)
            obj.relations.relative_positions.append({
                "target_id": other.id,
                "relation": rel,
            })

        # Determine interaction type based on label
        if obj.label in (ObjectType.UI_BUTTON, ObjectType.UI_ICON):
            obj.interaction = Interaction(type=InteractionType.CLICKABLE, clickable=True)
        elif obj.label in (ObjectType.UI_SLIDER,):
            obj.interaction = Interaction(type=InteractionType.SCROLLABLE, scrollable=True, direction="horizontal")
        elif obj.label in (ObjectType.UI_TOGGLE,):
            obj.interaction = Interaction(type=InteractionType.TOGGLE, toggle_state=False)

    return objects
