"""Main pipeline v2: SAM 3 + OmniParser + DINO-X + Depth Pro + BoT-SORT + VGGT + 3DGS."""

from __future__ import annotations

import base64
import logging
import time
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Suppress non-fatal numpy warnings from depth/trajectory calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')

from app.config import settings
from app.exporters.image_exporter import (
    export_crop_image,
    export_depth_visualization,
    export_detection_overlay,
    export_mask_image,
    export_mask_with_alpha,
    export_point_cloud_preview,
)
from app.models.completion_2d import get_completion_2d
from app.models.completion_3d import get_completion_3d
from app.models.depth_pro import DepthPro, get_depth_model
from app.models.gaussian_splatting import GaussianSplatPipeline, get_splat_pipeline
from app.models.grounding_dino import GroundingDINO, get_detector
from app.models.mask_accumulator import MaskAccumulator
from app.models.mast3r import MASt3RReconstructor, get_reconstructor
from app.models.mesh_reconstruction import reconstruct_object_meshes
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
    Mesh3D,
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


def clear_gpu_memory():
    """Clear GPU memory by deleting models and calling garbage collection."""
    import gc
    import torch

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    # Log memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def cleanup_all_models():
    """Delete all globally cached model instances to free GPU memory."""
    import gc
    import torch

    # Reset all global singletons
    from app.models import grounding_dino, sam3, depth_pro, strongsort_wrapper, mast3r, cotracker3
    from app.models import completion_2d, completion_3d, gaussian_splatting, omniparser
    from app.models import spann3r, object_gs, shape_of_motion

    # GroundingDINO
    grounding_dino._detector = None

    # SAM3
    sam3._sam3_predictor = None

    # Depth Pro
    depth_pro._depth_model = None

    # Tracker
    strongsort_wrapper._tracker = None

    # 3D Reconstruction
    mast3r._reconstructor = None

    # Spann3R
    if hasattr(spann3r, '_spann3r'):
        spann3r._spann3r = None

    # CoTracker3
    cotracker3._cotracker_predictor = None

    # ObjectGS
    if hasattr(object_gs, '_objectgs_pipeline'):
        object_gs._objectgs_pipeline = None

    # Shape of Motion
    if hasattr(shape_of_motion, '_shape_of_motion'):
        shape_of_motion._shape_of_motion = None

    # Completion models
    if hasattr(completion_2d, '_completion_2d'):
        completion_2d._completion_2d = None
    if hasattr(completion_3d, '_completion_3d'):
        completion_3d._completion_3d = None

    # Gaussian Splatting
    if hasattr(gaussian_splatting, '_splat_pipeline'):
        gaussian_splatting._splat_pipeline = None

    # OmniParser
    if hasattr(omniparser, '_omniparser'):
        omniparser._omniparser = None

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"All models cleaned up. GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def _check_gpu_health():
    """Verify GPU is in a usable state after subprocess failures.

    A subprocess segfault (e.g. Spann3R) can corrupt the shared CUDA context.
    This probes the GPU with a trivial allocation; if it fails, we force a
    CUDA context reset so downstream models don't inherit a broken state.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        # Tiny probe to verify CUDA context is intact
        _probe = torch.zeros(1, device='cuda')
        del _probe
        torch.cuda.synchronize()
        logger.info("GPU health check passed")
    except Exception as e:
        logger.warning("GPU health check failed (%s), attempting CUDA context reset…", e)
        try:
            import torch
            # Reset CUDA context by clearing all cached allocations
            torch.cuda.empty_cache()
            # Re-initialize by allocating and freeing on device 0
            _init = torch.zeros(1, device='cuda')
            del _init
            torch.cuda.synchronize()
            logger.info("CUDA context reset complete")
        except Exception as e2:
            logger.error("CUDA context reset failed: %s. Subsequent GPU stages may fail.", e2)

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
    model_versions: dict[str, str] = {}

    # ─── Extract frames (persist to output dir so MASt3R/3DGS can access) ──
    frame_dir = settings.output_dir / f"{file_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    img_dir = _ensure_export_dir(file_path)
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

    # Data collection for per-object 3D mesh reconstruction
    per_frame_obj_data = []
    all_depth_maps = []

    for frame_path in frame_paths:
        img, img_info = preprocess_image(frame_path)

        # Detection on first frame (or every N frames)
        # detection_frequency > 0 overrides the default keyframe cadence
        detect_interval = config.detection_frequency if config.detection_frequency > 0 else max(1, int(fps))
        if frame_idx % detect_interval == 0:
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

        # Collect per-frame object data for 3D mesh reconstruction
        frame_obj_list = []
        for track in tracked:
            bbox = track["bbox"]
            tid = track["id"]
            # Try to get mask from seg_results (keyframe) or SAM propagation
            seg_mask = None
            for seg in seg_results:
                seg_bbox = seg.get("bbox", [])
                if len(seg_bbox) == 4 and _calc_iou(bbox, seg_bbox) > 0.3:
                    seg_mask = seg.get("mask")
                    break
            frame_obj_list.append({"id": tid, "bbox": list(bbox), "mask": seg_mask})
        per_frame_obj_data.append({"frame_idx": frame_idx, "objects": frame_obj_list})
        all_depth_maps.append(depth_map)

        frame_idx += 1

    # ─── Propagate SAM 3 masks through video ────────────────
    if inference_state is not None:
        propagation_results = sam3.propagate_video(inference_state)
        # Merge propagated masks into tracked objects
        _merge_propagated_masks(all_frame_objects, propagation_results)

    # ─── Stage 1: Multi-frame Mask Accumulation ─────────────
    accumulator = MaskAccumulator(
        frame_width=video_meta["width"],
        frame_height=video_meta["height"],
    )
    for frame_entry in per_frame_obj_data:
        for obj_info in frame_entry["objects"]:
            if obj_info.get("mask") is not None:
                accumulator.accumulate(
                    obj_info["id"], obj_info["mask"], obj_info["bbox"], frame_entry["frame_idx"]
                )

    # Assess completeness for each tracked object
    final_objects = list(all_frame_objects.values())
    incomplete_objects = {}
    threshold = config.completeness_threshold

    for obj in final_objects:
        label = obj.label_custom or obj.label.value if hasattr(obj.label, "value") else "object"
        result = accumulator.get_completeness(obj.id, label=label, threshold=threshold)
        obj.temporal.completeness_score = result.score
        obj.temporal.is_complete = result.is_complete
        obj.temporal.accumulated_mask_frames = result.frames_contributed

        if not result.is_complete:
            incomplete_objects[obj.id] = {
                "score": result.score,
                "accumulated_mask": result.accumulated_mask,
                "frames": result.frames_contributed,
            }
        logger.info(
            "Object %s: completeness=%.2f (complete=%s, frames=%d)",
            obj.id, result.score, result.is_complete, result.frames_contributed,
        )

    # Build dict of accumulated masks for mesh reconstruction
    accumulated_masks = {
        tid: accumulator.get_accumulated_mask(tid)
        for tid in all_frame_objects
        if accumulator.get_accumulated_mask(tid) is not None
    }

    # ─── Stage 2a: 2D Completion (optional) ─────────────────
    if incomplete_objects and config.enable_completion_2d:
        logger.info("Stage 2a: Running 2D completion for %d incomplete objects", len(incomplete_objects))
        completion_2d = get_completion_2d()
        for obj in final_objects:
            if obj.id not in incomplete_objects:
                continue
            comp_info = incomplete_objects[obj.id]
            acc_mask = comp_info.get("accumulated_mask")
            if acc_mask is None:
                continue

            # Crop the object from the last frame it appeared in
            bbox = obj.bbox
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            if w <= 0 or h <= 0:
                continue

            # Find the last frame where this object was detected
            last_frame_idx = obj.temporal.frame_index
            last_frame_path = None
            for fe in per_frame_obj_data:
                for oi in fe["objects"]:
                    if oi["id"] == obj.id and fe["frame_idx"] <= last_frame_idx:
                        frame_file = frame_paths[fe["frame_idx"]] if fe["frame_idx"] < len(frame_paths) else None
                        if frame_file:
                            last_frame_path = frame_file

            if last_frame_path is None or not last_frame_path.exists():
                continue

            try:
                from PIL import Image
                frame_img = np.array(Image.open(last_frame_path).convert("RGB"))
                crop = frame_img[y:y+h, x:x+w]
                partial = acc_mask[y:y+h, x:x+w]

                if crop.size == 0 or partial.size == 0:
                    continue

                completed_img, completed_mask = completion_2d.complete(crop, partial)

                # Update the accumulated mask with completed region
                completed_full = np.zeros((video_meta["height"], video_meta["width"]), dtype=np.uint8)
                ch, cw = completed_mask.shape[:2]
                completed_full[y:y+ch, x:x+cw] = completed_mask
                accumulated_masks[obj.id] = completed_full

                # Update object's mask_base64
                completed_base64 = mask_to_base64(completed_mask)
                obj.mask_base64 = completed_base64
                obj.temporal.is_complete = True  # Mark as completed
                logger.info("2D completion applied to %s", obj.id)
            except Exception as e:
                logger.warning("2D completion failed for %s: %e", obj.id, e)

    # ─── 3D Reconstruction (MASt3R / Spann3R) ───────────────
    pc_data = {"points": [], "colors": []}
    camera_poses = []
    reconstruction_backend = "none"

    # Clean up ALL models (including global singletons) before loading 3D reconstruction.
    # SAM3's global singleton holds ~28GB; just `del sam3` only removes the local reference.
    logger.info("Clearing GPU memory before 3D reconstruction...")
    cleanup_all_models()

    # Diagnostic: log GPU memory state after cleanup
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "GPU after cleanup: %.1f GB free / %.1f GB total (%.1f%% free)",
            free / 1024**3, total / 1024**3, free / total * 100,
        )

    # Try Spann3R first if enabled (spatial memory for better long sequences)
    if config.enable_spann3r and config.enable_mast3r:
        try:
            from app.models.spann3r import get_spann3r
            spann3r = get_spann3r()
            if spann3r.available:
                pc_data, camera_poses = spann3r.reconstruct(
                    frame_dir, sample_interval=config.mast3r_sample_interval,
                )
                reconstruction_backend = "spann3r"
                logger.info("3D reconstruction via Spann3R (spatial memory)")
            else:
                logger.info("Spann3R not available, trying MASt3R/VGGT")
        except Exception as e:
            logger.warning("Spann3R failed (%s), falling back to MASt3R/VGGT", e)
            _check_gpu_health()

    # Try MASt3R/VGGT
    if reconstruction_backend == "none" and config.enable_mast3r:
        reconstructor = get_reconstructor()
        pc_data, camera_poses = reconstructor.reconstruct(frame_dir)
        reconstruction_backend = "vggt" if reconstructor._backend == "vggt" else "mast3r"

    # ─── Per-Object 3D Mesh Reconstruction ──────────────────
    mesh_results = {}
    scene_mesh_path = None
    if config.enable_3d_reconstruction and all_depth_maps and per_frame_obj_data:
        mesh_dir = img_dir / "meshes"
        mesh_results = reconstruct_object_meshes(
            frame_paths=frame_paths,
            per_frame_objects=per_frame_obj_data,
            depth_maps=all_depth_maps,
            frame_width=video_meta["width"],
            frame_height=video_meta["height"],
            output_dir=mesh_dir,
            camera_poses=camera_poses,
            accumulated_masks=accumulated_masks,
        )

    # ─── Stage 2b: 3D Completion (optional) ─────────────────
    if incomplete_objects and config.enable_completion_3d and mesh_results:
        logger.info("Stage 2b: Running 3D completion for incomplete objects")
        completion_3d = get_completion_3d()
        for obj in final_objects:
            if obj.id not in mesh_results:
                continue
            if obj.id not in incomplete_objects and obj.temporal.is_complete:
                continue
            mdata = mesh_results[obj.id]
            vertices = np.array(mdata.get("vertices", []))
            colors = np.array(mdata.get("colors", []))
            if len(vertices) < 10:
                continue

            original_count = len(vertices)
            completed_verts, completed_colors, method = completion_3d.complete(vertices, colors)

            if len(completed_verts) > original_count:
                mdata["vertices"] = completed_verts.tolist()
                mdata["colors"] = completed_colors.tolist()
                mdata["completion_applied"] = True
                mdata["completion_method"] = method
                mdata["original_point_count"] = original_count
                logger.info(
                    "3D completion applied to %s: %d -> %d points",
                    obj.id, original_count, len(completed_verts),
                )

    # ─── 3D Gaussian Splatting (optional) ───────────────────
    gs_data = None
    if config.enable_gaussian_splatting:
        # Clean up all models before Gaussian Splatting
        logger.info("Clearing GPU memory before 3D Gaussian Splatting...")
        cleanup_all_models()

        gs_pipe = get_splat_pipeline()
        gs_data = gs_pipe.train(frame_dir, settings.output_dir / "gs_training")

    # ─── ObjectGS: Per-object 3D Gaussian Splatting (optional) ──
    objectgs_data = None
    if config.enable_objectgs:
        try:
            logger.info("Clearing GPU memory before ObjectGS...")
            cleanup_all_models()
            from app.models.object_gs import get_objectgs_pipeline
            objectgs_pipe = get_objectgs_pipeline()
            if objectgs_pipe.available:
                masks_dir = img_dir / "objectgs_masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                objectgs_data = objectgs_pipe.train(
                    frame_dir=frame_dir,
                    masks_dir=masks_dir,
                    output_dir=settings.output_dir / "objectgs_training",
                )
                model_versions["object_gs"] = "per_object_3dgs"
                logger.info("ObjectGS: trained %d per-object Gaussians",
                            len(objectgs_data.get("object_meshes", {})))
        except Exception as e:
            logger.warning("ObjectGS failed: %s", e)
            _check_gpu_health()

    # ─── CoTracker3: Dense point tracking (optional) ─────────
    cotracker_data = None
    if config.enable_cotracker3 and len(frame_paths) >= 2:
        try:
            # Clean up all models before CoTracker3
            logger.info("Clearing GPU memory before CoTracker3...")
            cleanup_all_models()

            from app.models.cotracker3 import get_cotracker
            cotracker = get_cotracker(mode="offline")
            if cotracker.model is not None:
                # Load frames as numpy array for CoTracker3
                # Limit frames and grid_size to avoid CUDA OOM:
                # CoTracker3 loads entire video tensor (T,3,H,W) into GPU.
                from PIL import Image as PILImage
                max_cotracker_frames = min(50, config.max_video_frames)
                cotracker_frame_paths = frame_paths[:max_cotracker_frames]
                cotracker_frames = []
                for fp in cotracker_frame_paths:
                    fimg = np.array(PILImage.open(fp).convert("RGB"))
                    cotracker_frames.append(fimg)
                cotracker_frames = np.stack(cotracker_frames)
                logger.info("CoTracker3: processing %d frames with grid_size=20 (500 points)",
                            len(cotracker_frames))
                cotracker_data = cotracker.track_video(
                    frames=cotracker_frames,
                    grid_size=20,
                    query_frame=0,
                )
                model_versions["cotracker3"] = "offline"
                logger.info("CoTracker3: tracked %d points across %d frames",
                            cotracker_data.get("num_points", 0), len(cotracker_frames))
                # Free CoTracker3 model memory for downstream stages
                cleanup_all_models()
        except Exception as e:
            logger.warning("CoTracker3 failed: %s", e)
            _check_gpu_health()

    # ─── Stage 6: 4D Trajectory Extraction ───────────────────
    trajectories_4d = {}
    trajectory_backend = "none"

    # Option A: Shape of Motion (end-to-end 4D, replaces depth+tracking+ICP)
    if config.enable_shape_of_motion and config.enable_4d_trajectory:
        try:
            from app.models.shape_of_motion import get_shape_of_motion
            som_pipe = get_shape_of_motion()
            som_result = som_pipe.reconstruct_4d(
                video_path=file_path,
                output_dir=settings.output_dir / "shape_of_motion",
                num_frames=min(len(frame_paths), config.max_video_frames),
            )
            # Extract trajectories from Shape of Motion output
            som_pcs = som_result.get("per_frame_pointclouds", [])
            if som_pcs:
                som_trajectories = som_pipe.extract_object_trajectories(som_pcs)
                for obj_id, traj_data in som_trajectories.items():
                    from app.schemas import ObjectTrajectory4D, TrajectoryKeyframe
                    kfs = [TrajectoryKeyframe(**kf) for kf in traj_data.get("keyframes", [])
                           if "position" in kf and "rotation" not in kf]
                    # Shape of Motion gives positions but not rotations — fill defaults
                    full_kfs = []
                    for kf_data in traj_data.get("keyframes", []):
                        full_kfs.append(TrajectoryKeyframe(
                            timestamp=kf_data["timestamp"],
                            frame_idx=kf_data["frame_idx"],
                            position=kf_data["position"],
                            rotation=(1.0, 0.0, 0.0, 0.0),
                        ))
                    if full_kfs:
                        traj = ObjectTrajectory4D(
                            object_id=obj_id,
                            keyframes=full_kfs,
                            motion_type=traj_data.get("motion_type", "rigid"),
                            duration=full_kfs[-1].timestamp - full_kfs[0].timestamp if len(full_kfs) > 1 else 0.0,
                        )
                        trajectories_4d[obj_id] = traj
                trajectory_backend = "shape_of_motion"
                model_versions["trajectory_4d"] = "shape_of_motion"
                logger.info("Shape of Motion: extracted %d trajectories (end-to-end 4D)",
                            len(trajectories_4d))
        except Exception as e:
            logger.warning("Shape of Motion failed: %s, falling back to ICP-based extraction", e)
            _check_gpu_health()

    # Option B: ICP-based trajectory extraction (with CoTracker3 enhancement)
    if not trajectories_4d and config.enable_4d_trajectory and all_depth_maps and per_frame_obj_data and camera_poses:
        try:
            from app.models.trajectory_4d import TrajectoryExtractor4D
            traj_extractor = TrajectoryExtractor4D(config)
            trajectories_4d = traj_extractor.extract_trajectories(
                per_frame_objects=per_frame_obj_data,
                depth_maps=all_depth_maps,
                camera_poses=camera_poses,
                frame_width=video_meta["width"],
                frame_height=video_meta["height"],
                fps=fps,
            )
            trajectory_backend = "icp_pca"
            model_versions["trajectory_4d"] = "icp_pca"
            logger.info("4D Trajectory (ICP): extracted %d object trajectories", len(trajectories_4d))
        except Exception as e:
            logger.warning("4D Trajectory extraction failed: %s", e)

    # Enhance trajectories with CoTracker3 data if available
    if cotracker_data and trajectories_4d and trajectory_backend == "icp_pca":
        logger.info("CoTracker3 tracks available for trajectory refinement (%d points)",
                    cotracker_data.get("num_points", 0))
        # Store CoTracker data for downstream use (e.g., deformation estimation)
        model_versions["cotracker3_enhanced"] = "true"

    # ─── Stage 7: 4D Gaussian Splatting (optional, heavy) ────
    gs4d_data = None
    if config.enable_4dgs:
        try:
            logger.info("Clearing GPU memory before 4D Gaussian Splatting...")
            cleanup_all_models()
            from app.models.gaussian_splatting_4d import GaussianSplat4DPipeline
            gs4d_pipe = GaussianSplat4DPipeline(config=config)
            gs4d_output_dir = settings.output_dir / "4dgs_training"
            gs4d_data = gs4d_pipe.train(
                frame_dir=frame_dir,
                camera_poses=camera_poses,
                output_dir=gs4d_output_dir,
            )
            model_versions["gaussian_splatting_4d"] = "hexplane"
        except Exception as e:
            logger.warning("4D Gaussian Splatting failed: %s", e)
            _check_gpu_health()

    # ─── Finalize objects ───────────────────────────────────
    # Compute velocity for tracked objects
    if tracker:
        for obj in final_objects:
            vel = tracker.compute_velocity(obj.id)
            if vel:
                obj.temporal.velocity = vel

    final_objects = _compute_relations(final_objects)

    # Attach per-object meshes to StructuredObjects
    for obj in final_objects:
        if obj.id in mesh_results:
            mdata = mesh_results[obj.id]
            obj.mesh_3d = Mesh3D(
                vertices=[tuple(v) for v in mdata.get("vertices", [])],
                faces=[tuple(f) for f in mdata.get("faces", [])],
                normals=[tuple(n) for n in mdata["normals"]] if mdata.get("normals") else None,
                bounds=mdata.get("bounds"),
                point_count=mdata.get("point_count", 0),
                texture_path=mdata.get("texture_path"),
                texture_base64=mdata.get("texture_base64"),
                completion_applied=mdata.get("completion_applied", False),
                completion_method=mdata.get("completion_method"),
            )
            # Save per-object mesh export path
            if mdata.get("obj_path"):
                obj.mesh_obj_file = mdata["obj_path"]

    # Attach 4D trajectories to StructuredObjects
    for obj in final_objects:
        if obj.id in trajectories_4d:
            obj.trajectory_4d = trajectories_4d[obj.id]

    # ─── Stage 8: Dynamic Scene Graph ────────────────────────
    scene_graph = None
    if config.enable_scene_graph and trajectories_4d:
        try:
            from app.scene.scene_graph_4d import SceneGraph4DBuilder
            graph_builder = SceneGraph4DBuilder(fps=fps)
            scene_graph = graph_builder.build(final_objects, trajectories_4d)
            model_versions["scene_graph"] = "4d_temporal"
        except Exception as e:
            logger.warning("Scene graph construction failed: %s", e)

    # ─── Stage 9: Animated Export ────────────────────────────
    animated_gltf_path = None
    usd_path = None
    blend_path = None
    scene_graph_json_path = None

    if config.enable_animated_export and trajectories_4d:
        img_dir = _ensure_export_dir(file_path)
        anim_dir = img_dir / "animated"
        anim_dir.mkdir(parents=True, exist_ok=True)

        # Animated glTF
        try:
            from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
            gltf_exp = AnimatedGLTFExporter()
            animated_gltf_path = gltf_exp.export(
                objects=final_objects,
                trajectories=trajectories_4d,
                camera_poses=[CameraPose(**pose) for pose in camera_poses] if camera_poses else None,
                output_dir=anim_dir,
                filename=f"{file_path.stem}_animated.glb",
            )
            model_versions["animated_gltf"] = "v1"
        except Exception as e:
            logger.warning("Animated glTF export failed: %s", e)

        # USD scene
        try:
            from app.exporters.usd_exporter import USDExporter
            usd_exp = USDExporter()
            usd_path = usd_exp.export(
                objects=final_objects,
                trajectories=trajectories_4d,
                camera_poses=[CameraPose(**pose) for pose in camera_poses] if camera_poses else None,
                output_dir=anim_dir,
                filename=f"{file_path.stem}_scene.usda",
            )
        except Exception as e:
            logger.warning("USD export failed: %s", e)

        # Blender scene
        try:
            from app.exporters.blender_exporter import BlenderExporter
            blend_exp = BlenderExporter()
            blend_path = blend_exp.export(
                objects=final_objects,
                trajectories=trajectories_4d,
                camera_poses=[CameraPose(**pose) for pose in camera_poses] if camera_poses else None,
                output_dir=anim_dir,
                filename=file_path.stem,
            )
        except Exception as e:
            logger.warning("Blender export failed: %s", e)

    # Scene graph JSON
    if scene_graph is not None:
        img_dir = _ensure_export_dir(file_path)
        scene_graph_json_path = img_dir / f"{file_path.stem}_scene_graph.json"
        import json as _json
        with open(scene_graph_json_path, "w") as _f:
            _json.dump(scene_graph.model_dump(), _f, indent=2, default=str)

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
    # NOTE: model_versions dict is populated throughout the pipeline above
    # (including 4D stages: cotracker3, trajectory_4d, scene_graph, animated_gltf)
    # Do NOT reset it here.
    if config.enable_sam3:
        model_versions["segmentation"] = "sam3"
    if config.enable_omniparser and config.mode == "ui":
        model_versions["detection"] = "omniparser"
    elif config.enable_grounding_dino:
        model_versions["detection"] = "grounding_dino"
    if config.enable_strongsort:
        model_versions["tracking"] = "botsort"
    if config.enable_depth_pro:
        model_versions["depth"] = "depth_pro"
    if config.enable_mast3r:
        model_versions["reconstruction"] = "vggt"
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

    # Set combined scene mesh path
    if mesh_results:
        scene_obj = mesh_dir / "scene_mesh.obj"
        if scene_obj.exists():
            output.scene_mesh_path = str(scene_obj)

    # Set 4D scene data
    if scene_graph:
        output.scene_graph_4d = scene_graph
    if gs4d_data:
        # Use first object's 4DGS data
        for key, gs4d in gs4d_data.items():
            output.gaussian_splats_4d = gs4d
            break
    if animated_gltf_path:
        output.animated_gltf_path = str(animated_gltf_path)
    if usd_path:
        output.usd_path = str(usd_path)
    if blend_path:
        output.blend_path = str(blend_path)
    if scene_graph_json_path:
        output.scene_graph_json_path = str(scene_graph_json_path)

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


def _estimate_simple_camera_poses(frame_paths: list[Path], width: int, height: int) -> list[dict]:
    """Estimate simple camera poses when no 3D backend is available.

    Uses a circular orbit assumption: camera moves in a 90° arc around the scene.
    Returns list of camera pose dicts compatible with downstream 3D reconstruction.
    """
    from app.utils.camera import estimate_intrinsics, rt_matrix_to_position, rt_matrix_to_quaternion

    K = estimate_intrinsics(width, height)
    n = len(frame_paths)
    poses = []

    for i in range(n):
        t = i / max(n - 1, 1)
        angle = t * np.pi * 0.5  # 90 degree arc
        radius = 3.0

        # Camera position on arc
        cx = radius * np.sin(angle)
        cy = 0.5
        cz = radius * np.cos(angle)

        # Look-at rotation (camera looking at origin)
        target = np.array([0, 0, 0])
        camera_pos = np.array([cx, cy, cz])
        forward = target - camera_pos
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 1, 0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.column_stack([right, up, -forward])
        T = -R.T @ camera_pos

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T

        poses.append({
            "frame_idx": i,
            "intrinsics": K.tolist(),
            "extrinsics": extrinsics.tolist(),
            "position": tuple(float(x) for x in rt_matrix_to_position(R, T)),
            "rotation": tuple(float(x) for x in rt_matrix_to_quaternion(R)),
        })

    return poses


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
