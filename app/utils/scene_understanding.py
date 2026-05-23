"""Scene understanding — floor/wall detection, gravity alignment, scene layout analysis."""

from __future__ import annotations

import numpy as np


def detect_ground_plane(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    threshold: float = 0.05,
) -> dict | None:
    """Detect the ground plane from a point cloud using RANSAC.

    Args:
        points: (N, 3) point cloud in world coordinates.
        colors: (N, 3) optional point colors.
        threshold: inlier distance threshold in meters.

    Returns:
        dict with {normal, point_on_plane, inlier_indices, plane_eq} or None.
    """
    if len(points) < 10:
        return None

    # RANSAC plane fitting
    best_inliers = []
    best_normal = None
    best_point = None

    rng = np.random.RandomState(42)
    n_iterations = min(100, len(points) // 3)

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]

        # Compute normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len

        # Ensure normal points up (positive Y)
        if normal[1] < 0:
            normal = -normal

        # Count inliers
        distances = np.abs(np.dot(points - p0, normal))
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_point = p0

    if len(best_inliers) < len(points) * 0.1:
        return None  # Not enough inliers

    # Refine plane with all inliers
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)

    # PCA on inliers to get refined normal
    centered = inlier_points - centroid
    _, eigvecs = np.linalg.eigh(np.dot(centered.T, centered) / len(inlier_points))
    refined_normal = eigvecs[:, 0]  # Smallest eigenvector = plane normal

    if refined_normal[1] < 0:
        refined_normal = -refined_normal

    return {
        "normal": refined_normal.tolist(),
        "point": centroid.tolist(),
        "inlier_count": int(len(best_inliers)),
        "inlier_ratio": round(len(best_inliers) / len(points), 3),
        "plane_eq": [refined_normal[0], refined_normal[1], refined_normal[2],
                     -np.dot(refined_normal, centroid)],
    }


def align_to_gravity(
    points: np.ndarray,
    ground_plane: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate points so that the ground plane normal aligns with world Y-up.

    Args:
        points: (N, 3) point cloud.
        ground_plane: dict from detect_ground_plane with "normal" and "point".

    Returns:
        (aligned_points, rotation_matrix_4x4).
    """
    normal = np.array(ground_plane["normal"])
    plane_point = np.array(ground_plane["point"])

    target = np.array([0.0, 1.0, 0.0])  # Y-up

    # Compute rotation from normal to Y-up using Rodrigues' formula
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)

    if s < 1e-6:
        # Already aligned or opposite
        if c > 0:
            return points, np.eye(4)
        # Flip 180 degrees around X
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    else:
        v_normalized = v / s
        K = np.array([
            [0, -v_normalized[2], v_normalized[1]],
            [v_normalized[2], 0, -v_normalized[0]],
            [-v_normalized[1], v_normalized[0], 0],
        ])
        R = np.eye(3) + s * K + (1 - c) * K @ K

    # Build 4x4 transform: translate to plane, rotate, translate back
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ plane_point + np.array([0.0, plane_point[1], 0.0])

    # Apply transform
    ones = np.ones((len(points), 1))
    pts_hom = np.concatenate([points, ones], axis=1)
    aligned = (T @ pts_hom.T).T[:, :3]

    return aligned, T


def classify_scene_layout(
    objects: list,
    point_cloud_points: list | None = None,
) -> dict:
    """Analyze scene layout: identify floor, walls, furniture, etc.

    Args:
        objects: list of StructuredObject.
        point_cloud_points: optional (N, 3) point cloud.

    Returns:
        dict with scene layout analysis.
    """
    layout = {
        "floor_detected": False,
        "wall_count": 0,
        "furniture_objects": [],
        "npc_objects": [],
        "item_objects": [],
        "scene_type": "unknown",
    }

    for obj in objects:
        label = obj.label.value if hasattr(obj.label, "value") else str(obj.label)

        if "floor" in label:
            layout["floor_detected"] = True
        elif "wall" in label:
            layout["wall_count"] += 1
        elif "furniture" in label or "table" in label or "chair" in label:
            layout["furniture_objects"].append(obj.id)
        elif "npc" in label or "character" in label:
            layout["npc_objects"].append(obj.id)
        elif "item" in label or "prop" in label:
            layout["item_objects"].append(obj.id)

    # Classify scene type
    if layout["wall_count"] >= 3 and layout["floor_detected"]:
        layout["scene_type"] = "interior_room"
    elif layout["wall_count"] >= 2:
        layout["scene_type"] = "corridor"
    elif layout["floor_detected"]:
        layout["scene_type"] = "open_space"
    elif layout["npc_objects"]:
        layout["scene_type"] = "character_scene"
    else:
        layout["scene_type"] = "object_scene"

    # Estimate room dimensions if floor and walls detected
    if point_cloud_points and layout["floor_detected"]:
        pts = np.array(point_cloud_points)
        layout["room_bounds"] = {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "size": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
        }

    return layout
