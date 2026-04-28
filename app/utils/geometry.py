"""Geometry utilities: collision detection, relative position, spatial relationships."""

from __future__ import annotations


def compute_relative_position(bbox1: list[float], bbox2: list[float]) -> str:
    """Determine relative position of bbox1 with respect to bbox2.
    Returns: above, below, left_of, right_of, inside, overlap.
    bbox = [x, y, w, h]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Centers
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

    # Check overlap
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = ix * iy
    area1 = w1 * h1
    if area1 > 0 and overlap_area / area1 > 0.7:
        return "inside"
    if overlap_area > 0:
        return "overlap"

    dx = cx1 - cx2
    dy = cy1 - cy2

    if abs(dx) > abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"


def check_collision(bbox1: list[float], bbox2: list[float]) -> bool:
    """Check if two bboxes intersect. bbox = [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def compute_z_index(objects_bboxes: list[tuple[str, list[float]]]) -> dict[str, int]:
    """Compute z-index based on vertical position (lower y = further back = lower z-index).
    Returns {object_id: z_index}.
    """
    sorted_objects = sorted(objects_bboxes, key=lambda ob: ob[1][1])  # Sort by y
    return {obj_id: idx for idx, (obj_id, _) in enumerate(sorted_objects)}
