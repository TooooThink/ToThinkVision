"""UV unwrapping and multi-view texture baking."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def unwrap_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "sphere",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate UV coordinates for a mesh.

    Args:
        vertices: (V, 3) vertex positions in meters.
        faces: (F, 3) triangle face indices.
        method: "sphere", "box", or "xatlas".

    Returns:
        uv_coords: (V, 2) UV coordinates in [0, 1].
        uv_face_map: (F, 3) face-to-UV-vertex index mapping.

    Note:
        "xatlas" requires `pip install xatlas`. Falls back to "box" if unavailable.
    """
    if method == "xatlas":
        try:
            import xatlas
            vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
            return uvs, indices
        except ImportError:
            logger.warning("xatlas not installed, falling back to box projection")
            return _box_projection(vertices, faces)

    if method == "box":
        return _box_projection(vertices, faces)

    return _sphere_projection(vertices, faces)


def _sphere_projection(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical UV projection.

    Projects vertices onto a unit sphere and uses (theta, phi) as UV.
    Works well for convex, roughly spherical objects.
    """
    center = vertices.mean(axis=0)
    v_rel = vertices - center  # relative to centroid

    # Spherical coordinates
    r = np.linalg.norm(v_rel, axis=1, keepdims=True)
    r = np.maximum(r, 1e-6)

    # Normalize to unit sphere
    v_norm = v_rel / r  # (V, 3)

    # UV from spherical coordinates: u=longitude, v=latitude
    # u in [0, 1], v in [0, 1]
    u = 0.5 + np.arctan2(v_norm[:, 0], v_norm[:, 2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(v_norm[:, 1], -1, 1)) / np.pi

    uv_coords = np.stack([u, v], axis=1)  # (V, 2)

    # For sphere projection, each face maps to its original vertex indices
    uv_face_map = faces  # (F, 3)

    return uv_coords, uv_face_map


def _box_projection(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Box (6-face) UV projection.

    Projects each face onto the dominant axis plane (XY, YZ, or XZ)
    based on the face normal. Results in 6 separate UV islands.
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Face normals
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)  # (F, 3)

    # Dominant axis: which component has largest absolute value?
    abs_n = np.abs(normals)
    dominant = np.argmax(abs_n, axis=1)  # 0=X, 1=Y, 2=Z per face

    # Compute UV bounds per projection plane
    uv_coords = np.zeros((len(vertices), 2), dtype=np.float32)
    uv_face_map = np.zeros_like(faces)

    # Track UV vertex count per face
    uv_vertex_count = 0

    for axis in range(3):
        mask = dominant == axis
        if not mask.any():
            continue

        # UV axes: for each dominant axis, pick the other two
        uv_axes = [a for a in range(3) if a != axis]
        ua, va = uv_axes

        face_indices = np.where(mask)[0]

        for fi in face_indices:
            for vi_local in range(3):
                vi = faces[fi, vi_local]
                uv_coords[vi] = vertices[vi, [ua, va]]
                uv_face_map[fi, vi_local] = vi

    # Normalize UVs to [0, 1]
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.maximum(uv_range, 1e-6)
    uv_coords = (uv_coords - uv_min) / uv_range

    return uv_coords, uv_face_map


def bake_texture_from_mesh_colors(
    uv_coords: np.ndarray,
    uv_face_map: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int = 512,
) -> np.ndarray:
    """Bake a texture atlas from vertex colors.

    Uses barycentric interpolation within each UV triangle to
    fill the texture image.

    Args:
        uv_coords: (V, 2) UV coordinates.
        uv_face_map: (F, 3) face-to-UV indices.
        vertex_colors: (V, 3) RGB colors [0-255].
        texture_size: texture atlas resolution (square).

    Returns:
        texture: (texture_size, texture_size, 3) RGB uint8.
    """
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    faces = uv_face_map

    if len(faces) == 0 or len(uv_coords) == 0:
        return texture

    # Get UV positions for each face
    uv0 = uv_coords[faces[:, 0]]  # (F, 2)
    uv1 = uv_coords[faces[:, 1]]
    uv2 = uv_coords[faces[:, 2]]

    # Get colors for each face
    c0 = vertex_colors[faces[:, 0]]  # (F, 3)
    c1 = vertex_colors[faces[:, 1]]
    c2 = vertex_colors[faces[:, 2]]

    # Compute face bounding boxes in UV space
    uv_min = np.minimum(np.minimum(uv0, uv1), uv2)  # (F, 2)
    uv_max = np.maximum(np.maximum(uv0, uv1), uv2)

    # Rasterize each face
    # Scale to pixel coords
    px_min = (uv_min * (texture_size - 1)).astype(int)
    px_max = (uv_max * (texture_size - 1)).astype(int)

    for fi in range(len(faces)):
        x0 = max(0, px_min[fi, 0])
        y0 = max(0, px_min[fi, 1])
        x1 = min(texture_size - 1, px_max[fi, 0])
        y1 = min(texture_size - 1, px_max[fi, 1])

        if x1 <= x0 or y1 <= y0:
            continue

        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)

        # Convert to UV
        uvs = pixels.astype(np.float32) / (texture_size - 1)

        # Barycentric coordinates
        w = _barycentric_coords(uvs, uv0[fi], uv1[fi], uv2[fi])

        # Inside triangle?
        inside = (w[:, 0] >= -1e-4) & (w[:, 1] >= -1e-4) & (w[:, 2] >= -1e-4)
        w_valid = w[inside]

        # Normalize weights
        w_sum = w_valid.sum(axis=1, keepdims=True)
        w_sum = np.maximum(w_sum, 1e-6)
        w_valid = w_valid / w_sum

        # Interpolate colors
        colors = (
            w_valid[:, 0:1] * c0[fi].astype(np.float32)
            + w_valid[:, 1:2] * c1[fi].astype(np.float32)
            + w_valid[:, 2:3] * c2[fi].astype(np.float32)
        )
        colors = np.clip(colors, 0, 255).astype(np.uint8)

        # Write to texture
        valid_pixels = pixels[inside]
        texture[valid_pixels[:, 1], valid_pixels[:, 0]] = colors

    return texture


def _barycentric_coords(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates for points in triangles (a, b, c).

    Args:
        points: (N, 2) query points.
        a, b, c: (2,) triangle vertices.

    Returns:
        (N, 3) barycentric weights.
    """
    v0 = b - a  # (2,)
    v1 = c - a
    v2 = points - a  # (N, 2)

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.sum(v2 * v0, axis=1)  # (N,)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = max(abs(denom), 1e-10) * np.sign(denom)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return np.stack([w0, w1, w2], axis=1)


def bake_texture_multiview(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    uv_face_map: np.ndarray,
    frame_paths: list[Path],
    camera_poses: list[dict] | None = None,
    frame_intrinsics: np.ndarray | None = None,
    texture_size: int = 512,
) -> np.ndarray:
    """Bake texture by projecting mesh vertices into source frames.

    For each vertex, finds visible frames and samples colors weighted by
    view angle and distance. Then barycentric-interpolates into UV atlas.

    Args:
        vertices: (V, 3) world-space vertex positions.
        faces: (F, 3) triangle indices.
        uv_coords: (V, 2) UV coordinates.
        uv_face_map: (F, 3) face-to-UV indices.
        frame_paths: paths to source frame images.
        camera_poses: list of {extrinsics: 4x4, position: tuple, ...}.
        frame_intrinsics: (3, 3) camera intrinsic matrix.
        texture_size: output texture resolution.

    Returns:
        texture: (texture_size, texture_size, 3) RGB uint8.
    """
    import cv2

    n_frames = len(frame_paths)
    n_verts = len(vertices)

    if n_frames == 0 or n_verts == 0:
        return np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

    # Load all frames
    frames_rgb = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            frames_rgb.append(None)
        else:
            frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if frame_intrinsics is None:
        if len(frames_rgb) > 0:
            h, w = frames_rgb[0].shape[:2]
            frame_intrinsics = np.array([
                [w, 0, w / 2],
                [0, h, h / 2],
                [0, 0, 1],
            ], dtype=np.float64)
        else:
            frame_intrinsics = np.eye(3)

    K = frame_intrinsics
    K_inv = np.linalg.inv(K)

    # For each vertex, find best color from visible frames
    vertex_colors = np.zeros((n_verts, 3), dtype=np.float32)
    vertex_weights = np.zeros(n_verts, dtype=np.float32)

    for fi, frame_rgb in enumerate(frames_rgb):
        if frame_rgb is None:
            continue

        frame_h, frame_w = frame_rgb.shape[:2]

        # Get camera-to-world transform
        if camera_poses and fi < len(camera_poses):
            pose = camera_poses[fi]
            if hasattr(pose, "extrinsics"):
                RT = np.array(pose.extrinsics)
            elif isinstance(pose, dict) and "extrinsics" in pose:
                RT = np.array(pose["extrinsics"])
            else:
                RT = np.eye(4)
        else:
            RT = np.eye(4)

        # World-to-camera: invert extrinsics
        try:
            RT_inv = np.linalg.inv(RT)
        except np.linalg.LinAlgError:
            continue

        R_wc = RT_inv[:3, :3]
        t_wc = RT_inv[:3, 3]

        # Transform vertices to camera space
        v_cam = (R_wc @ vertices.T + t_wc[:, None]).T  # (V, 3)

        # Project to pixel coords
        x_proj = K[0, 0] * v_cam[:, 0] / np.maximum(v_cam[:, 2], 1e-6) + K[0, 2]
        y_proj = K[1, 1] * v_cam[:, 1] / np.maximum(v_cam[:, 2], 1e-6) + K[1, 2]

        # Check visibility: in front of camera, within frame bounds, not too oblique
        in_front = v_cam[:, 2] > 0.01
        in_frame = (
            (x_proj >= 0) & (x_proj < frame_w)
            & (y_proj >= 0) & (y_proj < frame_h)
        )
        valid = in_front & in_frame

        if not valid.any():
            continue

        # Compute view angle weight (dot product of vertex normal with view direction)
        view_dir = vertices[valid] - pose.get("position", (0, 0, 0)) if camera_poses else v_cam[valid]
        view_norm = np.linalg.norm(view_dir, axis=1, keepdims=True)
        view_norm = np.maximum(view_norm, 1e-6)
        view_dir_norm = view_dir / view_norm

        # Weight by view angle (prefer frontal views)
        angle_weight = np.abs(view_dir_norm[:, 2])  # prefer facing camera

        # Weight by distance (closer is better)
        dist_weight = 1.0 / (1.0 + view_norm[:, 0] * 0.1)

        weight = angle_weight * dist_weight

        # Sample colors
        px = np.clip(x_proj[valid].astype(int), 0, frame_w - 1)
        py = np.clip(y_proj[valid].astype(int), 0, frame_h - 1)
        colors = frame_rgb[py, px].astype(np.float32)

        # Accumulate weighted colors
        idx = np.where(valid)[0]
        vertex_colors[idx] += colors * weight[:, None]
        vertex_weights[idx] += weight

    # Average accumulated colors
    valid_verts = vertex_weights > 0
    if valid_verts.any():
        vertex_colors[valid_verts] /= vertex_weights[valid_verts, None]

    # Fill in non-visible vertices with fallback (use nearest visible)
    if not valid_verts.all():
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices[valid_verts])
        _, nn = tree.query(vertices[~valid_verts], k=1)
        vertex_colors[~valid_verts] = vertex_colors[valid_verts][nn]

    vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)

    # Bake to UV atlas
    return bake_texture_from_mesh_colors(
        uv_coords, uv_face_map, vertex_colors, texture_size
    )
