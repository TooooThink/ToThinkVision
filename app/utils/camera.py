"""Camera utilities: intrinsics estimation, pose conversion, coordinate transforms."""

from __future__ import annotations

import numpy as np


def estimate_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Estimate camera intrinsics from image size and FOV.

    Args:
        width: image width in pixels
        height: image height in pixels
        fov_deg: horizontal field of view in degrees

    Returns:
        K: (3, 3) intrinsic matrix
    """
    focal = width / (2 * np.tan(np.deg2rad(fov_deg / 2)))
    cx, cy = width / 2, height / 2
    return np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def rt_matrix_to_position(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Extract camera center from R, t.

    Camera center C = -R^T @ t

    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        C: (3,) camera center in world coordinates
    """
    return -R.T @ t


def rt_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        w = s / 4
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = s / 4
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = s / 4
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = s / 4
    return (float(w), float(x), float(y), float(z))


def build_extrinsics(position: np.ndarray, look_at: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build 4x4 extrinsics matrix from camera position and look-at point.

    Args:
        position: (3,) camera position
        look_at: (3,) point camera is looking at
        up: (3,) up vector, default (0,1,0)

    Returns:
        RT: (4, 4) extrinsics matrix (world → camera)
    """
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float64)

    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.column_stack([right, up, -forward]).T  # (3, 3)
    t = -R @ position

    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = t
    return RT


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform 3D points by a 4x4 matrix.

    Args:
        points: (N, 3)
        T: (4, 4) transformation matrix

    Returns:
        transformed: (N, 3)
    """
    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])  # (N, 4)
    return (T @ homogeneous.T).T[:, :3]  # (N, 3)


def invert_extrinsics(RT: np.ndarray) -> np.ndarray:
    """Invert a 4x4 extrinsics matrix (world→camera to camera→world)."""
    T_inv = np.eye(4)
    R = RT[:3, :3]
    t = RT[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
