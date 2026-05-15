"""Point cloud utilities: back-projection, filtering, normals, PLY export."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def backproject_depth(depth_map: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to 3D points using camera intrinsics.

    Args:
        depth_map: (H, W) depth in meters
        K: (3, 3) camera intrinsic matrix
        mask: (H, W) optional binary mask to filter points

    Returns:
        points: (N, 3) xyz in camera coordinates
        colors: (N, 3) rgb (from image if provided, else zeros)
    """
    h, w = depth_map.shape
    y, x = np.mgrid[0:h, 0:w]

    # Unproject: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth_map
    valid = (z > 0) & (z < 100)  # Filter invalid depth
    if mask is not None:
        valid = valid & (mask > 0)

    z_valid = z[valid]
    x_valid = (x[valid] - cx) * z_valid / fx
    y_valid = (y[valid] - cy) * z_valid / fy

    points = np.stack([x_valid, y_valid, z_valid], axis=-1)  # (N, 3)
    return points


def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Estimate surface normals using PCA on k nearest neighbors.

    Args:
        points: (N, 3) point cloud
        k: number of neighbors

    Returns:
        normals: (N, 3) per-point normal vectors
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    normals = np.zeros_like(points)

    # Process in batches to avoid memory issues
    batch_size = min(5000, len(points))
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        _, indices = tree.query(points[start:end], k=min(k + 1, len(points)))
        neighbors = points[indices]

        # PCA: normal = eigenvector with smallest eigenvalue
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / k
        _, eigvecs = np.linalg.eigh(cov)
        normals[start:end] = eigvecs[:, :, 0]

    # Orient normals towards camera (positive z)
    facing_camera = normals[:, 2] < 0
    normals[facing_camera] *= -1

    return normals


def filter_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    min_z: float = 0.01,
    max_z: float = 50.0,
    voxel_size: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filter and optionally downsample a point cloud.

    Args:
        points: (N, 3)
        colors: (N, 3) RGB
        min_z, max_z: depth range filter
        voxel_size: if > 0, voxel grid downsampling

    Returns:
        filtered_points, filtered_colors
    """
    valid = (points[:, 2] > min_z) & (points[:, 2] < max_z)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    if voxel_size > 0:
        points, colors = _voxel_downsample(points, colors, voxel_size)

    return points, colors


def _voxel_downsample(
    points: np.ndarray, colors: np.ndarray | None, voxel_size: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Simple voxel grid downsampling."""
    voxels = (points / voxel_size).astype(np.int32)
    _, indices = np.unique(voxels.view([("", voxels.dtype)] * 3), return_index=True)

    points = points[indices]
    if colors is not None:
        colors = colors[indices]
    return points, colors


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None):
    """Save point cloud as PLY file.

    Args:
        path: output file path
        points: (N, 3) xyz
        colors: (N, 3) RGB 0-255
        normals: (N, 3) normals
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)
    has_color = colors is not None and len(colors) == n
    has_normal = normals is not None and len(normals) == n

    elements = "element vertex " + str(n)
    props = ["property float x", "property float y", "property float z"]
    if has_color:
        props += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_normal:
        props += ["property float nx", "property float ny", "property float nz"]

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(elements + "\n")
        f.write("\n".join(props) + "\n")
        f.write("end_header\n")
        for i in range(n):
            parts = [f"{points[i,0]:.6f}", f"{points[i,1]:.6f}", f"{points[i,2]:.6f}"]
            if has_color:
                parts += [str(int(c)) for c in colors[i]]
            if has_normal:
                parts += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
            f.write(" ".join(parts) + "\n")


def save_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray | None = None):
    """Save point cloud as binary PLY (more compact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)
    has_color = colors is not None and len(colors) == n

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")

        for i in range(n):
            f.write(struct.pack("<fff", points[i, 0], points[i, 1], points[i, 2]))
            if has_color:
                f.write(struct.pack("<BBB", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
