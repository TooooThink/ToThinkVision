"""3D Mesh Reconstruction — per-object mesh from multi-view depth fusion + meshing + UV + texture."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.utils.camera import estimate_intrinsics
from app.utils.texture_bake import (
    bake_texture_from_mesh_colors,
    bake_texture_multiview,
    unwrap_uv,
)

logger = logging.getLogger(__name__)


def reconstruct_object_meshes(
    frame_paths: list[Path],
    per_frame_objects: list[dict],
    depth_maps: list[np.ndarray] | None,
    frame_width: int,
    frame_height: int,
    output_dir: Path,
    camera_poses: list[dict] | None = None,
    accumulated_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Reconstruct per-object 3D meshes from depth maps across video frames.

    Pipeline:
    1. Collect per-frame object regions with depth
    2. Back-project depth to 3D points in world coordinates (using camera poses)
    3. Merge per-object point clouds
    4. Filter outliers, build mesh (Poisson / marching cubes / alpha shape)
    5. UV unwrap the mesh
    6. Bake texture from multi-view frame projection
    7. Refine mesh (Laplacian smoothing, simplify)
    8. Export per-object OBJ + texture + combined scene

    Args:
        frame_paths: paths to video frames (in order)
        per_frame_objects: [{"frame_idx": int, "objects": [{"id": str, "bbox": [...], "mask": ...}]}]
        depth_maps: per-frame depth maps in meters, or None
        frame_width, frame_height: frame dimensions
        output_dir: directory to save mesh files and textures
        camera_poses: optional list of {extrinsics: 4x4, position: tuple, ...} from MASt3R

    Returns:
        dict mapping object_id -> mesh_data dict
    """
    if depth_maps is None or len(depth_maps) == 0:
        return {}

    K = estimate_intrinsics(frame_width, frame_height)

    # ─── Step 1: Collect per-object 3D points in world coordinates ────
    object_points: dict[str, list] = {}
    K_inv = np.linalg.inv(K)

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx >= len(depth_maps):
            break

        depth_map = depth_maps[frame_idx]
        if depth_map is None:
            continue

        # Get camera-to-world transform for this frame
        RT_cw = None  # camera-to-world 4x4 matrix
        if camera_poses and frame_idx < len(camera_poses):
            pose = camera_poses[frame_idx]
            if hasattr(pose, "extrinsics"):
                RT_cw = np.array(pose.extrinsics)
            elif isinstance(pose, dict) and "extrinsics" in pose:
                RT_cw = np.array(pose["extrinsics"])

        # Find objects for this frame
        frame_obj_data = None
        for fod in per_frame_objects:
            if fod.get("frame_idx") == frame_idx:
                frame_obj_data = fod
                break
        if frame_obj_data is None:
            continue

        # Load frame image for color
        try:
            import cv2
            frame_img = cv2.imread(str(frame_path))
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB) if frame_img is not None else None
        except Exception:
            frame_img = None

        for obj_info in frame_obj_data.get("objects", []):
            obj_id = obj_info["id"]
            bbox = obj_info.get("bbox")
            mask = obj_info.get("mask")

            if bbox is None:
                continue

            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            x, y = max(0, x), max(0, y)
            x2, y2 = min(x + w, frame_width), min(y + h, frame_height)

            if x2 <= x or y2 <= y:
                continue

            obj_depth = depth_map[y:y2, x:x2]

            # Use accumulated mask (union of all frames) if available,
            # otherwise fall back to per-frame mask or full bbox
            if accumulated_masks and obj_id in accumulated_masks:
                acc_mask = accumulated_masks[obj_id][y:y2, x:x2]
                obj_mask = (acc_mask > 0) | (mask[y:y2, x:x2] > 0) if mask is not None else (acc_mask > 0)
            else:
                obj_mask = mask[y:y2, x:x2] if mask is not None else np.ones((y2 - y, x2 - x), dtype=bool)

            # Back-project to 3D in camera coordinates
            pts_3d_cam = _backproject_object_depth(obj_depth, K, x, y)
            valid_mask = obj_mask & (obj_depth > 0) & (obj_depth < 50)

            if valid_mask.sum() == 0:
                continue

            # Transform to world coordinates if camera pose available
            if RT_cw is not None:
                pts_valid_cam = pts_3d_cam[valid_mask]
                ones = np.ones((len(pts_valid_cam), 1))
                pts_hom = np.concatenate([pts_valid_cam, ones], axis=1)
                pts_3d_world = (RT_cw @ pts_hom.T).T[:, :3]
            else:
                pts_3d_world = pts_3d_cam[valid_mask]

            # Get colors
            if frame_img is not None:
                obj_img = frame_img[y:y2, x:x2]
                colors_valid = obj_img[valid_mask]
            else:
                colors_valid = np.full((len(pts_3d_world), 3), 128, dtype=np.uint8)

            if obj_id not in object_points:
                object_points[obj_id] = []

            object_points[obj_id].append({
                "points": pts_3d_world,
                "colors": colors_valid,
                "frame_idx": frame_idx,
            })

    # ─── Step 2: Build meshes with UV + texture ────
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for obj_id, point_data_list in object_points.items():
        if not point_data_list:
            continue

        # Merge all points from all frames
        all_pts = np.vstack([pd["points"] for pd in point_data_list])
        all_colors = np.vstack([pd["colors"] for pd in point_data_list])

        # Filter outliers
        all_pts, all_colors = _filter_outliers(all_pts, all_colors)

        if len(all_pts) < 50:
            continue

        # Build mesh
        mesh_data = _build_mesh(all_pts, all_colors, obj_id, output_dir)
        if mesh_data is None:
            continue

        # ─── UV Unwrap ────
        vertices = np.array(mesh_data["vertices"], dtype=np.float64)
        faces_arr = np.array(mesh_data["faces"], dtype=np.int32)
        vertex_colors = np.array(mesh_data.get("colors", all_colors[:len(vertices)]), dtype=np.uint8)

        try:
            uv_coords, uv_face_map = unwrap_uv(vertices, faces_arr, method="box")
            mesh_data["uv_coords"] = uv_coords.tolist()
            mesh_data["uv_face_map"] = uv_face_map.tolist()
        except Exception as e:
            logger.warning(f"UV unwrap failed for {obj_id}: {e}")

        # ─── Bake Texture ────
        # Try multi-view baking first, fall back to vertex-color baking
        texture_baked = None
        if uv_coords is not None and uv_face_map is not None:
            try:
                # Use multi-view texture if frames and poses available
                if len(frame_paths) > 0 and frame_idx is not None:
                    tex_K = estimate_intrinsics(frame_width, frame_height)
                    texture_baked = bake_texture_multiview(
                        vertices=vertices,
                        faces=faces_arr,
                        uv_coords=uv_coords,
                        uv_face_map=uv_face_map,
                        frame_paths=frame_paths,
                        camera_poses=camera_poses,
                        frame_intrinsics=tex_K,
                        texture_size=512,
                    )
            except Exception as e:
                logger.warning(f"Multi-view texture bake failed for {obj_id}: {e}")

            # Fallback: bake from vertex colors
            if texture_baked is None:
                try:
                    texture_baked = bake_texture_from_mesh_colors(
                        uv_coords, uv_face_map, vertex_colors, texture_size=512
                    )
                except Exception as e:
                    logger.warning(f"Vertex color texture bake failed for {obj_id}: {e}")

        # Save texture
        if texture_baked is not None:
            tex_pil = Image.fromarray(texture_baked)
            texture_path = output_dir / f"{obj_id}_texture.png"
            tex_pil.save(texture_path)
            mesh_data["texture_path"] = str(texture_path)

            # Encode as base64 for API response
            try:
                buf = BytesIO()
                tex_pil.save(buf, format="PNG")
                mesh_data["texture_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

        # ─── Export per-object OBJ with UV ────
        obj_path = _export_single_object_mesh(mesh_data, obj_id, output_dir)
        if obj_path:
            mesh_data["obj_path"] = obj_path

        # ─── Mesh Refinement ────
        mesh_data = _refine_mesh(mesh_data)

        results[obj_id] = mesh_data

    # ─── Step 3: Export combined scene mesh ────
    _export_combined_scene_mesh(results, output_dir)

    return results


def _backproject_object_depth(depth_patch: np.ndarray, K: np.ndarray,
                              offset_x: int, offset_y: int) -> np.ndarray:
    """Back-project a depth patch to 3D points in camera coordinates.

    Args:
        depth_patch: (H, W) depth values in meters
        K: (3, 3) camera intrinsic matrix
        offset_x, offset_y: top-left offset of the patch in the full image

    Returns:
        (H, W, 3) 3D point array
    """
    h, w = depth_patch.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    y_grid, x_grid = np.mgrid[offset_y:offset_y + h, offset_x:offset_x + w]

    z = depth_patch
    X = (x_grid - cx) * z / fx
    Y = (y_grid - cy) * z / fy

    return np.stack([X, Y, z], axis=-1)


def _filter_outliers(points: np.ndarray, colors: np.ndarray | None,
                     percentile: float = 98) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove outlier points using statistical filtering."""
    if len(points) < 10:
        return points, colors

    # Filter by depth range
    z = points[:, 2]
    z_low = np.percentile(z, 2)
    z_high = np.percentile(z, percentile)
    mask = (z > z_low) & (z < z_high)

    # Filter by spatial distance from median
    median = np.median(points[mask], axis=0)
    dists = np.linalg.norm(points[mask] - median, axis=1)
    dist_thresh = np.percentile(dists, 95) * 1.5
    spatial_mask = dists < dist_thresh

    combined_mask = mask.copy()
    combined_mask[mask] = spatial_mask

    points = points[combined_mask]
    if colors is not None:
        colors = colors[combined_mask]

    return points, colors


def _build_mesh(points: np.ndarray, colors: np.ndarray,
                obj_id: str, output_dir: Path) -> dict | None:
    """Convert point cloud to triangle mesh with color inheritance."""
    if len(points) < 50:
        return None

    # Try Poisson surface reconstruction
    try:
        return _poisson_reconstruction(points, colors, obj_id, output_dir)
    except Exception as e:
        logger.warning(f"Poisson reconstruction failed for {obj_id}: {e}, trying marching cubes")

    # Fallback: marching cubes
    try:
        return _marching_cubes_mesh(points, colors, obj_id, output_dir)
    except Exception as e:
        logger.warning(f"Marching cubes failed for {obj_id}: {e}")

    # Final fallback: alpha shape
    try:
        return _alpha_shape_mesh(points, colors, obj_id, output_dir)
    except Exception as e:
        logger.warning(f"Alpha shape mesh failed for {obj_id}: {e}")

    # Last resort: convex hull
    try:
        return _convex_hull_mesh(points, colors, obj_id, output_dir)
    except Exception as e:
        logger.warning(f"Convex hull mesh failed for {obj_id}: {e}")

    return None


def _poisson_reconstruction(points: np.ndarray, colors: np.ndarray,
                            obj_id: str, output_dir: Path) -> dict:
    """Poisson surface reconstruction.

    Uses Open3D if available (best quality), or alpha-shape fallback.
    """
    # Try Open3D first (best quality)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd.normals = o3d.utility.Vector3dVector(_estimate_normals_pca(points, k=20))

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )

        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Simplify if too many vertices
        if len(mesh.vertices) > 50000:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

        return _open3d_mesh_to_dict(mesh, points, colors)
    except ImportError:
        pass

    # Fallback: alpha shape
    return _alpha_shape_mesh(points, colors, obj_id, output_dir)


def _marching_cubes_mesh(points: np.ndarray, colors: np.ndarray,
                         obj_id: str, output_dir: Path) -> dict:
    """Marching cubes surface reconstruction on a voxel grid."""
    from scipy.spatial import cKDTree

    # Create voxel grid
    padding = 0.1
    mins = points.min(axis=0) - padding
    maxs = points.max(axis=0) + padding

    resolution = min(64, max(16, int(len(points) ** 0.33) * 2))
    grid_shape = (resolution, resolution, resolution)

    # Rasterize points to occupancy grid
    grid = np.zeros(grid_shape, dtype=np.float32)
    voxel_size = (maxs - mins) / np.array(grid_shape)

    for pt in points:
        vi = ((pt - mins) / voxel_size).astype(int)
        if np.all(vi >= 0) and np.all(vi < np.array(grid_shape)):
            grid[vi[0], vi[1], vi[2]] = 1.0

    # Smooth the grid for better marching cubes
    from scipy.ndimage import gaussian_filter
    grid = gaussian_filter(grid.astype(np.float64), sigma=1.0)

    # Marching cubes
    from skimage import measure
    verts, faces, normals, _ = measure.marching_cubes(grid, level=0.3)

    # Convert back to world coordinates
    verts = verts * voxel_size + mins

    # Color each vertex by nearest point
    tree = cKDTree(points)
    _, indices = tree.query(verts, k=1)
    vertex_colors = colors[indices]

    return {
        "vertices": verts.tolist(),
        "faces": faces.tolist(),
        "normals": normals.tolist(),
        "colors": vertex_colors.tolist(),
        "point_count": len(points),
        "bounds": {
            "min": verts.min(axis=0).tolist(),
            "max": verts.max(axis=0).tolist(),
        },
    }


def _alpha_shape_mesh(points: np.ndarray, colors: np.ndarray,
                      obj_id: str, output_dir: Path) -> dict:
    """Alpha shape surface reconstruction (Delaunay-based)."""
    from scipy.spatial import Delaunay

    # 2D projection for simpler cases
    xy = points[:, :2]

    # Subsample for efficiency
    if len(xy) > 5000:
        indices = np.random.choice(len(xy), 5000, replace=False)
        xy = xy[indices]
        points = points[indices]
        colors = colors[indices]

    tri = Delaunay(xy)
    faces = tri.simplices.tolist()
    normals = _estimate_normals_pca(points, k=10)

    return {
        "vertices": points.tolist(),
        "faces": faces,
        "normals": normals.tolist(),
        "colors": colors.tolist(),
        "point_count": len(points),
        "bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }


def _convex_hull_mesh(points: np.ndarray, colors: np.ndarray,
                      obj_id: str, output_dir: Path) -> dict:
    """Convex hull mesh as last resort."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)

    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, indices = tree.query(points[hull.vertices], k=1)
    hull_colors = colors[hull.vertices]

    return {
        "vertices": points[hull.vertices].tolist(),
        "faces": hull.simplices.tolist(),
        "normals": None,
        "colors": hull_colors.tolist(),
        "point_count": len(points),
        "bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }


def _estimate_normals_pca(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate surface normals using PCA on k nearest neighbors."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    n = len(points)
    normals = np.zeros((n, 3))

    batch_size = min(5000, n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, indices = tree.query(points[start:end], k=min(k + 1, n))
        neighbors = points[indices]

        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / k
        _, eigvecs = np.linalg.eigh(cov)
        normals[start:end] = eigvecs[:, :, 0]

    # Orient normals towards camera (positive z)
    facing_camera = normals[:, 2] < 0
    normals[facing_camera] *= -1

    return normals


def _open3d_mesh_to_dict(mesh, points: np.ndarray, colors: np.ndarray) -> dict:
    """Convert Open3D mesh to dict format with color inheritance."""
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

    # Color vertices by nearest original point
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, indices = tree.query(vertices, k=1)
    vertex_colors = colors[indices]

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "normals": normals.tolist() if normals is not None else None,
        "colors": vertex_colors.tolist(),
        "point_count": len(points),
        "bounds": {
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        },
    }


def _export_single_object_mesh(mesh_data: dict, obj_id: str,
                                output_dir: Path) -> str | None:
    """Export a single object mesh as an individual OBJ file with UV + MTL."""
    vertices = mesh_data.get("vertices", [])
    faces = mesh_data.get("faces", [])
    normals = mesh_data.get("normals")
    uv_coords = mesh_data.get("uv_coords")
    uv_face_map = mesh_data.get("uv_face_map")
    texture_path = mesh_data.get("texture_path")

    if not vertices:
        return None

    out_path = output_dir / f"{obj_id}.obj"
    mtl_name = f"{obj_id}.mtl"
    tex_name = Path(texture_path).name if texture_path else None

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# ToThinkVision Object Mesh: {obj_id}\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        if tex_name:
            f.write(f"mtllib {mtl_name}\n")
        f.write("\n")

        # Write MTL if texture exists
        if tex_name:
            mtl_path = output_dir / mtl_name
            with open(mtl_path, "w", encoding="utf-8") as mf:
                mf.write(f"newmtl mat_{obj_id}\n")
                mf.write("Ka 0.2 0.2 0.2\n")
                mf.write("Kd 1.0 1.0 1.0\n")
                mf.write("Ks 0.5 0.5 0.5\n")
                mf.write("Ns 96.078431\n")
                mf.write("d 1.0\n")
                mf.write("illum 2\n")
                mf.write(f"map_Kd {tex_name}\n")
            f.write(f"usemtl mat_{obj_id}\n\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # UV coordinates
        if uv_coords:
            f.write("\n")
            for uv in uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Normals
        if normals:
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Faces
        if faces:
            f.write("\n")
            has_uv = uv_coords is not None and uv_face_map is not None
            has_normal = normals is not None
            for face in faces:
                if has_uv and has_normal:
                    uv_f = uv_face_map[0] if isinstance(uv_face_map, list) and len(uv_face_map) > 0 else face
                    if isinstance(uv_face_map, list) and len(uv_face_map) == len(faces):
                        uv_f = uv_face_map[faces.index(face)] if face in faces else face
                    # Simplified: use same indices for UV and normal
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                            f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                            f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                elif has_uv:
                    f.write(f"f {face[0]+1}/{face[0]+1} "
                            f"{face[1]+1}/{face[1]+1} "
                            f"{face[2]+1}/{face[2]+1}\n")
                elif has_normal:
                    f.write(f"f {face[0]+1}//{face[0]+1} "
                            f"{face[1]+1}//{face[1]+1} "
                            f"{face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return str(out_path)


def _export_combined_scene_mesh(results: dict[str, dict], output_dir: Path):
    """Export all object meshes as a single combined OBJ file with UV + MTL."""
    out_path = output_dir / "scene_mesh.obj"
    mtl_path = output_dir / "scene_mesh.mtl"

    # Write MTL file
    with open(mtl_path, "w", encoding="utf-8") as mf:
        mf.write("# ToThinkVision Scene Materials\n\n")
        for obj_id, mesh_data in results.items():
            tex_path = mesh_data.get("texture_path")
            mat_name = f"mat_{obj_id}"
            mf.write(f"newmtl {mat_name}\n")
            mf.write("Ka 0.2 0.2 0.2\n")
            mf.write("Kd 1.0 1.0 1.0\n")
            mf.write("Ks 0.5 0.5 0.5\n")
            mf.write("Ns 96.078431\n")
            mf.write("d 1.0\n")
            mf.write("illum 2\n")
            if tex_path:
                tex_name = Path(tex_path).name
                mf.write(f"map_Kd {tex_name}\n")
            mf.write("\n")

    vertex_offset = 0
    uv_offset = 0
    normal_offset = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ToThinkVision Scene Mesh\n")
        f.write(f"# Objects: {len(results)}\n")
        f.write(f"mtllib scene_mesh.mtl\n\n")

        for obj_id, mesh_data in results.items():
            f.write(f"usemtl mat_{obj_id}\n")
            f.write(f"o {obj_id}\n\n")

            vertices = mesh_data.get("vertices", [])
            faces = mesh_data.get("faces", [])
            normals = mesh_data.get("normals")
            uv_coords = mesh_data.get("uv_coords")
            uv_face_map = mesh_data.get("uv_face_map")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # UV coordinates
            if uv_coords:
                f.write("\n")
                for uv in uv_coords:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Normals
            if normals:
                f.write("\n")
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Faces
            if faces:
                f.write("\n")
                has_uv = uv_coords is not None and uv_face_map is not None and len(uv_face_map) == len(faces)
                has_normal = normals is not None

                for fi, face in enumerate(faces):
                    if has_uv and has_normal:
                        uv_fi = fi  # Use face index for UV lookup
                        f.write(f"f {face[0]+vertex_offset+1}/{uv_fi+uv_offset+1}/{face[0]+normal_offset+1} "
                                f"{face[1]+vertex_offset+1}/{uv_fi+uv_offset+1}/{face[1]+normal_offset+1} "
                                f"{face[2]+vertex_offset+1}/{uv_fi+uv_offset+1}/{face[2]+normal_offset+1}\n")
                    elif has_uv:
                        uv_fi = fi
                        f.write(f"f {face[0]+vertex_offset+1}/{uv_fi+uv_offset+1} "
                                f"{face[1]+vertex_offset+1}/{uv_fi+uv_offset+1} "
                                f"{face[2]+vertex_offset+1}/{uv_fi+uv_offset+1}\n")
                    elif has_normal:
                        f.write(f"f {face[0]+vertex_offset+1}//{face[0]+normal_offset+1} "
                                f"{face[1]+vertex_offset+1}//{face[1]+normal_offset+1} "
                                f"{face[2]+vertex_offset+1}//{face[2]+normal_offset+1}\n")
                    else:
                        f.write(f"f {face[0]+vertex_offset+1} "
                                f"{face[1]+vertex_offset+1} "
                                f"{face[2]+vertex_offset+1}\n")

            vertex_offset += len(vertices)
            if uv_coords:
                uv_offset += len(uv_coords)
            if normals:
                normal_offset += len(normals)
            f.write("\n")

    logger.info(f"Combined scene mesh exported: {out_path}")
    return str(out_path)


def _refine_mesh(mesh_data: dict) -> dict:
    """Refine mesh: Laplacian smoothing, vertex count optimization."""
    vertices = np.array(mesh_data["vertices"], dtype=np.float64)
    faces = np.array(mesh_data["faces"], dtype=np.int32)
    normals = mesh_data.get("normals")
    colors = np.array(mesh_data.get("colors", []))

    if len(vertices) < 10 or len(faces) == 0:
        return mesh_data

    # Laplacian smoothing
    vertices = _laplacian_smooth(vertices, faces, iterations=2, weight=0.3)

    # Update normals if they exist
    if normals is not None:
        normals = _estimate_normals_pca(vertices, k=10)

    # Simplify if too many vertices (quadric decimation via edge collapse)
    if len(faces) > 30000:
        vertices, faces = _decimate_mesh(vertices, faces, target_faces=20000)

    mesh_data["vertices"] = vertices.tolist()
    mesh_data["faces"] = faces.tolist()
    if normals is not None:
        mesh_data["normals"] = normals.tolist()
    if len(colors) > 0:
        mesh_data["colors"] = colors[:len(vertices)].tolist()
    mesh_data["bounds"] = {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
    }

    return mesh_data


def _laplacian_smooth(vertices: np.ndarray, faces: np.ndarray,
                      iterations: int = 2, weight: float = 0.3) -> np.ndarray:
    """Laplacian smoothing: move each vertex toward the centroid of its neighbors."""
    v = vertices.copy()
    n = len(v)

    # Build adjacency
    adj: list[set] = [set() for _ in range(n)]
    for face in faces:
        for i in range(3):
            j = (i + 1) % 3
            adj[face[i]].add(face[j])
            adj[face[j]].add(face[i])

    for _ in range(iterations):
        delta = np.zeros_like(v)
        count = np.zeros(n)

        for i in range(n):
            if not adj[i]:
                continue
            neighbors = np.array(list(adj[i]))
            centroid = v[neighbors].mean(axis=0)
            delta[i] = centroid - v[i]
            count[i] = 1

        mask = count > 0
        v[mask] += delta[mask] * weight

    return v


def _decimate_mesh(vertices: np.ndarray, faces: np.ndarray,
                   target_faces: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Simplify mesh by removing low-detail triangles (edge collapse)."""
    if len(faces) <= target_faces:
        return vertices, faces

    # Simple approach: keep vertices that are part of most faces
    face_count_per_vertex = np.zeros(len(vertices), dtype=np.int32)
    for face in faces:
        for vi in face:
            face_count_per_vertex[vi] += 1

    # Keep top vertices by importance
    n_keep = max(target_faces * 3 // 2, len(vertices) // 2)
    top_indices = np.argsort(face_count_per_vertex)[-n_keep:]
    top_set = set(top_indices.tolist())

    # Build new face list, reindexing
    new_index = {}
    new_vertices = []
    new_faces = []

    for face in faces:
        if all(vi in top_set for vi in face):
            new_face = []
            for vi in face:
                if vi not in new_index:
                    new_index[vi] = len(new_vertices)
                    new_vertices.append(vertices[vi])
                new_face.append(new_index[vi])
            new_faces.append(new_face)

    if len(new_faces) == 0:
        return vertices, faces

    return np.array(new_vertices), np.array(new_faces, dtype=np.int32)
