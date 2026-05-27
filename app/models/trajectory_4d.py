"""4D Trajectory Extraction: per-object 6DoF motion from video.

Extracts position + rotation + scale trajectories for each tracked object
using depth maps, segmentation masks, and camera poses.

Algorithm:
1. Per-frame: mask ∩ depth → backproject to 3D → transform to world coords
2. First frame: record initial pose (PCA orientation + centroid)
3. Subsequent frames: ICP alignment → rigid transform (R, t) + residual
4. B-spline smoothing for noise reduction
5. Motion classification (static / rigid / deformable / disappearing)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation

from app.schemas import (
    CameraPose,
    ObjectTrajectory4D,
    PipelineConfig,
    TrajectoryKeyframe,
)
from app.utils.camera import estimate_intrinsics, invert_extrinsics, transform_points
from app.utils.pointcloud import backproject_depth

logger = logging.getLogger(__name__)

# Try Open3D for fast ICP, fallback to scipy-based implementation
_HAS_OPEN3D = False
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    pass


class TrajectoryExtractor4D:
    """Extract per-object 6DoF trajectories from video data."""

    def __init__(self, config: PipelineConfig):
        self.icp_threshold = config.icp_distance_threshold
        self.deform_threshold = config.deformation_threshold
        self.smoothing = config.trajectory_smoothing

        # Relax thresholds for AI-generated video
        if config.is_world_model_video:
            self.icp_threshold *= 2.0
            self.deform_threshold *= 1.5
            self.smoothing = min(1.0, self.smoothing + 0.2)
            logger.info("World model video mode: relaxed thresholds, smoothing=%.2f", self.smoothing)

    def extract_trajectories(
        self,
        per_frame_objects: list[dict[str, Any]],
        depth_maps: list[np.ndarray],
        camera_poses: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
    ) -> dict[str, ObjectTrajectory4D]:
        """Extract 6DoF trajectories for all tracked objects.

        Args:
            per_frame_objects: [{frame_idx, objects: [{id, bbox, mask}]}]
            depth_maps: per-frame depth maps (H, W) in meters
            camera_poses: per-frame camera pose dicts (with extrinsics/intrinsics)
            frame_width: video frame width
            frame_height: video frame height
            fps: video framerate

        Returns:
            dict mapping object_id → ObjectTrajectory4D
        """
        K = estimate_intrinsics(frame_width, frame_height)

        # Group observations by object ID
        object_observations: dict[str, list[dict]] = {}
        for frame_entry in per_frame_objects:
            fidx = frame_entry["frame_idx"]
            if fidx >= len(depth_maps):
                continue
            depth = depth_maps[fidx]

            # Get camera-to-world transform
            cam_to_world = self._get_camera_to_world(camera_poses, fidx, K)

            for obj_info in frame_entry["objects"]:
                oid = obj_info["id"]
                mask = obj_info.get("mask")

                # Extract 3D point cloud for this object in this frame
                points_world = self._extract_object_pointcloud(
                    mask=mask,
                    depth_map=depth,
                    K=K,
                    cam_to_world=cam_to_world,
                    bbox=obj_info.get("bbox"),
                )

                if points_world is None or len(points_world) < 5:
                    continue

                if oid not in object_observations:
                    object_observations[oid] = []

                object_observations[oid].append({
                    "frame_idx": fidx,
                    "timestamp": fidx / fps,
                    "points": points_world,
                })

        # Extract trajectory for each object
        trajectories = {}
        for oid, observations in object_observations.items():
            if len(observations) < 2:
                continue
            try:
                traj = self._build_trajectory(oid, observations, fps)
                if traj is not None:
                    trajectories[oid] = traj
                    logger.info(
                        "Object %s: %d keyframes, motion=%s, distance=%.3fm",
                        oid, len(traj.keyframes), traj.motion_type, traj.total_distance,
                    )
            except Exception as e:
                logger.warning("Trajectory extraction failed for %s: %s", oid, e)

        return trajectories

    def _get_camera_to_world(
        self, camera_poses: list[dict], frame_idx: int, K: np.ndarray
    ) -> np.ndarray:
        """Get camera-to-world 4x4 transform for a frame."""
        if frame_idx < len(camera_poses):
            pose = camera_poses[frame_idx]
            ext = pose.get("extrinsics", np.eye(4).tolist())
            if isinstance(ext, list):
                ext = np.array(ext)
            if ext.shape == (4, 4):
                return invert_extrinsics(ext)

        # Fallback: identity (camera at origin)
        return np.eye(4)

    def _extract_object_pointcloud(
        self,
        mask: np.ndarray | None,
        depth_map: np.ndarray,
        K: np.ndarray,
        cam_to_world: np.ndarray,
        bbox: list | None = None,
    ) -> np.ndarray | None:
        """Back-project masked depth pixels to 3D world coordinates."""
        if mask is None and bbox is None:
            return None

        # Create effective mask
        h, w = depth_map.shape
        if mask is not None:
            if mask.shape != (h, w):
                # Resize mask to depth map size
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((w, h), PILImage.NEAREST)
                mask = np.array(mask_img).astype(np.float32) / 255.0
            effective_mask = mask > 0.5
        else:
            # Use bbox as rectangular mask
            effective_mask = np.zeros((h, w), dtype=bool)
            x, y, bw, bh = [int(v) for v in bbox]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            effective_mask[y1:y2, x1:x2] = True

        # Back-project to camera coordinates
        points_cam = backproject_depth(depth_map, K, mask=effective_mask.astype(np.uint8))
        if len(points_cam) < 5:
            return None

        # Transform to world coordinates
        points_world = transform_points(points_cam, cam_to_world)
        return points_world

    def _build_trajectory(
        self,
        object_id: str,
        observations: list[dict],
        fps: float,
    ) -> ObjectTrajectory4D | None:
        """Build complete 6DoF trajectory from per-frame observations."""
        # Sort by frame index
        observations.sort(key=lambda x: x["frame_idx"])

        # Reference: first observation's point cloud
        ref_points = observations[0]["points"]
        ref_centroid = ref_points.mean(axis=0)
        ref_orientation = self._estimate_orientation(ref_points)

        keyframes = []

        for i, obs in enumerate(observations):
            points = obs["points"]
            centroid = points.mean(axis=0)

            if i == 0:
                # First frame: identity transform
                position = tuple(centroid.tolist())
                rotation = ref_orientation
                deform_score = 0.0
                is_rigid = True
                scale = (1.0, 1.0, 1.0)
            else:
                # ICP alignment: align current to reference
                T, residual = self._estimate_rigid_transform(points, ref_points)

                # Extract position and rotation from transform
                R = T[:3, :3]
                t = T[:3, 3]

                # Position = reference centroid transformed
                position = tuple((centroid).tolist())

                # Rotation from ICP
                rot = Rotation.from_matrix(R)
                q = rot.as_quat()  # (x, y, z, w)
                rotation = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))  # (w, x, y, z)

                # Deformation score from ICP residual
                deform_score = min(1.0, residual / max(self.deform_threshold, 1e-6))
                is_rigid = residual < self.deform_threshold

                # Scale estimation from point cloud extent
                scale = self._estimate_scale(points, ref_points)

            kf = TrajectoryKeyframe(
                timestamp=obs["timestamp"],
                frame_idx=obs["frame_idx"],
                position=position,
                rotation=rotation,
                scale=scale,
                is_rigid=is_rigid,
                deformation_score=deform_score,
            )
            keyframes.append(kf)

        if not keyframes:
            return None

        # Apply B-spline smoothing
        keyframes = self._smooth_trajectory(keyframes)

        # Compute velocities
        keyframes = self._compute_velocities(keyframes)

        # Compute statistics
        total_dist, max_speed, avg_speed = self._compute_statistics(keyframes)
        duration = keyframes[-1].timestamp - keyframes[0].timestamp if len(keyframes) > 1 else 0.0
        motion_type = self._classify_motion(keyframes)

        return ObjectTrajectory4D(
            object_id=object_id,
            keyframes=keyframes,
            motion_type=motion_type,
            total_distance=total_dist,
            max_speed=max_speed,
            avg_speed=avg_speed,
            duration=duration,
        )

    def _estimate_rigid_transform(
        self, src: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Estimate rigid transform from src to tgt using ICP.

        Returns:
            T: (4, 4) transformation matrix
            residual: mean ICP error (meters)
        """
        if _HAS_OPEN3D:
            return self._icp_open3d(src, tgt)
        return self._icp_scipy(src, tgt)

    def _icp_open3d(
        self, src: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """ICP using Open3D."""
        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(src)
        tgt_pc = o3d.geometry.PointCloud()
        tgt_pc.points = o3d.utility.Vector3dVector(tgt)

        # Initial alignment: translate centroids
        src_center = src.mean(axis=0)
        tgt_center = tgt.mean(axis=0)
        init_T = np.eye(4)
        init_T[:3, 3] = tgt_center - src_center

        result = o3d.pipelines.registration.registration_icp(
            src_pc, tgt_pc,
            max_correspondence_distance=self.icp_threshold * 5,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )

        residual = result.inlier_rmse if result.inlier_rmse > 0 else self.icp_threshold
        return np.array(result.transformation), float(residual)

    def _icp_scipy(
        self, src: np.ndarray, tgt: np.ndarray, max_iter: int = 50
    ) -> tuple[np.ndarray, float]:
        """ICP using scipy (no Open3D dependency).

        Standard point-to-point ICP with iterative closest point matching.
        Uses the closed-form SVD solution for rigid alignment (Arun et al., 1987).
        """
        from scipy.spatial import cKDTree

        current = src.copy()
        T_total = np.eye(4)

        prev_error = float("inf")

        for iteration in range(max_iter):
            # Find nearest neighbors in target
            tree = cKDTree(tgt)
            distances, indices = tree.query(current, k=1)

            # Adaptive threshold: use percentile to reject outliers
            # Start generous (90th percentile) and tighten over iterations
            pct = max(50, 90 - iteration * 2)
            threshold = np.percentile(distances, pct)
            threshold = max(threshold, self.icp_threshold * 10)

            inlier_mask = distances < threshold
            n_inliers = np.sum(inlier_mask)
            if n_inliers < 3:
                inlier_mask = np.ones(len(current), dtype=bool)

            src_pts = current[inlier_mask]
            tgt_pts = tgt[indices[inlier_mask]]

            # Closed-form rigid alignment using SVD
            src_center = src_pts.mean(axis=0)
            tgt_center = tgt_pts.mean(axis=0)

            src_centered = src_pts - src_center
            tgt_centered = tgt_pts - tgt_center

            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)

            # Ensure proper rotation (det = +1)
            d = np.linalg.det(Vt.T @ U.T)
            sign_matrix = np.diag([1, 1, np.sign(d)])
            R = Vt.T @ sign_matrix @ U.T

            # Optimal translation
            t = tgt_center - R @ src_center

            # Build 4x4 transform for this iteration
            T_iter = np.eye(4)
            T_iter[:3, :3] = R
            T_iter[:3, 3] = t

            # Apply transform to current points
            current = (R @ current.T).T + t

            # Accumulate total transform
            T_total = T_iter @ T_total

            # Check convergence
            mean_error = float(np.mean(distances[inlier_mask]))
            if abs(prev_error - mean_error) < 1e-8:
                break
            prev_error = mean_error

        # Final residual
        tree = cKDTree(tgt)
        distances, _ = tree.query(current, k=1)
        residual = float(np.mean(distances))

        return T_total, residual

    def _estimate_orientation(self, points: np.ndarray) -> tuple[float, float, float, float]:
        """Estimate object orientation using PCA.

        Returns quaternion (w, x, y, z).
        """
        if len(points) < 3:
            return (1.0, 0.0, 0.0, 0.0)

        centered = points - points.mean(axis=0)
        cov = centered.T @ centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Build rotation matrix from principal axes
        # Column 0 = principal axis (longest), Column 1 = second, Column 2 = normal
        R = eigenvectors
        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # (x, y, z, w)
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    def _estimate_scale(
        self, points: np.ndarray, ref_points: np.ndarray
    ) -> tuple[float, float, float]:
        """Estimate scale change relative to reference."""
        if len(points) < 3 or len(ref_points) < 3:
            return (1.0, 1.0, 1.0)

        # PCA-based scale: extent along principal axes
        def _extent(pts):
            centered = pts - pts.mean(axis=0)
            cov = centered.T @ centered / len(pts)
            eigenvalues = np.linalg.eigvalsh(cov)
            return np.sqrt(np.maximum(eigenvalues, 1e-10))

        ref_ext = _extent(ref_points)
        cur_ext = _extent(points)

        scales = cur_ext / np.maximum(ref_ext, 1e-10)
        # Clamp to reasonable range
        scales = np.clip(scales, 0.1, 10.0)
        return (float(scales[0]), float(scales[1]), float(scales[2]))

    def _smooth_trajectory(
        self, keyframes: list[TrajectoryKeyframe]
    ) -> list[TrajectoryKeyframe]:
        """Apply B-spline smoothing to position trajectory."""
        if len(keyframes) < 4 or self.smoothing <= 0:
            return keyframes

        timestamps = np.array([kf.timestamp for kf in keyframes])
        positions = np.array([kf.position for kf in keyframes])

        # Smoothness parameter (higher = smoother)
        s = self.smoothing * len(keyframes) * 0.01

        try:
            smoothed_positions = np.zeros_like(positions)
            for dim in range(3):
                spline = UnivariateSpline(timestamps, positions[:, dim], s=s, k=3)
                smoothed_positions[:, dim] = spline(timestamps)

            # Update keyframes with smoothed positions
            result = []
            for i, kf in enumerate(keyframes):
                result.append(TrajectoryKeyframe(
                    timestamp=kf.timestamp,
                    frame_idx=kf.frame_idx,
                    position=tuple(smoothed_positions[i].tolist()),
                    rotation=kf.rotation,
                    scale=kf.scale,
                    is_rigid=kf.is_rigid,
                    deformation_score=kf.deformation_score,
                    confidence=kf.confidence,
                ))
            return result
        except Exception as e:
            logger.warning("B-spline smoothing failed: %s", e)
            return keyframes

    def _compute_velocities(
        self, keyframes: list[TrajectoryKeyframe]
    ) -> list[TrajectoryKeyframe]:
        """Compute linear and angular velocities between keyframes."""
        if len(keyframes) < 2:
            return keyframes

        result = [keyframes[0]]
        for i in range(1, len(keyframes)):
            prev = keyframes[i - 1]
            curr = keyframes[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                dt = 1e-6

            # Linear velocity
            dp = np.array(curr.position) - np.array(prev.position)
            velocity = tuple((dp / dt).tolist())

            # Angular velocity from quaternion difference
            q_prev = Rotation.from_quat([prev.rotation[1], prev.rotation[2], prev.rotation[3], prev.rotation[0]])
            q_curr = Rotation.from_quat([curr.rotation[1], curr.rotation[2], curr.rotation[3], curr.rotation[0]])
            q_diff = q_prev.inv() * q_curr
            rotvec = q_diff.as_rotvec()
            angular_velocity = tuple((rotvec / dt).tolist())

            result.append(TrajectoryKeyframe(
                timestamp=curr.timestamp,
                frame_idx=curr.frame_idx,
                position=curr.position,
                rotation=curr.rotation,
                scale=curr.scale,
                velocity=velocity,
                angular_velocity=angular_velocity,
                is_rigid=curr.is_rigid,
                deformation_score=curr.deformation_score,
                confidence=curr.confidence,
            ))

        # Set first keyframe velocity to match second
        if len(result) > 1:
            result[0] = TrajectoryKeyframe(
                timestamp=result[0].timestamp,
                frame_idx=result[0].frame_idx,
                position=result[0].position,
                rotation=result[0].rotation,
                scale=result[0].scale,
                velocity=result[1].velocity,
                angular_velocity=result[1].angular_velocity,
                is_rigid=result[0].is_rigid,
                deformation_score=result[0].deformation_score,
                confidence=result[0].confidence,
            )

        return result

    def _compute_statistics(
        self, keyframes: list[TrajectoryKeyframe]
    ) -> tuple[float, float, float]:
        """Compute total distance, max speed, average speed."""
        if len(keyframes) < 2:
            return 0.0, 0.0, 0.0

        total_dist = 0.0
        speeds = []

        for i in range(1, len(keyframes)):
            dp = np.array(keyframes[i].position) - np.array(keyframes[i - 1].position)
            dist = float(np.linalg.norm(dp))
            total_dist += dist

            dt = keyframes[i].timestamp - keyframes[i - 1].timestamp
            if dt > 0:
                speeds.append(dist / dt)

        max_speed = max(speeds) if speeds else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        return total_dist, max_speed, avg_speed

    def _classify_motion(self, keyframes: list[TrajectoryKeyframe]) -> str:
        """Classify motion type based on trajectory statistics."""
        if len(keyframes) < 2:
            return "static"

        total_dist, max_speed, avg_speed = self._compute_statistics(keyframes)

        # Static: minimal movement
        if total_dist < 0.01:  # < 1cm total movement
            return "static"

        # Check deformation scores
        avg_deform = np.mean([kf.deformation_score for kf in keyframes])
        if avg_deform > self.deform_threshold:
            return "deformable"

        # Rigid: consistent shape, significant movement
        if max_speed > 0.001:  # > 1mm/s
            return "rigid"

        return "static"
