"""3D point cloud completion.

For objects still incomplete after mask accumulation + 2D completion,
this module predicts the missing 3D geometry.

Three backends with graceful degradation:
1. PVD (Point Voxel Diffusion) — best quality, needs separate weights
2. ShapeFormer — shape prior-based completion
3. Heuristic (always available) — mirror symmetry via PCA + convex hull fill.
   Works well for furniture (chairs, tables, people) with bilateral symmetry.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_instance = None


class Completion3D:
    """3D point cloud completion with graceful degradation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._backend = None  # "pvd", "shapeformer", "heuristic"
        self._init_model()

    def _init_model(self):
        """Try loading 3D completion models."""
        if settings.mock_mode:
            logger.info("Completion3D: using mock mode")
            return

        # Try PVD (Point Voxel Diffusion)
        try:
            from pvd.models.pvd import PVD
            self.model = PVD(device=self.device)
            self._backend = "pvd"
            logger.info("Completion3D loaded: PVD")
            return
        except Exception as e:
            logger.info(f"PVD load failed: {e}")

        # Try ShapeFormer
        try:
            from shapeformer.models import ShapeFormerModel
            self.model = ShapeFormerModel(device=self.device)
            self._backend = "shapeformer"
            logger.info("Completion3D loaded: ShapeFormer")
            return
        except Exception as e:
            logger.info(f"ShapeFormer load failed: {e}")

        # Fallback: heuristic (always available, no model needed)
        self._backend = "heuristic"
        self.model = True  # heuristic doesn't need a model object
        logger.info("Completion3D: using heuristic symmetry completion")

    def complete(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        bbox_3d: dict | None = None,
        symmetry_axis: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete a partial point cloud.

        Args:
            points: (N, 3) partial point cloud in world coordinates
            colors: (N, 3) RGB colors
            bbox_3d: optional 3D bounding box for scale reference
            symmetry_axis: optional axis for mirror symmetry ('x', 'y', 'z')
                          if None, auto-detected via PCA

        Returns:
            (completed_points, completed_colors):
                completed_points: (M, 3) with predicted points added
                completed_colors: (M, 3)
        """
        if self._backend == "pvd":
            return self._complete_pvd(points, colors)
        elif self._backend == "shapeformer":
            return self._complete_shapeformer(points, colors)
        else:
            return self._complete_heuristic(points, colors, symmetry_axis)

    def _complete_heuristic(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        symmetry_axis: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mirror symmetry completion + convex hull fill.

        Steps:
        1. Estimate principal axes via PCA
        2. Detect symmetry plane (axis with highest point density symmetry)
        3. Mirror points across symmetry plane
        4. Fill gaps between original and mirrored with convex hull sampling
        5. Assign colors by nearest-neighbor from original points
        """
        from scipy.spatial import cKDTree

        if len(points) < 10:
            return points, colors

        # Step 1: PCA for principal axes
        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal axes (sorted by eigenvalue descending)
        axes = eigenvectors[:, ::-1]  # (3, 3)

        # Step 2: Detect symmetry axis
        if symmetry_axis is None:
            symmetry_axis = self._detect_symmetry_axis(points, centroid, axes)

        # Step 3: Mirror points across symmetry plane
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(symmetry_axis, 0)
        symmetry_normal = axes[:, axis_idx]

        # Project points onto symmetry normal
        projections = centered @ symmetry_normal
        mirrored_centered = centered - 2 * np.outer(projections, symmetry_normal)
        mirrored_points = mirrored_centered + centroid

        # Step 4: Only keep mirrored points on the "empty" side
        # Determine which side has more points
        side_original = np.sign(projections)
        positive_count = (side_original > 0).sum()
        negative_count = (side_original < 0).sum()

        # Keep mirrored points that fill the sparser side
        if positive_count >= negative_count:
            # Original is on positive side, keep mirrored on negative side
            keep_mask = projections < 0
        else:
            keep_mask = projections > 0

        # Filter mirrored points to only those that don't overlap with original
        mirrored_points_valid = mirrored_points[keep_mask]
        mirrored_colors_valid = colors[keep_mask]

        # Step 5: Remove mirrored points that are too close to original points
        tree = cKDTree(points)
        dists, _ = tree.query(mirrored_points_valid, k=1)
        min_dist = np.median(dists) * 0.3  # threshold: 30% of median distance
        non_overlap = dists > min_dist

        mirrored_points_final = mirrored_points_valid[non_overlap]
        mirrored_colors_final = mirrored_colors_valid[non_overlap]

        # Step 6: Fill gaps with convex hull sampling (optional, for very sparse regions)
        combined_points = np.vstack([points, mirrored_points_final])
        combined_colors = np.vstack([colors, mirrored_colors_final])

        # Add a few convex hull boundary points for shape regularization
        if len(combined_points) > 50:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(combined_points[:, :2])  # XY projection
                hull_pts = combined_points[hull.vertices]
                hull_clr = combined_colors[hull.vertices]

                # Sample midpoints along hull edges
                mid_points = []
                mid_colors = []
                for i in range(len(hull_pts)):
                    j = (i + 1) % len(hull_pts)
                    mid = (hull_pts[i] + hull_pts[j]) / 2
                    mid_clr = (hull_clr[i] + hull_clr[j]) / 2
                    mid_points.append(mid)
                    mid_colors.append(mid_clr.astype(np.uint8))

                if mid_points:
                    combined_points = np.vstack([combined_points, np.array(mid_points)])
                    combined_colors = np.vstack([combined_colors, np.array(mid_colors)])
            except Exception:
                pass

        return combined_points, combined_colors

    def _detect_symmetry_axis(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        axes: np.ndarray,
    ) -> str:
        """Detect the axis of approximate bilateral symmetry.

        For each principal axis, mirror points and measure overlap.
        The axis with highest overlap (most symmetric) is chosen.
        """
        from scipy.spatial import cKDTree

        centered = points - centroid
        best_axis = "x"
        best_score = 0

        for axis_name, axis_idx in [("x", 0), ("y", 1), ("z", 2)]:
            normal = axes[:, axis_idx]
            projections = centered @ normal

            # Mirror points
            mirrored = centered - 2 * np.outer(projections, normal)

            # Measure overlap: how many mirrored points are close to original?
            tree = cKDTree(centered)
            dists, _ = tree.query(mirrored, k=1)
            overlap = (dists < np.median(dists) * 0.5).sum()
            score = overlap / len(points)

            if score > best_score:
                best_score = score
                best_axis = axis_name

        return best_axis

    def _complete_pvd(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete using PVD (Point Voxel Diffusion)."""
        # Normalize points to unit sphere
        centroid = points.mean(axis=0)
        scale = np.max(np.linalg.norm(points - centroid, axis=1))
        if scale > 0:
            normalized = (points - centroid) / scale
        else:
            normalized = points - centroid

        # PVD expects (N, 3) point cloud
        completed = self.model.complete(normalized)

        # Denormalize
        completed = completed * scale + centroid

        # Color completed points by nearest original point
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, indices = tree.query(completed, k=1)
        completed_colors = colors[indices]

        return completed, completed_colors

    def _complete_shapeformer(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete using ShapeFormer."""
        # Similar to PVD: normalize, complete, denormalize
        centroid = points.mean(axis=0)
        scale = np.max(np.linalg.norm(points - centroid, axis=1))
        if scale > 0:
            normalized = (points - centroid) / scale
        else:
            normalized = points - centroid

        completed = self.model.complete(normalized)
        completed = completed * scale + centroid

        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, indices = tree.query(completed, k=1)
        completed_colors = colors[indices]

        return completed, completed_colors


def get_completion_3d(device: str = "cuda") -> Completion3D:
    """Get or create 3D completion instance."""
    global _instance
    if _instance is None:
        _instance = Completion3D(device)
    return _instance


def complete_object_3d(
    points: np.ndarray,
    colors: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function for 3D completion."""
    return get_completion_3d(device).complete(points, colors)
