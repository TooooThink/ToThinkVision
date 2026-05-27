"""Dynamic 4D Scene Graph builder.

Constructs a time-varying graph of object relationships:
- Spatial relations (above, below, left_of, right_of, in_front, behind, near)
- Interaction events (collisions, contact, proximity changes)
- Static vs dynamic classification
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.schemas import (
    InteractionEvent,
    ObjectTrajectory4D,
    SceneGraph4D,
    SceneGraphEdge,
    SceneGraphNode,
    StructuredObject,
)

logger = logging.getLogger(__name__)

# Spatial relation thresholds (meters)
NEAR_DISTANCE = 0.5  # Objects within 50cm are "near"
CONTACT_DISTANCE = 0.02  # Objects within 2cm are "in contact"


class SceneGraph4DBuilder:
    """Build dynamic 4D scene graph from objects and trajectories."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def build(
        self,
        objects: list[StructuredObject],
        trajectories: dict[str, ObjectTrajectory4D],
    ) -> SceneGraph4D:
        """Build complete dynamic scene graph.

        Args:
            objects: Detected objects with mesh and label info
            trajectories: Per-object 6DoF trajectories

        Returns:
            SceneGraph4D with nodes, edges, and interaction events
        """
        # Build nodes
        nodes = []
        static_ids = self._find_static_objects(trajectories)

        for obj in objects:
            traj = trajectories.get(obj.id)
            label = obj.label_custom or (obj.label.value if hasattr(obj.label, "value") else str(obj.label))

            node = SceneGraphNode(
                object_id=obj.id,
                label=label,
                trajectory=traj,
                mesh_3d=obj.mesh_3d,
                is_static=obj.id in static_ids,
                category=self._categorize(label),
            )
            nodes.append(node)

        # Build edges (time-varying relationships)
        edges = []
        objects_with_traj = [
            (obj, trajectories[obj.id])
            for obj in objects
            if obj.id in trajectories
        ]

        for i, (obj_a, traj_a) in enumerate(objects_with_traj):
            for j, (obj_b, traj_b) in enumerate(objects_with_traj):
                if i >= j:
                    continue
                pair_edges = self._compute_temporal_relations(obj_a, traj_a, obj_b, traj_b)
                edges.extend(pair_edges)

        # Detect interaction events
        interaction_events = self._detect_interactions(objects_with_traj)

        # Time range
        all_times = []
        for traj in trajectories.values():
            for kf in traj.keyframes:
                all_times.append(kf.timestamp)

        time_range = (min(all_times), max(all_times)) if all_times else (0.0, 0.0)

        num_static = len(static_ids)
        num_dynamic = len(trajectories) - num_static

        scene_graph = SceneGraph4D(
            nodes=nodes,
            edges=edges,
            interaction_events=interaction_events,
            time_range=time_range,
            num_static_objects=num_static,
            num_dynamic_objects=num_dynamic,
        )

        logger.info(
            "Scene graph: %d nodes (%d static, %d dynamic), %d edges, %d events",
            len(nodes), num_static, num_dynamic, len(edges), len(interaction_events),
        )

        return scene_graph

    def _find_static_objects(
        self, trajectories: dict[str, ObjectTrajectory4D]
    ) -> set[str]:
        """Identify static objects (minimal movement)."""
        static_ids = set()
        for oid, traj in trajectories.items():
            if traj.motion_type == "static":
                static_ids.add(oid)
            elif traj.total_distance < 0.02:  # < 2cm total movement
                static_ids.add(oid)
        return static_ids

    def _compute_temporal_relations(
        self,
        obj_a: StructuredObject,
        traj_a: ObjectTrajectory4D,
        obj_b: StructuredObject,
        traj_b: ObjectTrajectory4D,
    ) -> list[SceneGraphEdge]:
        """Compute time-varying spatial relations between two objects."""
        edges = []

        # Build time-indexed position maps
        pos_a = {kf.frame_idx: np.array(kf.position) for kf in traj_a.keyframes}
        pos_b = {kf.frame_idx: np.array(kf.position) for kf in traj_b.keyframes}

        # Find common frames
        common_frames = sorted(set(pos_a.keys()) & set(pos_b.keys()))
        if not common_frames:
            return edges

        # Track relation intervals
        current_relations: dict[str, list[int]] = {}  # relation → [start_frame]

        for frame in common_frames:
            pa = pos_a[frame]
            pb = pos_b[frame]
            relations = self._compute_spatial_relations(pa, pb)

            # Check for new relations
            for rel in relations:
                if rel not in current_relations:
                    current_relations[rel] = [frame]

            # Check for ended relations
            ended = [r for r in current_relations if r not in relations]
            for rel in ended:
                start_frame = current_relations[rel][0]
                end_frame = frame
                t_start = start_frame / self.fps
                t_end = end_frame / self.fps

                if t_end - t_start > 0.05:  # Only keep relations lasting > 50ms
                    dist = float(np.linalg.norm(pa - pb))
                    edges.append(SceneGraphEdge(
                        source_id=obj_a.id,
                        target_id=obj_b.id,
                        relation=rel,
                        time_range=(t_start, t_end),
                        confidence=1.0,
                        distance=round(dist, 4),
                    ))
                del current_relations[rel]

        # Close remaining open relations
        if common_frames:
            last_frame = common_frames[-1]
            for rel, start_frames in current_relations.items():
                t_start = start_frames[0] / self.fps
                t_end = last_frame / self.fps
                if t_end - t_start > 0.05:
                    edges.append(SceneGraphEdge(
                        source_id=obj_a.id,
                        target_id=obj_b.id,
                        relation=rel,
                        time_range=(t_start, t_end),
                        confidence=1.0,
                    ))

        return edges

    def _compute_spatial_relations(
        self, pos_a: np.ndarray, pos_b: np.ndarray
    ) -> list[str]:
        """Compute spatial relations from A's perspective to B.

        Uses Y-up coordinate system (common in game engines).
        """
        diff = pos_b - pos_a  # vector from A to B
        dist = float(np.linalg.norm(diff))
        relations = []

        # Vertical (Y-axis, up)
        if diff[1] > 0.1:
            relations.append("below")  # B is above A → A is below B
        elif diff[1] < -0.1:
            relations.append("above")

        # Horizontal (X-axis, right)
        if diff[0] > 0.1:
            relations.append("left_of")  # B is to the right → A is left of B
        elif diff[0] < -0.1:
            relations.append("right_of")

        # Depth (Z-axis, forward)
        if diff[2] > 0.1:
            relations.append("in_front")
        elif diff[2] < -0.1:
            relations.append("behind")

        # Proximity
        if dist < CONTACT_DISTANCE:
            relations.append("in_contact")
        elif dist < NEAR_DISTANCE:
            relations.append("near")

        return relations

    def _detect_interactions(
        self,
        objects_with_traj: list[tuple[StructuredObject, ObjectTrajectory4D]],
    ) -> list[InteractionEvent]:
        """Detect discrete interaction events between objects."""
        events = []

        for i, (obj_a, traj_a) in enumerate(objects_with_traj):
            for j, (obj_b, traj_b) in enumerate(objects_with_traj):
                if i >= j:
                    continue

                pos_a = {kf.frame_idx: np.array(kf.position) for kf in traj_a.keyframes}
                pos_b = {kf.frame_idx: np.array(kf.position) for kf in traj_b.keyframes}
                common_frames = sorted(set(pos_a.keys()) & set(pos_b.keys()))

                was_in_contact = False

                for frame in common_frames:
                    dist = float(np.linalg.norm(pos_a[frame] - pos_b[frame]))
                    in_contact = dist < CONTACT_DISTANCE

                    if in_contact and not was_in_contact:
                        events.append(InteractionEvent(
                            timestamp=frame / self.fps,
                            frame_idx=frame,
                            event_type="contact_start",
                            object_ids=[obj_a.id, obj_b.id],
                            description=f"{obj_a.id} contacts {obj_b.id}",
                            position=tuple(pos_a[frame].tolist()),
                        ))
                    elif not in_contact and was_in_contact:
                        events.append(InteractionEvent(
                            timestamp=frame / self.fps,
                            frame_idx=frame,
                            event_type="contact_end",
                            object_ids=[obj_a.id, obj_b.id],
                            description=f"{obj_a.id} separates from {obj_b.id}",
                            position=tuple(pos_a[frame].tolist()),
                        ))

                    was_in_contact = in_contact

        # Sort by time
        events.sort(key=lambda e: e.timestamp)
        return events

    @staticmethod
    def _categorize(label: str) -> str:
        """Categorize object label into semantic category."""
        label_lower = label.lower()

        if any(kw in label_lower for kw in ["floor", "ground", "terrain"]):
            return "ground"
        if any(kw in label_lower for kw in ["wall", "ceiling", "room"]):
            return "structure"
        if any(kw in label_lower for kw in ["table", "desk", "shelf", "cabinet"]):
            return "furniture"
        if any(kw in label_lower for kw in ["person", "human", "man", "woman", "people"]):
            return "person"
        if any(kw in label_lower for kw in ["car", "truck", "vehicle", "bus"]):
            return "vehicle"
        if any(kw in label_lower for kw in ["chair", "sofa", "bed", "couch"]):
            return "seating"
        if any(kw in label_lower for kw in ["door", "window", "gate"]):
            return "opening"
        if any(kw in label_lower for kw in ["tree", "plant", "bush", "flower"]):
            return "vegetation"

        return "object"
