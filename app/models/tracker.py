"""Cross-frame object tracker using IoU and feature matching."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class ObjectTracker:
    """Simple IoU-based object tracker for cross-frame association."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks: dict[str, dict] = {}  # track_id -> {bbox, label, history}
        self.next_id = 0

    @staticmethod
    def iou(bbox1: list[float], bbox2: list[float]) -> float:
        """Compute IoU between two bboxes [x, y, w, h]."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        if union == 0:
            return 0.0
        return inter / union

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        """Associate detections with existing tracks. Returns updated track list."""
        if not self.tracks:
            # First frame: create tracks for all detections
            for det in detections:
                track_id = f"obj_{self.next_id:04d}"
                self.next_id += 1
                bbox = det["bbox"]
                self.tracks[track_id] = {
                    "id": track_id,
                    "bbox": bbox,
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.0),
                    "appear_frame": frame_idx,
                    "disappear_frame": -1,
                    "history": [{"x": bbox[0] + bbox[2] / 2, "y": bbox[1] + bbox[3] / 2, "t": frame_idx}],
                }
            return list(self.tracks.values())

        # Compute IoU cost matrix
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets))

        for i, tid in enumerate(track_ids):
            track_bbox = self.tracks[tid]["bbox"]
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self.iou(track_bbox, det["bbox"])

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                tid = track_ids[r]
                det = detections[c]
                self.tracks[tid]["bbox"] = det["bbox"]
                self.tracks[tid]["confidence"] = det.get("confidence", self.tracks[tid]["confidence"])
                self.tracks[tid]["disappear_frame"] = -1
                self.tracks[tid]["history"].append({
                    "x": det["bbox"][0] + det["bbox"][2] / 2,
                    "y": det["bbox"][1] + det["bbox"][3] / 2,
                    "t": frame_idx,
                })
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched tracks: mark as disappeared
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                if self.tracks[tid]["disappear_frame"] == -1:
                    self.tracks[tid]["disappear_frame"] = frame_idx - 1

        # Unmatched detections: create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track_id = f"obj_{self.next_id:04d}"
                self.next_id += 1
                bbox = det["bbox"]
                self.tracks[track_id] = {
                    "id": track_id,
                    "bbox": bbox,
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.0),
                    "appear_frame": frame_idx,
                    "disappear_frame": -1,
                    "history": [{"x": bbox[0] + bbox[2] / 2, "y": bbox[1] + bbox[3] / 2, "t": frame_idx}],
                }

        return list(self.tracks.values())

    def get_all_tracks(self) -> list[dict]:
        """Return all tracked objects with their histories."""
        return list(self.tracks.values())

    def compute_velocity(self, track_id: str) -> dict[str, float] | None:
        """Compute average velocity for a track."""
        track = self.tracks.get(track_id)
        if not track or len(track["history"]) < 2:
            return None
        hist = track["history"]
        dt = hist[-1]["t"] - hist[0]["t"]
        if dt == 0:
            return {"vx": 0.0, "vy": 0.0}
        dx = hist[-1]["x"] - hist[0]["x"]
        dy = hist[-1]["y"] - hist[0]["y"]
        return {"vx": dx / dt, "vy": dy / dt}

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
