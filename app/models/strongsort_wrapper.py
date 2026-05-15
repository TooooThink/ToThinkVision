"""StrongSORT — robust multi-object tracking with ReID + Kalman + GMC."""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_tracker = None


class StrongSORTTracker:
    """StrongSORT tracker wrapper for multi-object tracking.

    Uses ReID features + Kalman filter + Global Motion Compensation
    for robust tracking with minimal ID switches.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.tracker = None
        self.next_track_id = 0
        self.tracks: dict[str, dict] = {}
        self._init_model()

    def _init_model(self):
        """Initialize StrongSORT tracker."""
        if settings.mock_mode:
            logger.info("StrongSORT: using mock mode")
            return

        try:
            from strongsort import StrongSORT
            from strongsort.utils.reid_model import extract_reid_model

            self.tracker = StrongSORT(
                model_weights=str(self._get_reid_weights()),
                device=settings.device,
                fp16=False,
            )
            self.tracker.gmc = self._init_gmc()
            logger.info("StrongSORT tracker initialized")
        except ImportError:
            logger.warning("strongsort package not installed, using IoU fallback")
        except Exception as e:
            logger.warning(f"StrongSORT init failed: {e}")

    def _get_reid_weights(self):
        from pathlib import Path
        return Path(settings.model_cache_dir) / "osnet_x1_0_msmt17.pt"

    def _init_gmc(self):
        """Initialize Global Motion Compensation."""
        try:
            from strongsort.utils.gmc import GMC
            return GMC(method="sparseOptFlow")
        except ImportError:
            return None

    def update(self, detections: list[dict], frame: np.ndarray | None = None,
               frame_idx: int = 0) -> list[dict]:
        """Update tracker with new detections.

        Args:
            detections: list of {bbox, label, confidence, feature?}
            frame: current frame image (for GMC)
            frame_idx: current frame number

        Returns:
            list of tracked objects with persistent IDs
        """
        if self.tracker is not None:
            return self._update_strongsort(detections, frame, frame_idx)
        else:
            return self._update_iou(detections, frame_idx)

    def _update_strongsort(self, detections: list[dict], frame: np.ndarray | None,
                           frame_idx: int) -> list[dict]:
        """Update using StrongSORT."""
        try:
            # Convert detections to format: [x1, y1, x2, y2, conf, feature?]
            bboxes = []
            for det in detections:
                x, y, w, h = det["bbox"]
                bboxes.append([x, y, x + w, y + h, det.get("confidence", 0.5)])
            bboxes = np.array(bboxes)

            features = None
            if all("feature" in d for d in detections):
                features = np.array([d["feature"] for d in detections])

            # Run tracker
            tracks = self.tracker.update(bboxes, frame, features)

            results = []
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
                track_id_str = f"obj_{int(track_id):04d}"
                if track_id_str not in self.tracks:
                    self.tracks[track_id_str] = {"history": []}

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                self.tracks[track_id_str]["history"].append({
                    "x": float(cx), "y": float(cy), "t": frame_idx
                })

                results.append({
                    "id": track_id_str,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "confidence": float(conf),
                    "history": self.tracks[track_id_str]["history"],
                })
            return results
        except Exception as e:
            logger.error(f"StrongSORT update failed: {e}, falling back to IoU")
            return self._update_iou(detections, frame_idx)

    def _update_iou(self, detections: list[dict], frame_idx: int) -> list[dict]:
        """Simple IoU-based fallback tracking."""
        if not self.tracks:
            for det in detections:
                track_id = f"obj_{self.next_track_id:04d}"
                self.next_track_id += 1
                bbox = det["bbox"]
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                self.tracks[track_id] = {
                    "bbox": bbox,
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.5),
                    "appear_frame": frame_idx,
                    "disappear_frame": -1,
                    "history": [{"x": cx, "y": cy, "t": frame_idx}],
                }
            return self._format_tracks()

        # IoU matching
        track_ids = list(self.tracks.keys())
        matched_dets = set()
        for i, det in enumerate(detections):
            best_tid, best_iou = None, 0
            for tid in track_ids:
                iou = self._calc_iou(det["bbox"], self.tracks[tid]["bbox"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_tid = tid

            if best_tid:
                self.tracks[best_tid]["bbox"] = det["bbox"]
                self.tracks[best_tid]["disappear_frame"] = -1
                cx = det["bbox"][0] + det["bbox"][2] / 2
                cy = det["bbox"][1] + det["bbox"][3] / 2
                self.tracks[best_tid]["history"].append({"x": cx, "y": cy, "t": frame_idx})
                matched_dets.add(i)
            else:
                track_id = f"obj_{self.next_track_id:04d}"
                self.next_track_id += 1
                bbox = det["bbox"]
                self.tracks[track_id] = {
                    "bbox": bbox,
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.5),
                    "appear_frame": frame_idx,
                    "disappear_frame": -1,
                    "history": [{"x": bbox[0] + bbox[2] / 2, "y": bbox[1] + bbox[3] / 2, "t": frame_idx}],
                }

        # Mark unmatched tracks as disappeared
        for tid in track_ids:
            bbox = self.tracks[tid]["bbox"]
            if not any(self._calc_iou(d["bbox"], bbox) > 0.3 for i, d in enumerate(detections) if i not in matched_dets):
                if self.tracks[tid]["disappear_frame"] == -1:
                    self.tracks[tid]["disappear_frame"] = frame_idx - 1

        return self._format_tracks()

    def _calc_iou(self, b1: list[float], b2: list[float]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = ix * iy
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _format_tracks(self) -> list[dict]:
        results = []
        for tid, track in self.tracks.items():
            bbox = track["bbox"]
            results.append({
                "id": tid,
                "bbox": bbox,
                "confidence": track.get("confidence", 0.5),
                "history": track.get("history", []),
                "appear_frame": track.get("appear_frame", 0),
                "disappear_frame": track.get("disappear_frame", -1),
            })
        return results

    def compute_velocity(self, track_id: str) -> dict[str, float] | None:
        """Compute average velocity for a track."""
        track = self.tracks.get(track_id)
        if not track or len(track["history"]) < 2:
            return None
        hist = track["history"]
        dt = hist[-1]["t"] - hist[0]["t"]
        if dt == 0:
            return {"vx": 0.0, "vy": 0.0}
        return {
            "vx": (hist[-1]["x"] - hist[0]["x"]) / dt,
            "vy": (hist[-1]["y"] - hist[0]["y"]) / dt,
        }

    def get_all_tracks(self) -> list[dict]:
        return self._format_tracks()

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 0


def get_tracker(fps: float = 30.0) -> StrongSORTTracker:
    """Get or create tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = StrongSORTTracker(fps=fps)
    return _tracker
