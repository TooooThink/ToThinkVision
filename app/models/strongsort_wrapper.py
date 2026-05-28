"""BoT-SORT — robust multi-object tracking with ReID + camera motion compensation.

Upgraded from StrongSORT: higher MOTA/MOTP, better camera motion compensation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_tracker = None


class BoTSORTTracker:
    """BoT-SORT tracker wrapper for multi-object tracking.

    Uses ReID features + Kalman filter + camera motion compensation
    for robust tracking with higher MOTA/MOTP than StrongSORT.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.tracker = None
        self.next_track_id = 0
        self.tracks: dict[str, dict] = {}
        self._init_model()

    def _init_model(self):
        """Initialize BoT-SORT tracker."""
        # Try Ultralytics BoT-SORT first (most common install path)
        #
        # BOTSORT.__init__() API varies wildly across ultralytics versions:
        #   - old:  BOTSORT(model_weights=None, track_high_thresh=0.5, ...)
        #   - mid:  BOTSORT(track_high_thresh=0.5, ...)
        #   - new:  BOTSORT()  ← no params, reads from cfg
        #
        # Strategy: inspect __init__ signature and only pass accepted kwargs
        try:
            import inspect
            from ultralytics.trackers.bot_sort import BOTSORT

            sig = inspect.signature(BOTSORT.__init__)
            accepted = set(sig.parameters.keys()) - {"self"}
            logger.debug("BOTSORT.__init__ accepts: %s", accepted)

            # Build kwargs from what the constructor actually accepts
            candidate_kwargs = {
                "track_high_thresh": 0.5,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.6,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.25,
                "fuse_first_frame": True,
                "model_weights": None,
            }

            # If **kwargs in signature, pass everything
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            # Check if 'args' is required (some versions need argparse.Namespace)
            args_param = sig.parameters.get("args")
            needs_args = args_param and args_param.default == inspect.Parameter.empty

            if needs_args:
                # Construct a Namespace-like object with tracker config
                from argparse import Namespace
                tracker_args = Namespace(
                    tracking_method="botsort",
                    tracking_model="yolov8n.pt",
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=30,
                    match_thresh=0.8,
                    proximity_thresh=0.5,
                    appearance_thresh=0.25,
                    fuse_first_frame=True,
                    with_reid=True,
                )
                self.tracker = BOTSORT(tracker_args)
            elif has_var_keyword:
                use_kwargs = candidate_kwargs
                self.tracker = BOTSORT(**use_kwargs)
            else:
                use_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
                self.tracker = BOTSORT(**use_kwargs)

            self._backend = "ultralytics"
            logger.info("BoT-SORT initialized (ultralytics, %s)",
                        "args=Namespace" if needs_args else f"params: {list(use_kwargs.keys()) or 'none'}")
            return
        except ImportError:
            logger.warning("ultralytics BOTSORT not available, trying standalone botsort")

        # Try standalone botsort package
        try:
            from botsort import BoTSORT

            self.tracker = BoTSORT(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
            )
            self._backend = "botsort"
            logger.info("BoT-SORT initialized (botsort package)")
            return
        except ImportError:
            raise RuntimeError(
                "BoT-SORT is required but not installed. Install with: "
                "pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"BoT-SORT init failed: {e}")

    def update(self, detections: list[dict], frame: np.ndarray | None = None,
               frame_idx: int = 0) -> list[dict]:
        """Update tracker with new detections.

        Args:
            detections: list of {bbox, label, confidence, feature?}
            frame: current frame image (for camera motion compensation)
            frame_idx: current frame number

        Returns:
            list of tracked objects with persistent IDs
        """
        return self._update_botsort(detections, frame, frame_idx)

    def _update_botsort(self, detections: list[dict], frame: np.ndarray | None,
                        frame_idx: int) -> list[dict]:
        """Update using BoT-SORT."""
        # Convert detections to format: [x1, y1, x2, y2, conf]
        bboxes = []
        for det in detections:
            x, y, w, h = det["bbox"]
            bboxes.append([x, y, x + w, y + h, det.get("confidence", 0.5)])
        bboxes = np.array(bboxes)

        if self._backend == "ultralytics":
            tracks = self.tracker.update(bboxes, frame)
        else:
            tracks = self.tracker.update(bboxes, frame)

        results = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, *_ = track[:7]
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


# Backward-compatible alias
StrongSORTTracker = BoTSORTTracker


def get_tracker(fps: float = 30.0) -> BoTSORTTracker:
    """Get or create tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = BoTSORTTracker(fps=fps)
    return _tracker
