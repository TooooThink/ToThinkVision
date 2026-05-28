"""Tests for the main pipeline and utilities."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.models.tracker import ObjectTracker
from app.utils.color import extract_dominant_color, rgb_to_hex
from app.utils.geometry import check_collision, compute_relative_position, compute_z_index
from app.schemas import BBox2D, ObjectType, StructuredObject


@pytest.fixture(autouse=True)
def mock_mode():
    """Force mock mode for all tests."""
    os.environ["MOCK_MODE"] = "true"
    yield
    os.environ.pop("MOCK_MODE", None)


class TestObjectTracker:
    def test_iou_perfect_overlap(self):
        bbox = [0.0, 0.0, 100.0, 100.0]
        assert ObjectTracker.iou(bbox, bbox) == 1.0

    def test_iou_no_overlap(self):
        b1 = [0.0, 0.0, 50.0, 50.0]
        b2 = [100.0, 100.0, 50.0, 50.0]
        assert ObjectTracker.iou(b1, b2) == 0.0

    def test_iou_partial(self):
        b1 = [0.0, 0.0, 100.0, 100.0]
        b2 = [50.0, 50.0, 100.0, 100.0]
        iou = ObjectTracker.iou(b1, b2)
        assert 0.0 < iou < 1.0

    def test_first_frame_creates_tracks(self):
        tracker = ObjectTracker()
        dets = [
            {"bbox": [10, 20, 50, 50], "label": "button", "confidence": 0.9},
            {"bbox": [100, 200, 80, 40], "label": "text", "confidence": 0.85},
        ]
        tracks = tracker.update(dets, frame_idx=0)
        assert len(tracks) == 2
        assert tracks[0]["appear_frame"] == 0

    def test_tracking_across_frames(self):
        tracker = ObjectTracker()
        dets_0 = [{"bbox": [10, 20, 50, 50], "label": "obj", "confidence": 0.9}]
        dets_1 = [{"bbox": [12, 22, 50, 50], "label": "obj", "confidence": 0.88}]
        tracker.update(dets_0, frame_idx=0)
        tracks = tracker.update(dets_1, frame_idx=1)
        assert len(tracks) == 1
        assert len(tracks[0]["history"]) == 2

    def test_new_object_appears(self):
        tracker = ObjectTracker()
        tracker.update([{"bbox": [10, 20, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=0)
        tracks = tracker.update([
            {"bbox": [12, 22, 50, 50], "label": "a", "confidence": 0.88},
            {"bbox": [200, 300, 40, 40], "label": "b", "confidence": 0.8},
        ], frame_idx=1)
        assert len(tracks) == 2

    def test_disappeared_object(self):
        tracker = ObjectTracker()
        tracker.update([{"bbox": [10, 20, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=0)
        tracks = tracker.update([], frame_idx=1)
        assert tracks[0]["disappear_frame"] == 0

    def test_velocity(self):
        tracker = ObjectTracker()
        tracker.update([{"bbox": [0, 0, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=0)
        tracker.update([{"bbox": [10, 0, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=1)
        tracker.update([{"bbox": [20, 0, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=2)
        vel = tracker.compute_velocity(tracker.get_all_tracks()[0]["id"])
        assert vel is not None
        assert vel["vx"] == 10.0

    def test_reset(self):
        tracker = ObjectTracker()
        tracker.update([{"bbox": [10, 20, 50, 50], "label": "a", "confidence": 0.9}], frame_idx=0)
        tracker.reset()
        assert len(tracker.tracks) == 0


class TestGeometry:
    def test_collision_overlapping(self):
        b1 = [0, 0, 100, 100]
        b2 = [50, 50, 100, 100]
        assert check_collision(b1, b2) is True

    def test_collision_separate(self):
        b1 = [0, 0, 50, 50]
        b2 = [100, 100, 50, 50]
        assert check_collision(b1, b2) is False

    def test_relative_above(self):
        top = [50, 0, 50, 50]
        bottom = [50, 100, 50, 50]
        assert compute_relative_position(top, bottom) == "above"

    def test_relative_below(self):
        top = [50, 100, 50, 50]
        bottom = [50, 0, 50, 50]
        assert compute_relative_position(top, bottom) == "below"

    def test_relative_left(self):
        left = [0, 50, 50, 50]
        right = [100, 50, 50, 50]
        assert compute_relative_position(left, right) == "left_of"

    def test_relative_right(self):
        left = [100, 50, 50, 50]
        right = [0, 50, 50, 50]
        assert compute_relative_position(left, right) == "right_of"

    def test_z_index_order(self):
        bboxes = [("b", [0, 100, 50, 50]), ("a", [0, 0, 50, 50]), ("c", [0, 200, 50, 50])]
        z = compute_z_index(bboxes)
        assert z["a"] < z["b"] < z["c"]


class TestColor:
    def test_extract_dominant_color(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:40, 10:60] = [255, 0, 0]  # Red region
        dominant, palette = extract_dominant_color(img, [10, 10, 50, 30])
        assert dominant is not None
        assert isinstance(palette, list)

    def test_rgb_to_hex(self):
        assert rgb_to_hex(255, 0, 0) == "#ff0000"
        assert rgb_to_hex(0, 255, 0) == "#00ff00"
        assert rgb_to_hex(0, 0, 255) == "#0000ff"
        assert rgb_to_hex(255, 255, 255) == "#ffffff"


class TestPipelineIntegration:
    """Integration tests for the full pipeline in mock mode."""

    def test_process_image_mock(self):
        """Test that process_file runs without errors in mock mode on a synthetic image."""
        os.makedirs("/tmp/ttv_test", exist_ok=True)
        # Create a simple test image
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[20:80, 20:120] = [0, 100, 200]
        img[100:150, 50:200] = [200, 100, 0]

        img_path = Path("/tmp/ttv_test/test_image.png")
        Image.fromarray(img).save(str(img_path))

        from app.pipeline import process_file
        result = process_file(img_path, mode="general")
        assert result.source_type == "image"
        assert result.frame_count == 1
        assert len(result.objects) > 0

    def test_process_image_mock_generates_objects(self):
        """Verify mock pipeline produces objects with all required fields."""
        os.makedirs("/tmp/ttv_test", exist_ok=True)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img_path = Path("/tmp/ttv_test/test_simple.png")
        Image.fromarray(img).save(str(img_path))

        from app.pipeline import process_file
        result = process_file(img_path, mode="general")
        assert len(result.objects) > 0
        obj = result.objects[0]
        assert obj.id is not None
        assert obj.bbox is not None
        assert obj.bbox.w > 0
        assert obj.bbox.h > 0
        assert obj.dominant_color is not None or obj.depth_value is not None
