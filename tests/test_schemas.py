"""Tests for v2 schemas, exporters, and pipeline components."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Force mock mode
os.environ["MOCK_MODE"] = "true"

from app.schemas import (
    BBox2D, BBox3D, CameraPose, ExportFormat, GaussianSplatData,
    Interaction, InteractionType, ObjectRelation, ObjectType,
    PointCloud, PSDLayer, StructuredObject, StructuredOutput,
    TemporalInfo, VideoMetadata,
)


# ─── Schema Tests ─────────────────────────────────────────────

def test_bbox_2d():
    bbox = BBox2D(x=10, y=20, w=100, h=50)
    assert bbox.x == 10 and bbox.w == 100


def test_bbox_3d():
    bbox = BBox3D(x=1.5, y=2.5, z=3.0)
    assert bbox.z == 3.0


def test_point_cloud():
    pc = PointCloud(
        points=[(0, 0, 1), (1, 0, 1), (0, 1, 1)],
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        normals=[(0, 0, 1), (0, 0, 1), (0, 0, 1)],
        confidence=[0.9, 0.8, 0.7],
    )
    assert len(pc.points) == 3
    assert len(pc.colors) == 3


def test_camera_pose():
    pose = CameraPose(
        frame_idx=5,
        intrinsics=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        extrinsics=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        position=(0.0, 0.0, -2.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )
    assert pose.frame_idx == 5
    assert pose.position == (0.0, 0.0, -2.0)


def test_gaussian_splat_data():
    gs = GaussianSplatData(
        means=[(0, 0, 1)],
        quats=[(1, 0, 0, 0)],
        scales=[(0.1, 0.1, 0.1)],
        opacities=[0.8],
        sh_coeffs=[(0.5, 0.3, 0.2)],
    )
    assert len(gs.means) == 1


def test_psd_layer():
    layer = PSDLayer(
        name="Button_1",
        position=(100, 200),
        size=(120, 40),
        opacity=0.8,
        blend_mode="multiply",
        children=[PSDLayer(name="Text_1", position=(110, 210), size=(100, 20))],
    )
    assert len(layer.children) == 1


def test_structured_object_v2():
    obj = StructuredObject(
        id="obj_0001",
        label=ObjectType.UI_BUTTON,
        bbox=BBox2D(x=50, y=100, w=120, h=40),
        bbox_3d=BBox3D(x=110, y=120, z=2.5),
        depth_value=2.5,
        mask_base64="base64data...",
        crop_image_base64="base64crop...",
        point_cloud_indices=[0, 1, 2],
        temporal=TemporalInfo(
            frame_index=10,
            trajectory=[{"x": 110, "y": 120, "t": i} for i in range(11)],
            depth_per_frame=[2.5] * 11,
        ),
    )
    assert obj.mask_base64 is not None
    assert obj.crop_image_base64 is not None
    assert len(obj.temporal.depth_per_frame) == 11


def test_structured_output_v2():
    obj = StructuredObject(id="obj_0001", bbox=BBox2D(x=0, y=0, w=100, h=100))
    pc = PointCloud(points=[(0, 0, 1), (1, 0, 1)], colors=[(255, 0, 0), (0, 255, 0)])
    output = StructuredOutput(
        source_file="test.png",
        source_type="image",
        objects=[obj],
        point_cloud=pc,
        camera_poses=[CameraPose(frame_idx=0, position=(0, 0, -2))],
        gaussian_splats=GaussianSplatData(
            means=[(0, 0, 1)],
            quats=[(1, 0, 0, 0)],
            scales=[(0.1, 0.1, 0.1)],
            opacities=[0.8],
            sh_coeffs=[(0.5, 0.3, 0.2)],
        ),
    )
    assert len(output.point_cloud.points) == 2
    assert len(output.gaussian_splats.means) == 1


# ─── Exporter Tests ───────────────────────────────────────────

@pytest.fixture
def sample_video_output_v2():
    return StructuredOutput(
        source_file="test_video.mp4",
        source_type="video",
        metadata=VideoMetadata(fps=30.0, total_frames=90, width=1920, height=1080, duration_seconds=3.0),
        objects=[
            StructuredObject(
                id="obj_0000",
                label=ObjectType.VIDEO_OBJECT,
                label_custom="Player",
                confidence=0.95,
                bbox=BBox2D(x=100, y=200, w=80, h=120),
                bbox_3d=BBox3D(x=140, y=260, z=5.0),
                temporal=TemporalInfo(
                    frame_index=0, appear_frame=0, disappear_frame=89,
                    trajectory=[{"x": 140 + i * 5, "y": 260 + i * 2, "t": i} for i in range(10)],
                    velocity={"vx": 5.0, "vy": 2.0},
                ),
                z_index=3,
            ),
        ],
        frame_count=10,
        processing_time_seconds=5.0,
        point_cloud=PointCloud(
            points=[(float(i), 0.0, 1.0 + i * 0.1) for i in range(100)],
            colors=[(255, 0, 0)] * 100,
        ),
        camera_poses=[
            CameraPose(frame_idx=i, position=(0.0, 0.0, float(-2 - i * 0.5)))
            for i in range(10)
        ],
        gaussian_splats=GaussianSplatData(
            means=[(0.0, 0.0, 1.0)] * 50,
            quats=[(1.0, 0.0, 0.0, 0.0)] * 50,
            scales=[(0.1, 0.1, 0.1)] * 50,
            opacities=[0.8] * 50,
            sh_coeffs=[(0.5, 0.3, 0.2)] * 50,
        ),
    )


def test_splat_export_from_splat_data(sample_video_output_v2, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.splat_exporter import SplatExporter
    exporter = SplatExporter(ExportFormat.UNITY_SPLAT)
    path = exporter.export(sample_video_output_v2)
    assert path.exists()
    assert path.suffix == ".splat"


def test_splat_export_ply(sample_video_output_v2, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.splat_exporter import SplatExporter
    exporter = SplatExporter(ExportFormat.GLTF)
    path = exporter.export(sample_video_output_v2)
    assert path.exists()
    assert path.suffix == ".ply"


def test_psd_export_static(tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.psd_exporter import PSDExporter
    data = StructuredOutput(
        source_file="test.png", source_type="image",
        metadata=VideoMetadata(width=800, height=600),
        objects=[
            StructuredObject(
                id="obj_0000", label=ObjectType.UI_BUTTON,
                bbox=BBox2D(x=50, y=50, w=100, h=40),
                label_custom="Submit",
            ),
        ],
    )
    exporter = PSDExporter(ExportFormat.PSD_STATIC)
    path = exporter.export(data)
    assert path.exists()
    assert path.suffix == ".psd"


def test_psd_export_animated(sample_video_output_v2, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.psd_exporter import PSDExporter
    exporter = PSDExporter(ExportFormat.PSD_ANIMATED)
    path = exporter.export(sample_video_output_v2)
    assert path.exists()
    assert path.suffix == ".psd"


def test_ae_project_export(sample_video_output_v2, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.ae_project_exporter import AEProjectExporter
    exporter = AEProjectExporter(ExportFormat.AE_PROJECT)
    path = exporter.export(sample_video_output_v2)
    assert path.exists()
    assert path.suffix == ".jsx"
    # Check companion JSON was also created
    json_path = path.with_suffix(".ae_data.json")
    assert json_path.exists()
    jdata = json.loads(json_path.read_text())
    assert jdata["format"] == "ae_project_data"
    assert jdata["fps"] == 30.0


def test_video_trajectory_export_v2(sample_video_output_v2, tmp_path):
    os.environ["TTV_OUTPUT_DIR"] = str(tmp_path)
    from app.exporters.video_exporter import VideoExporter
    exporter = VideoExporter(ExportFormat.VIDEO_TRAJECTORY)
    path = exporter.export(sample_video_output_v2)
    assert path.exists()
    content = path.read_text()
    assert "object_id" in content


# ─── Pipeline Utility Tests ──────────────────────────────────

def test_pointcloud_backproject():
    from app.utils.pointcloud import backproject_depth, compute_normals, filter_pointcloud
    from app.utils.camera import estimate_intrinsics

    depth = np.ones((100, 100), dtype=np.float32) * 2.0  # 2m depth everywhere
    K = estimate_intrinsics(100, 100)
    points = backproject_depth(depth, K)
    assert points.shape == (10000, 3)
    assert np.allclose(points[:, 2], 2.0)  # All z should be 2m


def test_pointcloud_filter():
    from app.utils.pointcloud import filter_pointcloud
    points = np.array([[0, 0, 0.5], [0, 0, 5], [0, 0, 100]])
    filtered, _ = filter_pointcloud(points, min_z=1.0, max_z=50.0)
    assert len(filtered) == 1
    assert filtered[0, 2] == 5.0


def test_camera_intrinsics():
    from app.utils.camera import estimate_intrinsics
    K = estimate_intrinsics(1920, 1080, fov_deg=60)
    assert K.shape == (3, 3)
    assert K[0, 2] == 960.0  # cx = width/2
    assert K[1, 2] == 540.0  # cy = height/2


def test_camera_extrinsics():
    from app.utils.camera import build_extrinsics, rt_matrix_to_position, rt_matrix_to_quaternion
    import numpy as np

    position = np.array([0, 0, -5])
    look_at = np.array([0, 0, 0])
    RT = build_extrinsics(position, look_at)
    assert RT.shape == (4, 4)

    R = RT[:3, :3]
    t = RT[:3, 3]
    recovered = rt_matrix_to_position(R, t)
    np.testing.assert_allclose(recovered, position, atol=1e-10)

    quat = rt_matrix_to_quaternion(np.eye(3))
    assert quat == (1.0, 0.0, 0.0, 0.0)


def test_ply_write(tmp_path):
    from app.utils.pointcloud import save_ply
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    path = tmp_path / "test.ply"
    save_ply(path, points, colors)
    assert path.exists()
    content = path.read_text()
    assert "ply" in content
    assert "element vertex 3" in content
