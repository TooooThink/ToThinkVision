"""Tests for 4D trajectory extraction and animated export.

Tests cover:
- ICP alignment with known transforms
- B-spline smoothing
- Motion classification
- Full trajectory extraction with synthetic data
- Animated glTF export round-trip
- USDA export validity
- Blender script generation
- Scene graph construction
- World model adapter
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.schemas import (
    CameraPose,
    InteractionEvent,
    Mesh3D,
    ObjectTrajectory4D,
    PipelineConfig,
    SceneGraph4D,
    SceneGraphEdge,
    SceneGraphNode,
    StructuredObject,
    TrajectoryKeyframe,
    BBox2D,
)


# ─── Fixtures ────────────────────────────────────────────────

def make_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(**kwargs)


def make_cube_points(center=(0, 0, 5), size=0.5, n=50) -> np.ndarray:
    """Generate a random point cloud in a cube."""
    points = np.random.randn(n, 3) * size + np.array(center)
    return points


def make_trajectory(n_keyframes=10, motion="rigid") -> ObjectTrajectory4D:
    """Create a synthetic trajectory."""
    keyframes = []
    for i in range(n_keyframes):
        t = i * 0.1  # 10fps
        if motion == "static":
            pos = (0.0, 0.0, 5.0)
        elif motion == "rigid":
            pos = (float(i * 0.1), 0.0, 5.0)  # Moving along X
        else:  # deformable
            pos = (float(np.sin(t)), float(np.cos(t)), 5.0)

        kf = TrajectoryKeyframe(
            timestamp=t,
            frame_idx=i,
            position=pos,
            rotation=(1.0, 0.0, 0.0, 0.0),
            is_rigid=(motion != "deformable"),
            deformation_score=0.5 if motion == "deformable" else 0.0,
        )
        keyframes.append(kf)

    # Compute total_distance correctly
    total_dist = 0.0
    if motion == "rigid":
        total_dist = (n_keyframes - 1) * 0.1
    elif motion == "deformable":
        total_dist = 2.0

    return ObjectTrajectory4D(
        object_id="test_obj",
        keyframes=keyframes,
        motion_type=motion,
        total_distance=total_dist,
        max_speed=1.0 if motion != "static" else 0.0,
        duration=n_keyframes * 0.1,
    )


def make_mesh() -> Mesh3D:
    """Create a simple triangle mesh."""
    return Mesh3D(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        normals=[(0, 0, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0)],
        point_count=4,
    )


def make_object(obj_id="obj_001", label="test_object") -> StructuredObject:
    """Create a structured object with mesh."""
    return StructuredObject(
        id=obj_id,
        bbox=BBox2D(x=100, y=100, w=50, h=50),
        label_custom=label,
        mesh_3d=make_mesh(),
    )


# ─── Schema Tests ────────────────────────────────────────────

class TestSchemas4D:
    def test_trajectory_keyframe(self):
        kf = TrajectoryKeyframe(
            timestamp=0.5,
            frame_idx=15,
            position=(1.0, 2.0, 3.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        )
        assert kf.timestamp == 0.5
        assert kf.position == (1.0, 2.0, 3.0)
        assert kf.is_rigid is True
        assert kf.deformation_score == 0.0

    def test_object_trajectory_4d(self):
        traj = make_trajectory()
        assert traj.object_id == "test_obj"
        assert len(traj.keyframes) == 10
        assert traj.motion_type == "rigid"

    def test_scene_graph_node(self):
        node = SceneGraphNode(
            object_id="obj_1",
            label="table",
            is_static=True,
            category="furniture",
        )
        assert node.is_static is True

    def test_scene_graph_edge(self):
        edge = SceneGraphEdge(
            source_id="obj_1",
            target_id="obj_2",
            relation="above",
            time_range=(0.0, 5.0),
        )
        assert edge.relation == "above"

    def test_interaction_event(self):
        event = InteractionEvent(
            timestamp=1.5,
            frame_idx=45,
            event_type="collision",
            object_ids=["obj_1", "obj_2"],
        )
        assert event.event_type == "collision"

    def test_pipeline_config_4d(self):
        config = PipelineConfig(
            enable_4d_trajectory=True,
            enable_4dgs=False,
            enable_scene_graph=True,
            trajectory_smoothing=0.8,
            icp_distance_threshold=0.1,
        )
        assert config.enable_4d_trajectory is True
        assert config.enable_4dgs is False
        assert config.trajectory_smoothing == 0.8

    def test_world_model_config(self):
        config = PipelineConfig(is_world_model_video=True)
        assert config.is_world_model_video is True


# ─── Trajectory Extraction Tests ─────────────────────────────

class TestTrajectoryExtractor:
    def test_init_default(self):
        config = make_config()
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)
        assert ext.icp_threshold == 0.05
        assert ext.deform_threshold == 0.3
        assert ext.smoothing == 0.5

    def test_init_world_model(self):
        config = make_config(is_world_model_video=True)
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)
        # Thresholds should be relaxed
        assert ext.icp_threshold == 0.1  # 2x default
        assert ext.smoothing == 0.7  # +0.2

    def test_icp_identity(self):
        """ICP with identical point clouds should return identity transform."""
        config = make_config()
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)

        points = make_cube_points(n=100)
        T, residual = ext._estimate_rigid_transform(points, points)

        # Should be close to identity
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=0.01)
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=0.01)
        assert residual < 0.01

    def test_icp_translation(self):
        """ICP should detect pure translation."""
        config = make_config()
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)

        # Use 3D point cloud (not coplanar) for reliable ICP
        np.random.seed(42)
        src = np.random.randn(200, 3) * 2.0 + np.array([0, 0, 5])

        translation = np.array([1.0, 0.5, -0.3])
        tgt = src + translation

        T, residual = ext._estimate_rigid_transform(src, tgt)

        # Translation should be close to the applied translation
        np.testing.assert_allclose(T[:3, 3], translation, atol=0.15)
        assert residual < 0.15

    def test_icp_rotation(self):
        """ICP should detect rotation."""
        config = make_config()
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)

        src = make_cube_points(center=(0, 0, 5), n=200)

        # Apply 30 degree rotation around Z axis
        angle = np.radians(30)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        tgt = (R @ src.T).T

        T, residual = ext._estimate_rigid_transform(src, tgt)
        assert residual < 0.1

    def test_orientation_pca(self):
        """PCA orientation should produce a valid quaternion."""
        config = make_config()
        from app.models.trajectory_4d import TrajectoryExtractor4D
        ext = TrajectoryExtractor4D(config)

        # Elongated point cloud along X axis
        points = np.random.randn(100, 3) * np.array([3.0, 0.5, 0.5])
        q = ext._estimate_orientation(points)

        # Should be a valid quaternion
        assert len(q) == 4
        norm = sum(x**2 for x in q)
        assert abs(norm - 1.0) < 0.01

    def test_motion_classification_static(self):
        """Static objects should be classified as static."""
        from app.models.trajectory_4d import TrajectoryExtractor4D
        config = make_config()
        ext = TrajectoryExtractor4D(config)

        traj = make_trajectory(motion="static")
        motion = ext._classify_motion(traj.keyframes)
        assert motion == "static"

    def test_motion_classification_rigid(self):
        """Moving objects with low deformation should be classified as rigid."""
        from app.models.trajectory_4d import TrajectoryExtractor4D
        config = make_config()
        ext = TrajectoryExtractor4D(config)

        # Create keyframes with actual movement
        keyframes = []
        for i in range(10):
            kf = TrajectoryKeyframe(
                timestamp=i * 0.1,
                frame_idx=i,
                position=(float(i * 0.5), 0.0, 5.0),  # Moving 0.5m per frame
                rotation=(1.0, 0.0, 0.0, 0.0),
                is_rigid=True,
                deformation_score=0.0,
            )
            keyframes.append(kf)

        motion = ext._classify_motion(keyframes)
        assert motion == "rigid"

    def test_bspline_smoothing(self):
        """B-spline smoothing should reduce noise."""
        from app.models.trajectory_4d import TrajectoryExtractor4D
        config = make_config(trajectory_smoothing=0.8)
        ext = TrajectoryExtractor4D(config)

        # Create noisy trajectory
        keyframes = []
        for i in range(20):
            noise = np.random.randn(3) * 0.05
            pos = (float(i * 0.1) + noise[0], noise[1], 5.0 + noise[2])
            kf = TrajectoryKeyframe(
                timestamp=i * 0.1,
                frame_idx=i,
                position=pos,
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
            keyframes.append(kf)

        smoothed = ext._smooth_trajectory(keyframes)
        assert len(smoothed) == len(keyframes)

        # Smoothed positions should be closer to a line
        orig_positions = np.array([kf.position for kf in keyframes])
        smooth_positions = np.array([kf.position for kf in smoothed])

        # Variance from linear fit should be smaller for smoothed
        t = np.linspace(0, 1, len(keyframes))
        orig_residual = np.std(orig_positions[:, 0] - np.polyval(np.polyfit(t, orig_positions[:, 0], 1), t))
        smooth_residual = np.std(smooth_positions[:, 0] - np.polyval(np.polyfit(t, smooth_positions[:, 0], 1), t))
        # Smoothed should have less or equal residual (more linear)
        assert smooth_residual <= orig_residual + 0.01

    def test_velocity_computation(self):
        """Velocities should be computed between keyframes."""
        from app.models.trajectory_4d import TrajectoryExtractor4D
        config = make_config()
        ext = TrajectoryExtractor4D(config)

        keyframes = [
            TrajectoryKeyframe(timestamp=0.0, frame_idx=0, position=(0, 0, 0)),
            TrajectoryKeyframe(timestamp=1.0, frame_idx=30, position=(1, 0, 0)),
            TrajectoryKeyframe(timestamp=2.0, frame_idx=60, position=(2, 0, 0)),
        ]

        result = ext._compute_velocities(keyframes)
        assert len(result) == 3
        # Second keyframe should have velocity ~(1, 0, 0) m/s
        assert result[1].velocity is not None
        np.testing.assert_allclose(result[1].velocity, (1.0, 0.0, 0.0), atol=0.01)

    def test_scale_estimation(self):
        """Scale estimation should detect size changes."""
        from app.models.trajectory_4d import TrajectoryExtractor4D
        config = make_config()
        ext = TrajectoryExtractor4D(config)

        ref = make_cube_points(size=1.0, n=100)
        larger = make_cube_points(size=2.0, n=100)

        scale = ext._estimate_scale(larger, ref)
        assert len(scale) == 3
        # Scale should be roughly 2x (with some noise)
        for s in scale:
            assert 1.0 < s < 4.0  # Reasonable range


# ─── Animated glTF Export Tests ──────────────────────────────

class TestAnimatedGLTFExporter:
    def test_export_basic(self):
        from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
        exporter = AnimatedGLTFExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            assert path.exists()
            assert path.suffix == ".glb"
            assert path.stat().st_size > 100  # Non-trivial file

    def test_glb_header(self):
        """GLB file should have correct magic bytes."""
        from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
        exporter = AnimatedGLTFExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            with open(path, "rb") as f:
                magic = f.read(4)
                version = int.from_bytes(f.read(4), "little")
                length = int.from_bytes(f.read(4), "little")

            assert magic == b"glTF"
            assert version == 2
            assert length == path.stat().st_size

    def test_export_without_trajectory(self):
        """Should export static scene when no trajectories available."""
        from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
        exporter = AnimatedGLTFExporter()

        obj = make_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={},
                output_dir=Path(tmpdir),
            )
            assert path.exists()

    def test_export_with_camera(self):
        """Should include camera animation when poses provided."""
        from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
        exporter = AnimatedGLTFExporter()

        obj = make_object()
        traj = make_trajectory()
        poses = [
            CameraPose(frame_idx=i, position=(float(i), 0, 0), rotation=(1, 0, 0, 0))
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                camera_poses=poses,
                output_dir=Path(tmpdir),
            )
            assert path.exists()

    def test_multiple_objects(self):
        """Should handle multiple animated objects."""
        from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
        exporter = AnimatedGLTFExporter()

        objects = [make_object(f"obj_{i:03d}", f"object_{i}") for i in range(3)]
        trajectories = {obj.id: make_trajectory() for obj in objects}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=objects,
                trajectories=trajectories,
                output_dir=Path(tmpdir),
            )
            assert path.exists()


# ─── USD Export Tests ────────────────────────────────────────

class TestUSDExporter:
    def test_export_basic(self):
        from app.exporters.usd_exporter import USDExporter
        exporter = USDExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            assert path.exists()
            assert path.suffix == ".usda"

    def test_usda_header(self):
        """USDA file should start with correct header."""
        from app.exporters.usd_exporter import USDExporter
        exporter = USDExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            content = path.read_text()
            assert content.startswith("#usda 1.0")
            assert "defaultPrim" in content
            assert "World" in content

    def test_usda_animated_transform(self):
        """USDA should contain time-sampled transforms."""
        from app.exporters.usd_exporter import USDExporter
        exporter = USDExporter()

        obj = make_object()
        traj = make_trajectory()
        # Match trajectory key to object ID
        traj_matched = ObjectTrajectory4D(
            object_id=obj.id,
            keyframes=traj.keyframes,
            motion_type=traj.motion_type,
            total_distance=traj.total_distance,
            max_speed=traj.max_speed,
            duration=traj.duration,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={obj.id: traj_matched},
                output_dir=Path(tmpdir),
            )

            content = path.read_text()
            assert "timeSamples" in content
            assert "xformOp:translate" in content

    def test_quat_to_euler(self):
        """Quaternion to Euler conversion should be correct."""
        from app.exporters.usd_exporter import USDExporter
        # Identity quaternion
        euler = USDExporter._quat_to_euler((1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(euler, (0, 0, 0), atol=0.01)


# ─── Blender Export Tests ────────────────────────────────────

class TestBlenderExporter:
    def test_export_basic(self):
        from app.exporters.blender_exporter import BlenderExporter
        exporter = BlenderExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            assert path.exists()
            assert path.suffix == ".py"

    def test_script_content(self):
        """Generated script should contain key Blender operations."""
        from app.exporters.blender_exporter import BlenderExporter
        exporter = BlenderExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            content = path.read_text()
            assert "import bpy" in content
            assert "keyframe_insert" in content
            assert "Quaternion" in content
            assert "obj_import" in content

    def test_obj_files_created(self):
        """OBJ mesh files should be created."""
        from app.exporters.blender_exporter import BlenderExporter
        exporter = BlenderExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            mesh_dir = Path(tmpdir) / "meshes"
            assert mesh_dir.exists()
            obj_files = list(mesh_dir.glob("*.obj"))
            assert len(obj_files) > 0

    def test_trajectory_json_created(self):
        """Trajectory JSON should be created."""
        from app.exporters.blender_exporter import BlenderExporter
        exporter = BlenderExporter()

        obj = make_object()
        traj = make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export(
                objects=[obj],
                trajectories={"test_obj": traj},
                output_dir=Path(tmpdir),
            )

            traj_files = list(Path(tmpdir).glob("*_trajectories.json"))
            assert len(traj_files) > 0

            with open(traj_files[0]) as f:
                data = json.load(f)
            assert "test_obj" in data


# ─── Scene Graph Tests ───────────────────────────────────────

class TestSceneGraph4D:
    def test_build_basic(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)
        obj = make_object()
        traj = make_trajectory()
        # Match trajectory key to object ID
        traj_matched = ObjectTrajectory4D(
            object_id=obj.id,
            keyframes=traj.keyframes,
            motion_type=traj.motion_type,
            total_distance=traj.total_distance,
            max_speed=traj.max_speed,
            duration=traj.duration,
        )

        graph = builder.build([obj], {obj.id: traj_matched})

        assert isinstance(graph, SceneGraph4D)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].object_id == obj.id

    def test_static_detection(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)
        obj = make_object()
        traj = make_trajectory(motion="static")
        traj_matched = ObjectTrajectory4D(
            object_id=obj.id,
            keyframes=traj.keyframes,
            motion_type="static",
            total_distance=0.0,
            max_speed=0.0,
            duration=traj.duration,
        )

        graph = builder.build([obj], {obj.id: traj_matched})

        assert graph.nodes[0].is_static is True
        assert graph.num_static_objects == 1

    def test_spatial_relations(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)

        # Object A is below object B
        pos_a = np.array([0, 0, 0])
        pos_b = np.array([0, 2, 0])  # B is 2m above A

        relations = builder._compute_spatial_relations(pos_a, pos_b)
        assert "below" in relations  # A is below B

    def test_contact_detection(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)

        pos_a = np.array([0, 0, 0])
        pos_b = np.array([0.01, 0, 0])  # Very close

        relations = builder._compute_spatial_relations(pos_a, pos_b)
        assert "in_contact" in relations

    def test_multiple_objects_edges(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)
        objects = [
            make_object("obj_001", "table"),
            make_object("obj_002", "cup"),
        ]
        trajs = {
            "obj_001": make_trajectory(motion="static"),
            "obj_002": make_trajectory(motion="rigid"),
        }

        graph = builder.build(objects, trajs)
        assert len(graph.nodes) == 2

    def test_categorization(self):
        from app.scene.scene_graph_4d import SceneGraph4DBuilder

        builder = SceneGraph4DBuilder(fps=30.0)
        assert builder._categorize("table") == "furniture"
        assert builder._categorize("floor") == "ground"
        assert builder._categorize("person") == "person"
        assert builder._categorize("car") == "vehicle"
        assert builder._categorize("something") == "object"

    def test_scene_graph_serialization(self):
        """Scene graph should be serializable to JSON."""
        graph = SceneGraph4D(
            nodes=[SceneGraphNode(object_id="obj_1", label="table", is_static=True)],
            edges=[SceneGraphEdge(source_id="obj_1", target_id="obj_2", relation="above", time_range=(0.0, 5.0))],
            interaction_events=[InteractionEvent(timestamp=1.0, frame_idx=30, event_type="contact_start", object_ids=["obj_1", "obj_2"])],
            time_range=(0.0, 5.0),
            num_static_objects=1,
            num_dynamic_objects=1,
        )

        data = graph.model_dump()
        json_str = json.dumps(data, default=str)
        parsed = json.loads(json_str)

        assert len(parsed["nodes"]) == 1
        assert len(parsed["edges"]) == 1
        assert parsed["num_static_objects"] == 1


# ─── World Model Adapter Tests ───────────────────────────────

class TestWorldModelAdapter:
    def test_adjust_config_for_ai_video(self):
        from app.scene.world_model_adapter import WorldModelAdapter

        adapter = WorldModelAdapter()
        adapter.is_ai_generated = True

        config = make_config()
        adjusted = adapter.adjust_config(config)

        assert adjusted.is_world_model_video is True
        assert adjusted.icp_distance_threshold > config.icp_distance_threshold
        assert adjusted.deformation_threshold > config.deformation_threshold
        assert adjusted.trajectory_smoothing > config.trajectory_smoothing

    def test_no_adjustment_for_real_video(self):
        from app.scene.world_model_adapter import WorldModelAdapter

        adapter = WorldModelAdapter()
        adapter.is_ai_generated = False

        config = make_config()
        adjusted = adapter.adjust_config(config)

        # Should return same config
        assert adjusted.icp_distance_threshold == config.icp_distance_threshold


# ─── New Model Wrapper Tests ─────────────────────────────────

class TestCoTracker3:
    """Tests for CoTracker3 dense point tracking wrapper."""

    def test_init(self):
        from app.models.cotracker3 import CoTracker3Predictor
        predictor = CoTracker3Predictor(device="cpu")
        assert predictor.device == "cpu"
        assert predictor.mode == "offline"

    def test_global_instance(self):
        from app.models.cotracker3 import get_cotracker
        c1 = get_cotracker()
        c2 = get_cotracker()
        assert c1 is c2


class TestObjectGS:
    """Tests for ObjectGS per-object 3D Gaussian Splatting wrapper."""

    def test_init(self):
        from app.models.object_gs import ObjectGSPipeline
        pipe = ObjectGSPipeline()
        assert pipe.device == "cuda"
        assert pipe.available is False  # No repo cloned

    def test_unavailable_raises(self):
        from app.models.object_gs import ObjectGSPipeline
        pipe = ObjectGSPipeline()
        with pytest.raises(RuntimeError):
            pipe.train(Path("/tmp/fake_frames"))

    def test_global_instance(self):
        from app.models.object_gs import get_objectgs_pipeline
        p1 = get_objectgs_pipeline()
        p2 = get_objectgs_pipeline()
        assert p1 is p2


class TestSpann3R:
    """Tests for Spann3R 3D reconstruction with spatial memory."""

    def test_init(self):
        from app.models.spann3r import Spann3RReconstructor
        recon = Spann3RReconstructor()
        assert recon.device == "cuda"
        assert recon.available is False

    def test_unavailable_raises(self):
        from app.models.spann3r import Spann3RReconstructor
        recon = Spann3RReconstructor()
        with pytest.raises(RuntimeError):
            recon.reconstruct(Path("/tmp/fake_frames"), sample_interval=5)

    def test_nerfstudio_poses_parser(self):
        from app.models.spann3r import Spann3RReconstructor
        recon = Spann3RReconstructor()

        transform_data = {
            "fl_x": 500, "fl_y": 500, "cx": 320, "cy": 240,
            "w": 640, "h": 480,
            "frames": [
                {"transform_matrix": np.eye(4).tolist()},
                {"transform_matrix": np.eye(4).tolist()},
            ],
        }
        poses = recon._parse_nerfstudio_poses(transform_data, sample_interval=5)
        assert len(poses) == 2
        assert poses[0]["frame_idx"] == 0
        assert poses[1]["frame_idx"] == 5

    def test_global_instance(self):
        from app.models.spann3r import get_spann3r
        s1 = get_spann3r()
        s2 = get_spann3r()
        assert s1 is s2


class TestShapeOfMotion:
    """Tests for Shape of Motion end-to-end 4D reconstruction."""

    def test_init(self):
        from app.models.shape_of_motion import ShapeOfMotionPipeline
        pipe = ShapeOfMotionPipeline()
        assert pipe.device == "cuda"
        assert pipe.available is False

    def test_unavailable_raises(self):
        from app.models.shape_of_motion import ShapeOfMotionPipeline
        pipe = ShapeOfMotionPipeline()
        with pytest.raises(RuntimeError):
            pipe.reconstruct_4d(video_path=Path("/tmp/fake_video.mp4"), num_frames=10)

    def test_extract_object_trajectories(self):
        from app.models.shape_of_motion import ShapeOfMotionPipeline
        pipe = ShapeOfMotionPipeline()

        result = pipe.reconstruct_4d(
            video_path=Path("/tmp/fake_video.mp4"),
            num_frames=10,
        )

        trajectories = pipe.extract_object_trajectories(
            result["per_frame_pointclouds"]
        )
        assert "scene" in trajectories
        traj = trajectories["scene"]
        assert "keyframes" in traj
        assert "motion_type" in traj
        assert len(traj["keyframes"]) > 0

    def test_motion_classification(self):
        from app.models.shape_of_motion import ShapeOfMotionPipeline
        pipe = ShapeOfMotionPipeline()

        # Static
        static_kfs = [{"position": (0, 0, 0)} for _ in range(5)]
        assert pipe._classify_simple_motion(static_kfs) == "static"

        # Rigid (small movement: total dist = 4*0.05 = 0.2 < 0.5)
        moving_kfs = [{"position": (i * 0.05, 0, 0)} for i in range(5)]
        assert pipe._classify_simple_motion(moving_kfs) == "rigid"

        # Deformable (large movement: total dist > 0.5)
        fast_kfs = [{"position": (i * 0.2, 0, 0)} for i in range(10)]
        assert pipe._classify_simple_motion(fast_kfs) == "deformable"

    def test_global_instance(self):
        from app.models.shape_of_motion import get_shape_of_motion
        s1 = get_shape_of_motion()
        s2 = get_shape_of_motion()
        assert s1 is s2


# ─── Pipeline Config New Model Toggles ────────────────────────

class TestNewModelConfig:
    """Test that new model toggles work correctly in PipelineConfig."""

    def test_default_toggles(self):
        config = PipelineConfig()
        assert config.enable_cotracker3 is True  # Default on
        assert config.enable_objectgs is False  # Heavy, default off
        assert config.enable_spann3r is False  # Needs repo, default off
        assert config.enable_shape_of_motion is False  # Needs repo, default off

    def test_all_enabled(self):
        config = PipelineConfig(
            enable_cotracker3=True,
            enable_objectgs=True,
            enable_spann3r=True,
            enable_shape_of_motion=True,
        )
        assert config.enable_cotracker3 is True
        assert config.enable_objectgs is True
        assert config.enable_spann3r is True
        assert config.enable_shape_of_motion is True

    def test_independent_toggles(self):
        """Each toggle should be independently controllable."""
        config = PipelineConfig(
            enable_cotracker3=False,
            enable_objectgs=True,
            enable_spann3r=True,
            enable_shape_of_motion=False,
        )
        assert config.enable_cotracker3 is False
        assert config.enable_objectgs is True
        assert config.enable_spann3r is True
        assert config.enable_shape_of_motion is False

