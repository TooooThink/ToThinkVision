#!/bin/bash
#SBATCH --job-name=ttv_real
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ttv_real_%j.out
#SBATCH --error=logs/ttv_real_%j.err

# ============================================================================
# ToThinkVision — Real Model Inference Test
# ============================================================================
# Three end-to-end tests:
#
#   Test 1: Image → 2D (detect + segment + LaMa + PSD layers)
#           Input:  test_input.png
#           Output: per-object crops, masks, PSD file for Photoshop
#
#   Test 2: Video → 3D (full pipeline: detect + segment + depth + 3DGS + mesh)
#           Input:  test_input.mp4
#           Output: per-object 3D meshes, point cloud, glTF, OBJ+MTL
#
#   Test 3: Video → 4D (full 3D pipeline + CoTracker3 + 4D trajectory + scene graph + animated export)
#           Input:  test_input.mp4
#           Output: animated glTF (.glb), USD (.usda), Blender scene, scene graph JSON
#
# Before sbatch:
#   1. Place your test file(s) in the project dir:
#      - test_input.png   (image test)
#      - test_input.mp4   (video test, used by Test 2 and Test 3)
#   2. Install models:
#      bash install_models.sh          # v1/v2 models (Test 1+2)
#      bash install_models_v3.sh       # v3 models: CoTracker3, ObjectGS, Spann3R (Test 3)
#   3. Submit:
#      sbatch run_test_real.sh
# ============================================================================

set -e

mkdir -p logs outputs

# ── Clear stale __pycache__ (NFS caching on compute nodes can keep corrupted .pyc) ──
find app/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Force correct HF endpoint — hf-mirror.com doesn't resolve on compute nodes
HF_ENDPOINT="https://huggingface.co"
export HF_ENDPOINT

# Offline mode — all models must be pre-downloaded on login node
export HF_HUB_OFFLINE=1

# Point transformers cache to our model cache
CACHE_DIR="${TTV_MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

echo "=== ToThinkVision Real Model Inference ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Environment ──
echo ""
echo "── Activating conda environment ──"

# Init conda for non-interactive shell
eval "$(conda shell.bash hook 2>/dev/null)" || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true

conda activate ttv 2>/dev/null || conda activate base

# ── External tool paths (project-level config) ──
# COLMAP: required by ObjectGS and 3D Gaussian Splatting
COLMAP_BIN="$(which colmap 2>/dev/null)"
if [ -n "$COLMAP_BIN" ]; then
    export COLMAP_BIN
    echo "COLMAP: $COLMAP_BIN ($($COLMAP_BIN --version 2>/dev/null | head -1))"
else
    echo "WARNING: colmap not found in PATH. ObjectGS will fail."
fi

echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

# Add PyTorch lib to LD_LIBRARY_PATH (needed for GroundingDINO _C extension)
TORCH_LIB="$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
    echo "Added $TORCH_LIB to LD_LIBRARY_PATH"
fi

# Add MASt3R to PYTHONPATH (no setup.py, needs manual path)
MAST3R_PATH="$REPOS_DIR/mast3r"
if [ -d "$MAST3R_PATH" ]; then
    export PYTHONPATH="$MAST3R_PATH:$MAST3R_PATH/dust3r:$PYTHONPATH"
    echo "Added MASt3R to PYTHONPATH"
fi

# ── Model repo env vars (set by install_models_v3.sh, re-export for SLURM) ──
REPOS_DIR="${MODEL_REPOS_DIR:-$HOME/.local/share/tothinkvision/repos}"
[ -d "$REPOS_DIR/co-tracker" ] && export COTRACKER_REPO="$REPOS_DIR/co-tracker"
[ -d "$REPOS_DIR/ObjectGS" ] && export OBJECT_GS_PATH="$REPOS_DIR/ObjectGS"
[ -d "$REPOS_DIR/Spann3R" ] && export SPANN3R_PATH="$REPOS_DIR/Spann3R"
[ -d "$REPOS_DIR/shape-of-motion" ] && export SHAPE_OF_MOTION_PATH="$REPOS_DIR/shape-of-motion"

echo ""
echo "── Model repos ──"
echo "COTRACKER_REPO=${COTRACKER_REPO:-(not set)}"
echo "OBJECT_GS_PATH=${OBJECT_GS_PATH:-(not set)}"
echo "SPANN3R_PATH=${SPANN3R_PATH:-(not set)}"
echo "SHAPE_OF_MOTION_PATH=${SHAPE_OF_MOTION_PATH:-(not set)}"

# ── Check model weights ──
echo ""
echo "── Checking model weights ──"
CACHE_DIR="${TTV_MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
echo "Model cache: $CACHE_DIR"
if [ -d "$CACHE_DIR" ]; then
    echo "Cache contents:"
    ls -lh "$CACHE_DIR" 2>/dev/null | head -20
else
    echo "WARNING: No model cache found. Pipeline will raise if models are needed."
fi

# ── Run real inference ──
echo ""
echo "── Starting real model inference ──"
python -c "
import time, os, sys
from pathlib import Path

from app.pipeline import process_file
from app.schemas import PipelineConfig
from app.exporters.psd_exporter import PSDExporter
from app.schemas import ExportFormat

# ── Test 1: Image — 2D path (detect + segment + LaMa completion + PSD) ──
if os.path.exists('test_input.png'):
    print('='*60)
    print('TEST 1: Image — 2D path — test_input.png')
    print('='*60)
    start = time.time()

    # 2D-only config: skip depth, 3D reconstruction, mesh
    config_2d = PipelineConfig(
        enable_sam3=True,
        enable_grounding_dino=True,
        enable_omniparser=False,
        enable_depth_pro=False,
        enable_mast3r=False,
        enable_3d_reconstruction=False,
        enable_gaussian_splatting=False,
        enable_completion_2d=True,   # LaMa inpainting for occluded regions
        enable_completion_3d=False,
        mode='general',
    )
    result = process_file(Path('test_input.png'), mode='image', config=config_2d)
    elapsed = time.time() - start

    print(f'Duration: {elapsed:.1f}s')
    print(f'Objects: {len(result.objects)}')
    for obj in result.objects:
        print(f'  [{obj.id}] {obj.label}')
        print(f'    bbox: {obj.bbox}')
        if obj.mask_base64:
            print(f'    mask: yes (base64, {len(obj.mask_base64)} chars)')
        if obj.crop_png_path:
            print(f'    crop_png: {obj.crop_png_path}')
        elif obj.crop_image_base64:
            print(f'    crop: yes (base64, {len(obj.crop_image_base64)} chars)')
        if obj.temporal and obj.temporal.trajectory:
            print(f'    trajectory: {len(obj.temporal.trajectory)} frames')

    print()
    print('── Exports ──')

    print('Exporting PSD (static, each object = layer)...')
    t0 = time.time()
    p = PSDExporter(fmt=ExportFormat.PSD_STATIC).export(result)
    print(f'  -> {p} ({time.time()-t0:.1f}s)')

    print(f'\\nImage 2D test DONE in {elapsed:.1f}s')

    # Clean up models from Test 1 before Test 2
    from app.pipeline import cleanup_all_models
    print('\\nCleaning up GPU memory after Test 1...')
    cleanup_all_models()
else:
    print('SKIPPED: test_input.png not found')

# ── Test 2: Video — 3D path (full pipeline) ──
if os.path.exists('test_input.mp4'):
    print()
    print('='*60)
    print('TEST 2: Video — 3D path — test_input.mp4')
    print('='*60)
    from app.exporters.gltf_exporter import GltfExporter
    from app.exporters.obj_exporter import ObjExporter
    from app.exporters.game_exporter import GameExporter

    start = time.time()
    result = process_file(Path('test_input.mp4'), mode='video')
    elapsed = time.time() - start

    print(f'Duration: {elapsed:.1f}s')
    print(f'Objects: {len(result.objects)}')
    print(f'Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} points')
    print(f'Camera poses: {len(result.camera_poses)} frames')

    for obj in result.objects:
        print(f'  [{obj.id}] {obj.label}')
        if obj.mesh_3d:
            print(f'    mesh: {len(obj.mesh_3d.vertices)}v / {len(obj.mesh_3d.faces)}f / texture={obj.mesh_3d.texture_path}')
        if obj.temporal and obj.temporal.trajectory:
            print(f'    trajectory: {len(obj.temporal.trajectory)} frames')

    print()
    print('── Exports ──')
    print('Exporting glTF...')
    p = GltfExporter().export(result)
    if p:
        print(f'  -> {p}')
    else:
        print('  (skipped: no 3D data)')
    print('Exporting OBJ+MTL...')
    p = ObjExporter().export(result)
    if p:
        print(f'  -> {p}')
    else:
        print('  (skipped: no 3D data)')

    # Also export PSD for video (each frame = group, each object = layer)
    print('Exporting PSD (animated, frame groups)...')
    p = PSDExporter(fmt=ExportFormat.PSD_ANIMATED).export(result)
    print(f'  -> {p}')

    print(f'\\nVideo 3D test DONE in {elapsed:.1f}s')

    # Clean up models from Test 2 before Test 3
    from app.pipeline import cleanup_all_models
    print('\\nCleaning up GPU memory after Test 2...')
    cleanup_all_models()
else:
    print()
    print('SKIPPED: test_input.mp4 not found')

# ── Test 3: Video — 4D path (full pipeline + 4D trajectory + animated export) ──
if os.path.exists('test_input.mp4'):
    print()
    print('='*60)
    print('TEST 3: Video — 4D path — test_input.mp4')
    print('  Pipeline: detect → segment → track → depth → 3D recon → mesh')
    print('         → CoTracker3 → 4D trajectory → scene graph → animated export')
    print('  Output:   animated glTF + USD + Blender + scene graph JSON')
    print('='*60)

    from app.exporters.animated_gltf_exporter import AnimatedGLTFExporter
    from app.exporters.usd_exporter import USDExporter
    from app.exporters.blender_exporter import BlenderExporter

    start = time.time()

    # Full 4D config: everything enabled
    config_4d = PipelineConfig(
        # ── Perception ──
        enable_sam3=True,
        enable_grounding_dino=True,
        enable_omniparser=False,
        enable_strongsort=True,
        # ── Depth + 3D reconstruction ──
        enable_depth_pro=True,
        enable_mast3r=True,
        enable_3d_reconstruction=True,
        enable_spann3r=True,            # Spann3R first, fallback MASt3R/VGGT
        # ── Mesh + splatting ──
        enable_gaussian_splatting=True,
        enable_objectgs=True,           # Per-object 3DGS
        enable_completion_2d=True,
        enable_completion_3d=True,
        # ── 4D stages ──
        enable_cotracker3=True,         # Dense point tracking
        enable_shape_of_motion=False,   # End-to-end 4D (optional, heavy)
        enable_4d_trajectory=True,      # Per-object 6DoF trajectories
        enable_4dgs=False,              # 4D Gaussian Splatting (very heavy, multi-GPU)
        enable_scene_graph=True,        # Dynamic scene graph
        enable_animated_export=True,    # Animated glTF / USD / Blender
        # ── Tuning ──
        trajectory_smoothing=0.5,
        icp_distance_threshold=0.05,
        deformation_threshold=0.3,
        mode='general',
    )

    result = process_file(Path('test_input.mp4'), mode='video', config=config_4d)
    elapsed = time.time() - start

    print(f'Duration: {elapsed:.1f}s')
    print(f'Objects: {len(result.objects)}')
    print(f'Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} points')
    print(f'Camera poses: {len(result.camera_poses)} frames')
    print(f'Model versions: {result.model_versions}')
    print()

    # ── Per-object details ──
    print('── Objects ──')
    for obj in result.objects:
        print(f'  [{obj.id}] {obj.label}')
        if obj.mesh_3d:
            print(f'    mesh: {len(obj.mesh_3d.vertices)}v / {len(obj.mesh_3d.faces)}f')
        if obj.trajectory_4d:
            t = obj.trajectory_4d
            print(f'    4D trajectory:')
            print(f'      motion_type: {t.motion_type}')
            print(f'      keyframes: {len(t.keyframes)}')
            print(f'      total_distance: {t.total_distance:.4f} m')
            print(f'      max_speed: {t.max_speed:.4f} m/s')
            if t.keyframes:
                kf0 = t.keyframes[0]
                kfn = t.keyframes[-1]
                print(f'      first: pos={kf0.position} rot={kf0.rotation}')
                print(f'      last:  pos={kfn.position} rot={kfn.rotation}')
        if obj.temporal and obj.temporal.trajectory:
            print(f'    2D trajectory: {len(obj.temporal.trajectory)} frames')

    # ── Scene graph ──
    print()
    print('── Scene Graph ──')
    if result.scene_graph_4d:
        sg = result.scene_graph_4d
        print(f'  Nodes: {len(sg.nodes)}')
        print(f'  Edges: {len(sg.edges)}')
        print(f'  Time range: {sg.time_range}')
        if sg.interaction_events:
            print(f'  Interactions: {len(sg.interaction_events)}')
            for evt in sg.interaction_events[:5]:
                print(f'    {evt}')
        for edge in sg.edges[:10]:
            print(f'  {edge.source_id} → {edge.relation} → {edge.target_id}  '
                  f'(t={edge.time_range}, conf={edge.confidence:.2f})')
    else:
        print('  (no scene graph built)')

    # ── Animated exports ──
    print()
    print('── Animated Exports ──')
    if result.animated_gltf_path:
        p = Path(result.animated_gltf_path)
        sz = p.stat().st_size if p.exists() else 0
        print(f'  Animated glTF: {p} ({sz/1024:.1f} KB)')
    else:
        print('  Animated glTF: (not generated)')

    if result.usd_path:
        p = Path(result.usd_path)
        sz = p.stat().st_size if p.exists() else 0
        print(f'  USD scene:     {p} ({sz/1024:.1f} KB)')
    else:
        print('  USD scene:     (not generated)')

    if result.blend_path:
        p = Path(result.blend_path)
        sz = p.stat().st_size if p.exists() else 0
        print(f'  Blender scene: {p} ({sz/1024:.1f} KB)')
    else:
        print('  Blender scene: (not generated)')

    if result.scene_graph_json_path:
        p = Path(result.scene_graph_json_path)
        sz = p.stat().st_size if p.exists() else 0
        print(f'  Scene graph:   {p} ({sz/1024:.1f} KB)')
    else:
        print('  Scene graph:   (not generated)')

    # ── Summary ──
    n_traj = sum(1 for o in result.objects if o.trajectory_4d is not None)
    n_mesh = sum(1 for o in result.objects if o.mesh_3d is not None)
    n_export = sum(1 for p in [result.animated_gltf_path, result.usd_path,
                                result.blend_path, result.scene_graph_json_path] if p)
    print()
    print(f'  Summary: {n_mesh}/{len(result.objects)} objects with mesh, '
          f'{n_traj} with 4D trajectory, {n_export}/4 exports')
    print(f'\\nVideo 4D test DONE in {elapsed:.1f}s')
else:
    print()
    print('SKIPPED: test_input.mp4 not found (needed for 4D test)')

print()
print('='*60)
print('ALL TESTS COMPLETE')
print('='*60)
"

# ── Show outputs ──
echo ""
echo "── Output files ──"
find outputs/ -type f -exec ls -lh {} \; 2>/dev/null || echo "No outputs found"

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Log: ttv_real_${SLURM_JOB_ID}.out"
