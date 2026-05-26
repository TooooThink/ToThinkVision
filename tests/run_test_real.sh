#!/bin/bash
#SBATCH --job-name=ttv_real
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ttv_real_%j.out
#SBATCH --error=logs/ttv_real_%j.err

# ============================================================================
# ToThinkVision — Real Model Inference Test
# ============================================================================
# Before sbatch:
#   1. Place your test file(s) in the project dir:
#      - test_input.png   (image test)
#      - test_input.mp4   (video test, optional)
#   2. Download model weights via install_models.sh or manually:
#      bash install_models.sh
#   3. Submit:
#      sbatch run_test_real.sh
# ============================================================================

set -e

mkdir -p logs outputs

# Force correct HF endpoint — hf-mirror.com doesn't resolve on compute nodes
HF_ENDPOINT="https://huggingface.co"
export HF_ENDPOINT

echo "=== ToThinkVision Real Model Inference ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

# ── Environment ──
echo ""
echo "── Activating conda environment ──"

# Init conda for non-interactive shell
eval "$(conda shell.bash hook 2>/dev/null)" || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true

conda activate ttv 2>/dev/null || conda activate base
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

# ── Check model weights ──
echo ""
echo "── Checking model weights ──"
CACHE_DIR="${TTV_MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
echo "Model cache: $CACHE_DIR"
if [ -d "$CACHE_DIR" ]; then
    echo "Cache contents:"
    ls -lh "$CACHE_DIR" 2>/dev/null | head -20
else
    echo "WARNING: No model cache found. Models will fallback to mock."
fi

# ── Run real inference ──
echo ""
echo "── Starting real model inference ──"
python -c "
import time, os, sys
from pathlib import Path

os.environ['MOCK_MODE'] = 'false'

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
    print(f'  -> {p}')
    print('Exporting OBJ+MTL...')
    p = ObjExporter().export(result)
    print(f'  -> {p}')

    # Also export PSD for video (each frame = group, each object = layer)
    print('Exporting PSD (animated, frame groups)...')
    p = PSDExporter(fmt=ExportFormat.PSD_ANIMATED).export(result)
    print(f'  -> {p}')

    print(f'\\nVideo 3D test DONE in {elapsed:.1f}s')
else:
    print()
    print('SKIPPED: test_input.mp4 not found')

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
