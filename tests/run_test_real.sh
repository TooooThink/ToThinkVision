#!/bin/bash
#SBATCH --job-name=ttv_real
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ttv_real_%j.out
#SBATCH --error=ttv_real_%j.err

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

os.environ['MOCK_MODE'] = 'false'

from app.pipeline import process_file
from app.exporters.gltf_exporter import GltfExporter
from app.exporters.obj_exporter import ObjExporter
from app.exporters.game_exporter import GameExporter

# ── Test 1: Image ──
if os.path.exists('test_input.png'):
    print('='*60)
    print('TEST 1: Image — test_input.png')
    print('='*60)
    start = time.time()
    result = process_file('test_input.png', mode='image')
    elapsed = time.time() - start

    print(f'Duration: {elapsed:.1f}s')
    print(f'Objects: {len(result.objects)}')
    print(f'Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} points')
    print(f'Camera poses: {len(result.camera_poses)} frames')
    print(f'Gaussian splats: {\"yes\" if result.gaussian_splats else \"no\"}')

    for obj in result.objects:
        print(f'  [{obj.id}] {obj.label}')
        print(f'    bbox_2d: {obj.bbox_2d}')
        if obj.mesh_3d:
            print(f'    mesh: {obj.mesh_3d.vertex_count} verts, {obj.mesh_3d.face_count} faces')
            print(f'    UV coords: {\"yes\" if obj.mesh_3d.uv_coords else \"no\"}')
            print(f'    texture: {obj.mesh_3d.texture_path}')
            print(f'    texture size: {obj.mesh_3d.texture_size}')
            print(f'    bounds: {obj.mesh_3d.bounds}')
        if obj.temporal_data:
            print(f'    trajectory: {len(obj.temporal_data)} frames')

    print()
    print('── Exports ──')

    print('Exporting glTF...')
    t0 = time.time()
    p = GltfExporter().export(result)
    print(f'  -> {p} ({time.time()-t0:.1f}s)')

    print('Exporting OBJ+MTL...')
    t0 = time.time()
    p = ObjExporter().export(result)
    print(f'  -> {p} ({time.time()-t0:.1f}s)')

    print('Exporting Unity JSON...')
    t0 = time.time()
    p = GameExporter().export_unity(result)
    print(f'  -> {p} ({time.time()-t0:.1f}s)')

    print(f'\\nImage test DONE in {elapsed:.1f}s')
else:
    print('SKIPPED: test_input.png not found')

# ── Test 2: Video ──
if os.path.exists('test_input.mp4'):
    print()
    print('='*60)
    print('TEST 2: Video — test_input.mp4')
    print('='*60)
    start = time.time()
    result = process_file('test_input.mp4', mode='video')
    elapsed = time.time() - start

    print(f'Duration: {elapsed:.1f}s')
    print(f'Objects: {len(result.objects)}')
    print(f'Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} points')
    print(f'Camera poses: {len(result.camera_poses)} frames')

    for obj in result.objects:
        print(f'  [{obj.id}] {obj.label}')
        if obj.mesh_3d:
            print(f'    mesh: {obj.mesh_3d.vertex_count}v / {obj.mesh_3d.face_count}f / texture={obj.mesh_3d.texture_path}')
        if obj.temporal_data:
            print(f'    trajectory: {len(obj.temporal_data)} frames')

    print()
    print('── Exports ──')
    print('Exporting glTF...')
    p = GltfExporter().export(result)
    print(f'  -> {p}')
    print('Exporting OBJ+MTL...')
    p = ObjExporter().export(result)
    print(f'  -> {p}')

    print(f'\\nVideo test DONE in {elapsed:.1f}s')
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
