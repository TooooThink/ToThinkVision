#!/bin/bash
#SBATCH --job-name=ttv_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=ttv_test_%j.out
#SBATCH --error=ttv_test_%j.err

set -e

echo "=== ToThinkVision Test Job ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

# ── Environment ──
echo ""
echo "── Setting up environment ──"

# Use system Python or create venv if needed
PYTHON=python3
if ! command -v $PYTHON &>/dev/null; then
    PYTHON=python
fi

# Create virtualenv if not already done
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

echo "Python: $(which python)"
echo "Packages OK"

# ── Step 1: Unit Tests (mock mode) ──
echo ""
echo "── Step 1: Running 51 unit tests (mock mode) ──"
MOCK_MODE=true pytest tests/ -v --tb=short 2>&1 | tail -20
echo "Unit tests done: $(date)"

# ── Step 2: Mock Mode End-to-End Pipeline ──
echo ""
echo "── Step 2: Mock mode end-to-end test ──"
python -c "
from app.pipeline import process_file
import json, os, time

print('Running mock pipeline...')
start = time.time()
result = process_file('test_input.png', mode='image')
elapsed = time.time() - start

print(f'  Objects detected: {len(result.objects)}')
print(f'  Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} points')
print(f'  Camera poses: {len(result.camera_poses)} frames')
print(f'  Gaussian splats: {\"yes\" if result.gaussian_splats else \"no\"}')
print(f'  Scene mesh: {result.scene_mesh_path or \"none\"}')
if result.objects:
    obj = result.objects[0]
    print(f'  First object: id={obj.id}, label={obj.label}')
    print(f'    bbox_2d: {obj.bbox_2d}')
    if obj.mesh_3d:
        print(f'    mesh: {obj.mesh_3d.vertex_count} verts, {obj.mesh_3d.face_count} faces')
        print(f'    texture: {obj.mesh_3d.texture_path}')
        print(f'    uv_coords: {\"yes\" if obj.mesh_3d.uv_coords else \"no\"}')

# Export a few formats
from app.exporters.gltf_exporter import GltfExporter
from app.exporters.obj_exporter import ObjExporter
from app.exporters.game_exporter import GameExporter

print()
print('Exporting glTF...')
gltf = GltfExporter()
p = gltf.export(result)
print(f'  -> {p}')

print('Exporting OBJ+MTL...')
obj_exp = ObjExporter()
p = obj_exp.export(result)
print(f'  -> {p}')

print('Exporting Unity JSON...')
game = GameExporter()
p = game.export_unity(result)
print(f'  -> {p}')

print(f'\\nTotal time: {elapsed:.1f}s')
print('Mock pipeline OK')
"

echo ""
echo "── Step 2 done: $(date) ──"

# ── Step 3: Real Model Test (if weights exist) ──
if [ -f "test_input.png" ]; then
    echo ""
    echo "── Step 3: Real model test with test_input.png ──"
    python -c "
from app.pipeline import process_file
import time

print('Running REAL pipeline on test_input.png...')
start = time.time()
result = process_file('test_input.png', mode='image')
elapsed = time.time() - start

print(f'  Objects: {len(result.objects)}')
print(f'  Point cloud: {len(result.point_cloud.points) if result.point_cloud else 0} pts')
print(f'  Time: {elapsed:.1f}s')

if result.objects:
    for obj in result.objects[:3]:
        print(f'  - {obj.id}: {obj.label} bbox={obj.bbox_2d}')
        if obj.mesh_3d:
            print(f'      mesh={obj.mesh_3d.vertex_count}v/{obj.mesh_3d.face_count}f texture={obj.mesh_3d.texture_path}')
print('Real pipeline OK')
"
    echo "── Step 3 done: $(date) ──"
else
    echo ""
    echo "── Step 3: SKIPPED (no test_input.png found) ──"
    echo "To test with real models, place a test image as test_input.png and re-run."
fi

# ── Summary ──
echo ""
echo "=== Test Summary ==="
echo "End: $(date)"
echo "Output dir: $(ls -la outputs/ 2>/dev/null || echo 'no outputs/')"
echo "Log: ttv_test_${SLURM_JOB_ID}.out"
