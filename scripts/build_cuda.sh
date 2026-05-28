#!/bin/bash
#SBATCH --job-name=compile_gdino
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/compile_gdino_%j.out
#SBATCH --error=logs/compile_gdino_%j.err

# Load CUDA
module load cuda/12.1
export CUDA_HOME=/public/software/compiler/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== Environment ==="
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version
/public/home/xlwang/jyy/anaconda/envs/ttv/bin/python -c "
import torch
print('PyTorch CUDA available:', torch.cuda.is_available())
print('PyTorch CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

echo ""
echo "=== Compiling GroundingDINO ==="
cd ~/.cache/tothinkvision/GroundingDINO
rm -rf build/ groundingdino/_C*.so

/public/home/xlwang/jyy/anaconda/envs/ttv/bin/python setup.py build_ext --inplace
echo "EXIT_CODE=$?"

echo ""
echo "=== Verifying ==="
ls -la groundingdino/_C*.so 2>/dev/null && echo "CUDA extension compiled!" || echo "FAILED: no .so found"

/public/home/xlwang/jyy/anaconda/envs/ttv/bin/python -c "
from groundingdino.util.inference import load_model
print('GroundingDINO import OK')