#!/bin/bash
#SBATCH --job-name=test_gdino
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=test_gdino_%j.out
#SBATCH --error=test_gdino_%j.err

export CUDA_HOME=/public/software/compiler/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export LD_LIBRARY_PATH=/public/home/xlwang/jyy/anaconda/envs/ttv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /public/home/xlwang/genalyu/ToThinkVision

# 测试 GroundingDINO 加载和检测
python -c "
from app.models.grounding_dino import GroundingDINO
import numpy as np
print('Loading GroundingDINO...')
detector = GroundingDINO()
print('Backend:', detector._backend)
print('Testing detection with random image...')
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
results = detector.detect(img, 'general')
print('Detections:', len(results))
print('GroundingDINO OK!')
"
