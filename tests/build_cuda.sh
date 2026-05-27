#!/bin/bash
#SBATCH --job-name=build_gdino
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=build_gdino_%j.out
#SBATCH --error=build_gdino_%j.err

export CUDA_HOME=/public/software/compiler/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

cd /public/home/xlwang/.cache/tothinkvision/GroundingDINO

# 只编译 C++ 扩展，不触发 pip 安装
python setup.py build_ext --inplace
echo "Build complete"
