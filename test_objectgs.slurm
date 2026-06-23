#!/bin/bash
#SBATCH --job-name=test_objectgs
#SBATCH --partition=a100
#SBATCH --output=test_objectgs_%j.out
#SBATCH --error=test_objectgs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Load conda environment
source ~/jyy/anaconda/etc/profile.d/conda.sh
conda activate ttv

# Set environment variables
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_VISIBLE_DEVICES=0

# Navigate to ObjectGS repo
cd ~/.local/share/tothinkvision/repos/ObjectGS

# Run training
echo "Starting ObjectGS training..."
bash train_3d.sh /public/home/xlwang/genalyu/ToThinkVision/outputs/2848682/objectgs_training/data

echo "Training completed with exit code: $?"
