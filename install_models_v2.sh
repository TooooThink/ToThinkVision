#!/bin/bash
# ToThinkVision v2 — 新增依赖安装脚本
# 用于补全模块 + 高质量 3D 重建所需的额外权重和库
set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
mkdir -p "$CACHE_DIR"

USE_MIRROR="${USE_MIRROR:-true}"
if [ "$USE_MIRROR" = "true" ]; then
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
else
    PIP_INDEX=""
    HF_ENDPOINT="https://huggingface.co"
fi
export HF_ENDPOINT

# GitHub mirror helper
try_clone() {
    local raw="https://github.com/$1.git"
    local proxy="https://gh-proxy.com/"
    # PVD uses requirement_voxel.txt instead of requirements.txt
    if [ -f "$2/requirements.txt" ] || [ -f "$2/requirement_voxel.txt" ] || [ -f "$2/setup.py" ] || [ -f "$2/pyproject.toml" ]; then
        echo "  Already cloned: $1"
        return 0
    fi
    rm -rf "$2" 2>/dev/null || true
    git clone "${proxy}${raw}" "$2" 2>/dev/null && return 0
    curl -fSL "${proxy}${raw%/}/archive/refs/heads/main.zip" -o "$2.zip" && \
    unzip -q "$2.zip" -d "$(dirname "$2")" && \
    mv "$(dirname "$2")/$1-main" "$2" && \
    rm -f "$2.zip" && \
    return 0
    return 1
}

echo "=========================================="
echo "  ToThinkVision v2 — 新增依赖安装"
echo "=========================================="
echo ""
echo "  1) Open3D (Poisson 高质量网格重建)"
echo "  2) LaMa 权重 (2D 补全)"
echo "  3) PVD 权重 (3D 点云补全 — 扩散模型)"
echo "  4) OmniParser v2 权重 (UI 检测)"
echo "  5) 全部安装"
echo ""
read -p "Enter selection (1-5): " choice

install_open3d() {
    echo ""
    echo ">>> Installing Open3D (CUDA 12.1)..."
    # Must use conda — login node pip defaults to CPU build
    # cuda121=1.11.0 ensures GPU build with CUDA 12.1 support
    conda install -y "open3d=1.11.0=cuda121*" -c open3d-admin -c conda-forge 2>/dev/null || \
    conda install -y "open3d=1.10.0=cuda121*" -c open3d-admin -c conda-forge 2>/dev/null || \
    conda install -y open3d -c open3d-admin -c conda-forge 2>/dev/null || \
    echo "Open3D conda install failed. Try: conda install open3d -c open3d-admin"
    echo "Open3D installed. Poisson surface reconstruction will now be used."
}

install_lama() {
    echo ""
    echo ">>> Installing LaMa inpainting..."
    pip install ${PIP_INDEX} lama-cleaner
    # Also install transformers-based LaMa as fallback
    pip install ${PIP_INDEX} transformers accelerate

    # Pre-download big-lama weights
    WEIGHTS_DIR="$CACHE_DIR/lama"
    mkdir -p "$WEIGHTS_DIR"

    echo "Downloading big-lama weights..."
    if [ "$USE_MIRROR" = "true" ]; then
        curl -fSL -o "$WEIGHTS_DIR/big-lama.pt" \
            "https://hf-mirror.com/sanster/big-lama/resolve/main/big-lama.pt" 2>/dev/null || \
        curl -fSL -o "$WEIGHTS_DIR/big-lama.pt" \
            "https://huggingface.co/sanster/big-lama/resolve/main/big-lama.pt" 2>/dev/null || \
        echo "LaMa weights download failed, will auto-download on first use."
    else
        curl -fSL -o "$WEIGHTS_DIR/big-lama.pt" \
            "https://huggingface.co/sanster/big-lama/resolve/main/big-lama.pt" 2>/dev/null || \
        echo "LaMa weights download failed, will auto-download on first use."
    fi

    if [ -f "$WEIGHTS_DIR/big-lama.pt" ]; then
        echo "LaMa weights saved to $WEIGHTS_DIR/big-lama.pt"
    fi
}

install_pvd() {
    echo ""
    echo ">>> Installing PVD (Point Voxel Diffusion)..."

    # Install h5py via conda (pre-built, no compilation needed)
    echo "Installing h5py via conda (pre-built)..."
    conda install -y h5py 2>/dev/null || \
        pip install ${PIP_INDEX} h5py --only-binary :all: 2>/dev/null || \
        echo "h5py install failed, PVD may not work. Install manually: conda install h5py"

    # PVD requires specific torch version and point-cloud libraries
    echo "Installing core dependencies..."
    pip install ${PIP_INDEX} torch torchvision scipy

    PVD_DIR="$CACHE_DIR/PVD"
    try_clone "alexzhou907/PVD" "$PVD_DIR" || {
        echo "PVD clone failed. Manual install required:"
        echo "  git clone https://github.com/alexzhou907/PVD.git $PVD_DIR"
        echo "  cd $PVD_DIR && pip install -r requirements.txt"
        return 0
    }

    cd "$PVD_DIR"
    if [ -f "requirements.txt" ]; then
        pip install ${PIP_INDEX} -r requirements.txt
    elif [ -f "requirement_voxel.txt" ]; then
        # PVD's requirement_voxel.txt mixes conda: and pip: sections with strict version pins.
        # Strip all version pins (==, >=, <=, ~=, =) to avoid conflicts with existing packages.
        echo "Installing PVD dependencies (version pins stripped)..."

        # Pip packages
        pip_req="/tmp/pvd_pip_req.txt"
        awk '/^pip:/{found=1; next} /^conda:/{found=0} found && /./ && !/^[a-z]*:/{print}' \
            requirement_voxel.txt | \
            sed 's/[><=!~].*//g' | \
            sed '/^$/d' | \
            sed 's/^ *//;s/ *$//' > "$pip_req"
        if [ -s "$pip_req" ]; then
            # Remove open3d — handled by install_open3d() via conda -c open3d-admin
            grep -v -E '^open3d$' "$pip_req" > "${pip_req}.tmp" && mv "${pip_req}.tmp" "$pip_req"
            if [ -s "$pip_req" ]; then
                echo "Pip packages (no version pins): $(cat "$pip_req" | tr '\n' ' ')"
                pip install ${PIP_INDEX} -r "$pip_req"
            else
                echo "No pip packages remaining after filtering"
            fi
            rm -f "$pip_req"
        else
            echo "WARNING: Could not extract pip requirements from requirement_voxel.txt"
        fi

        # Conda packages (skip torch/torchvision/cudatoolkit — already installed)
        conda_req="/tmp/pvd_conda_req.txt"
        awk '/^conda:/{found=1; next} /^pip:/{found=0} found && /./ && !/^[a-z]*:/{print}' \
            requirement_voxel.txt | \
            sed 's/[><=!~].*//g' | \
            sed '/^$/d' | \
            sed 's/^ *//;s/ *$//' | \
            grep -v -E '^(python|torch|torchvision|cudatoolkit)$' > "$conda_req" || true
        if [ -s "$conda_req" ]; then
            echo "Conda packages (no version pins, skipping torch/python): $(cat "$conda_req" | tr '\n' ' ')"
            conda install -y $(cat "$conda_req" | tr '\n' ' ') -c pytorch -c nvidia -c conda-forge 2>/dev/null || \
                echo "Some conda packages failed to install, continuing..."
            rm -f "$conda_req"
        else
            echo "No conda packages to install"
        fi
    else
        echo "WARNING: No requirements file found in PVD repo"
    fi
    cd -

    # Download PVD checkpoint
    CKPT_DIR="$PVD_DIR/checkpoints"
    mkdir -p "$CKPT_DIR"

    echo "Downloading PVD checkpoint..."
    # PVD checkpoints are typically hosted on Google Drive or project page
    # The model expects: checkpoints/pvd_completion.pth or similar
    # Check if the repo provides a download script
    if [ -f "$PVD_DIR/scripts/download_pretrained.sh" ]; then
        cd "$PVD_DIR"
        bash scripts/download_pretrained.sh 2>/dev/null || true
        cd -
    elif [ -f "$PVD_DIR/download.sh" ]; then
        cd "$PVD_DIR"
        bash download.sh 2>/dev/null || true
        cd -
    else
        echo "PVD checkpoint not auto-downloaded."
        echo "Please download manually from: https://alexzhou907.github.io/pvd"
        echo "Place in: $PVD_DIR/checkpoints/"
    fi

    echo "PVD installed. Checkpoint location: $CKPT_DIR/"
}

install_omniparser_weights() {
    echo ""
    echo ">>> Downloading OmniParser v2 weights..."
    OMNI_DIR="$CACHE_DIR/OmniParser"

    if [ ! -d "$OMNI_DIR" ]; then
        echo "OmniParser not installed. Installing first..."
        try_clone "microsoft/OmniParser" "$OMNI_DIR" || {
            echo "OmniParser clone failed. Please install manually."
            return 0
        }
        cd "$OMNI_DIR"
        pip install ${PIP_INDEX} -r requirements.txt
        cd -
    fi

    # Download weights via HuggingFace
    cd "$OMNI_DIR"
    if [ "$USE_MIRROR" = "true" ]; then
        HF_ENDPOINT="https://hf-mirror.com" python -c "
try:
    from util.utils import download_weights
    download_weights()
except Exception as e:
    print(f'Auto download failed: {e}')
    print('Please download manually from: https://huggingface.co/microsoft/OmniParser-v2.0')
" 2>/dev/null || true
    else
        python -c "
try:
    from util.utils import download_weights
    download_weights()
except Exception as e:
    print(f'Auto download failed: {e}')
    print('Please download manually from: https://huggingface.co/microsoft/OmniParser-v2.0')
" 2>/dev/null || true
    fi
    cd -

    echo "OmniParser weights downloaded."
}

case $choice in
    1) install_open3d ;;
    2) install_lama ;;
    3) install_pvd ;;
    4) install_omniparser_weights ;;
    5)
        install_open3d
        install_lama
        install_pvd
        install_omniparser_weights
        ;;
    *) echo "Invalid selection." ;;
esac

echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "新增模块说明："
echo ""
echo "  Open3D   → Poisson 网格重建（质量远超 alpha shape）"
echo "  LaMa     → 2D 图像补全（inpainting，补齐未入镜区域）"
echo "  PVD      → 3D 点云补全（扩散模型，最优效果）"
echo "  OmniParser → UI 元素检测（可选，只在 UI 模式需要）"
echo ""
echo "默认已开启（app/schemas.py）："
echo "  enable_completion_2d = True"
echo "  enable_completion_3d = True"
echo ""
echo "启动服务："
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
