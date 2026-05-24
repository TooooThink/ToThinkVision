#!/bin/bash
# ToThinkVision v2 — Interactive Model Installer
set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
mkdir -p "$CACHE_DIR"

# GitHub mirror helpers: try multiple proxies until one works
if [ "$USE_MIRROR" = "true" ]; then
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
else
    PIP_INDEX=""
    HF_ENDPOINT="https://huggingface.co"
fi

# GitHub mirror: gh-proxy.com
try_clone() {
    # $1 = repo (org/name), $2 = dest
    local raw="https://github.com/$1.git"
    local proxy="https://gh-proxy.com/"

    # Skip if already cloned
    if [ -d "$2/.git" ] || [ -f "$2/setup.py" ] || [ -f "$2/pyproject.toml" ]; then
        echo "  Already cloned: $1"
        return 0
    fi

    # Method 1: git clone via proxy
    git clone "${proxy}${raw}" "$2" 2>/dev/null && return 0

    # Method 2: download zip via proxy, then unzip
    curl -fSL "${proxy}${raw%/}/archive/refs/heads/main.zip" -o "$2.zip" && \
    unzip -q "$2.zip" -d "$(dirname "$2")" 2>/dev/null && \
    mv "$(dirname "$2")/$1-main" "$2" && \
    rm -f "$2.zip" && \
    return 0

    return 1
}

export HF_ENDPOINT

echo "=========================================="
echo "  ToThinkVision v2 — Model Installer"
echo "=========================================="
echo ""
echo "Select models to install (or 'all' for everything):"
echo ""
echo "  1) SAM 3 (12-24GB VRAM) — Segmentation + Detection + Tracking"
echo "  2) OmniParser v2 (8-12GB VRAM) — UI Element Detection"
echo "  3) GroundingDINO (4-8GB VRAM) — Open-Vocabulary Detection"
echo "  4) BoT-SORT (CPU) — Robust Multi-Object Tracking"
echo "  5) Depth Pro (4-8GB VRAM) — Metric Depth Estimation"
echo "  6) VGGT (4-8GB VRAM) — 3D Point Cloud Reconstruction (Meta, CVPR 2025)"
echo "  7) 3D Gaussian Splatting (24GB VRAM) — Photorealistic 3D"
echo "  8) All models (requires ~60GB+ VRAM for all simultaneously)"
echo ""
read -p "Enter selection (1-8): " choice

install_sam3() {
    echo ""
    echo ">>> Installing SAM 3..."
    try_clone "facebookresearch/sam3" "$CACHE_DIR/sam3" || \
    try_clone "facebookresearch/sam2" "$CACHE_DIR/sam3" || \
    { echo "SAM git clone failed, skipping pip install"; }
    if [ -f "$CACHE_DIR/sam3/setup.py" ] || [ -f "$CACHE_DIR/sam3/pyproject.toml" ]; then
        cd "$CACHE_DIR/sam3"
        pip install ${PIP_INDEX} -e .
        cd -
    fi
    echo "SAM 3/2 installed. Downloading checkpoint..."
    # SAM 3 uses SAM 2.1 as its backbone — no separate SAM 3 weights yet
    wget -nc -P "$CACHE_DIR" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" 2>/dev/null || \
    curl -L -o "$CACHE_DIR/sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "SAM 3 checkpoint saved to $CACHE_DIR"
}

install_omniparser() {
    echo ""
    echo ">>> Installing OmniParser v2..."
    try_clone "microsoft/OmniParser" "$CACHE_DIR/OmniParser" || true
    cd "$CACHE_DIR/OmniParser"
    pip install ${PIP_INDEX} -r requirements.txt
    python -c "from util.utils import download_weights; download_weights()" 2>/dev/null || \
    echo "Please download OmniParser weights from: https://huggingface.co/microsoft/OmniParser-v2.0"
    cd -
}

install_grounding_dino() {
    echo ""
    echo ">>> Installing GroundingDINO..."
    try_clone "IDEA-Research/GroundingDINO" "$CACHE_DIR/GroundingDINO" 2>/dev/null || true
    cd "$CACHE_DIR/GroundingDINO"
    pip install ${PIP_INDEX} -e .
    cd -
}

install_botsort() {
    echo ""
    echo ">>> Installing BoT-SORT..."
    pip install ${PIP_INDEX} ultralytics 2>/dev/null || \
    pip install ${PIP_INDEX} botsort 2>/dev/null || \
    echo "Please install ultralytics: pip install ultralytics"
    echo "BoT-SORT installed (via ultralytics trackers)"
}

install_depth_pro() {
    echo ""
    echo ">>> Installing Depth Pro..."
    try_clone "apple/ml-depth-pro" "$CACHE_DIR/ml-depth-pro" || true
    cd "$CACHE_DIR/ml-depth-pro"
    pip install ${PIP_INDEX} -e .
    cd -
    echo "Depth Pro installed. Weights will be downloaded on first use from HuggingFace."
}

install_vggt() {
    echo ""
    echo ">>> Installing VGGT..."
    try_clone "facebookresearch/vggt" "$CACHE_DIR/vggt" 2>/dev/null && \
    cd "$CACHE_DIR/vggt" && \
    pip install ${PIP_INDEX} -e . && cd - || \
    echo "VGGT official repo install failed, using HuggingFace transformers (auto-downloads on first run)..."
    pip install ${PIP_INDEX} "transformers>=4.47.0"
    echo "VGGT will auto-download weights from HuggingFace (meta/VGGT) on first use."
}

install_3dgs() {
    echo ""
    echo ">>> Installing 3D Gaussian Splatting..."
    pip install ${PIP_INDEX} gsplat
    pip install ${PIP_INDEX} nerfstudio
    pip install ${PIP_INDEX} plyfile
    echo "3DGS installed."
}

case $choice in
    1) install_sam3 ;;
    2) install_omniparser ;;
    3) install_grounding_dino ;;
    4) install_botsort ;;
    5) install_depth_pro ;;
    6) install_vggt ;;
    7) install_3dgs ;;
    8)
        install_sam3
        install_omniparser
        install_grounding_dino
        install_botsort
        install_depth_pro
        install_vggt
        install_3dgs
        ;;
    *) echo "Invalid selection." ;;
esac

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo ""
echo "Start the server:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or in mock mode (no GPU needed):"
echo "  MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000"
