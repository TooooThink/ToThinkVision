#!/bin/bash
# ToThinkVision v2 — Interactive Model Installer
set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
mkdir -p "$CACHE_DIR"

# Mirror configuration (set to "true" to use mirrors)
USE_MIRROR="${USE_MIRROR:-true}"

if [ "$USE_MIRROR" = "true" ]; then
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    GH_MIRROR="https://gitclone.com/github.com/"
    HF_ENDPOINT="https://hf-mirror.com"
else
    PIP_INDEX=""
    GH_MIRROR="https://github.com/"
    HF_ENDPOINT="https://huggingface.co"
fi

export HF_ENDPOINT

echo "=========================================="
echo "  ToThinkVision v2 — Model Installer"
echo "=========================================="
echo ""
echo "Select models to install (or 'all' for everything):"
echo ""
echo "  1) SAM 3 (12-24GB VRAM) — Segmentation + Detection + Tracking"
echo "  2) OmniParser v2 (8-12GB VRAM) — UI Element Detection"
echo "  3) DINO-X (4-8GB VRAM) — Open-Vocabulary Detection (SOTA, 56.0 COCO AP)"
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
    git clone ${GH_MIRROR}facebookresearch/sam3.git "$CACHE_DIR/sam3" 2>/dev/null || \
    git clone ${GH_MIRROR}facebookresearch/sam2.git "$CACHE_DIR/sam3" 2>/dev/null || \
    { echo "SAM git clone failed, installing from PyPI fallback"; cd "$CACHE_DIR"; }
    if [ -f "$CACHE_DIR/sam3/setup.py" ] || [ -f "$CACHE_DIR/sam3/pyproject.toml" ]; then
        cd "$CACHE_DIR/sam3"
        pip install ${PIP_INDEX} -e .
        cd -
    fi
    echo "SAM 3/2 installed. Downloading checkpoint..."
    wget -nc -P "$CACHE_DIR" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" 2>/dev/null || \
    curl -L -o "$CACHE_DIR/sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "SAM 3 checkpoint saved to $CACHE_DIR"
}

install_omniparser() {
    echo ""
    echo ">>> Installing OmniParser v2..."
    git clone ${GH_MIRROR}microsoft/OmniParser.git "$CACHE_DIR/OmniParser" 2>/dev/null || true
    cd "$CACHE_DIR/OmniParser"
    pip install ${PIP_INDEX} -r requirements.txt
    python -c "from util.utils import download_weights; download_weights()" 2>/dev/null || \
    echo "Please download OmniParser weights from: https://huggingface.co/microsoft/OmniParser-v2.0"
    cd -
}

install_dino_x() {
    echo ""
    echo ">>> Installing DINO-X..."
    pip install ${PIP_INDEX} git+${GH_MIRROR}IDEA-Research/dino-x-api.git 2>/dev/null || \
    echo "DINO-X API install failed, installing Grounding DINO fallback..."
    git clone ${GH_MIRROR}IDEA-Research/GroundingDINO.git "$CACHE_DIR/GroundingDINO" 2>/dev/null || true
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
    git clone ${GH_MIRROR}apple/ml-depth-pro.git "$CACHE_DIR/ml-depth-pro" 2>/dev/null || true
    cd "$CACHE_DIR/ml-depth-pro"
    pip install ${PIP_INDEX} -e .
    cd -
    echo "Depth Pro installed. Weights will be downloaded on first use from HuggingFace."
}

install_vggt() {
    echo ""
    echo ">>> Installing VGGT..."
    pip install ${PIP_INDEX} git+${GH_MIRROR}facebookresearch/vggt.git 2>/dev/null || \
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
    3) install_dino_x ;;
    4) install_botsort ;;
    5) install_depth_pro ;;
    6) install_vggt ;;
    7) install_3dgs ;;
    8)
        install_sam3
        install_omniparser
        install_dino_x
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
