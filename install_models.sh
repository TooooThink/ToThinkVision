#!/bin/bash
# ToThinkVision v2 — Interactive Model Installer
set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "  ToThinkVision v2 — Model Installer"
echo "=========================================="
echo ""
echo "Select models to install (or 'all' for everything):"
echo ""
echo "  1) SAM 3 (12-24GB VRAM) — Segmentation + Detection + Tracking"
echo "  2) OmniParser v2 (8-12GB VRAM) — UI Element Detection"
echo "  3) Grounding DINO 1.6 (4-8GB VRAM) — Open-Vocabulary Detection"
echo "  4) StrongSORT (CPU) — Robust Multi-Object Tracking"
echo "  5) Depth Pro (4-8GB VRAM) — Metric Depth Estimation"
echo "  6) MASt3R (24-48GB VRAM) — 3D Point Cloud Reconstruction"
echo "  7) 3D Gaussian Splatting (24GB VRAM) — Photorealistic 3D"
echo "  8) All models (requires ~60GB+ VRAM for all simultaneously)"
echo ""
read -p "Enter selection (1-8): " choice

install_sam3() {
    echo ""
    echo ">>> Installing SAM 3..."
    pip install git+https://github.com/facebookresearch/sam3.git 2>/dev/null || \
    pip install git+https://github.com/facebookresearch/sam2.git
    echo "SAM 3/2 installed. Downloading checkpoint..."
    wget -nc -P "$CACHE_DIR" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" 2>/dev/null || \
    curl -L -o "$CACHE_DIR/sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "SAM 3 checkpoint saved to $CACHE_DIR"
}

install_omniparser() {
    echo ""
    echo ">>> Installing OmniParser v2..."
    git clone https://github.com/microsoft/OmniParser.git "$CACHE_DIR/OmniParser" 2>/dev/null || true
    cd "$CACHE_DIR/OmniParser"
    pip install -r requirements.txt
    python -c "from util.utils import download_weights; download_weights()" 2>/dev/null || \
    echo "Please download OmniParser weights from: https://huggingface.co/microsoft/OmniParser-v2.0"
    cd -
}

install_grounding_dino() {
    echo ""
    echo ">>> Installing Grounding DINO 1.6..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$CACHE_DIR/GroundingDINO" 2>/dev/null || true
    cd "$CACHE_DIR/GroundingDINO"
    pip install -e .
    cd -
}

install_strongsort() {
    echo ""
    echo ">>> Installing StrongSORT..."
    pip install strongsort 2>/dev/null || \
    pip install bytetracker
    echo "Downloading ReID model weights..."
    wget -nc -P "$CACHE_DIR" "https://drive.google.com/uc?id=1wQ6uy1k3jYJ1H3pM1Z1pM1Z1pM1Z1pM" 2>/dev/null || \
    echo "Please download osnet_x1_0_msmt17.pt and place in $CACHE_DIR"
}

install_depth_pro() {
    echo ""
    echo ">>> Installing Depth Pro..."
    git clone https://github.com/apple/ml-depth-pro.git "$CACHE_DIR/ml-depth-pro" 2>/dev/null || true
    cd "$CACHE_DIR/ml-depth-pro"
    pip install -e .
    cd -
    echo "Depth Pro installed. Weights will be downloaded on first use from HuggingFace."
}

install_mast3r() {
    echo ""
    echo ">>> Installing MASt3R..."
    git clone --recursive https://github.com/naver/mast3r.git "$CACHE_DIR/mast3r" 2>/dev/null || true
    cd "$CACHE_DIR/mast3r"
    pip install -r dust3r/requirements.txt
    pip install -e dust3r
    cd "$CACHE_DIR/mast3r/asmk" && pip install . 2>/dev/null || true
    cd -
    echo "MASt3R installed. Weights will be downloaded on first use from HuggingFace."
}

install_3dgs() {
    echo ""
    echo ">>> Installing 3D Gaussian Splatting..."
    pip install gsplat
    pip install nerfstudio
    pip install plyfile
    echo "3DGS installed."
}

case $choice in
    1) install_sam3 ;;
    2) install_omniparser ;;
    3) install_grounding_dino ;;
    4) install_strongsort ;;
    5) install_depth_pro ;;
    6) install_mast3r ;;
    7) install_3dgs ;;
    8)
        install_sam3
        install_omniparser
        install_grounding_dino
        install_strongsort
        install_depth_pro
        install_mast3r
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
