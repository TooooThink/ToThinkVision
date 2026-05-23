#!/bin/bash
# ToThinkVision v2 — Interactive Model Installer
set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
mkdir -p "$CACHE_DIR"

# ── Git mirror detection (China-friendly) ──
GIT_PREFIX=""
try_mirror() {
    local repo=$1
    # Try direct first
    git ls-remote --quiet "https://github.com/$repo.git" 2>/dev/null && { echo "github"; return; }
    # Try gitee mirror
    git ls-remote --quiet "https://gitee.com/mirrors/$repo.git" 2>/dev/null && { echo "gitee"; return; }
    # Try ghproxy
    git ls-remote --quiet "https://ghproxy.com/https://github.com/$repo.git" 2>/dev/null && { echo "ghproxy"; return; }
    echo "none"
}

clone_with_mirrors() {
    local repo=$1
    local dest=$2
    local mirror
    mirror=$(try_mirror "$repo")
    echo "  Git mirror: $mirror"
    case $mirror in
        github)  git clone --quiet "https://github.com/$repo.git" "$dest" ;;
        gitee)   git clone --quiet "https://gitee.com/mirrors/$repo.git" "$dest" ;;
        ghproxy) git clone --quiet "https://ghproxy.com/https://github.com/$repo.git" "$dest" ;;
        *)
            echo "  ERROR: Cannot clone $repo, all mirrors failed."
            return 1
            ;;
    esac
}

pip_install_git() {
    local repo=$1
    local fallback=$2
    local mirror
    mirror=$(try_mirror "$repo")
    echo "  Git mirror: $mirror"
    case $mirror in
        github)
            pip install "git+https://github.com/$repo.git" || { [ -n "$fallback" ] && echo "$fallback" && return 1; return 1; }
            ;;
        gitee)
            pip install "git+https://gitee.com/mirrors/$repo.git" || { [ -n "$fallback" ] && echo "$fallback" && return 1; return 1; }
            ;;
        ghproxy)
            pip install "git+https://ghproxy.com/https://github.com/$repo.git" || { [ -n "$fallback" ] && echo "$fallback" && return 1; return 1; }
            ;;
        *)
            [ -n "$fallback" ] && echo "$fallback"
            return 1
            ;;
    esac
}

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

# ── pip index URL (China) ──
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

install_sam3() {
    echo ""
    echo ">>> Installing SAM 3..."
    pip_install_git "facebookresearch/sam3" "  SAM 3 failed, trying SAM 2 fallback..."
    if [ $? -ne 0 ]; then
        pip_install_git "facebookresearch/sam2" "  SAM 2 also failed. Install manually later."
    fi
    echo "Downloading SAM checkpoint..."
    wget -nc -P "$CACHE_DIR" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" 2>/dev/null || \
    curl -L -o "$CACHE_DIR/sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" || \
    echo "  Download checkpoint manually: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "SAM checkpoint saved to $CACHE_DIR"
}

install_omniparser() {
    echo ""
    echo ">>> Installing OmniParser v2..."
    clone_with_mirrors "microsoft/OmniParser" "$CACHE_DIR/OmniParser" 2>/dev/null || \
    echo "  OmniParser clone failed. Download weights from: https://huggingface.co/microsoft/OmniParser-v2.0"
    cd "$CACHE_DIR/OmniParser" && pip install --index-url "$PIP_INDEX" -r requirements.txt 2>/dev/null && cd -
    python -c "from util.utils import download_weights; download_weights()" 2>/dev/null || \
    echo "  Please download OmniParser weights from: https://huggingface.co/microsoft/OmniParser-v2.0"
}

install_dino_x() {
    echo ""
    echo ">>> Installing DINO-X..."
    pip_install_git "IDEA-Research/dino-x-api" "  DINO-X failed, trying Grounding DINO fallback..."
    if [ $? -ne 0 ]; then
        echo "  Installing Grounding DINO fallback..."
        clone_with_mirrors "IDEA-Research/GroundingDINO" "$CACHE_DIR/GroundingDINO" 2>/dev/null || true
        cd "$CACHE_DIR/GroundingDINO" && pip install --index-url "$PIP_INDEX" -e . 2>/dev/null && cd -
    fi
}

install_botsort() {
    echo ""
    echo ">>> Installing BoT-SORT..."
    pip install --index-url "$PIP_INDEX" ultralytics 2>/dev/null || \
    pip install --index-url "$PIP_INDEX" botsort 2>/dev/null || \
    echo "  Please install ultralytics: pip install ultralytics"
    echo "BoT-SORT installed (via ultralytics trackers)"
}

install_depth_pro() {
    echo ""
    echo ">>> Installing Depth Pro..."
    clone_with_mirrors "apple/ml-depth-pro" "$CACHE_DIR/ml-depth-pro" 2>/dev/null || \
    echo "  Depth Pro clone failed. Weights auto-download from HuggingFace on first use."
    cd "$CACHE_DIR/ml-depth-pro" && pip install --index-url "$PIP_INDEX" -e . 2>/dev/null && cd -
    echo "Depth Pro installed. Weights will be downloaded on first use from HuggingFace."
}

install_vggt() {
    echo ""
    echo ">>> Installing VGGT..."
    pip_install_git "facebookresearch/vggt" "  VGGT repo install failed."
    if [ $? -ne 0 ]; then
        echo "  Using HuggingFace transformers (auto-downloads meta/VGGT on first run)..."
    fi
    pip install --index-url "$PIP_INDEX" "transformers>=4.47.0"
}

install_3dgs() {
    echo ""
    echo ">>> Installing 3D Gaussian Splatting..."
    pip install --index-url "$PIP_INDEX" gsplat nerfstudio plyfile
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
