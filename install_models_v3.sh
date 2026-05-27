#!/bin/bash
# ToThinkVision v2.1 — Advanced 3D/4D Model Installer
#
# Installs the 4 new models added in v2.1:
#   1. CoTracker3   (Meta, weights only, torch.hub auto-download)
#   2. ObjectGS     (ICCV 2025, repo clone + dependencies)
#   3. Spann3R      (3DV 2025, repo clone + DUSt3R submodules)
#   4. Shape of Motion (ICCV 2025, repo clone + weights)
#
# All 4 models have mock fallback — you only need to install them if you
# want real inference. Without them, the pipeline still runs (mock mode).

set -e

CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/tothinkvision}"
REPOS_DIR="${MODEL_REPOS_DIR:-$HOME/.local/share/tothinkvision/repos}"
mkdir -p "$CACHE_DIR" "$REPOS_DIR"

# ─── Mirror helpers (China / proxy) ──────────────────────────
if [ "$USE_MIRROR" = "true" ]; then
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
    echo "  Mirror mode: Tsinghua PyPI + hf-mirror.com"
else
    PIP_INDEX=""
    HF_ENDPOINT="https://huggingface.co"
fi
export HF_ENDPOINT

# GitHub proxy: gh-proxy.com (works in mainland China)
try_clone() {
    # $1 = repo (org/name), $2 = dest, $3 = optional branch
    local branch="${3:-main}"
    local raw="https://github.com/$1.git"
    local proxy="https://gh-proxy.com/"

    if [ -d "$2/.git" ] || [ -f "$2/setup.py" ] || [ -f "$2/pyproject.toml" ] || [ -f "$2/README.md" ]; then
        echo "  Already cloned: $1 → $2"
        return 0
    fi

    mkdir -p "$(dirname "$2")"

    # Method 1: direct git clone
    echo "  Cloning $1 (direct)..."
    git clone --recursive --branch "$branch" --depth 1 "$raw" "$2" 2>/dev/null && return 0

    # Method 2: via gh-proxy.com
    echo "  Cloning $1 (via gh-proxy.com)..."
    git clone --recursive --branch "$branch" --depth 1 "${proxy}${raw}" "$2" 2>/dev/null && return 0

    # Method 3: download tarball
    echo "  Downloading $1 as tarball..."
    local url="https://github.com/$1/archive/refs/heads/$branch.tar.gz"
    curl -fSL "${proxy}${url}" -o "$2.tar.gz" && \
        mkdir -p "$2" && \
        tar -xzf "$2.tar.gz" --strip-components=1 -C "$2" && \
        rm -f "$2.tar.gz" && \
        return 0

    echo "  ❌ Failed to clone $1"
    return 1
}

download_weights() {
    # $1 = URL, $2 = destination
    if [ -f "$2" ]; then
        echo "  Already downloaded: $(basename "$2")"
        return 0
    fi
    mkdir -p "$(dirname "$2")"
    echo "  Downloading: $(basename "$2")"
    curl -fSL "$1" -o "$2" || wget -q "$1" -O "$2"
}

echo "=========================================="
echo "  ToThinkVision v2.1 — 4D Model Installer"
echo "=========================================="
echo ""
echo "This installs the 4 advanced models added in v2.1."
echo "All 4 have automatic mock fallback — you can skip any of them"
echo "and the pipeline will still run (using mock data for those stages)."
echo ""
echo "Select models to install:"
echo ""
echo "  1) CoTracker3 (Meta, 8-12GB VRAM)"
echo "     Dense point tracking (265×265) — improves trajectory accuracy"
echo "     Install: torch.hub weight download only (~300MB)"
echo ""
echo "  2) ObjectGS (ICCV 2025, 24GB VRAM)"
echo "     Per-object 3D Gaussian Splatting — individual 3D Gaussians"
echo "     Install: repo clone + submodules + 3DGS rasterizer (~4GB)"
echo ""
echo "  3) Spann3R (3DV 2025, 24-48GB VRAM)"
echo "     3D reconstruction with spatial memory (built on DUSt3R)"
echo "     Install: repo clone + DUSt3R submodules + weights (~8GB)"
echo ""
echo "  4) Shape of Motion (ICCV 2025, 24GB VRAM)"
echo "     End-to-end 4D reconstruction from monocular video"
echo "     Install: repo clone + weights (~6GB)"
echo ""
echo "  5) ALL of the above (~18GB total)"
echo "  0) Exit"
echo ""
echo "Tip: set USE_MIRROR=true before running for mainland China mirrors"
echo ""

read -r -p "Your choice [0-5]: " CHOICE

install_cotracker3() {
    echo ""
    echo "=== Installing CoTracker3 ==="
    # CoTracker3 通过 torch.hub 加载，但 torch.hub 会直连 GitHub（国内不稳）
    # 解决方案：先手动 clone 仓库，然后用 source='local' 加载

    local repo_dest="$REPOS_DIR/co-tracker"

    # Step 1: Clone repo
    if ! try_clone "facebookresearch/co-tracker" "$repo_dest"; then
        echo "  ❌ Failed to clone CoTracker3"
        return 1
    fi

    cd "$repo_dest"

    # Step 2: Install dependencies (requirements.txt + optional CUDA extensions)
    if [ -f "requirements.txt" ]; then
        echo "  Installing CoTracker3 dependencies..."
        pip install $PIP_INDEX -r requirements.txt 2>&1 | tail -3 || true
    fi

    # Step 3: Download pretrained weights from HuggingFace (走 hf-mirror 镜像)
    local weights_dir="$CACHE_DIR/CoTracker3"
    mkdir -p "$weights_dir"
    local weights_url="${HF_ENDPOINT}/facebook/cotracker3/resolve/main/cotracker3_offline.pth"
    local weights_dest="$weights_dir/cotracker3_offline.pth"
    download_weights "$weights_url" "$weights_dest" 2>/dev/null || \
        echo "  ⚠ Offline weights download skipped"
    local weights_url_online="${HF_ENDPOINT}/facebook/cotracker3/resolve/main/cotracker3_online.pth"
    local weights_dest_online="$weights_dir/cotracker3_online.pth"
    download_weights "$weights_url_online" "$weights_dest_online" 2>/dev/null || \
        echo "  ⚠ Online weights download skipped"

    # Step 4: 验证加载（用 source='local' 避免再次下载 zip）
    echo "  Verifying CoTracker3 load (source=local)..."
    python -c "
import sys
sys.path.insert(0, '$repo_dest')
try:
    import torch
    # 方式1: source='local' 直接用本地仓库
    model = torch.hub.load('$repo_dest', 'cotracker3_offline', source='local')
    print('  ✓ CoTracker3 offline loaded (local)')
except Exception as e:
    print(f'  ⚠ Load failed: {e}')
    print('  Will fall back to mock mode at runtime.')
" 2>&1 || true

    echo ""
    echo "  ✓ CoTracker3 installed at: $repo_dest"
    echo ""
    echo "  Add to your shell profile:"
    echo "    export COTRACKER_REPO=\"$repo_dest\""
    echo ""

    if [ -t 0 ]; then
        read -r -p "  Auto-add to ~/.bashrc? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy] ]]; then
            echo "export COTRACKER_REPO=\"$repo_dest\"" >> "$HOME/.bashrc"
            echo "  Added. Run 'source ~/.bashrc' or open a new shell."
        fi
    fi
}

install_objectgs() {
    echo ""
    echo "=== Installing ObjectGS ==="
    local dest="$REPOS_DIR/ObjectGS"

    try_clone "RuijieZhu94/ObjectGS" "$dest" || return 1

    cd "$dest"

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "  Installing ObjectGS dependencies..."
        pip install $PIP_INDEX -r requirements.txt 2>&1 | tail -3 || true
    fi

    # Install 3DGS rasterizer (submodule)
    if [ -d "submodules/simple-knn" ]; then
        echo "  Building simple-knn..."
        (cd submodules/simple-knn && pip install . 2>&1 | tail -2) || true
    fi
    if [ -d "submodules/diff-gaussian-rasterization" ]; then
        echo "  Building diff-gaussian-rasterization..."
        (cd submodules/diff-gaussian-rasterization && pip install . 2>&1 | tail -2) || true
    fi

    # Initialize submodules if needed
    if [ -f ".gitmodules" ]; then
        git submodule update --init --recursive 2>/dev/null || true
    fi

    echo ""
    echo "  ✓ ObjectGS installed at: $dest"
    echo ""
    echo "  Add to your shell profile (.bashrc / .zshrc):"
    echo "    export OBJECT_GS_PATH=\"$dest\""
    echo ""

    # Auto-append to .bashrc if interactive
    if [ -t 0 ]; then
        read -r -p "  Auto-add to ~/.bashrc? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy] ]]; then
            echo "export OBJECT_GS_PATH=\"$dest\"" >> "$HOME/.bashrc"
            echo "  Added. Run 'source ~/.bashrc' or open a new shell."
        fi
    fi
}

install_spann3r() {
    echo ""
    echo "=== Installing Spann3R ==="
    local dest="$REPOS_DIR/Spann3R"

    try_clone "HengyiWang/Spann3R" "$dest" || return 1

    cd "$dest"

    # Init submodules (DUSt3R is a submodule)
    if [ -f ".gitmodules" ]; then
        echo "  Initializing submodules (includes DUSt3R)..."
        git submodule update --init --recursive 2>/dev/null || true
    fi

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "  Installing Spann3R dependencies..."
        pip install $PIP_INDEX -r requirements.txt 2>&1 | tail -3 || true
    fi

    # Install DUSt3R (if present as submodule)
    if [ -d "dust3r" ] && [ -f "dust3r/setup.py" ]; then
        echo "  Installing DUSt3R..."
        (cd dust3r && pip install . 2>&1 | tail -2) || true
    fi

    # Download weights (Spann3R checkpoint via HuggingFace)
    local weights_url="${HF_ENDPOINT}/hengyiwang/Spann3R/resolve/main/spann3r_checkpoint.pth"
    local weights_dest="$CACHE_DIR/Spann3R/spann3r_checkpoint.pth"
    download_weights "$weights_url" "$weights_dest" 2>/dev/null || \
        echo "  ⚠ Weight download failed; set SPANN3R_WEIGHTS manually"

    echo ""
    echo "  ✓ Spann3R installed at: $dest"
    echo ""
    echo "  Add to your shell profile:"
    echo "    export SPANN3R_PATH=\"$dest\""
    echo ""

    if [ -t 0 ]; then
        read -r -p "  Auto-add to ~/.bashrc? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy] ]]; then
            echo "export SPANN3R_PATH=\"$dest\"" >> "$HOME/.bashrc"
            echo "  Added."
        fi
    fi
}

install_shape_of_motion() {
    echo ""
    echo "=== Installing Shape of Motion ==="
    local dest="$REPOS_DIR/shape-of-motion"

    try_clone "vye16/shape-of-motion" "$dest" || return 1

    cd "$dest"

    # Init submodules
    if [ -f ".gitmodules" ]; then
        echo "  Initializing submodules..."
        git submodule update --init --recursive 2>/dev/null || true
    fi

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "  Installing dependencies..."
        pip install $PIP_INDEX -r requirements.txt 2>&1 | tail -3 || true
    fi

    # Build CUDA extensions if present
    if [ -d "cuda_ext" ] && [ -f "cuda_ext/setup.py" ]; then
        echo "  Building CUDA extensions..."
        (cd cuda_ext && pip install . 2>&1 | tail -2) || true
    fi

    # Download pretrained weights via HuggingFace (if available)
    local weights_url="${HF_ENDPOINT}/vye16/shape-of-motion/resolve/main/som_pretrained.pth"
    local weights_dest="$CACHE_DIR/ShapeOfMotion/som_pretrained.pth"
    download_weights "$weights_url" "$weights_dest" 2>/dev/null || \
        echo "  ⚠ Weight download skipped; run the repo's official download script if needed"

    echo ""
    echo "  ✓ Shape of Motion installed at: $dest"
    echo ""
    echo "  Add to your shell profile:"
    echo "    export SHAPE_OF_MOTION_PATH=\"$dest\""
    echo ""

    if [ -t 0 ]; then
        read -r -p "  Auto-add to ~/.bashrc? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy] ]]; then
            echo "export SHAPE_OF_MOTION_PATH=\"$dest\"" >> "$HOME/.bashrc"
            echo "  Added."
        fi
    fi
}

case "$CHOICE" in
    1) install_cotracker3 ;;
    2) install_objectgs ;;
    3) install_spann3r ;;
    4) install_shape_of_motion ;;
    5)
        install_cotracker3
        install_objectgs
        install_spann3r
        install_shape_of_motion
        ;;
    0) echo "Exit."; exit 0 ;;
    *) echo "Invalid choice: $CHOICE"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo "  Installation complete"
echo "=========================================="
echo ""
echo "Verify in Python:"
echo "  python -c 'from app.models.cotracker3 import get_cotracker; print(get_cotracker().model is not None)'"
echo "  python -c 'from app.models.object_gs import get_objectgs_pipeline; print(get_objectgs_pipeline().available)'"
echo "  python -c 'from app.models.spann3r import get_spann3r; print(get_spann3r().available)'"
echo "  python -c 'from app.models.shape_of_motion import get_shape_of_motion; print(get_shape_of_motion().available)'"
echo ""
echo "Enable in API call:"
echo "  curl -X POST http://localhost:8000/api/process \\"
echo "    -F 'file=@demo.mp4' \\"
echo "    -F 'enable_cotracker3=true' \\"
echo "    -F 'enable_objectgs=true' \\"
echo "    -F 'enable_shape_of_motion=true' \\"
echo "    -F 'export_formats=[\"animated_gltf\", \"scene_graph_json\"]'"
echo ""
