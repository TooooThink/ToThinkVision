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
# Default: USE_MIRROR=true (always use mirrors for mainland China)
# Set USE_MIRROR=false to disable (direct GitHub + huggingface.co)
if [ "${USE_MIRROR:-true}" = "false" ]; then
    PIP_INDEX=""
    HF_ENDPOINT="https://huggingface.co"
else
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
    echo "  Mirror mode ON (default): Tsinghua PyPI + hf-mirror.com + gh-proxy.com"
    echo "  Set USE_MIRROR=false to use direct connections instead."
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
    # $1 = URL, $2 = destination, $3 = expected size in bytes (optional)
    #
    # Features:
    #   - Checks both file existence AND size (catches partial downloads)
    #   - Cleans up incomplete files on failure
    #   - Retries up to 3 times with backoff
    #   - Shows clear error messages (no silent suppression)
    if [ -f "$2" ]; then
        # If expected size provided, verify completeness
        if [ -n "$3" ]; then
            local actual_size
            actual_size=$(stat -c%s "$2" 2>/dev/null || stat -f%z "$2" 2>/dev/null || echo "0")
            if [ "$actual_size" -ge "$3" ]; then
                echo "  ✓ Already downloaded (complete): $(basename "$2")"
                return 0
            else
                echo "  ⚠ Found incomplete file $(basename "$2") ($actual_size / $3 bytes), re-downloading..."
                rm -f "$2"
            fi
        else
            # No size check — just verify non-empty
            local actual_size
            actual_size=$(stat -c%s "$2" 2>/dev/null || stat -f%z "$2" 2>/dev/null || echo "0")
            if [ "$actual_size" -gt 0 ]; then
                echo "  ✓ Already downloaded: $(basename "$2")"
                return 0
            else
                echo "  ⚠ Found empty file $(basename "$2"), re-downloading..."
                rm -f "$2"
            fi
        fi
    fi
    mkdir -p "$(dirname "$2")"
    echo "  Downloading: $(basename "$2")"
    echo "  From: $1"

    local max_retries=3
    local attempt=0
    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        if curl -fSL --retry 3 --retry-delay 5 --connect-timeout 30 -o "$2" "$1"; then
            echo "  ✓ Downloaded: $(basename "$2")"
            return 0
        fi
        echo "  ⚠ curl attempt $attempt/$max_retries failed"
        rm -f "$2"  # Clean up partial download
        sleep $((attempt * 2))
    done

    # Fallback: wget (if available)
    if command -v wget &>/dev/null; then
        echo "  Trying wget..."
        if wget --tries=3 --timeout=30 -O "$2" "$1"; then
            echo "  ✓ Downloaded (wget): $(basename "$2")"
            return 0
        fi
        rm -f "$2"
    fi

    echo "  ❌ Download failed: $(basename "$2")"
    return 1
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
echo "Note: Mirrors are enabled by default (hf-mirror.com + gh-proxy.com + Tsinghua PyPI)"
echo "      Set USE_MIRROR=false to use direct connections instead"
echo ""

read -r -p "Your choice [0-5]: " CHOICE

install_cotracker3() {
    echo ""
    echo "=== Installing CoTracker3 ==="
    #
    # CoTracker3 加载流程（国内环境特殊处理）:
    #   1. torch.hub.load() 会先从 GitHub 下载 repo zip → 我们用 source='local' 跳过
    #   2. hubconf.py 内部会再从 huggingface.co 下载权重 → 我们预先用镜像下载，传 checkpoint_path 跳过
    #
    # 关键：必须预先下载权重并传 checkpoint_path，否则 hubconf.py 会直连 huggingface.co（外网）
    #

    local repo_dest="$REPOS_DIR/co-tracker"

    # Step 1: Clone repo (via gh-proxy if needed)
    if ! try_clone "facebookresearch/co-tracker" "$repo_dest"; then
        echo "  ❌ Failed to clone CoTracker3"
        return 1
    fi

    cd "$repo_dest"

    # Step 2: Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "  Installing CoTracker3 dependencies..."
        pip install $PIP_INDEX -r requirements.txt 2>&1 | tail -3 || true
    fi

    # Step 3: Download pretrained weights
    #
    # 权重真实地址（~300MB each）:
    #   https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
    #   https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
    #
    # 走 HF_ENDPOINT 镜像（USE_MIRROR=true 时为 hf-mirror.com）
    # 如果镜像也失败，尝试 huggingface.co 直连（万一用户有梯子）
    # 最后给出手动下载指引
    #
    local weights_dir="$CACHE_DIR/CoTracker3"
    mkdir -p "$weights_dir"

    local offline_weights="$weights_dir/scaled_offline.pth"
    local online_weights="$weights_dir/scaled_online.pth"

    echo ""
    echo "  === Downloading CoTracker3 weights ==="
    echo "  Target dir: $weights_dir"
    echo "  Using HF endpoint: $HF_ENDPOINT"
    echo ""

    # --- Offline weights (primary, ~300MB) ---
    local mirror_url="${HF_ENDPOINT}/facebook/cotracker3/resolve/main/scaled_offline.pth"
    if ! download_weights "$mirror_url" "$offline_weights"; then
        echo "  ⚠ Mirror download failed."
        if [ "$HF_ENDPOINT" != "https://huggingface.co" ]; then
            echo "  Trying direct huggingface.co (needs VPN/foreign access)..."
            download_weights "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" "$offline_weights" || true
        fi
    fi

    # --- Online weights (secondary, ~300MB) ---
    local mirror_url_online="${HF_ENDPOINT}/facebook/cotracker3/resolve/main/scaled_online.pth"
    if ! download_weights "$mirror_url_online" "$online_weights"; then
        echo "  ⚠ Online weights download failed (non-critical, offline mode still works)"
    fi

    # Step 4: Verify installation
    #
    # 关键：hubconf.py 会忽略 checkpoint_path 参数，始终从 huggingface.co 下载
    # 解决方案：把权重复制到 torch.hub 缓存目录 (~/.cache/torch/hub/checkpoints/)
    # 这样 hubconf.py 下载时发现文件已存在，会跳过下载
    #
    local torch_cache="$HOME/.cache/torch/hub/checkpoints"
    mkdir -p "$torch_cache"

    echo ""
    echo "  === Preparing torch.hub cache ==="

    if [ -f "$offline_weights" ]; then
        local cache_dest="$torch_cache/scaled_offline.pth"
        if [ ! -f "$cache_dest" ]; then
            echo "  Copying offline weights to torch.hub cache..."
            cp "$offline_weights" "$cache_dest"
            echo "  ✓ Cached: $cache_dest"
        else
            echo "  ✓ Already in torch.hub cache: $cache_dest"
        fi
    fi

    if [ -f "$online_weights" ]; then
        local cache_dest_online="$torch_cache/scaled_online.pth"
        if [ ! -f "$cache_dest_online" ]; then
            echo "  Copying online weights to torch.hub cache..."
            cp "$online_weights" "$cache_dest_online"
            echo "  ✓ Cached: $cache_dest_online"
        else
            echo "  ✓ Already in torch.hub cache: $cache_dest_online"
        fi
    fi

    export COTRACKER_REPO="$repo_dest"
    local ckpt_path=""
    if [ -f "$offline_weights" ]; then
        ckpt_path="$offline_weights"
    fi

    python -c "
import os, sys
sys.path.insert(0, '$repo_dest')

ckpt_path = '$ckpt_path'.strip()
if not ckpt_path or not os.path.isfile(ckpt_path):
    ckpt_path = None

print()
print('  === Verifying CoTracker3 installation ===')
print(f'  Repo: $repo_dest')
print(f'  Weights: {ckpt_path or \"(none)\"}')
print()

# Step 1: Check if repo and dependencies are OK
try:
    import torch
    print('  ✓ PyTorch available')
except ImportError:
    print('  ❌ PyTorch not installed')
    print('    Install: pip install torch torchvision')
    sys.exit(0)

if not os.path.isdir('$repo_dest'):
    print('  ❌ Repo not found: $repo_dest')
    sys.exit(0)
print('  ✓ Repo cloned: $repo_dest')

# Step 2: Only try loading if weights exist
if not ckpt_path:
    print()
    print('  ⚠ Skipping model load (no weights file)')
    print()
    print('  权重未下载。hubconf.py 会尝试从 huggingface.co 下载（外网不可达），')
    print('  所以这里不执行加载测试。下载权重后再运行此脚本验证。')
    print()
    print('  解决方案 — 在有外网的机器下载后 scp 到本节点:')
    print('    # 在有网机器:')
    print('    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth \\\\')
    print('         -O $weights_dir/scaled_offline.pth')
    print('    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth \\\\')
    print('         -O $weights_dir/scaled_online.pth')
    print()
    print('    # scp 到集群:')
    print('    scp $weights_dir/*.pth your_cluster:$weights_dir/')
    print()
    print('  CoTracker3 will use mock mode at runtime (pipeline still works).')
    sys.exit(0)

# Step 3: Load model (hubconf.py will find weights in torch.hub cache)
try:
    print('  Loading model...')
    print('  (weights pre-cached in ~/.cache/torch/hub/checkpoints/ → no download)')
    model = torch.hub.load('$repo_dest', 'cotracker3_offline', source='local')
    print('  ✓ CoTracker3 loaded successfully!')
    print(f'    Weights: ~/.cache/torch/hub/checkpoints/scaled_offline.pth')
except Exception as e:
    print(f'  ❌ Load failed: {e}')
    print()
    print('  权重已缓存但加载失败，检查:')
    print('    - requirements.txt 依赖是否装全:')
    print('      pip install -r $repo_dest/requirements.txt')
    print()
    print('  CoTracker3 will use mock mode at runtime (pipeline still works).')
" 2>&1 || true

    echo ""
    echo "  ✓ CoTracker3 repo installed at: $repo_dest"
    echo ""
    echo "  Add to your shell profile (~/.bashrc or ~/.zshrc):"
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
    download_weights "$weights_url" "$weights_dest" || {
        echo "  ⚠ Weight download failed; set SPANN3R_WEIGHTS manually or download from:"
        echo "    ${HF_ENDPOINT}/hengyiwang/Spann3R/resolve/main/spann3r_checkpoint.pth"
    }

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
    download_weights "$weights_url" "$weights_dest" || \
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
