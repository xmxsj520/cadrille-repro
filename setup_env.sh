#!/usr/bin/env bash
# =============================================================================
# cadrille 复现 —— 环境一键安装脚本
# =============================================================================
# 参考：官方仓库 AIRI-Institute/cadrille 的 Dockerfile（版本全部钉死）
#
# 用法（在租来的 Linux 服务器上，建议在 tmux 里跑）：
#   tmux new -s build
#   bash setup_env.sh                 # 全量安装
#   bash setup_env.sh --skip-flash    # 跳过 flash-attn（最快第一轮先这样）
#   bash setup_env.sh --step 4        # 只重跑第 4 步（失败后修复用）
#   bash setup_env.sh --list          # 看有哪些步骤
#
# 重入性：每一步都会检查是否已完成，重复执行不会重复编译。
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
WORKDIR="${WORKDIR:-/root/autodl-tmp}"
REPO_DIR="${REPO_DIR:-$WORKDIR/cadrille}"
LOG_DIR="${LOG_DIR:-$WORKDIR/setup_logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 官方钉死的版本
TORCH_VER="2.5.1"
TORCH_CUDA="cu124"
OPEN3D_COMMIT="8e434558a9b1ecacba7854da7601a07e8bdceb26"
PYTORCH3D_COMMIT="06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f"
CADQUERY_COMMIT="e99a15df3cf6a88b69101c405326305b5db8ed94"
FLASH_ATTN_VER="2.7.2.post1"

# 依赖包（与官方 Dockerfile 完全一致，不要擅自升级）
PIP_PKGS=(
  accelerate==0.34.2
  cadquery-ocp==7.7.2
  casadi==3.6.7
  contourpy==1.3.1
  einops==0.8.0
  ezdxf==1.3.5
  huggingface-hub==0.27.0
  imageio==2.36.1
  lazy_loader==0.4
  manifold3d==3.0.0
  matplotlib==3.10.0
  multimethod==1.12
  nlopt==2.9.0
  numpy==2.2.0
  path==17.0.0
  pyparsing==3.2.0
  regex==2024.11.6
  safetensors==0.4.5
  scikit-image==0.25.0
  scipy==1.14.1
  tifffile==2024.12.12
  tokenizers==0.21.0
  transformers==4.50.3
  trimesh==4.5.3
  typish==1.9.3
  qwen-vl-utils==0.0.10
)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
C_RESET='\033[0m'; C_RED='\033[0;31m'; C_GRN='\033[0;32m'
C_YEL='\033[0;33m'; C_BLU='\033[0;34m'

info()  { echo -e "${C_BLU}[INFO]${C_RESET} $*"; }
ok()    { echo -e "${C_GRN}[ OK ]${C_RESET} $*"; }
warn()  { echo -e "${C_YEL}[WARN]${C_RESET} $*"; }
err()   { echo -e "${C_RED}[FAIL]${C_RESET} $*"; }
die()   { err "$*"; echo; err "修复后可用: bash $0 --step <步骤号> 只重跑该步"; exit 1; }

banner() {
  echo
  echo "============================================================================="
  echo "  $*"
  echo "============================================================================="
}

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# 步骤 0：预检
# ---------------------------------------------------------------------------
step_0() {
  banner "第 0 步：预检"

  [[ $EUID -eq 0 ]] || die "请用 root 运行（或全程加 sudo）"

  for c in git curl wget python3 pip3; do
    command -v "$c" >/dev/null 2>&1 || die "缺少命令: $c"
  done
  ok "基础命令齐全"

  python3 -c 'import sys; v=sys.version_info; assert (v.major,v.minor) >= (3,10), f"Python {v.major}.{v.minor} too old"; print(f"Python {v.major}.{v.minor}.{v.micro}")' \
    || die "Python 版本不满足（需 >= 3.10）"

  if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader \
      | sed 's/^/       /'
    # 50 系(sm_120) 不受 torch 2.5.1 支持
    if nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
       | grep -qE '^12\.'; then
      die "检测到 sm_120（50 系显卡）。torch 2.5.1 / pytorch3d / flash-attn 均不支持。
     这是硬件选型问题，无法用脚本绕过 —— 请换 A100 / H100 / 4090 / 3090 实例。"
    fi
  else
    warn "没有 nvidia-smi，可能不是 GPU 实例，后续只能跑 CPU 部分"
  fi

  # 可用的 nvcc 版本
  if command -v nvcc >/dev/null 2>&1; then
    info "nvcc: $(nvcc --version | grep -oP 'release \K[\d.]+' || echo '?')"
  else
    warn "找不到 nvcc。编译 pytorch3d / flash-attn 需要 CUDA toolkit。"
  fi

  df -h "$WORKDIR" 2>/dev/null | tail -1 | awk '{print "       磁盘 "$4" 可用 ("$6")"}'
  [[ -d "$WORKDIR" ]] || die "$WORKDIR 不存在，请用 WORKDIR=... 指定"
  ok "预检通过"
}

# ---------------------------------------------------------------------------
# 步骤 1：系统库
# ---------------------------------------------------------------------------
step_1() {
  banner "第 1 步：系统库（apt）"
  # 注意: 官方 Dockerfile 把 git-lfs 拼错成 git-lsf，这里用正确的
  local pkgs=(
    git git-lfs wget curl cmake build-essential pkg-config
    libgl1-mesa-glx libosmesa6-dev libglu1-mesa-dev libglew-dev
    libglfw3-dev libeigen3-dev libjpeg-dev libpng-dev tmux htop
  )
  info "更新apt索引..."
  apt-get update -qq || warn "apt-get update 失败，继续尝试安装"

  local missing=()
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "系统库已齐（${#pkgs[@]} 个）"
  else
    info "安装 ${#missing[@]} 个包: ${missing[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}" \
      || die "apt 安装失败，看上面的报错（常见: 软件源问题、磁盘满）"
    ok "系统库安装完成"
  fi

  git lfs install >/dev/null 2>&1 && ok "git-lfs 已初始化"
  [[ "$(git lfs version 2>/dev/null)" ]] || die "git-lfs 仍不可用"
}

# ---------------------------------------------------------------------------
# 步骤 2：PyTorch
# ---------------------------------------------------------------------------
step_2() {
  banner "第 2 步：PyTorch ${TORCH_VER}+${TORCH_CUDA}"

  local cur
  cur=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo "none")
  if [[ "$cur" == "${TORCH_VER} 12.4" ]]; then
    ok "torch 已是 ${TORCH_VER}+cu124，跳过"
    return 0
  fi

  warn "当前 torch: $cur … 将安装 ${TORCH_VER}+${TORCH_CUDA}"
  pip3 install --no-cache-dir \
    torch=="${TORCH_VER}" torchvision \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    2>&1 | tee "$LOG_DIR/step2_torch.log" | tail -5
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "torch 安装失败，见 $LOG_DIR/step2_torch.log"

  python3 - <<'PY' || die "torch 装上了但 CUDA 不可用（检查驱动）"
import torch, sys
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"  cuda_available = {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA 不可用"
assert torch.cuda.is_bf16_supported(), "GPU 不支持 bf16"
print("  bf16 OK")
PY
  ok "PyTorch 就绪"
}

# ---------------------------------------------------------------------------
# 步骤 3：Open3D（源码编译 headless 版）—— 最耗时
# ---------------------------------------------------------------------------
step_3() {
  banner "第 3 步：Open3D headless（源码编译，30-90 分钟）"

  if python3 -c 'import open3d' 2>/dev/null; then
    local v; v=$(python3 -c 'import open3d;print(open3d.__version__)')
    if [[ -f "$WORKDIR/.open3d_headless_built" ]]; then
      ok "Open3D $v 已编译过（标记文件存在），跳过"
      return 0
    fi
    warn "检测到 pip 版 Open3D $v，但它没有 headless 渲染能力，将重新编译"
  fi

  mkdir -p "$WORKDIR"
  local src="$WORKDIR/Open3D"
  if [[ ! -d "$src/.git" ]]; then
    info "克隆 Open3D..."
    git clone --filter=blob:none https://github.com/isl-org/Open3D.git "$src" \
      || die "克隆 Open3D 失败（网络？试 git clone 时加代理）"
  fi
  cd "$src" || die "进不去 $src"
  git fetch --depth 1 origin "$OPEN3D_COMMIT" >/dev/null 2>&1 || true
  git checkout -q "$OPEN3D_COMMIT" 2>/dev/null \
    || die "切不到 commit $OPEN3D_COMMIT"
  ok "Open3D 源码就位 ($(git rev-parse --short HEAD))"

  mkdir -p build && cd build || die "建 build 目录失败"
  info "cmake 配置（headless 渲染 + 关闭 GUI）..."
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_HEADLESS_RENDERING=ON \
    -DBUILD_GUI=OFF \
    -DUSE_SYSTEM_GLEW=OFF \
    -DUSE_SYSTEM_GLFW=OFF \
    -DBUILD_PYTHON_MODULE=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    .. 2>&1 | tee "$LOG_DIR/step3_cmake.log" | tail -8
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "cmake 配置失败，见 $LOG_DIR/step3_cmake.log"

  local nproc_use; nproc_use=$(nproc)
  info "开始编译（make -j${nproc_use}），这一步很慢，可以去泡杯茶..."
  make -j"$nproc_use" 2>&1 | tee "$LOG_DIR/step3_make.log" | tail -5
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "Open3D 编译失败，见 $LOG_DIR/step3_make.log（常见: 内存不足，改用 make -j4）"

  info "安装 Python 包..."
  make install-pip-package 2>&1 | tee -a "$LOG_DIR/step3_make.log" | tail -5
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "install-pip-package 失败"

  touch "$WORKDIR/.open3d_headless_built"
  cd "$WORKDIR" || true
  ok "Open3D headless 编译完成"
}

# ---------------------------------------------------------------------------
# 步骤 4：pytorch3d + cadquery
# ---------------------------------------------------------------------------
step_4() {
  banner "第 4 步：pytorch3d / cadquery（源码编译）"

  # --- pytorch3d ---
  if python3 -c 'from pytorch3d.ops import sample_farthest_points' 2>/dev/null; then
    ok "pytorch3d 已可用，跳过"
  else
    local arch; arch=$(detect_arch)
    info "pytorch3d 将针对 TORCH_CUDA_ARCH_LIST=$arch 编译"
    export TORCH_CUDA_ARCH_LIST="$arch"
    export FORCE_CUDA=1
    export MAX_JOBS="$(nproc)"
    pip3 install --no-build-isolation --no-deps \
      "git+https://github.com/facebookresearch/pytorch3d@${PYTORCH3D_COMMIT}" \
      2>&1 | tee "$LOG_DIR/step4_pytorch3d.log" | tail -8
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "pytorch3d 编译失败，见 $LOG_DIR/step4_pytorch3d.log"
    python3 -c 'from pytorch3d.ops import sample_farthest_points; print("  pytorch3d import OK")' \
      || die "pytorch3d 装上了但 import 失败（多半是 CUDA 架构不匹配）"
    ok "pytorch3d 就绪"
  fi

  # --- cadquery ---
  if python3 -c 'import cadquery' 2>/dev/null; then
    ok "cadquery 已可用，跳过"
  else
    pip3 install --no-build-isolation --no-deps \
      "git+https://github.com/CadQuery/cadquery.git@${CADQUERY_COMMIT}" \
      2>&1 | tee "$LOG_DIR/step4_cadquery.log" | tail -5
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "cadquery 安装失败，见 $LOG_DIR/step4_cadquery.log"
    python3 -c 'import cadquery; print("  cadquery", cadquery.__version__)' \
      || die "cadquery import 失败（注意: 官方用 --no-deps，缺 typish/multimethod 会在第 5 步补）"
    ok "cadquery 就绪"
  fi
}

# ---------------------------------------------------------------------------
# 步骤 5：全部 Python 依赖（钉死版本）
# ---------------------------------------------------------------------------
step_5() {
  banner "第 5 步：Python 依赖 ${#PIP_PKGS[@]} 个（版本钉死）"

  info "先补 --no-deps 会漏掉的基础依赖..."
  pip3 install --no-cache-dir setuptools wheel ninja packaging 2>&1 | tail -2

  local failed=()
  for pkg in "${PIP_PKGS[@]}"; do
    local name="${pkg%%==*}"
    if pip3 show "$name" >/dev/null 2>&1; then
      local got; got=$(pip3 show "$name" 2>/dev/null | awk '/^Version:/{print $2}')
      if [[ "$got" == "${pkg##*==}" ]]; then
        echo "  [skip] $pkg（已是该版本）"
        continue
      fi
      warn "$name 当前 $got，需要 ${pkg##*==}，将重装"
    fi
    echo "  [inst] $pkg"
    if ! pip3 install --no-cache-dir --no-deps "$pkg" >>"$LOG_DIR/step5_pip.log" 2>&1; then
      failed+=("$pkg")
    fi
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    die "以下包安装失败: ${failed[*]}  见 $LOG_DIR/step5_pip.log"
  fi
  ok "Python 依赖安装完成"

  if command -v hf_transfer >/dev/null 2>&1 || pip3 show hf_transfer >/dev/null 2>&1; then
    ok "hf_transfer 可用"
  else
    info "安装 huggingface 加速工具..."
    pip3 install --no-cache-dir -U "huggingface_hub[cli]" hf_transfer 2>&1 | tail -2 || \
      warn "hf_transfer 安装失败，不影响主流程"
  fi
}

# ---------------------------------------------------------------------------
# 步骤 6：flash-attn（可选，编译很慢）
# ---------------------------------------------------------------------------
step_6() {
  banner "第 6 步：flash-attn ${FLASH_ATTN_VER}（可选，30-90 分钟）"
  if [[ "$SKIP_FLASH" == "1" ]]; then
    warn "已按 --skip-flash 跳过。注意: 训练/推理要用 attn_implementation='eager' 或 'sdpa'"
    return 0
  fi
  if python3 -c "import flash_attn" 2>/dev/null; then
    ok "flash-attn 已安装，跳过"
    return 0
  fi

  export TORCH_CUDA_ARCH_LIST="$(detect_arch)"
  info "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

  # 先试预编译 wheel（快得多），失败再源码编译
  if pip3 install --no-cache-dir --no-build-isolation "flash-attn==${FLASH_ATTN_VER}" \
       --no-deps --only-binary flash-attn 2>/dev/null; then
    ok "flash-attn 用预编译 wheel 装上了"
  else
    warn "没有可用 wheel，转源码编译（很慢且吃内存，建议机器内存 >= 32GB）"
    MAX_JOBS=4 pip3 install --no-cache-dir --no-build-isolation \
      "flash-attn==${FLASH_ATTN_VER}" --no-deps \
      2>&1 | tee "$LOG_DIR/step6_flash.log" | tail -8
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "flash-attn 编译失败，见 $LOG_DIR/step6_flash.log（可用 --skip-flash 先绕过）"
  fi
  python3 -c 'import flash_attn; print("  flash_attn", flash_attn.__version__)' \
    || die "flash-attn import 失败"
  ok "flash-attn 就绪"
}

# ---------------------------------------------------------------------------
# 步骤 7：拉仓库 + 最终自检
# ---------------------------------------------------------------------------
step_7() {
  banner "第 7 步：拉 cadrille 仓库 + 最终自检"

  mkdir -p "$WORKDIR"
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    info "克隆 cadrille 到 $REPO_DIR"
    git clone https://github.com/AIRI-Institute/cadrille.git "$REPO_DIR" \
      || die "克隆 cadrille 失败"
  else
    ok "cadrille 仓库已存在"
  fi
  cd "$REPO_DIR" || die "进不去 $REPO_DIR"

  export HF_ENDPOINT
  export HF_HUB_ENABLE_HF_TRANSFER=1
  info "HF_ENDPOINT=$HF_ENDPOINT"

  local check="step1_check_env.py"
  local check_path=""
  for cand in "$REPO_DIR/$check" "$WORKDIR/$check"; do
    [[ -f "$cand" ]] && { check_path="$cand"; break; }
  done
  if [[ -n "$check_path" ]]; then
    info "运行环境自检 $check_path ..."
    python3 "$check_path" --data-dir "$WORKDIR/data" || true
  else
    warn "没找到 $check（请把它上传到 $REPO_DIR 或 $WORKDIR 再手动运行）"
  fi

  cat <<'EOF'

=============================================================================
 环境安装结束。下一步：
 1) 如果自检有 FAIL，先修
 2) 下载数据（先只下这两个）：
      cd REPO_DIR/data
      git lfs install
      git clone https://hf-mirror.com/datasets/maksimko123/deepcad_test_mesh
 3) 官方权重（Table 1 用它）：
      huggingface-cli download maksimko123/cadrille --local-dir ../ckpt_sft
 4) 跑推理 + 评估（口径: n_samples=1, 即 IoU@1）
      python test.py --data-path ./data --split deepcad_test_mesh \
             --mode pc --checkpoint-path ./ckpt_sft --py-path ./work_dirs/tmp_py
      python evaluate.py --gt-mesh-path ./data/deepcad_test_mesh \
             --pred-py-path ./work_dirs/tmp_py
=============================================================================
EOF
}

# ---------------------------------------------------------------------------
# 架构检测
# ---------------------------------------------------------------------------
detect_arch() {
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$cc" ]]; then
    echo "8.0"; return
  fi
  local major="${cc%%.*}" minor="${cc##*.}"
  case "$major.$minor" in
    12.*) echo "12.0" ;;
    *)    echo "${major}.${minor}" ;;
  esac
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
SKIP_FLASH=0
ONLY_STEP=""

usage() {
  sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-flash) SKIP_FLASH=1; shift ;;
    --step)       [[ -n "${2:-}" ]] || die "--step 需要参数，如 --step 3"; ONLY_STEP="$2"; shift 2 ;;
    --list)
      echo "步骤: 0预检 1系统库 2torch 3Open3D 4pytorch3d+cadquery 5pip依赖 6flash-attn 7仓库+自检"
      exit 0 ;;
    --workdir)    [[ -n "${2:-}" ]] || die "--workdir 需要参数"; WORKDIR="$2"; REPO_DIR="$WORKDIR/cadrille"; LOG_DIR="$WORKDIR/setup_logs"; shift 2 ;;
    -h|--help) usage ;;
    *) die "未知参数: $1（用 --help 看用法）" ;;
  esac
done

STEPS=(step_0 step_1 step_2 step_3 step_4 step_5 step_6 step_7)

if [[ -n "$ONLY_STEP" ]]; then
  t="step_${ONLY_STEP}"
  declare -F "$t" >/dev/null || die "没有第 $ONLY_STEP 步（--list 查看）"
  export WORKDIR REPO_DIR LOG_DIR HF_ENDPOINT SKIP_FLASH
  "$t"
  ok "第 $ONLY_STEP 步完成"
  exit 0
fi

export WORKDIR REPO_DIR LOG_DIR HF_ENDPOINT SKIP_FLASH
for s in "${STEPS[@]}"; do
  "$s" || { err "$s 失败，修复后用 --step ${s#step_} 重跑该步"; exit 1; }
done

ok "全部步骤完成！"
