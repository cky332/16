#!/bin/bash
# ============================================================
# sweet-watermark 项目环境部署脚本
# 从 Anaconda/Miniconda 开始完整配置运行环境
# ============================================================

set -e

# ---------- 配置项 ----------
ENV_NAME="sweet-watermark"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"  # 根据你的 GPU 驱动修改: 11.8 / 12.1 / cpu

# ---------- 颜色输出 ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================================
# 第 1 步: 检查 conda 是否已安装
# ============================================================
if ! command -v conda &> /dev/null; then
    info "未检测到 conda，正在安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    # 初始化 conda
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    info "Miniconda 安装完成，请重新打开终端或执行: source ~/.bashrc"
else
    info "检测到 conda: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ============================================================
# 第 2 步: 创建 conda 虚拟环境
# ============================================================
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "环境 '${ENV_NAME}' 已存在，跳过创建"
else
    info "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 激活环境
info "激活环境: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# ============================================================
# 第 3 步: 安装 PyTorch (根据 CUDA 版本)
# ============================================================
info "安装 PyTorch (CUDA ${CUDA_VERSION})..."
if [ "${CUDA_VERSION}" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "${CUDA_VERSION}" = "11.8" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "${CUDA_VERSION}" = "12.1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    warn "未识别的 CUDA 版本 '${CUDA_VERSION}'，使用默认 PyTorch 安装"
    pip install torch torchvision torchaudio
fi

# ============================================================
# 第 4 步: 安装项目依赖
# ============================================================
info "安装项目依赖 (requirements.txt)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ============================================================
# 第 5 步: 安装 Cython 编译所需的系统依赖
# ============================================================
info "确保 Cython 编译环境就绪..."
# levenshtein.pyx 会在运行时通过 pyximport 自动编译
# 需要确保 gcc/g++ 和 python-dev 已安装
if ! command -v gcc &> /dev/null; then
    warn "未找到 gcc，请手动安装: sudo apt install build-essential python3-dev"
fi
# 清理 pyximport 缓存，避免损坏的 .so 文件导致 ImportError
if [ -d "$HOME/.pyxbld" ]; then
    info "清理 Cython 编译缓存 (~/.pyxbld/)..."
    rm -rf "$HOME/.pyxbld"
fi

# 预编译 levenshtein.pyx（避免多进程 accelerate 运行时编译导致竞争条件）
info "预编译 Cython 扩展 (levenshtein.pyx)..."
python "${SCRIPT_DIR}/build_ext.py"

# ============================================================
# 第 6 步: 下载 NLTK 数据
# ============================================================
info "下载 NLTK 数据..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ============================================================
# 第 7 步: 配置 accelerate
# ============================================================
info "配置 HuggingFace Accelerate..."
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    info "生成 accelerate 默认配置 (单 GPU)..."
    mkdir -p ~/.cache/huggingface/accelerate
    cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
    info "已生成默认配置 (单 GPU + bf16)，如需多 GPU 请运行: accelerate config"
else
    info "accelerate 配置已存在，跳过"
fi

# ============================================================
# 第 8 步: HuggingFace 登录 (使用环境变量方式，最可靠)
# ============================================================
warn ""
warn "如需使用 StarCoder 等受限模型，请设置 HuggingFace Token:"
warn "  1. 在 https://huggingface.co/settings/tokens 创建 Token"
warn "  2. 在 https://huggingface.co/bigcode/starcoder 接受模型协议"
warn "  3. 将以下内容添加到 ~/.bashrc:"
warn "     export HF_TOKEN=\"hf_你的token\""
warn "  4. 执行: source ~/.bashrc"
warn ""
warn "  或在 Python 交互模式下登录:"
warn "     python -c \"from huggingface_hub import login; login(token='hf_你的token')\""

# ============================================================
# 完成
# ============================================================
info "============================================"
info "环境部署完成！"
info "============================================"
info ""
info "使用方法:"
info "  conda activate ${ENV_NAME}"
info ""
info "运行示例 (WLLM 水印生成):"
info "  bash scripts/main/run_wllm_generation.sh"
info ""
info "运行示例 (WLLM 水印检测):"
info "  bash scripts/main/run_wllm_detection.sh"
info ""
