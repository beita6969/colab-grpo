#!/bin/bash
# =====================================================
# 环境配置脚本 - Google Colab / A100
# 用法: source setup_env.sh
# =====================================================

# ========== CUDA 环境 ==========
# 根据实际CUDA版本调整路径
if [ -d "/usr/local/cuda-12.6" ]; then
    export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d "/usr/local/cuda-12.5" ]; then
    export CUDA_HOME=/usr/local/cuda-12.5
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:$PATH

# ========== OpenAI API (必需) ==========
# 方式1: 在运行前设置环境变量
# export OPENAI_API_KEY='your-api-key-here'
#
# 方式2: 从 Colab Secrets 读取 (推荐)
# from google.colab import userdata
# os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
#
# 检查是否已设置
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: OPENAI_API_KEY 未设置!"
    echo "请运行: export OPENAI_API_KEY='your-api-key-here'"
fi

# ========== 禁用代理 (Colab 不需要) ==========
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export no_proxy="localhost,127.0.0.1"

# ========== HuggingFace 缓存 ==========
export HF_HOME=${HOME}/.cache/huggingface
export TRANSFORMERS_CACHE=${HOME}/.cache/huggingface/transformers
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

# ========== Python 配置 ==========
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ========== 验证环境 ==========
echo "========================================"
echo "       环境配置完成"
echo "========================================"
echo "CUDA_HOME:     ${CUDA_HOME:-未找到}"
echo "Python:        $(python3 --version 2>/dev/null || echo '未安装')"
echo "PyTorch GPU:   $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch未安装')"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OpenAI API:    ✅ 已配置"
else
    echo "OpenAI API:    ❌ 未配置"
fi
echo "HF_HOME:       $HF_HOME"
echo "========================================"
