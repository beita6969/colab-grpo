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
export OPENAI_API_KEY="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"

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
export HF_TOKEN="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"
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

# ========== 本地模型路径 ==========
export MODEL_PATH=/home/claude-user/models/Qwen2.5-7B-Instruct

# ========== 禁用 wandb ==========
export WANDB_DISABLED=true

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
