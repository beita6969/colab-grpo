# AFlow + GRPO 智能体工作流训练框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

本项目实现了 **AFlow + ROLL GRPO** 训练框架，用于训练大语言模型生成智能体工作流（Agent Workflow）。

### 核心特性

- 🚀 **GRPO 训练**: Group Relative Policy Optimization，无需 Critic 模型
- 🔧 **WA-GRPO**: Workflow-Aware 优势计算，考虑多样性和改进幅度
- 🎯 **LoRA 微调**: 低资源高效训练，仅需 40M 可训练参数
- 🤖 **LLM Judge**: 使用 OpenAI gpt-4o-mini 作为评估器
- 📊 **多领域支持**: 数学、编程、问答三大领域

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程                                  │
├─────────────────────────────────────────────────────────────┤
│  输入问题 → 模型生成工作流 → AFlow执行 → LLM评估 → GRPO更新  │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Qwen2.5    │    │    AFlow     │    │   OpenAI     │
│  7B-Instruct │ →  │   Executor   │ →  │  gpt-4o-mini │
│  (LoRA微调)  │    │  (算子执行)   │    │  (LLM Judge) │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 🖥️ 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|----------|
| GPU | V100 16GB | A100 40GB |
| 内存 | 32GB | 64GB |
| 存储 | 50GB | 100GB |

### 软件要求

| 软件 | 版本要求 | 测试版本 |
|------|----------|----------|
| Python | 3.10+ | 3.10.12 |
| CUDA | 12.0+ | 12.6 |
| PyTorch | 2.0+ | 2.9.0 |
| transformers | 4.40+ | 4.57.2 |
| peft | 0.10+ | 0.18.0 |
| openai | 1.0+ | 2.8.1 |

---

## 🚀 Google Colab 快速开始

### 方式一：一键启动 (推荐)

复制以下代码到 Colab 单元格并运行：

```python
#@title 🚀 AFlow + GRPO 一键启动
#@markdown ### 配置参数
OPENAI_API_KEY = "sk-your-api-key-here"  #@param {type:"string"}
USE_WANDB = False  #@param {type:"boolean"}
WANDB_API_KEY = ""  #@param {type:"string"}

import os

# ======== Step 1: 检查 GPU ========
print("🔍 检查 GPU...")
!nvidia-smi --query-gpu=name,memory.total --format=csv

# ======== Step 2: 克隆仓库 ========
print("\n📥 克隆仓库...")
!git clone https://github.com/beita6969/colab.git 2>/dev/null || (cd colab && git pull)
%cd colab

# ======== Step 3: 安装依赖 ========
print("\n📦 安装依赖 (约2-3分钟)...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers>=4.40.0 accelerate>=0.27.0 peft>=0.10.0
!pip install -q bitsandbytes>=0.42.0 scipy safetensors
!pip install -q openai httpx pyyaml tqdm wandb
!pip install -q datasets sentencepiece tiktoken huggingface-hub

# ======== Step 4: 配置环境变量 ========
print("\n⚙️ 配置环境...")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'
os.environ['PYTHONUNBUFFERED'] = '1'

if USE_WANDB and WANDB_API_KEY:
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    print("✅ WandB 已配置")

# ======== Step 5: 验证环境 ========
print("\n🔬 验证环境...")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ======== Step 6: 启动训练 ========
print("\n🚀 启动训练...")
print("="*50)
!python3 train.py --config config/training.yaml
```

### 方式二：分步执行

#### Step 1: 设置 Colab 运行时

1. 点击菜单 `运行时` → `更改运行时类型`
2. 硬件加速器选择 `GPU`
3. GPU 类型选择 `A100`（如有）或 `V100` / `T4`

#### Step 2: 检查 GPU

```python
!nvidia-smi

# 预期输出示例:
# NVIDIA A100-SXM4-40GB, 40960MiB
```

#### Step 3: 克隆仓库

```bash
!git clone https://github.com/beita6969/colab.git
%cd colab
```

#### Step 4: 安装依赖

```bash
# PyTorch (CUDA 12.6)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 核心依赖
!pip install -r requirements.txt
```

#### Step 5: 配置 API Key

**方法 A: 直接设置**
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-openai-api-key'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'
```

**方法 B: 使用 Colab Secrets (推荐，更安全)**
```python
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

#### Step 6: 启动训练

```bash
!python3 train.py --config config/training.yaml
```

---

## 📁 项目结构

```
.
├── train.py                    # 🚀 训练入口
├── requirements.txt            # 📦 Python 依赖列表
├── setup_env.sh               # ⚙️ 环境配置脚本 (bash)
├── COLAB_SETUP.md             # 📖 Colab 环境说明
│
├── config/                     # ⚙️ 配置文件目录
│   ├── training.yaml          # 主训练配置
│   ├── aflow_llm.yaml         # LLM API 配置
│   ├── operator.json          # AFlow 算子描述
│   ├── judge_prompts.yaml     # LLM Judge 提示词
│   └── datasets.yaml          # 数据集配置
│
├── src/                        # 🔧 核心代码
│   ├── grpo_trainer.py        # GRPO 训练器主逻辑
│   ├── aflow_executor.py      # AFlow 工作流执行器
│   ├── reward_computer.py     # 奖励计算模块
│   ├── wa_grpo.py             # WA-GRPO 优势估计
│   ├── answer_extractor.py    # 答案提取器
│   ├── data_manager.py        # 数据管理
│   ├── gpu_manager.py         # GPU 资源管理
│   └── ...
│
├── scripts/                    # 📜 工具脚本
│   ├── async_llm.py           # 异步 LLM 客户端 (OpenAI)
│   ├── operators.py           # AFlow 工作流算子
│   ├── evaluator.py           # 评估器 (DatasetType 枚举)
│   ├── download_datasets.py   # 下载数据集
│   └── ...
│
└── data/                       # 📊 数据目录
    ├── ready_to_train/        # 预处理后的训练数据
    │   ├── train_10k_final.jsonl
    │   └── test_500_preprocessed.jsonl
    ├── gsm8k/                 # GSM8K 数学数据
    ├── humaneval/             # HumanEval 代码数据
    └── hotpotqa/              # HotpotQA 问答数据
```

---

## ⚙️ 配置详解

### 训练配置 (`config/training.yaml`)

```yaml
# ========== GRPO 算法 ==========
num_return_sequences_in_group: 2   # K值: 每个问题生成K个工作流
rollout_batch_size: 5              # B值: 每批处理B个问题
# 实际每步样本数 = K × B = 2 × 5 = 10

# ========== 学习参数 ==========
learning_rate: 2.0e-5              # 学习率
max_steps: 500                     # 最大训练步数
warmup_steps: 100                  # 预热步数 (20%)
kl_loss_coef: 0.005                # KL 散度惩罚系数
clip_range: 0.20                   # PPO 裁剪范围

# ========== LoRA 配置 ==========
lora_rank: 64                      # LoRA 矩阵秩
lora_alpha: 64                     # LoRA 缩放因子
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"  # 目标模块
lora_dropout: 0.05                 # Dropout 率

# ========== WA-GRPO 配置 ==========
wa_grpo:
  diversity_weight: 0.35           # 工作流多样性权重
  revise_gain_weight: 0.25         # 修订改进权重
  exec_success_weight: 0.20        # 执行成功率权重
  efficiency_weight: 0.10          # 效率权重
  op_variety_weight: 0.10          # 算子多样性权重

# ========== 温度调度 ==========
temperature_schedule:
  enabled: true                    # 启用动态温度
  initial: 0.5                     # 初始温度 (高探索)
  final: 0.15                      # 最终温度 (低探索)
  warmup_steps: 150                # 衰减步数
```

### 显存配置建议

| GPU | 显存 | K | B | grad_accum | 说明 |
|-----|------|---|---|------------|------|
| T4 | 16GB | 2 | 2 | 8 | 最小配置 |
| V100 | 16GB | 2 | 3 | 6 | 推荐 |
| A100 | 40GB | 2 | 5 | 4 | **默认配置** |
| A100 | 80GB | 4 | 8 | 2 | 高吞吐 |

---

## 🔧 常见问题 (FAQ)

### Q1: CUDA 库找不到

**错误信息:**
```
OSError: libcudart.so.12: cannot open shared object file
```

**解决方案:**
```python
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'
```

或在终端运行:
```bash
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

### Q2: OpenAI API 认证失败

**错误信息:**
```
openai.AuthenticationError: Invalid API key provided
```

**解决方案:**
1. 检查 API Key 是否正确
2. 确保设置了环境变量:
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-xxx'  # 替换为你的 key
```

---

### Q3: 显存不足 (OOM)

**错误信息:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案:** 修改 `config/training.yaml`:
```yaml
rollout_batch_size: 2              # 减小批大小
gradient_accumulation_steps: 8     # 增加累积步数
gradient_checkpointing: true       # 启用梯度检查点
```

---

### Q4: 模型下载慢

**解决方案:** 使用 HuggingFace 镜像:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

### Q5: WandB 连接问题

**解决方案:** 禁用 WandB:
```yaml
# config/training.yaml
wandb:
  enabled: false
```

---

## 📊 监控训练

### 使用 WandB (推荐)

1. 注册账号: https://wandb.ai
2. 获取 API Key: https://wandb.ai/settings
3. 配置:
```yaml
# config/training.yaml
wandb:
  enabled: true
  project: "agent-prompt"
  api_key: "your-wandb-api-key"
```

### 查看本地日志

```bash
# 实时查看训练日志
tail -f training.log

# 筛选关键指标
grep -E "Step|reward|loss|accuracy" training.log | tail -50
```

---

## 🔄 恢复训练

如果 Colab 断开连接或训练中断:

```python
# 1. 查看已保存的 checkpoints
!ls -la checkpoints/

# 2. 从最新 checkpoint 恢复
!python3 train.py --config config/training.yaml --resume checkpoints/step_100
```

---

## 📚 AFlow 算子说明

| 算子 | 功能 | 适用场景 |
|------|------|----------|
| `Custom` | 自定义指令执行 | 通用问题 |
| `AnswerGenerate` | 步骤推理 | 数学题 |
| `Programmer` | 代码生成执行 | 编程题 |
| `Test` | 代码测试 | 验证代码 |
| `Review` | 解答审查 | 质量检查 |
| `Revise` | 解答修订 | 改进答案 |
| `ScEnsemble` | 自洽集成 | 多答案投票 |

---

## 📝 数据格式

训练数据 JSONL 格式:

```json
{"question": "What is 2 + 3?", "answer": "5", "source": "gsm8k"}
{"question": "def add(a, b): ...", "answer": "return a + b", "source": "humaneval"}
{"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "source": "hotpotqa"}
```

---

## 🙏 致谢

- [AFlow](https://github.com/geekan/MetaGPT) - 工作流框架
- [GRPO](https://arxiv.org/abs/2402.03300) - 训练算法论文
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基础模型
- [OpenAI](https://openai.com) - LLM Judge API
- [PEFT](https://github.com/huggingface/peft) - LoRA 实现

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)
