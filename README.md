# AFlow-GRPO: 开放式工作流组合训练系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **核心创新**：让模型自主学习如何组合 Operators 来解决问题，而不是从预定义选项中选择

## 🎯 项目理念

```
传统方法: "请选择最佳工作流: A) Custom B) Programmer C) Custom->Review"
本项目方法: "这是可用的Operators，请设计最优工作流 DSL"
```

模型学习生成 DSL (Domain Specific Language) 来组合 Operators，实现真正的**开放式工作流组合**。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AFlow-GRPO 训练系统                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   vLLM      │───>│   DSL       │───>│   Workflow      │  │
│  │  Generator  │    │   Parser    │    │   Executor      │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                                      │            │
│         v                                      v            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   GRPO      │<───│   Reward    │<───│   Evaluator     │  │
│  │   Trainer   │    │   Computer  │    │   (gpt-4o-mini) │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 训练流程

1. **输入问题** → 模型根据问题类型生成 DSL 工作流
2. **DSL 解析** → 转换为可执行的 Python 代码
3. **工作流执行** → 按照 DSL 逻辑执行各个 Operator (通过 OpenAI API)
4. **奖励计算** → 评估答案正确性、效率等
5. **GRPO 更新** → 使用 WA-GRPO 更新模型参数

---

## 📦 项目结构

```
.
├── train.py                    # 主训练入口
├── train_grouped.py            # 分组训练入口
├── run_train.sh                # 一键启动脚本
├── setup_env.sh                # 环境配置脚本
├── requirements.txt            # Python依赖
│
├── config/                     # 配置文件 (10个)
│   ├── training.yaml           # 主训练配置 (P30)
│   ├── aflow_llm.yaml          # LLM API 配置
│   ├── datasets.yaml           # 数据集配置
│   ├── judge_prompts.yaml      # 评估提示词
│   └── operator.json           # Operator 定义
│
├── src/                        # 核心代码 (23个模块)
│   ├── grpo_trainer.py         # GRPO 训练器 (1425行)
│   ├── vllm_workflow_generator.py  # DSL生成器 (1593行)
│   ├── aflow_executor.py       # 工作流执行器 (1197行)
│   ├── reward_computer.py      # 奖励计算 (2207行)
│   ├── wa_grpo.py              # WA-GRPO 优势估计
│   └── unified_evaluator.py    # 评估器
│
├── scripts/                    # 辅助脚本 (26个)
│   ├── train_improved.py       # 改进训练脚本
│   ├── inference.py            # 推理脚本
│   ├── monitor_training.py     # 训练监控
│   └── download_datasets.py    # 数据下载
│
├── docs/                       # 技术文档 (20个)
│   ├── GRPO_COLLAPSE_ANALYSIS.md   # K=2问题深度分析
│   └── ...
│
├── data/
│   └── ready_to_train/
│       ├── train_10k_final.jsonl   # 训练集 (10K样本)
│       └── test_500_preprocessed.jsonl  # 测试集
│
└── logs/                       # 训练日志
    └── training_p30.log        # 最新实验日志
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|----------|
| GPU | V100 16GB | A100 40GB |
| Python | 3.10+ | 3.10.12 |
| CUDA | 12.0+ | 12.6 |

### 1. 克隆仓库

```bash
git clone https://github.com/beita6969/colab-grpo.git
cd colab-grpo

# 如果有 LFS 大文件
git lfs pull
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境

```bash
# 配置 API Key
export OPENAI_API_KEY="your-openai-api-key"

# 或使用配置脚本
source setup_env.sh
```

### 4. 启动训练

```bash
# 方式1: 使用启动脚本
./run_train.sh

# 方式2: 直接运行
python train.py --config config/training.yaml
```

---

## 🖥️ Google Colab 一键启动

```python
#@title AFlow-GRPO 一键启动
OPENAI_API_KEY = "sk-your-api-key"  #@param {type:"string"}

import os

# 检查 GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv

# 克隆仓库
!git clone https://github.com/beita6969/colab-grpo.git 2>/dev/null || (cd colab-grpo && git pull)
%cd colab-grpo
!git lfs pull

# 安装依赖
!pip install -q -r requirements.txt

# 配置环境
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'

# 启动训练
!python3 train.py --config config/training.yaml
```

---

## 🔧 DSL 语法

模型生成的工作流使用 DSL (Domain Specific Language) 表示：

| 语法 | 含义 | 示例 |
|------|------|------|
| `->` | 顺序执行 | `Custom -> Review -> Revise` |
| `[...]` | 并行执行 | `[Custom, Custom, Custom] -> ScEnsemble` |
| `?` | 条件分支 | `Review ? Revise : done` |
| `* n` | 循环执行 | `(Review -> Revise) * 3` |

### 示例工作流

```python
# 数学问题 - 编程验证
"Custom -> Programmer -> Review ? Revise : done"

# 代码生成 - 测试驱动
"CustomCodeGenerate -> Test -> Format"

# 复杂问题 - 多路投票
"[Custom, Custom, Custom] -> ScEnsemble -> Review"

# 迭代优化
"AnswerGenerate -> (Review -> Revise) * 2 -> Format"
```

---

## 🛠️ 可用 Operators

| Operator | 功能 | 输入 → 输出 |
|----------|------|-------------|
| **Custom** | 通用生成 | `(input, instruction)` → `response` |
| **AnswerGenerate** | 思维链推理 | `(input)` → `thought, answer` |
| **Programmer** | 代码执行 | `(problem, analysis)` → `code, output` |
| **CustomCodeGenerate** | 代码生成 | `(problem, entry_point, instruction)` → `code` |
| **Test** | 测试验证 | `(problem, solution, entry_point)` → `result, solution` |
| **Review** | 解答审查 | `(problem, solution)` → `review_result, feedback` |
| **Revise** | 解答修改 | `(problem, solution, feedback)` → `solution` |
| **Format** | 格式化输出 | `(problem, solution)` → `solution` |
| **ScEnsemble** | 自洽集成 | `(solutions, problem)` → `response` |
| **MdEnsemble** | 多数投票 | `(solutions, problem)` → `solution` |

---

## ⚙️ 配置详解

### 主要参数 (`config/training.yaml`)

```yaml
# 实验配置
exp_name: "aflow_grpo_k2_b3_p30"

# GRPO 算法配置
num_return_sequences_in_group: 2   # K值: 每个问题生成K个工作流
rollout_batch_size: 3              # B值: 每批处理B个问题
learning_rate: 2.0e-6              # 学习率 (P30降低10倍)
kl_loss_coef: 0.005                # KL 散度惩罚系数
clip_range: 0.20                   # PPO 裁剪范围
gradient_accumulation_steps: 8     # 梯度累积

# LoRA 配置
lora_rank: 64
lora_alpha: 64
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"

# 温度调度
temperature_schedule:
  enabled: true
  initial: 0.3
  final: 0.15
  warmup_steps: 100
```

### 显存配置建议

| GPU | 显存 | K | B | grad_accum |
|-----|------|---|---|------------|
| T4 | 16GB | 2 | 2 | 8 |
| V100 | 16GB | 2 | 3 | 6 |
| A100 | 40GB | 4 | 4 | 4 |

---

## ⚠️ 已知问题与解决方案

### K=2 导致训练崩溃

**问题**: 当 `num_return_sequences_in_group=2` 时，97.4% 的梯度更新后模型输出崩溃。

**原因**: K=2 时组内归一化导致 advantage 恒为 ±1.0，极端值导致模型不稳定。

**解决方案** (详见 `docs/GRPO_COLLAPSE_ANALYSIS.md`):

```yaml
# 方案1: 增加 K 值 (推荐)
num_return_sequences_in_group: 8  # 从2改为8

# 方案2: 修改 advantage 计算
# 移除 std 归一化，只用 mean 归一化
```

---

## 📊 奖励系统

**5级奖励**：`[0, 0.2, 0.4, 0.7, 1.0]`

```yaml
reward_weights:
  correctness: 0.65    # 答案正确性
  efficiency: 0.15     # 执行效率
  simplicity: 0.10     # 工作流简洁度
  format: 0.05         # 输出格式
  repetition: 0.05     # 重复惩罚
```

---

## 📈 监控训练

```bash
# 实时日志
tail -f logs/training_p30.log

# 查看关键指标
grep -E "Step|reward|accuracy" logs/training_p30.log | tail -50

# 使用监控脚本
python scripts/monitor_training.py
```

---

## 📂 数据集格式

```json
{
  "question": "问题文本",
  "answer": "标准答案",
  "domain": "math|code|qa",
  "entry_point": "函数名 (仅code)"
}
```

**数据分布**：Math 33.3% / Code 33.3% / QA 33.4%

---

## 🔍 常见问题

### Q: DSL 解析失败？

系统会自动处理常见问题：
- `X ? Y : done` → 自动转换为 `X -> Y`
- `-> done` 后缀 → 自动移除

### Q: OOM (显存不足)？

```yaml
gradient_accumulation_steps: 8     # 增加累积
gradient_checkpointing: true       # 启用检查点
rollout_batch_size: 2              # 减少批次
```

### Q: OpenAI API 超时？

调整 `execution_timeout: 600` 或减少 `num_return_sequences_in_group`

---

## 🙏 致谢

- [AFlow](https://github.com/geekan/MetaGPT) - 工作流框架
- [GRPO](https://arxiv.org/abs/2402.03300) - 训练算法
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基础模型
- [PEFT](https://github.com/huggingface/peft) - LoRA 实现

---

## 📄 License

MIT License

---

**核心创新**：让模型学习 "如何组合工具"，而不是 "选择哪个预设方案"

---

*最后更新: 2025-12-05*
