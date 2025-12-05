# GRPO训练崩溃问题深度分析报告

**日期**: 2025-12-05
**实验**: P30 (aflow_grpo_k2_b3_p30)
**模型**: Qwen2.5-7B-Instruct + LoRA
**任务**: 工作流DSL生成

---

## 1. 问题现象

### 1.1 核心现象
训练过程中，**97.4%的梯度更新后立即导致模型输出崩溃**：
- 更新前：模型能输出有效DSL（如 `Custom -> Programmer -> Review ? Revise : done`）
- 更新后：模型输出无效文本（如 `### Step 1: Analyze the problem...`）

### 1.2 统计数据（P30实验，96步）
| 指标 | 数值 |
|-----|------|
| 总步数 | 96 |
| 触发梯度更新次数 | 38 |
| 更新后立即崩溃次数 | 37 |
| **崩溃率** | **97.4%** |

### 1.3 典型崩溃案例
```
Step 4: 准确率 50%, DSL输出正常
        → 触发梯度更新
Step 5: 准确率 0%, 6/6样本全部输出 "### Step 1:..."
        → P28机制跳过更新（因为全是fallback）
Step 6: 模型部分恢复...
```

---

## 2. 根本原因分析

### 2.1 直接原因：K=2导致极端Advantage值

当前配置：
```yaml
num_return_sequences_in_group: 2  # K=2
rollout_batch_size: 3             # B=3
# 每步总共 K×B = 6 个样本
```

**GRPO Advantage计算公式**（组内归一化）：
```python
for group_idx in range(batch_size):  # 3组
    group_rewards = rewards[group_idx * K : (group_idx + 1) * K]  # 每组2个
    mean = np.mean(group_rewards)
    std = np.std(group_rewards)
    advantages = (group_rewards - mean) / std
```

### 2.2 数学必然性

当 K=2 时，设组内奖励为 [r₁, r₂]：
```
mean = (r₁ + r₂) / 2
std = |r₁ - r₂| / 2

advantage₁ = (r₁ - mean) / std = (r₁ - r₂) / |r₁ - r₂| = ±1.0
advantage₂ = (r₂ - mean) / std = (r₂ - r₁) / |r₁ - r₂| = ∓1.0
```

**结论**：K=2时，只要两个样本奖励不同，advantage就是 **恰好 ±1.0**，没有任何中间值！

### 2.3 崩溃机制

以Step 4为例：
```
组0: 奖励=[1.0, 1.0] → mean=1.0, std=0 → advantage=[0, 0] (零方差)
组1: 奖励=[1.0, 0.2] → mean=0.6, std=0.4 → advantage=[+1.0, -1.0]
组2: 奖励=[0.2, 0.2] → mean=0.2, std=0 → advantage=[0, 0] (零方差)
```

**问题**：奖励=0.2的样本（生成了有效DSL！）获得了 **-1.0的负advantage**，被模型强烈"推离"！

模型学到：
1. 避免这些DSL输出（因为被惩罚了）
2. 但不知道应该输出什么替代品
3. 于是输出通用文本 "### Step 1:..." （这不在DSL词汇表中，不会被直接惩罚）

---

## 3. DSL多样性分析

### 3.1 总体统计
| 指标 | 数值 |
|-----|------|
| 总DSL输出 | 570 |
| 有效DSL (fallback=False) | 326 (57.2%) |
| Fallback输出 | 244 (42.8%) |
| 唯一DSL种类 | 20种 |

### 3.2 Top 10 DSL模式
| 排名 | 次数 | 占比 | DSL模式 |
|-----|------|------|---------|
| 1 | 115 | 35.3% | `Custom -> Programmer -> Review ? Revise : done` |
| 2 | 69 | 21.2% | `Custom -> Programmer -> Review -> Revise` |
| 3 | 47 | 14.4% | `Programmer` |
| 4 | 21 | 6.4% | `Custom -> Programmer` |
| 5 | 18 | 5.5% | `Custom -> Programmer -> Review` |
| 6 | 13 | 4.0% | `Programmer -> Review ? Revise : done` |
| 7 | 10 | 3.1% | `Programmer -> Review -> Revise` |
| 8 | 8 | 2.5% | `Custom` |
| 9 | 6 | 1.8% | `Programmer -> Review` |
| 10 | 5 | 1.5% | `Custom -> Review ? Revise : done` |

### 3.3 Operator使用频率
| Operator | 出现次数 | 占比 |
|----------|---------|------|
| Custom | 312 | 95.7% |
| Review | 284 | 87.1% |
| Revise | 284 | 87.1% |
| Programmer | 208 | 63.8% |
| ScEnsemble | 1 | 0.3% |

### 3.4 结构分类
| 类别 | 种类数 | 出现次数 |
|------|--------|---------|
| 条件分支 (? done) | 5种 | 152次 |
| 简单链 (->) | 10种 | 154次 |
| 单步操作 | 3种 | 60次 |
| 循环结构 (*) | 0种 | 0次 |

---

## 4. 业界研究结果

### 4.1 K值对比
| 来源 | K值 | 备注 |
|-----|-----|------|
| **我们** | **2** | 严重不足 |
| 业界最低推荐 | 4 | 最低限度 |
| 常见设置 | 8-16 | 通用推荐 |
| DeepSeekMath | 64 | 大规模训练 |
| OpenRLHF | 4-32 | 框架默认 |

### 4.2 GRPO的已知问题

1. **组内归一化的缺陷**
   - 小K值导致高方差advantage
   - 零方差组导致无信号
   - 组间不一致

2. **更优算法**
   | 算法 | 来源 | 改进点 |
   |-----|------|--------|
   | REINFORCE++ | OpenAI | 全局归一化 |
   | Dr. GRPO | 学术研究 | 移除std归一化 |
   | GSPO | Qwen团队 | 解决相似样本问题 |
   | GTPO | 学术研究 | Token级别奖励 |
   | DAPO | ByteDance | Decoupled clipping |

### 4.3 关键论文/资源
- GRPO原论文: DeepSeekMath团队
- REINFORCE++: https://arxiv.org/abs/2501.03262 (OpenAI)
- GSPO: Qwen团队博客
- Dr. GRPO: 移除std归一化的变体

---

## 5. 解决方案

### 方案A：增加K值（推荐首选）
```yaml
# config/training.yaml
num_return_sequences_in_group: 8  # 从2改为8
rollout_batch_size: 2             # 相应调整以控制显存
```

**优点**：最小改动，业界验证有效
**缺点**：需要更多显存

### 方案B：移除std归一化（Dr. GRPO）
```python
# src/grpo_trainer.py 约638-663行
# 修改前：
group_advantages = [(r - group_mean) / group_std for r in group_rewards]

# 修改后：
group_advantages = [r - group_mean for r in group_rewards]
# 可选：添加clipping
group_advantages = np.clip(group_advantages, -2.0, 2.0)
```

**优点**：消除±1.0极端值，改动简单
**缺点**：可能需要调整学习率

### 方案C：全局归一化（REINFORCE++风格）
```python
# src/grpo_trainer.py
# 替换组内归一化为全局归一化

# 修改前：按组计算mean/std
# 修改后：
all_rewards = np.array(all_rewards_flat)
global_mean = np.mean(all_rewards)
global_std = np.std(all_rewards) + 1e-8
advantages = (all_rewards - global_mean) / global_std
```

**优点**：理论最优，DeepSeek/OpenAI使用
**缺点**：需要更多代码改动

### 方案D：添加Advantage Clipping
```python
# 在现有代码基础上添加
advantages = np.clip(advantages, -3.0, 3.0)
```

**优点**：防止极端更新
**缺点**：治标不治本

### 方案E：切换到GSPO/GTPO
需要重新实现trainer，工作量较大，但长期效果最好。

---

## 6. 推荐实施顺序

1. **第一步**：增加K到8（方案A）
   - 改动最小
   - 如果显存不足，尝试K=4

2. **第二步**：如果K=8仍有问题，添加std移除（方案B）
   ```python
   # 组合使用
   K = 8
   advantages = rewards - mean  # 不除以std
   ```

3. **第三步**：如果需要更稳定，实现全局归一化（方案C）

4. **长期**：考虑切换到GSPO/GTPO

---

## 7. 相关代码位置

| 文件 | 行号 | 内容 |
|-----|------|------|
| `config/training.yaml` | - | K值配置 |
| `src/grpo_trainer.py` | 635-714 | Advantage计算 |
| `src/grpo_trainer.py` | 938-1073 | Policy更新 |
| `src/reward_functions.py` | - | 奖励函数定义 |

---

## 8. 实验记录

### P30配置
```yaml
exp_name: "aflow_grpo_k2_b3_p30"
learning_rate: 2.0e-6  # 比P29低10倍
num_return_sequences_in_group: 2
rollout_batch_size: 3
gradient_accumulation_steps: 8
kl_loss_coef: 0.005
entropy_coef: 0.0
```

### P30结果
- 运行96步
- 38次更新，37次崩溃
- 崩溃率97.4%
- 即使学习率降低10倍，问题仍然存在

### 关键发现
降低学习率**不能解决K=2的根本问题**，因为问题在于advantage的极端值，而不是学习率。

---

## 9. 附录：分析脚本

以下Python脚本用于分析训练日志：

### 分析有效DSL
```python
# /tmp/analyze_valid_dsl.py
import re
from collections import Counter

with open('/home/claude-user/new-colab/logs/training_p30.log', 'r') as f:
    content = f.read()

dsl_entries = re.findall(r'DSL质量: fallback=(True|False), ops=(\d+), chain=(True|False), DSL=([^\n]+)', content)
valid_entries = [(fb, ops, chain, dsl) for fb, ops, chain, dsl in dsl_entries if fb == 'False']

print(f"有效DSL: {len(valid_entries)}/{len(dsl_entries)} ({len(valid_entries)/len(dsl_entries)*100:.1f}%)")
```

### 分析崩溃模式
```python
# /tmp/detailed_analysis.py
# 分析 [更新→崩溃] 模式
```

### 分析Advantage
```python
# /tmp/analyze_advantage.py
# 模拟K=2的advantage计算
```

---

## 10. 下一步行动

1. [ ] 修改 `config/training.yaml`，将K从2改为8
2. [ ] 如需要，修改 `src/grpo_trainer.py` 中的advantage计算
3. [ ] 启动新实验 P31
4. [ ] 监控前20步是否仍有崩溃现象
5. [ ] 如果K=8仍有问题，尝试全局归一化

---

**文档作者**: Claude Code
**最后更新**: 2025-12-05
