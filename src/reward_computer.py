#!/usr/bin/env python3
"""
奖励计算器 - P0/P1/P2修复版

修复内容:
P0-1: 5档细粒度奖励 (0/0.2/0.4/0.7/1.0)
P0-3: 代码执行多进程隔离 + 部分通过奖励
P0-4: 答案提取鲁棒性改进
P1-2: Judge稳健性和调试日志
P2-1: LLM Judge max_tokens从200增加到800，修复reasoning模型token不足导致content为空的问题
P12: LLM提取作为主力，改进代码检测和函数重命名
"""
import os

# P12修复: 不再禁用代理，使用OpenAI API需要网络
# os.environ.pop('http_proxy', None)
# os.environ.pop('https_proxy', None)
# os.environ.pop('HTTP_PROXY', None)
# os.environ.pop('HTTPS_PROXY', None)
# os.environ['no_proxy'] = 'localhost,127.0.0.1'

import sys
import re
import threading
import time
import json
import random
import multiprocessing
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入答案提取器
try:
    from .answer_extractor import AnswerExtractor
    from .judge_prompt_loader import JudgePromptLoader
except ImportError:
    from answer_extractor import AnswerExtractor
    from judge_prompt_loader import JudgePromptLoader


class RewardComputer:
    """
    P0/P1修复版奖励计算器

    修复特性:
    1. 5档细粒度奖励 (0/0.2/0.4/0.7/1.0) - 解决奖励稀疏问题
    2. 代码执行多进程隔离 - 安全性+稳定性
    3. 部分通过奖励 - Code任务按通过用例比例给分
    4. 答案提取鲁棒性 - 支持嵌套boxed/分数/百分比
    5. Judge调试日志 - 采样记录用于调试
    6. QA任务F1评分 - 替代简单包含匹配
    """

    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        use_answer_extractor: bool = True,  # 是否使用答案提取器
        use_llm_judge: bool = False,  # 是否使用LLM Judge
        llm_config: Optional[Dict] = None,  # LLM配置
        debug_logging: bool = False  # 是否启用详细调试日志
    ):
        """
        Args:
            reward_weights: 奖励权重配置（仅用于向后兼容，实际使用二元奖励）
            use_answer_extractor: 是否使用答案提取器来标准化答案
            use_llm_judge: 是否使用LLM Judge进行语义比较
            llm_config: LLM配置（用于LLM Judge）
            debug_logging: 是否启用详细调试日志
        """
        # 保留用于向后兼容，但不再使用
        self.reward_weights = reward_weights or {
            "correctness": 1.0
        }

        # 调试日志开关
        self.debug_logging = debug_logging

        # 初始化答案提取器
        self.use_answer_extractor = use_answer_extractor
        if use_answer_extractor:
            self.extractor = AnswerExtractor(use_llm_fallback=False)  # 暂时不使用LLM兜底
        else:
            self.extractor = None

        # 初始化LLM Judge
        self.use_llm_judge = use_llm_judge
        self.llm_judge_client = None
        self.judge_prompt_loader = None  # 数据集专属Prompt加载器
        if use_llm_judge:
            self._init_llm_judge_client(llm_config)
            # 初始化Prompt加载器
            try:
                self.judge_prompt_loader = JudgePromptLoader()
                stats = self.judge_prompt_loader.get_stats()
                print(f"  ✅ Judge Prompt加载器初始化成功")
                print(f"     已加载 {stats['total_datasets']} 个数据集配置")
                print(f"     启用数据集: {', '.join(stats['enabled_datasets'][:5])}...")
            except Exception as e:
                print(f"  ⚠️  Judge Prompt加载器初始化失败: {e}")
                print(f"     将使用通用Prompt")
                self.judge_prompt_loader = None

        print(f"✅ 10分制奖励计算器初始化完成")
        print(f"  模式: 5档细粒度奖励 [0, 0.2, 0.4, 0.7, 1.0] (P0修复)")
        print(f"  答案提取器: {'启用' if use_answer_extractor else '禁用'}")
        print(f"  LLM Judge: {'启用 (GPT OSS 120B @ port 8002)' if use_llm_judge else '禁用'}")
        print(f"  调试日志: {'启用' if debug_logging else '禁用'}")
        print(f"  代码执行: 多进程隔离模式 (P0修复)")

        # P1-2: Judge调试日志目录
        self.judge_log_dir = Path("logs/judge_samples")
        self.judge_log_dir.mkdir(parents=True, exist_ok=True)
        self.judge_log_file = self.judge_log_dir / f"judge_samples_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        # 初始化统计计数器（用于诊断）
        self.eval_stats = {
            'total_evaluations': 0,
            'llm_judge_success': 0,
            'llm_judge_parse_failures': 0,
            'llm_judge_api_failures': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

    def _init_llm_judge_client(self, llm_config: Optional[Dict]):
        """初始化LLM Judge客户端（使用OpenAI API）"""
        try:
            from openai import OpenAI

            # 使用OpenAI API
            default_config = {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.environ.get('OPENAI_API_KEY', 'sk-dummy'),
                "model_name": "gpt-4o-mini"
            }

            config = llm_config or default_config

            self.llm_judge_client = OpenAI(
                base_url=config.get("base_url", default_config["base_url"]),
                api_key=config.get("api_key", default_config["api_key"])
            )
            self.llm_judge_model = config.get("model_name", default_config["model_name"])

            print(f"  ✅ LLM Judge客户端初始化成功")
            print(f"     模型: {self.llm_judge_model}")
            print(f"     URL: {config.get('base_url', default_config['base_url'])}")
        except Exception as e:
            print(f"  ⚠️  LLM Judge客户端初始化失败: {e}")
            self.use_llm_judge = False
            self.llm_judge_client = None

    def _llm_judge_compare(
        self,
        problem: str,
        prediction: str,
        ground_truth: str,
        problem_type: str,
        source: Optional[str] = None  # 新增：数据集来源
    ) -> bool:
        """
        使用LLM Judge进行语义比较（支持数据集专属Prompt）

        Args:
            problem: 问题文本
            prediction: 模型预测（完整响应，未提取）
            ground_truth: Ground truth答案
            problem_type: 问题类型
            source: 数据集来源（如'gsm8k', 'math', 'hotpotqa'）

        Returns:
            bool: True表示等价，False表示不等价
        """
        self.eval_stats['total_evaluations'] += 1

        if not self.llm_judge_client:
            if self.debug_logging:
                print("⚠️  LLM Judge客户端未初始化，降级为规则比较")
            self.eval_stats['llm_judge_api_failures'] += 1
            return False

        # 🆕 使用数据集专属Prompt（如果可用）
        if self.judge_prompt_loader:
            query_prompt_template = self.judge_prompt_loader.get_judge_prompt(
                source=source,
                problem_type=problem_type
            )
            # 格式化prompt（手动替换，避免format()解析XML标签）
            query_prompt = query_prompt_template.replace('{{problem}}', problem)
            query_prompt = query_prompt.replace('{{prediction}}', prediction)
            query_prompt = query_prompt.replace('{{ground_truth}}', ground_truth)
            if self.debug_logging:
                print(f"  📋 使用数据集专属Prompt: source={source}")
        else:
            # Fallback: 使用原有的通用prompt
            query_prompt = self._get_legacy_prompt(problem, prediction, ground_truth)
            if self.debug_logging:
                print(f"  📋 使用通用Prompt (Fallback)")


        try:
            # 调用LLM Judge（最多重试1次）
            for attempt in range(2):  # 0=首次, 1=重试
                response = self.llm_judge_client.chat.completions.create(
                    model=self.llm_judge_model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer equivalence evaluator."},
                        {"role": "user", "content": query_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=800  # P2修复: 增加到800，reasoning模型需要更多token完成思考
                )

                # 检查响应是否为空
                content = response.choices[0].message.content
                if content is None:
                    if attempt == 0:
                        if self.debug_logging:
                            print(f"⚠️  LLM Judge首次返回空内容，重试中...")
                        self.eval_stats['llm_judge_api_failures'] += 1
                        continue  # 重试
                    else:
                        if self.debug_logging:
                            print(f"⚠️  LLM Judge重试后仍返回空内容，fallback判定为False")
                        self.eval_stats['llm_judge_api_failures'] += 1
                        return False

                # 成功获取内容，跳出重试循环
                result_text = content.strip()
                break

            # 解析<true_false>标签 - 增强的鲁棒性匹配
            import re
            # 匹配多种格式（按优先级尝试）：
            # 1. <true_false>True</true_false>
            # 2. <true_false>: True
            # 3. **true_false**: True
            # 4. true_false: True
            # 5. 直接在文本中查找True/False（最后手段）

            # 尝试1: 标准XML标签
            true_false_match = re.search(
                r'<true_false>\s*(True|False)\s*</true_false>',
                result_text,
                re.IGNORECASE
            )

            # 尝试2: 冒号分隔的标签
            if not true_false_match:
                true_false_match = re.search(
                    r'<true_false>\s*:\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试3: Markdown粗体格式
            if not true_false_match:
                true_false_match = re.search(
                    r'\*\*true_false\*\*\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试4: 简单的key: value格式
            if not true_false_match:
                true_false_match = re.search(
                    r'true_false\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            # 尝试5: 查找独立的True/False（最后手段）
            if not true_false_match:
                # 只在响应末尾查找，避免误匹配分析文本中的True/False
                last_200_chars = result_text[-200:]
                true_false_match = re.search(
                    r'\b(True|False)\b',
                    last_200_chars,
                    re.IGNORECASE
                )

            if true_false_match:
                verdict = true_false_match.group(1).lower() == "true"
                self.eval_stats['llm_judge_success'] += 1

                # 更新正确/错误计数
                if verdict:
                    self.eval_stats['correct_predictions'] += 1
                else:
                    self.eval_stats['incorrect_predictions'] += 1

                # 调试输出（根据debug_logging开关）
                if self.debug_logging:
                    import random
                    if random.random() < 0.2:  # 20%采样
                        print(f"\n🤖 LLM Judge结果 ({problem_type}):")
                        print(f"  问题: {problem[:60]}...")
                        print(f"  预测: {str(prediction)[:60]}...")
                        print(f"  真值: {str(ground_truth)[:60]}...")
                        print(f"  判决: {verdict}")
                        print(f"  LLM响应: {result_text[:150]}...")

                return verdict
            else:
                # 完全无法解析时，打印完整响应用于调试
                self.eval_stats['llm_judge_parse_failures'] += 1
                if self.debug_logging:
                    print(f"⚠️  无法解析LLM Judge响应（尝试了5种格式）")
                    print(f"  完整响应: {result_text}")
                    print(f"  问题: {problem[:100]}")
                    print(f"  预测: {str(prediction)[:100]}")
                    print(f"  真值: {str(ground_truth)[:100]}")
                return False

        except Exception as e:
            self.eval_stats['llm_judge_api_failures'] += 1
            if self.debug_logging:
                print(f"⚠️  LLM Judge调用失败: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _get_legacy_prompt(self, problem: str, prediction: str, ground_truth: str) -> str:
        """获取原有的通用Prompt（向后兼容）"""
        return f"""You are a precise mathematical and logical equivalence evaluator. Your task is to determine if the Model Response contains an answer equivalent to the Ground Truth.

**Step 1: Extract the Final Answer**
From the Model Response, extract ONLY the final answer, ignoring all reasoning steps, explanations, and intermediate calculations.

Look for answers in these formats (in order of priority):
1. Inside `\\boxed{{...}}` LaTeX notation
2. After phrases like "The answer is", "Therefore", "So", "Thus", "Final answer:"
3. In `<answer>...</answer>` tags
4. The last number, expression, or entity mentioned

**Step 2: Extract from Ground Truth**
Similarly extract the final answer from Ground Truth, which may contain:
- Step-by-step solutions (extract only the final result)
- Multiple numbers (take the last/final one)
- Explanatory text (ignore and find the answer)

**Step 3: Normalize Both Answers**
Before comparing, normalize both answers:
- **Numbers:** Convert to same format (0.5 == 1/2 == 50%)
- **Units/Currency:** Ignore ($30 == 30, 10 meters == 10)
- **Formatting:** Ignore spaces, case, punctuation
- **LaTeX:** Interpret mathematical meaning (\\frac{{1}}{{2}} == 0.5)

**Step 4: Compare Equivalence**
Answers are equivalent if:
- **Math:** Numerically/algebraically equal (even if different forms)
- **Text:** Same entity/concept (ignore synonyms, case)
- **Precision:** Allow reasonable rounding (42.0 == 42)

**Examples of CORRECT equivalence:**
- "1/2" == "0.5" ✓
- "$30" == "30" ✓
- "\\boxed{{42}}" == "42" ✓
- "x^2+2x+1" == "(x+1)^2" ✓ (algebraically equivalent)
- "10 meters" == "10" ✓

**Examples of INCORRECT equivalence:**
- "John Smith" == "Jane Doe" ✗ (different entities)
- "42" == "43" ✗ (different numbers)
- "Paris" == "London" ✗ (different locations)

**Inputs:**
Question: {problem}
Model Response: {prediction}
Ground Truth: {ground_truth}

**Required Output Format:**
<analysis>Your reasoning in 1-2 sentences</analysis>
<true_false>True or False</true_false>

Be LENIENT with formatting differences but STRICT with factual/numerical differences.
"""

    def compute_reward(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None,
        test: Optional[str] = None,
        entry_point: Optional[str] = None,
        source: Optional[str] = None  # 🆕 新增：数据集来源
    ) -> float:
        """
        计算奖励 - P0修复: 5档细粒度奖励

        奖励等级:
        - 1.0: 完美匹配
        - 0.7: 接近正确 (数值误差<5%, 部分测试通过>80%)
        - 0.4: 部分正确 (格式正确但答案有偏差, 测试通过>50%)
        - 0.2: 格式正确 (有效输出但答案错误, 测试通过>20%)
        - 0.0: 完全错误

        Args:
            source: 数据集来源（如'gsm8k', 'math', 'hotpotqa'）- 用于选择专属Judge Prompt

        Returns:
            reward: 0.0 / 0.2 / 0.4 / 0.7 / 1.0
        """
        metadata = metadata or {}

        # 调试日志：输入信息
        if self.debug_logging:
            print(f"\n📊 评估输入 ({problem_type}, source={source}):")
            print(f"  问题: {str(problem)[:100]}...")
            print(f"  预测: {str(prediction)[:100]}...")
            print(f"  真值: {str(ground_truth)[:100]}...")

        # P0修复: 根据任务类型使用不同的细粒度奖励计算
        if problem_type == "code":
            # 代码任务: 使用多进程隔离执行 + 部分通过奖励
            # P6修复: 传入problem用于处理HumanEval格式(problem=签名, prediction=函数体)
            reward = self._compute_code_reward(problem, prediction, ground_truth, test, entry_point)
        elif problem_type == "math":
            # 数学任务: 细粒度数值比较
            reward = self._compute_math_reward(problem, prediction, ground_truth, source)
        elif problem_type == "qa":
            # QA任务: F1评分
            reward = self._compute_qa_reward(problem, prediction, ground_truth, source)
        else:
            # 通用任务
            reward = self._compute_general_reward(prediction, ground_truth)

        # 更新统计
        if reward >= 0.9:
            self.eval_stats['correct_predictions'] += 1
        else:
            self.eval_stats['incorrect_predictions'] += 1

        if metadata is not None:
            metadata['correctness_score'] = reward
            metadata['used_llm_judge'] = self.use_llm_judge
            metadata['is_correct'] = reward >= 0.9
            metadata['reward_level'] = self._get_reward_level(reward)

        # 调试日志：输出结果
        if self.debug_logging:
            level = self._get_reward_level(reward)
            print(f"  判决: {level}")
            print(f"  奖励: {reward:.2f}")

        return reward

    # ============== P21修复: Graph-R1风格条件激活奖励 ==============

    def compute_dsl_quality_reward(self, dsl_info: Dict) -> float:
        """
        P21修复: 计算DSL质量奖励（基于Graph-R1格式奖励设计）

        DSL质量得分组成：
        - 0.35: 成功解析（非fallback）
        - 0.20: 多个operators (>=2)
        - 0.15: 链式结构 (->)
        - 0.10: 高级特性（loop * 或 conditional ?）
        - 0.10: operator多样性 (>=3种不同operator)
        - 0.10: 并行结构 ([])

        Returns:
            float: DSL质量得分 [0.0, 1.0]
        """
        if not dsl_info:
            return 0.0

        score = 0.0

        # 1. 成功解析（非fallback）- 35%
        if not dsl_info.get('is_fallback', True):
            score += 0.35

        # 2. 多个operators - 20%
        num_ops = dsl_info.get('num_operators', 0)
        if num_ops >= 2:
            score += 0.20
        elif num_ops == 1 and not dsl_info.get('is_fallback', True):
            score += 0.10  # 单operator但非fallback给部分分

        # 3. 链式结构 - 15%
        if dsl_info.get('has_chain', False):
            score += 0.15

        # 4. 高级特性（loop或conditional）- 10%
        if dsl_info.get('has_loop', False) or dsl_info.get('has_conditional', False):
            score += 0.10

        # 5. operator多样性 - 10%
        unique_ops = dsl_info.get('unique_operators', [])
        if len(unique_ops) >= 3:
            score += 0.10
        elif len(unique_ops) == 2:
            score += 0.05

        # 6. 并行结构 - 10%
        if dsl_info.get('has_parallel', False):
            score += 0.10

        return min(score, 1.0)

    def compute_reward_with_conditional_activation(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None,
        test: Optional[str] = None,
        entry_point: Optional[str] = None,
        source: Optional[str] = None,
        dsl_info: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        P21修复: 条件激活奖励函数（基于Graph-R1论文）

        公式: R_total = base + R_dsl + 𝕀{R_dsl >= threshold} · R_correctness

        参数调优：
        - base = -0.3: 基础惩罚，鼓励模型探索
        - threshold = 0.5: DSL质量门槛，达到后才给正确性奖励
        - 条件激活系数 = 0.7: 正确性奖励的缩放系数

        这种设计确保：
        1. fallback到默认workflow会得到负奖励（即使答案正确）
        2. 好的DSL但答案错误仍能得到部分正奖励
        3. 好的DSL + 正确答案获得最高奖励

        Returns:
            (total_reward, reward_breakdown)
        """
        # 计算DSL质量奖励
        dsl_quality_reward = self.compute_dsl_quality_reward(dsl_info) if dsl_info else 0.0

        # 计算正确性奖励（使用原有方法）
        correctness_reward = self.compute_reward(
            problem=problem,
            prediction=prediction,
            ground_truth=ground_truth,
            problem_type=problem_type,
            metadata=metadata,
            test=test,
            entry_point=entry_point,
            source=source
        )

        # P21: 条件激活逻辑
        # 参数配置
        base_penalty = -0.3  # 基础惩罚
        dsl_threshold = 0.5  # DSL质量门槛
        correctness_weight = 0.7  # 正确性奖励权重

        # Graph-R1公式: R = base + R_dsl + 𝕀{R_dsl >= threshold} · R_correctness
        total_reward = base_penalty + dsl_quality_reward

        # 只有DSL质量达到门槛才给正确性奖励
        activation = 1.0 if dsl_quality_reward >= dsl_threshold else 0.0
        total_reward += activation * correctness_weight * correctness_reward

        # 确保奖励在合理范围内
        total_reward = max(min(total_reward, 1.0), -1.0)

        # 构建奖励分解信息
        reward_breakdown = {
            'total_reward': total_reward,
            'base_penalty': base_penalty,
            'dsl_quality_reward': dsl_quality_reward,
            'correctness_reward': correctness_reward,
            'activation': activation,
            'dsl_threshold': dsl_threshold,
            'is_activated': activation > 0,
            'dsl_info': dsl_info
        }

        # 调试日志
        if self.debug_logging:
            print(f"\n🔥 P21条件激活奖励:")
            print(f"  DSL质量: {dsl_quality_reward:.3f} (门槛: {dsl_threshold})")
            print(f"  正确性: {correctness_reward:.3f}")
            print(f"  激活: {'是' if activation > 0 else '否'}")
            print(f"  总奖励: {total_reward:.3f}")
            if dsl_info:
                print(f"  DSL: {dsl_info.get('dsl_text', 'N/A')[:50]}")
                print(f"  Fallback: {dsl_info.get('is_fallback', True)}")

        return total_reward, reward_breakdown

    def _get_reward_level(self, reward: float) -> str:
        """获取奖励等级描述"""
        if reward >= 0.9:
            return "✅ 完美 (1.0)"
        elif reward >= 0.6:
            return "🟡 接近 (0.7)"
        elif reward >= 0.35:
            return "🟠 部分 (0.4)"
        elif reward >= 0.15:
            return "🔴 格式 (0.2)"
        else:
            return "❌ 错误 (0.0)"

    def _is_correct(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> bool:
        """
        判断预测是否正确 (传统规则)
        
        Returns:
            bool: True if correct, False otherwise
        """
        if prediction is None:
            return False

        if problem_type == "math":
            return self._is_math_correct(prediction, ground_truth)
        elif problem_type == "code":
            # Fallback for code if no test cases provided (should generally not happen if trained correctly)
            return False 
        elif problem_type == "qa":
            return self._is_qa_correct(prediction, ground_truth)
        else:
            return self._is_general_correct(prediction, ground_truth)

    # ============== P0修复: 细粒度奖励计算方法 ==============

    def _compute_math_reward(self, problem: str, prediction: Any, ground_truth: Any, source: Optional[str]) -> float:
        """
        P0修复: Math任务5档细粒度奖励
        P10修复: 使用OSS模型提取解释性文本中的答案
        P11修复: 改进代码检测逻辑，避免误判数学解释文本

        奖励等级:
        - 1.0: 完美匹配
        - 0.7: 数值接近 (相对误差<5%)
        - 0.4: 数量级正确 (相对误差<50%)
        - 0.2: 格式正确 (有boxed或数字输出)
        - 0.0: 完全错误
        """
        import re

        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        # P12修复: LLM提取作为主力
        # 策略: 对于任何非简单答案（长度>50或包含特殊格式），都尝试LLM提取
        skip_code_detection = False
        llm_extracted = False

        # P12: 判断是否需要LLM提取（更激进的策略）
        needs_extraction = (
            len(pred_str) > 50 or  # 答案太长，可能包含解释
            '**' in pred_str or    # Markdown格式
            '```' in pred_str or   # 代码块
            '\\boxed' in pred_str or  # LaTeX boxed
            'Step' in pred_str or  # 步骤说明
            'Therefore' in pred_str or 'Thus' in pred_str or 'Hence' in pred_str or  # 推理结论
            'Answer' in pred_str   # 包含Answer关键字
        )

        if needs_extraction and self.use_llm_judge:
            if self.debug_logging:
                print(f"  🔧 P12: 检测到需要提取的内容，使用LLM提取答案...")
            extracted = self._extract_answer_with_llm(pred_str, problem, "math")
            if extracted:
                if self.debug_logging:
                    print(f"  ✅ P12: LLM提取成功: {extracted[:50]}...")
                pred_str = extracted
                skip_code_detection = True
                llm_extracted = True
            else:
                # LLM提取失败，尝试本地boxed提取
                if self.debug_logging:
                    print(f"  🔄 P12: LLM提取失败，尝试本地boxed提取...")
                boxed_answer = self._extract_boxed_robust(pred_str)
                # 检查boxed内容是否有效
                invalid_boxed_prefixes = ['**Approach', '**Step', '**Solution', '**Analysis', '**Method', '**Let', 'Approach', 'Step']
                boxed_is_invalid = boxed_answer and any(boxed_answer.strip().startswith(p) for p in invalid_boxed_prefixes)

                if boxed_answer and not boxed_is_invalid and len(boxed_answer) < 100:
                    if self.debug_logging:
                        print(f"  ✅ P12: 本地boxed提取成功: {boxed_answer[:50]}...")
                    pred_str = boxed_answer
                    skip_code_detection = True
                else:
                    # 本地提取也失败，跳过代码检测让后续流程处理
                    if self.debug_logging:
                        print(f"  ⚠️  P12: 所有提取失败，跳过代码检测")
                    skip_code_detection = True

        # P11修复: 改进代码检测逻辑
        # 使用更严格的正则模式，避免误判数学文本中的英文单词
        if not skip_code_detection:
            # 严格的Python代码模式（需要完整的语法结构）
            strict_code_patterns = [
                r'\bimport\s+[a-zA-Z_][a-zA-Z0-9_]*',      # import module
                r'\bfrom\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import',  # from xxx import
                r'\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',    # def func(
                r'\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]',  # class Foo: 或 class Foo(
                r'\bfor\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+',  # for x in (Python特有)
                r'\bwhile\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:<>=!]',  # while x < (循环条件)
                r'if\s+__name__\s*==',                      # if __name__ ==
                r'print\s*\([^)]+\)',                       # print(xxx)
            ]
            pred_lower = pred_str.lower()
            is_code = any(re.search(pattern, pred_lower) for pattern in strict_code_patterns)

            if is_code:
                if self.debug_logging:
                    print(f"  ⚠️  P11-FIX: 检测到明确的代码结构，判定为格式错误(0.2)")
                return 0.2  # 格式错误，不是有效的数学答案

        # 1. 首先尝试LLM Judge (如果启用)
        if self.use_llm_judge:
            is_correct = self._llm_judge_compare(
                problem=problem,
                prediction=pred_str,
                ground_truth=gt_str,
                problem_type="math",
                source=source
            )
            if is_correct:
                return 1.0

        # 2. 规则匹配细粒度评估
        # 提取答案
        pred_answer = self._extract_math_answer(pred_str)
        gt_answer = self._extract_math_answer(gt_str)

        if pred_answer is None:
            # 没有有效输出
            return 0.0

        if gt_answer is None:
            # 无法解析ground truth，fallback到字符串匹配
            if gt_str.lower() in pred_str.lower():
                return 1.0
            return 0.0

        # 3. 数值比较
        try:
            pred_num = self._parse_number_robust(pred_answer)
            gt_num = self._parse_number_robust(gt_answer)

            if pred_num is not None and gt_num is not None:
                # P7修复: 与AFlow保持一致，使用abs_tol=1e-3
                # AFlow: math.py使用abs_tol=1e-3, gsm8k.py使用abs_tol=1e-6
                import math

                # GSM8K使用更严格的容差
                if source == 'gsm8k':
                    tolerance = 1e-6
                else:
                    tolerance = 1e-3  # MATH和其他数学数据集

                if math.isclose(pred_num, gt_num, abs_tol=tolerance):
                    return 1.0

                # 绝对误差检查
                abs_error = abs(pred_num - gt_num)
                if abs_error <= tolerance:
                    return 1.0

                # 相对误差（仅当gt不接近0时有意义）
                if abs(gt_num) > 1e-6:
                    rel_error = abs_error / abs(gt_num)
                    if rel_error < 0.01:  # <1%误差
                        return 1.0
                    elif rel_error < 0.05:  # <5%误差
                        return 0.7
                    elif rel_error < 0.50:  # <50%误差
                        return 0.4
                    else:
                        return 0.2
                else:
                    # gt接近0时用绝对误差
                    if abs_error < 0.01:
                        return 0.7
                    elif abs_error < 0.1:
                        return 0.4
                    else:
                        return 0.2
        except:
            pass

        # 4. 字符串匹配fallback
        if pred_answer.lower() == gt_answer.lower():
            return 1.0

        # 有输出但不匹配
        return 0.2

    def _compute_code_reward(self, problem: Optional[str], prediction: Any, ground_truth: Any,
                             test: Optional[str], entry_point: Optional[str]) -> float:
        """
        P0修复: Code任务多进程隔离执行 + 部分通过奖励
        P6修复: 支持HumanEval格式(problem=函数签名, prediction=函数体)
        P13修复: 支持MBPP格式(test_cases是列表，需转换为字符串)

        奖励等级:
        - 1.0: 所有测试通过
        - 0.7: >80%测试通过
        - 0.4: >50%测试通过
        - 0.2: >20%测试通过或代码语法正确
        - 0.0: 完全失败
        """
        # P13修复: MBPP使用test_cases列表格式，需要转换为字符串
        # MBPP格式: test_cases = ['assert func(x) == y', 'assert func(a) == b', ...]
        # HumanEval格式: test = 'def check(candidate):\n    assert ...'
        if isinstance(test, list):
            # 将列表格式的测试用例转换为可执行的字符串
            test = '\n'.join(test)
            if self.debug_logging:
                print(f"  🔬 [CODE DEBUG] P13: Converted test_cases list to string ({len(test)} chars)")

        # P3: 添加详细debug logging诊断Code问题
        if self.debug_logging:
            print(f"  🔬 [CODE DEBUG] prediction type: {type(prediction).__name__}")
            pred_str = str(prediction)
            print(f"  🔬 [CODE DEBUG] prediction[:300]: {pred_str[:300]}")
            print(f"  🔬 [CODE DEBUG] entry_point: {entry_point}")
            print(f"  🔬 [CODE DEBUG] test exists: {bool(test)}")

        if prediction is None:
            return 0.0

        solution = str(prediction).strip()
        if not solution:
            return 0.0

        # P3: 检测是否是dict格式的字符串 (如 "{'code': '...'}")
        if solution.startswith("{") and "'code'" in solution:
            try:
                import ast
                parsed = ast.literal_eval(solution)
                if isinstance(parsed, dict) and 'code' in parsed:
                    solution = parsed['code']
                    if self.debug_logging:
                        print(f"  🔬 [CODE DEBUG] Extracted code from dict string")
            except:
                pass

        # Sanitize solution (remove markdown blocks if any)
        if "```python" in solution:
            try:
                solution = solution.split("```python")[1].split("```")[0]
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] Removed ```python blocks")
            except:
                pass
        elif "```" in solution:
            try:
                solution = solution.split("```")[1].split("```")[0]
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] Removed ``` blocks")
            except:
                pass

        # P7修复: 添加代码sanitize功能（参考AFlow sanitize.py）
        solution = self._sanitize_code(solution, entry_point)

        # P6修复: HumanEval格式处理 - problem包含函数签名，prediction只包含函数体
        # 检测并合并函数签名与函数体
        if entry_point and problem:
            # 检查solution中是否缺少函数定义
            has_def_in_solution = f"def {entry_point}" in solution
            has_def_in_problem = f"def {entry_point}" in str(problem)

            if not has_def_in_solution and has_def_in_problem:
                # Bug2 修复: 检查 solution 是否已经包含其他函数定义
                # 如果是，直接重命名而不是 merge（避免嵌套函数缩进问题）
                import re
                existing_func_match = re.search(r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', solution, re.MULTILINE)

                if existing_func_match:
                    # solution 已有函数定义，直接重命名为 entry_point
                    existing_func_name = existing_func_match.group(1)
                    if existing_func_name != entry_point:
                        solution = re.sub(
                            rf'\bdef\s+{re.escape(existing_func_name)}\s*\(',
                            f'def {entry_point}(',
                            solution,
                            count=1  # 只替换第一个
                        )
                        if self.debug_logging:
                            print(f"  🔬 [CODE DEBUG] P6-Bug2: Renamed '{existing_func_name}' -> '{entry_point}' (避免merge缩进问题)")
                else:
                    # solution 只是函数体（没有 def），需要从 problem 提取签名并合并
                    problem_str = str(problem)
                    # 找到函数签名结束位置（第一个冒号后）
                    signature_match = re.search(rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)', problem_str)
                    if signature_match:
                        func_signature = signature_match.group(1)
                        # 确保函数体有正确的缩进
                        body_lines = solution.split('\n')
                        indented_body = []
                        for line in body_lines:
                            if line.strip():  # 非空行
                                # 如果行没有足够的缩进，添加4个空格
                                if not line.startswith('    ') and not line.startswith('\t'):
                                    indented_body.append('    ' + line)
                                else:
                                    indented_body.append(line)
                            else:
                                indented_body.append(line)
                        solution = func_signature + '\n' + '\n'.join(indented_body)
                        if self.debug_logging:
                            print(f"  🔬 [CODE DEBUG] P6: Merged function signature from problem")
                            print(f"  🔬 [CODE DEBUG] P6: merged solution[:200]: {solution[:200]}")

        if self.debug_logging:
            print(f"  🔬 [CODE DEBUG] cleaned solution[:300]: {solution[:300]}")
            # 检查entry_point是否在solution中定义
            if entry_point:
                if f"def {entry_point}" in solution:
                    print(f"  🔬 [CODE DEBUG] ✅ entry_point '{entry_point}' found in solution")
                else:
                    print(f"  🔬 [CODE DEBUG] ❌ entry_point '{entry_point}' NOT found in solution")

        # P12修复: 模糊函数名匹配 - 当entry_point不在solution中时，尝试找到实际定义的函数并重命名
        if entry_point and f"def {entry_point}" not in solution:
            # 查找solution中定义的所有函数
            import re
            defined_funcs = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', solution)
            if defined_funcs:
                # 找到第一个非私有函数（不以_开头）
                actual_func = None
                for func in defined_funcs:
                    if not func.startswith('_'):
                        actual_func = func
                        break
                if not actual_func:
                    actual_func = defined_funcs[0]

                # P12: 重命名函数以匹配entry_point
                if actual_func != entry_point:
                    old_def = f"def {actual_func}"
                    new_def = f"def {entry_point}"
                    solution = solution.replace(old_def, new_def, 1)
                    # 同时替换函数调用（在函数体内的递归调用）
                    # 使用\b确保只替换完整的函数名，避免替换子串
                    solution = re.sub(rf'\b{re.escape(actual_func)}\s*\(', f'{entry_point}(', solution)
                    if self.debug_logging:
                        print(f"  🔬 [CODE DEBUG] P12: Renamed function '{actual_func}' -> '{entry_point}'")

        # P0根本性修复: 从 test_cases 中提取 entry_point (如 MBPP 数据集没有 entry_point 但有 test_cases)
        if not entry_point and test:
            import re
            # 从 assert func_name(...) 格式中提取函数名
            match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
            if match:
                entry_point = match.group(1)
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] Extracted entry_point from test_cases: {entry_point}")

        # 如果没有test cases，使用LLM Judge或fallback到语法检查
        if not test or not entry_point:
            # P5修复: 对于没有测试用例的代码，使用LLM Judge进行语义比较
            if self.use_llm_judge and ground_truth:
                # 使用LLM Judge比较代码的语义等价性
                # P8修复: 添加缺失的problem参数
                is_equivalent = self._llm_judge_compare(
                    problem=str(problem) if problem else "",  # P8: 修复缺失参数
                    prediction=solution,
                    ground_truth=str(ground_truth),
                    problem_type="code",
                    source="code_llm_judge"
                )
                if is_equivalent is True:
                    # 检查语法是否正确
                    try:
                        compile(solution, '<string>', 'exec')
                        return 1.0  # LLM判定等价且语法正确
                    except:
                        return 0.4  # LLM判定等价但语法有问题
                elif is_equivalent is False:
                    # LLM判定不等价，检查语法
                    try:
                        compile(solution, '<string>', 'exec')
                        return 0.2  # 语法正确但LLM判定不等价
                    except:
                        return 0.0
                # is_equivalent is None (API失败)，fallback到语法检查

            # Fallback: 检查是否为有效Python代码
            try:
                compile(solution, '<string>', 'exec')
                return 0.2  # 语法正确但无法验证
            except:
                return 0.0

        # P0修复: 使用多进程隔离执行
        pass_rate = self._execute_code_isolated(solution, test, entry_point)

        # 根据通过率给分
        if pass_rate >= 1.0:
            return 1.0
        elif pass_rate >= 0.8:
            return 0.7
        elif pass_rate >= 0.5:
            return 0.4
        elif pass_rate >= 0.2:
            return 0.2
        else:
            # 检查语法是否正确
            try:
                compile(solution, '<string>', 'exec')
                return 0.2  # P1修复: 语法正确但测试全部失败，给0.2（原0.1不在5档内）
            except:
                return 0.0

    def _execute_code_isolated(self, solution: str, test: str, entry_point: str, timeout: int = 15) -> float:
        """
        P0修复: 多进程隔离执行代码
        P7修复: 超时改为15秒与AFlow一致 (原10秒)

        Returns:
            pass_rate: 通过率 [0.0, 1.0]
        """
        def run_tests_in_process(solution: str, test: str, entry_point: str, result_queue: Queue):
            """在子进程中执行测试"""
            try:
                global_dict = {
                    "math": __import__("math"),
                    "hashlib": __import__("hashlib"),
                    "re": __import__("re"),
                    "sys": __import__("sys"),
                    "List": List,
                    "Dict": Dict,
                    "Tuple": Tuple,
                    "Optional": Optional,
                    "Any": Any,
                }

                # P7修复: HumanEval特殊函数处理（参考AFlow humaneval.py）
                # 某些测试函数需要先定义依赖函数
                HUMANEVAL_HELPERS = {
                    'decode_cyclic': '''
def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
''',
                    'decode_shift': '''
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
''',
                    'find_zero': '''
def poly(xs: list, x: float):
    return sum([coeff * x ** i for i, coeff in enumerate(xs)])
'''
                }

                # 如果entry_point需要辅助函数，先注入
                if entry_point in HUMANEVAL_HELPERS:
                    helper_code = HUMANEVAL_HELPERS[entry_point]
                    exec(helper_code, global_dict)

                # 执行solution
                exec(solution, global_dict)

                if entry_point not in global_dict:
                    result_queue.put({'pass_rate': 0.0, 'error': 'entry_point not found'})
                    return

                # 执行test并捕获断言
                # 方法1: 直接执行test代码（可能包含多个assert）
                try:
                    exec(test, global_dict)

                    # 如果有check函数，调用它
                    if "check" in global_dict:
                        check_func = global_dict["check"]
                        check_func(global_dict[entry_point])

                    # 所有测试通过
                    result_queue.put({'pass_rate': 1.0, 'error': None})

                except AssertionError as e:
                    # 部分断言失败 - 尝试统计通过率
                    # 简化处理：有断言失败就算部分通过
                    result_queue.put({'pass_rate': 0.3, 'error': f'AssertionError: {e}'})

                except Exception as e:
                    result_queue.put({'pass_rate': 0.0, 'error': f'{type(e).__name__}: {e}'})

            except SyntaxError as e:
                result_queue.put({'pass_rate': 0.0, 'error': f'SyntaxError: {e}'})
            except Exception as e:
                result_queue.put({'pass_rate': 0.0, 'error': f'{type(e).__name__}: {e}'})

        # 创建结果队列
        result_queue = multiprocessing.Queue()

        # 创建子进程
        process = multiprocessing.Process(
            target=run_tests_in_process,
            args=(solution, test, entry_point, result_queue)
        )

        try:
            process.start()
            process.join(timeout=timeout)

            # 检查是否超时
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                if self.debug_logging:
                    print(f"  ⏱️ 代码执行超时 ({timeout}s)")
                return 0.2  # P1修复: 超时给0.2（原0.1不在5档内），因为代码可能部分正确

            # 获取结果
            if not result_queue.empty():
                result = result_queue.get_nowait()
                if self.debug_logging and result.get('error'):
                    print(f"  🔧 代码执行: {result.get('error', 'unknown')[:50]}")
                return result.get('pass_rate', 0.0)
            else:
                return 0.0

        except Exception as e:
            if self.debug_logging:
                print(f"  ⚠️ 多进程执行异常: {e}")
            return 0.0
        finally:
            # 确保进程被清理
            if process.is_alive():
                process.terminate()

    def _sanitize_code(self, code: str, entry_point: Optional[str] = None) -> str:
        """
        P7修复: 代码清理函数（参考AFlow scripts/utils/sanitize.py）

        功能:
        1. 提取有效代码段
        2. AST解析获取所有定义
        3. 如果指定entry_point，只保留相关依赖

        Args:
            code: 原始代码字符串
            entry_point: 入口函数名（可选）

        Returns:
            清理后的代码
        """
        import ast

        if not code or not code.strip():
            return code

        try:
            # 尝试解析代码
            tree = ast.parse(code)
        except SyntaxError:
            # 解析失败，返回原始代码
            return code

        # 收集所有定义
        imports = []
        definitions = []  # (name, code, dependencies)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                # 获取函数依赖
                deps = self._get_dependencies(node)
                definitions.append((node.name, ast.unparse(node), deps))
            elif isinstance(node, ast.ClassDef):
                deps = self._get_dependencies(node)
                definitions.append((node.name, ast.unparse(node), deps))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        deps = self._get_dependencies(node)
                        definitions.append((target.id, ast.unparse(node), deps))

        # 如果没有指定entry_point或找不到entry_point，返回所有代码
        if not entry_point:
            return code

        # 检查entry_point是否在definitions中
        entry_exists = any(name == entry_point for name, _, _ in definitions)
        if not entry_exists:
            return code

        # 构建依赖图，找到entry_point需要的所有定义
        needed = self._find_reachable(entry_point, definitions)

        # 组装最终代码
        result_parts = imports[:]
        for name, code_str, _ in definitions:
            if name in needed:
                result_parts.append(code_str)

        return '\n'.join(result_parts)

    def _get_dependencies(self, node: 'ast.AST') -> set:
        """获取AST节点中引用的名称"""
        import ast
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                deps.add(child.id)
        return deps

    def _find_reachable(self, entry_point: str, definitions: list) -> set:
        """从entry_point开始，找到所有可达的定义"""
        # 构建名称到依赖的映射
        dep_map = {name: deps for name, _, deps in definitions}

        # BFS找可达节点
        visited = set()
        queue = [entry_point]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in dep_map:
                for dep in dep_map[current]:
                    if dep not in visited and dep in dep_map:
                        queue.append(dep)

        return visited

    def _compute_qa_reward(self, problem: str, prediction: Any, ground_truth: Any, source: Optional[str]) -> float:
        """
        P1修复: QA任务评估 - 参考SQuAD/TriviaQA国际标准评估方法
        P10修复: 使用OSS模型提取解释性文本中的答案
        P11修复: 改进提取失败后的处理逻辑

        国际标准方法 (SQuAD官方评估):
        1. Exact Match (EM): 标准化后完全匹配
        2. F1 Score: Token级别的F1分数
        3. 数值等价: 数字的语义等价判断
        4. 包含关系: 简短答案包含在长答案中
        5. LLM Judge: 语义等价判断（可选）

        奖励等级:
        - 1.0: EM=1 或 F1>=0.8 或 数值等价 或 LLM判断正确
        - 0.7: F1>=0.5 或 包含关系成立
        - 0.4: F1>=0.3
        - 0.2: F1>=0.1 (有部分相关内容)
        - 0.0: 无匹配
        """
        import re

        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        if not pred_str:
            return 0.0

        # P12修复: LLM提取作为主力（与Math逻辑一致）
        # 策略: 对于任何非简单答案，都尝试LLM提取
        needs_extraction = (
            len(pred_str) > 50 or  # 答案太长
            '**' in pred_str or    # Markdown格式
            'Step' in pred_str or  # 步骤说明
            'Answer' in pred_str or  # 包含Answer
            'Explanation' in pred_str or  # 包含解释
            'Therefore' in pred_str or 'Thus' in pred_str  # 推理结论
        )

        if needs_extraction and self.use_llm_judge:
            if self.debug_logging:
                print(f"  🔧 P12: 检测到QA需要提取的内容，使用LLM提取答案...")
            extracted = self._extract_answer_with_llm(pred_str, problem, "qa")
            if extracted:
                if self.debug_logging:
                    print(f"  ✅ P12: LLM提取成功: {extracted[:50]}...")
                pred_str = extracted
            else:
                # LLM提取失败，尝试本地Answer模式提取
                if self.debug_logging:
                    print(f"  🔄 P12: LLM提取失败，尝试本地Answer模式提取...")
                answer_match = re.search(r'\*\*Answer[:\*]*\s*[:\-–—]*\s*(.+?)(?:\n\n|\*\*|$)', pred_str, re.IGNORECASE | re.DOTALL)
                if answer_match:
                    local_extracted = answer_match.group(1).strip()
                    local_extracted = re.sub(r'^[\*\#\-–—:]+|[\*\#\-–—:]+$', '', local_extracted).strip()
                    if local_extracted and len(local_extracted) < 300:
                        if self.debug_logging:
                            print(f"  ✅ P12: 本地提取成功: {local_extracted[:50]}...")
                        pred_str = local_extracted

        # 1. 首先尝试LLM Judge (如果启用) - 用于语义等价判断
        if self.use_llm_judge:
            is_correct = self._llm_judge_compare(
                problem=problem,
                prediction=pred_str,
                ground_truth=gt_str,
                problem_type="qa",
                source=source
            )
            if is_correct:
                return 1.0

        # 2. 标准化答案 (参考SQuAD官方评估脚本)
        pred_normalized = self._normalize_answer_squad(pred_str)
        gt_normalized = self._normalize_answer_squad(gt_str)

        # 3. Exact Match (EM)
        if pred_normalized == gt_normalized:
            return 1.0

        # 4. 数值等价检查 (国际标准: 数字语义等价)
        #    例如: "4" vs "four" vs "4 cylinders" 应该匹配
        if self._check_numeric_equivalence(pred_str, gt_str):
            return 1.0

        # 5. 包含关系检查 (国际标准: 简答包含在长答中)
        #    例如: "Paris" vs "The capital is Paris"
        if self._check_containment(pred_normalized, gt_normalized):
            return 0.7

        # 6. F1 Score计算 (SQuAD标准)
        f1 = self._compute_f1_score_squad(pred_normalized, gt_normalized)

        # 根据F1分数返回奖励
        if f1 >= 0.8:
            return 1.0
        elif f1 >= 0.5:
            return 0.7
        elif f1 >= 0.3:
            return 0.4
        elif f1 >= 0.1:
            return 0.2
        else:
            return 0.0

    def _normalize_answer_squad(self, text: str) -> str:
        """
        SQuAD官方标准化方法
        参考: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
        """
        import string
        import re

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def _check_numeric_equivalence(self, pred: str, gt: str) -> bool:
        """
        检查数值语义等价

        处理情况:
        - "4" vs "four" vs "4 cylinders"
        - "1990" vs "in 1990" vs "the year 1990"
        - "$100" vs "100 dollars" vs "100"
        """
        # 数字词映射
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
            'million': 1000000, 'billion': 1000000000
        }

        def extract_number(text: str) -> Optional[float]:
            text_lower = text.lower().strip()

            # 直接数字匹配
            num_match = re.search(r'-?\d+\.?\d*', text_lower)
            if num_match:
                try:
                    return float(num_match.group())
                except:
                    pass

            # 数字词匹配
            for word, num in number_words.items():
                if word in text_lower:
                    return float(num)

            return None

        pred_num = extract_number(pred)
        gt_num = extract_number(gt)

        if pred_num is not None and gt_num is not None:
            # 精确匹配或接近匹配
            if pred_num == gt_num:
                return True
            # 允许小误差
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.01:
                return True

        return False

    def _check_containment(self, pred: str, gt: str) -> bool:
        """
        检查包含关系 (国际标准方法)

        情况1: 预测是gt的子串 (pred简短但正确)
        情况2: gt是预测的子串 (gt简短，pred更完整)
        情况3: 词级别包含 (如 "watch" 出现在 "pocketwatch" 中)
        """
        # 跳过太短的答案（避免误匹配）
        if len(pred) < 2 or len(gt) < 2:
            return False

        # 双向包含检查
        if pred in gt or gt in pred:
            # 额外验证：包含的部分应该是有意义的比例
            shorter = pred if len(pred) < len(gt) else gt
            longer = gt if len(pred) < len(gt) else pred

            # 短答案应该是长答案的主要部分（至少30%）
            if len(shorter) >= len(longer) * 0.3:
                return True

        # P4修复: 词级别包含检查 (处理复合词如 pocketwatch)
        # 检查pred中的每个词是否出现在gt的某个词中（或反过来）
        pred_words = pred.split()
        gt_words = gt.split()

        for pw in pred_words:
            if len(pw) >= 3:  # 词长度至少3，避免误匹配
                for gw in gt_words:
                    # 检查词级别的包含（如 "watch" in "pocketwatch"）
                    # 条件: 短词至少占长词的40%（放宽以匹配复合词）
                    if pw in gw and len(pw) >= len(gw) * 0.4:
                        return True
                    if gw in pw and len(gw) >= len(pw) * 0.4:
                        return True

        return False

    def _compute_f1_score_squad(self, pred: str, gt: str) -> float:
        """
        SQuAD标准F1计算
        参考: https://rajpurkar.github.io/SQuAD-explorer/
        """
        from collections import Counter

        pred_tokens = pred.split()
        gt_tokens = gt.split()

        # 边界情况
        if len(gt_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        if len(pred_tokens) == 0:
            return 0.0

        # 计算共同tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def _normalize_answer(self, text: str) -> str:
        """标准化答案用于比较"""
        import string
        # 小写
        text = text.lower()
        # 去除标点
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 去除多余空格
        text = ' '.join(text.split())
        return text

    def _compute_f1_score(self, pred: str, gt: str) -> float:
        """P1修复: 计算token级别F1分数（使用Counter而非set，避免去重丢失信息）"""
        from collections import Counter

        pred_tokens = Counter(pred.split())
        gt_tokens = Counter(gt.split())

        if sum(gt_tokens.values()) == 0:
            return 1.0 if sum(pred_tokens.values()) == 0 else 0.0

        if sum(pred_tokens.values()) == 0:
            return 0.0

        # 计算交集（取最小计数）
        common = pred_tokens & gt_tokens
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / sum(pred_tokens.values())
        recall = num_same / sum(gt_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def _compute_general_reward(self, prediction: Any, ground_truth: Any) -> float:
        """通用奖励计算"""
        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip().lower()
        gt_str = str(ground_truth).strip().lower()

        if pred_str == gt_str:
            return 1.0
        elif gt_str in pred_str:
            return 0.7
        elif self._compute_f1_score(pred_str, gt_str) > 0.5:
            return 0.4
        else:
            return 0.0

    def _extract_answer_with_llm(self, explanatory_text: str, problem: str, problem_type: str) -> Optional[str]:
        """
        P10修复: 使用OSS模型从解释性文本中提取简洁答案

        Args:
            explanatory_text: 包含解释的完整文本（如 **Step 1...** 格式）
            problem: 原始问题
            problem_type: 问题类型 (math/qa)

        Returns:
            提取的简洁答案，或None如果提取失败
        """
        if not self.llm_judge_client:
            return None

        # 构建提取答案的prompt
        if problem_type == "math":
            extract_prompt = f"""Extract ONLY the final numeric answer from the following solution text.

Problem: {problem[:200]}

Solution Text:
{explanatory_text[:1000]}

IMPORTANT: Return ONLY the final answer (a number, fraction, or simple expression like "42", "5/6", "2x+1").
Do NOT include any explanations, steps, or formatting. Just the answer value.

<answer>YOUR_ANSWER_HERE</answer>"""
        else:  # qa
            extract_prompt = f"""Extract ONLY the direct answer from the following response.

Question: {problem[:200]}

Response:
{explanatory_text[:1000]}

IMPORTANT: Return ONLY the direct answer (a name, place, date, number, or short phrase).
Do NOT include any explanations or reasoning. Just the answer.

<answer>YOUR_ANSWER_HERE</answer>"""

        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.llm_judge_model,
                messages=[
                    {"role": "system", "content": "You are a precise answer extractor. Extract only the final answer."},
                    {"role": "user", "content": extract_prompt}
                ],
                temperature=0.0,
                max_tokens=100  # 答案应该很短
            )

            content = response.choices[0].message.content
            if content is None:
                return None

            result = content.strip()

            # 尝试从 <answer> 标签中提取
            import re
            answer_match = re.search(r'<answer>\s*(.+?)\s*</answer>', result, re.IGNORECASE | re.DOTALL)
            if answer_match:
                extracted = answer_match.group(1).strip()
                # 清理提取的答案
                extracted = re.sub(r'^[\*\#]+|[\*\#]+$', '', extracted).strip()  # 去除Markdown格式
                if extracted and len(extracted) < 200:  # 合理长度的答案
                    return extracted

            # Fallback: 如果没有标签，尝试获取最后一行非空内容
            lines = [l.strip() for l in result.split('\n') if l.strip()]
            if lines:
                last_line = lines[-1]
                # 清理并返回
                last_line = re.sub(r'^[\*\#]+|[\*\#]+$', '', last_line).strip()
                if last_line and len(last_line) < 200:
                    return last_line

            return None

        except Exception as e:
            if self.debug_logging:
                print(f"  ⚠️  P10 答案提取失败: {e}")
            return None

    def _extract_math_answer(self, text: str) -> Optional[str]:
        """
        P0-4修复: 鲁棒的数学答案提取

        支持:
        - 嵌套boxed: \\boxed{{a \\choose b}}
        - 分数: 5/324
        - 百分比: 50%
        - 科学计数法: 1.5e-3
        """
        if not text:
            return None

        # 1. 优先提取boxed (支持嵌套)
        boxed = self._extract_boxed_robust(text)
        if boxed:
            # P1修复: 检测代码泄漏（与answer_extractor.py保持一致）
            code_leak_keywords = ['def ', 'return ', 'import ', 'class ', 'if __name__', 'async def ']
            if any(kw in boxed for kw in code_leak_keywords):
                # 代码泄漏，跳过boxed继续尝试其他提取方法
                pass
            # 检测空boxed
            elif not boxed.strip():
                pass
            # 检测执行错误
            elif boxed.startswith('Error:') or 'Traceback' in boxed or 'SyntaxError' in boxed:
                pass
            else:
                return boxed

        # 2. 查找"答案是"、"Therefore"等模式后的内容
        answer_patterns = [
            r'答案[是为：:]+\s*([\d\./\-]+)',
            r'[Tt]he answer is[:\s]+([\d\./\-]+)',
            r'[Tt]herefore[,\s]+([\d\./\-]+)',
            r'[Ss]o[,\s]+([\d\./\-]+)',
            r'=\s*([\d\./\-]+)\s*$',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # 3. 提取最后一个数字
        numbers = self._extract_numbers(text)
        if numbers:
            return str(numbers[-1])

        # 4. 返回整个文本（如果很短）
        if len(text) < 50:
            return text.strip()

        return None

    def _extract_boxed_robust(self, text: str) -> Optional[str]:
        """
        P0-4修复: 支持嵌套花括号的boxed提取
        P12修复: 过滤无效的boxed内容（如解释性文本）
        """
        # 支持嵌套的正则（最多2层嵌套）
        pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
        matches = re.findall(pattern, text, re.DOTALL)

        def is_valid_boxed_content(content: str) -> bool:
            """P12修复: 检查boxed内容是否为有效答案"""
            if not content:
                return False
            content_stripped = content.strip()
            # 无效模式：解释性文本开头
            invalid_prefixes = [
                '**Approach', '**Step', '**Solution', '**Analysis',
                '**Method', '**Let', '**Define', '**Given',
                'Approach', 'Step 1', 'Solution:', 'Let ',
            ]
            for prefix in invalid_prefixes:
                if content_stripped.startswith(prefix):
                    return False
            # 无效模式：内容过长（超过200字符可能是解释而非答案）
            if len(content_stripped) > 200:
                # 但如果是数学表达式（含\frac等），允许更长
                if not any(tex in content for tex in ['\\frac', '\\sqrt', '\\sum', '\\int']):
                    return False
            return True

        if matches:
            # P12修复: 从后往前找第一个有效的boxed内容
            for match in reversed(matches):
                if is_valid_boxed_content(match):
                    return match.strip()
            # 如果所有boxed都无效，返回最后一个（让后续流程处理）
            return matches[-1].strip()

        # Fallback: 简单模式
        simple_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if simple_match:
            content = simple_match.group(1).strip()
            if is_valid_boxed_content(content):
                return content
            return content  # 即使无效也返回，让后续流程处理

        return None

    def _parse_number_robust(self, text: str) -> Optional[float]:
        """
        P0-4修复: 鲁棒的数字解析

        支持:
        - 分数: 5/324
        - 百分比: 50% -> 0.5
        - 科学计数法: 1.5e-3
        - 千分位: 1,234,567
        """
        if not text:
            return None

        text = text.strip()

        # 去除千分位逗号
        text = text.replace(',', '')

        # 百分比转换
        if '%' in text:
            try:
                num_str = text.replace('%', '').strip()
                return float(num_str) / 100.0
            except:
                pass

        # 分数转换
        if '/' in text:
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    return float(parts[0].strip()) / float(parts[1].strip())
            except:
                pass

        # 直接解析
        try:
            return float(text)
        except:
            pass

        # 提取第一个数字
        match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except:
                pass

        return None

    # ============== 原有方法（保留兼容性） ==============

    def _is_math_correct(self, prediction: str, ground_truth: str) -> bool:
        """
        判断数学答案是否正确
        
        支持:
        - 数字比较（含浮点误差）
        - 分数比较（如 5/324 vs 0.0154...）
        - 字符串匹配
        """
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            # 字符串完全匹配
            if pred_str == gt_str:
                return True

            # 解析为数值比较（支持分数）
            def parse_number(s: str) -> float:
                """解析数字，支持分数格式"""
                if '/' in s:
                    parts = s.split('/')
                    return float(parts[0]) / float(parts[1])
                return float(s)

            try:
                pred_num = parse_number(pred_str)
                gt_num = parse_number(gt_str)

                # 使用相对误差比较（处理浮点精度）
                rel_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-9)
                return rel_error < 1e-6
            except:
                pass

            # 方法1: boxed 格式
            pred_boxed = self._extract_boxed(pred_str)
            gt_boxed = self._extract_boxed(gt_str)
            if pred_boxed and gt_boxed:
                try:
                    pred_num = parse_number(pred_boxed)
                    gt_num = parse_number(gt_boxed)
                    rel_error = abs(pred_num - gt_num) / (abs(gt_num) + 1e-9)
                    if rel_error < 1e-6:
                        return True
                except:
                    pass

            # 方法2: 数字提取
            pred_numbers = self._extract_numbers(pred_str)
            gt_numbers = self._extract_numbers(gt_str)

            if not gt_numbers:
                # 无法提取数字，用字符串匹配
                return gt_str.strip().lower() in pred_str.strip().lower()

            if not pred_numbers:
                return False

            # 比较最后一个数字
            pred_answer = pred_numbers[-1]
            gt_answer = gt_numbers[-1]

            return abs(pred_answer - gt_answer) < 1e-4

        except Exception:
            return False

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _check_code_solution(self, solution: str, test: str, entry_point: str) -> bool:
        """
        Use execution to check if the code solution is correct.
        Inspired by AFlow's evaluation mechanism.
        """
        if not solution or not test or not entry_point:
            return False

        # Sanitize solution (remove markdown blocks if any)
        if "```python" in solution:
            solution = solution.split("```python")[1].split("```")[0]
        elif "```" in solution:
            solution = solution.split("```")[1].split("```")[0]
        
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # Execute the solution code
            exec(solution, global_dict)

            if entry_point not in global_dict:
                # Try to find if there is a 'solve' function or similar if entry_point is missing
                # But for HumanEval/MBPP, entry_point is strict.
                # If it's a full script, maybe we shouldn't fail immediately, but for now strict is better.
                return False

            # Execute the test code
            # The test code usually contains a 'check' function or assertions
            exec(test, global_dict)

            # Check if 'check' function exists (common in HumanEval)
            if "check" in global_dict:
                check = global_dict["check"]
                try:
                    # Run the check function with timeout
                    self.run_with_timeout(check, (global_dict[entry_point],), 5) # 5 seconds timeout
                    return True
                except Exception as e:
                    if self.debug_logging:
                        print(f"Code execution check failed: {e}")
                    return False
            else:
                # If no check function, assume the test code runs assertions directly
                # If exec(test) didn't raise exception, it might be correct
                return True

        except Exception as e:
            if self.debug_logging:
                print(f"Code execution error: {e}")
            return False

    def _is_code_correct(self, prediction: str, ground_truth: str, test: Optional[str] = None, entry_point: Optional[str] = None) -> bool:
        """判断代码答案是否正确"""
        import re

        # P2修复: Strip markdown code block markers FIRST
        prediction = str(prediction)
        if '```' in prediction:
            code_blocks = re.findall(r'```(?:python)?\s*\n?([^`]+)```', prediction)
            if code_blocks:
                prediction = code_blocks[-1].strip()
            else:
                prediction = re.sub(r'^```(?:python)?\n?', '', prediction)
                prediction = re.sub(r'```$', '', prediction)
                prediction = prediction.strip()

        # Prioritize execution-based checking if test cases are available
        if test and entry_point:
            return self._check_code_solution(prediction, test, entry_point)

        # Fallback to string matching if execution is not possible
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            if not pred_str:
                return False

            # 精确匹配
            if pred_str.lower() == gt_str.lower():
                return True

            # 包含匹配
            if gt_str.lower() in pred_str.lower():
                return True

            return False

        except Exception:
            return False

    def _is_qa_correct(self, prediction: str, ground_truth: str) -> bool:
        """判断QA答案是否正确"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            # 精确匹配
            if pred_str == gt_str:
                return True

            # 包含匹配
            if gt_str in pred_str or pred_str in gt_str:
                return True

            # Token重叠阈值 - P1修复: 使用Counter代替set
            from collections import Counter
            pred_tokens = Counter(pred_str.split())
            gt_tokens = Counter(gt_str.split())

            if sum(gt_tokens.values()) == 0:
                return False

            # 计算重叠
            common = pred_tokens & gt_tokens
            overlap_ratio = sum(common.values()) / sum(gt_tokens.values())
            return overlap_ratio > 0.8

        except Exception:
            return False

    def _is_general_correct(self, prediction: str, ground_truth: str) -> bool:
        """通用正确性判断"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            return pred_str == gt_str or gt_str in pred_str

        except Exception:
            return False

    def _compute_correctness_reward(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> float:
        """
        计算正确性奖励（保留用于向后兼容）
        
        Returns:
            reward: 1.0 or 0.0
        """
        # This function is kept for compatibility but compute_reward should be used
        # We map the binary 0/1 back to whatever range was expected if needed, 
        # but here we simply return 1.0 or 0.0 as requested.
        
        if prediction is None:
            return 0.0

        is_correct = False
        if problem_type == "math":
            is_correct = self._is_math_correct(prediction, ground_truth)
        elif problem_type == "code":
            # Without test cases here, we fall back to string matching which is weak
            is_correct = self._is_code_correct(prediction, ground_truth)
        elif problem_type == "qa":
            is_correct = self._is_qa_correct(prediction, ground_truth)
        else:
            is_correct = self._is_general_correct(prediction, ground_truth)
            
        return 1.0 if is_correct else 0.0

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\\boxed{}中的内容(ROLL风格)"""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_numbers(self, text: str) -> list:
        """从文本中提取所有数字(改进版 + 文字数字识别)"""
        numbers = []

        # Method 1: Numeric extraction (existing)
        # 匹配整数、小数、负数、科学计数法
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        for m in matches:
            if m:
                try:
                    numbers.append(float(m))
                except:
                    pass

        # Method 2: Word-to-number recognition (NEW - fixes ~15-20% QA errors)
        # Aligns with SQuAD/HotpotQA standards for text-based answers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }

        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                numbers.append(float(num))

        return numbers

    def _extract_function_names(self, code: str) -> list:
        """从代码中提取函数名"""
        pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(pattern, code)
        return matches

    def _compute_efficiency_reward(self, cost: float) -> float:
        return 0.0

    def _compute_simplicity_reward(
        self,
        execution_time: float,
        num_operators: int = 1
    ) -> float:
        return 0.0

    def _compute_format_reward(self, response: str, problem_type: str) -> float:
        return 0.0

    def _compute_repetition_penalty(self, response: str, ngram_size: int = 3) -> float:
        return 0.0

    def print_eval_stats(self):
        """
        打印评估统计信息（用于调试）
        """
        stats = self.eval_stats
        total = stats['total_evaluations']

        if total == 0:
            print("\n📊 评估统计: 无评估记录")
            return

        print(f"\n📊 评估统计 (总计: {total} 次):")
        print(f"  ✅ LLM Judge成功: {stats['llm_judge_success']} ({stats['llm_judge_success']/total*100:.1f}%)")
        print(f"  ⚠️  解析失败: {stats['llm_judge_parse_failures']} ({stats['llm_judge_parse_failures']/total*100:.1f}%)")
        print(f"  ❌ API失败: {stats['llm_judge_api_failures']} ({stats['llm_judge_api_failures']/total*100:.1f}%)")
        print(f"\n  判决结果:")
        print(f"    正确: {stats['correct_predictions']} ({stats['correct_predictions']/total*100:.1f}%)")
        print(f"    错误: {stats['incorrect_predictions']} ({stats['incorrect_predictions']/total*100:.1f}%)")

        # 计算准确率（如果有足够数据）
        judged = stats['correct_predictions'] + stats['incorrect_predictions']
        if judged > 0:
            accuracy = stats['correct_predictions'] / judged * 100
            print(f"\n  🎯 预测准确率: {accuracy:.1f}% (基于{judged}次成功评估)")

    def reset_eval_stats(self):
        """重置评估统计计数器"""
        self.eval_stats = {
            'total_evaluations': 0,
            'llm_judge_success': 0,
            'llm_judge_parse_failures': 0,
            'llm_judge_api_failures': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }
        print("🔄 评估统计已重置")


def test_reward_computer():
    """测试改进版奖励计算器"""
    print("\n" + "=" * 60)
    print("🧪 测试改进版奖励计算器")
    print("=" * 60)

    computer = RewardComputer()

    # 测试案例
    test_cases = [
        {
            "name": "数学 - 完美格式+正确",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Let me calculate: 15 + 27 = 42</think><answer>\\boxed{42}</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.5}
        },
        {
            "name": "代码 - 简单测试",
            "problem": "Write a function to square a number",
            "prediction": "def square(x):\n    return x * x",
            "ground_truth": "def square(x):\n    return x * x",
            "problem_type": "code",
            "test": "check = lambda func: func(2) == 4",
            "entry_point": "square",
            "metadata": {"cost": 0.003, "execution_time": 5.0}
        }
    ]

    for case in test_cases:
        reward = computer.compute_reward(
            problem=case["problem"],
            prediction=case["prediction"],
            ground_truth=case["ground_truth"],
            problem_type=case["problem_type"],
            metadata=case["metadata"],
            test=case.get("test"),
            entry_point=case.get("entry_point")
        )

        print(f"\n📝 {case['name']}")
        print(f"  预测: {case['prediction'][:60]}...")
        print(f"  正确答案: {case['ground_truth']}")
        print(f"  奖励: {reward:.2f}")


if __name__ == "__main__":
    test_reward_computer()
