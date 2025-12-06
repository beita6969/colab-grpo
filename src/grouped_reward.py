#!/usr/bin/env python3
"""
分组奖励计算器 - 支持多问题加权评分 + 复杂度奖励

设计:
1. 每个 workflow 在一组问题 (2 easy + 2 hard) 上运行
2. 计算加权得分: score = Σ(weight_i * correctness_i)
3. 新增复杂度奖励: operator数量 + workflow长度 + 控制流多样性

公式:
- easy_weight = 0.3 (每题 0.15)
- hard_weight = 0.7 (每题 0.35)
- Total_Reward = Correctness_Score × 0.6 + Complexity_Score × 0.4
"""

import re
import ast
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import math


class GroupedRewardCalculator:
    """
    分组奖励计算器

    特性:
    1. 多问题加权评分
    2. 多样性 tie-breaker
    3. 保证组内有非零优势
    """

    def __init__(
        self,
        weight_easy: float = 0.3,
        weight_hard: float = 0.7,
        correctness_weight: float = 0.6,     # 正确性权重
        complexity_weight: float = 0.4,       # 复杂度权重
        diversity_threshold: float = 0.05,   # 分数差距阈值
        diversity_weight: float = 0.1,       # 多样性加分权重 (tie-breaker)
        debug: bool = False
    ):
        self.weight_easy = weight_easy
        self.weight_hard = weight_hard
        self.correctness_weight = correctness_weight
        self.complexity_weight = complexity_weight
        self.diversity_threshold = diversity_threshold
        self.diversity_weight = diversity_weight
        self.debug = debug

        # 已知的 operator 列表
        self.known_operators = {
            'AnswerGenerate', 'Programmer', 'ScEnsemble',
            'Test', 'Review', 'Revise', 'Custom'
        }

        # DSL控制流符号
        self.control_flow_symbols = {
            '->': 'sequence',      # 序列
            '?': 'condition',      # 条件判断
            ':': 'branch',         # 分支
            '*': 'loop'            # 循环
        }

    def calculate_weighted_score(
        self,
        problem_scores: List[Dict[str, Any]]
    ) -> float:
        """
        计算加权得分

        Args:
            problem_scores: 每个问题的评分结果
                [{
                    'difficulty': 'easy'/'hard',
                    'weight': 0.15/0.35,
                    'correctness': 0.0-1.0,
                    'problem_id': 'easy_0'
                }, ...]

        Returns:
            加权总分 (0.0 - 1.0)
        """
        total_score = 0.0
        for p in problem_scores:
            total_score += p['weight'] * p['correctness']
        return total_score

    def calculate_diversity_score(self, workflow_code: str) -> float:
        """
        计算 workflow 的多样性得分 (用于tie-breaker)

        考虑因素:
        1. 使用的 operator 数量和种类
        2. 控制流复杂度 (if/for/while)
        3. 代码结构多样性

        Returns:
            多样性得分 (0.0 - 1.0)
        """
        if not workflow_code:
            return 0.0

        scores = []

        # 1. Operator 多样性 (0-0.4)
        operators_used = set()
        for op in self.known_operators:
            pattern = rf'\b{op}\b'
            if re.search(pattern, workflow_code):
                operators_used.add(op)

        op_diversity = min(len(operators_used) / 4.0, 1.0) * 0.4
        scores.append(op_diversity)

        # 2. 控制流复杂度 (0-0.3)
        control_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bawait\b'
        ]
        control_count = sum(1 for p in control_patterns if re.search(p, workflow_code))
        control_score = min(control_count / 4.0, 1.0) * 0.3
        scores.append(control_score)

        # 3. 步骤数量 (0-0.3)
        # 计算 await 调用次数作为步骤数
        await_count = len(re.findall(r'await\s+self\.\w+', workflow_code))
        step_score = min(await_count / 5.0, 1.0) * 0.3
        scores.append(step_score)

        return sum(scores)

    def calculate_complexity_score(self, dsl_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        🔧 P41修复: 计算 DSL workflow 的多样性得分（而非数量）

        多样性组成:
        1. diversity_score (50%): 不同种类operator的数量（多样性）
        2. efficiency_score (20%): 惩罚过度重复模式
        3. flow_score (30%): 控制流多样性奖励

        Args:
            dsl_text: DSL格式的workflow (如 "Custom -> Review ? Revise : done")

        Returns:
            (complexity_score, details)
            - complexity_score: 0.0 - 1.0
            - details: 各项得分明细
        """
        if not dsl_text or not dsl_text.strip():
            return 0.0, {'diversity_score': 0, 'efficiency_score': 0, 'flow_score': 0}

        dsl_text = dsl_text.strip()

        # 1. 🔧 P41: Operator多样性得分 (0-1.0, 权重50%)
        # 统计不同种类的operator（而非总数量）
        unique_operators = set()
        total_operator_count = 0
        for op in self.known_operators:
            pattern = rf'\b{op}\b'
            matches = re.findall(pattern, dsl_text)
            if matches:
                unique_operators.add(op)
                total_operator_count += len(matches)

        num_unique = len(unique_operators)

        # 评分基于种类数: 1种=0.0, 2种=0.4, 3种=0.7, 4种=0.9, 5种+=1.0
        if num_unique <= 1:
            diversity_score = 0.0
        elif num_unique == 2:
            diversity_score = 0.4
        elif num_unique == 3:
            diversity_score = 0.7
        elif num_unique == 4:
            diversity_score = 0.9
        else:
            diversity_score = 1.0

        # 2. 🔧 P41: 效率得分 - 惩罚过度重复 (0-1.0, 权重20%)
        # 检测重复模式: `* N` 和嵌套循环 `* N * M`
        loop_pattern = r'\*\s*(\d+)'
        loop_matches = re.findall(loop_pattern, dsl_text)

        # 计算总循环次数
        total_loops = 1
        for match in loop_matches:
            total_loops *= int(match)

        # 计算重复率 = 总operator出现次数 / 种类数
        if num_unique > 0:
            repetition_ratio = total_operator_count / num_unique
        else:
            repetition_ratio = 1.0

        # 效率评分: 重复率低=高分, 重复率高=低分
        # repetition_ratio=1 (无重复) -> 1.0
        # repetition_ratio=2 (每个用2次) -> 0.7
        # repetition_ratio=3+ -> 0.3
        # 加上循环惩罚
        if repetition_ratio <= 1.2 and total_loops <= 3:
            efficiency_score = 1.0
        elif repetition_ratio <= 2.0 and total_loops <= 6:
            efficiency_score = 0.7
        elif repetition_ratio <= 3.0 and total_loops <= 9:
            efficiency_score = 0.4
        else:
            efficiency_score = 0.1  # 严重重复惩罚

        # 3. 控制流多样性得分 (0-1.0, 权重30%)
        has_sequence = '->' in dsl_text
        has_condition = '?' in dsl_text and ':' in dsl_text
        has_loop = '*' in dsl_text

        # 评分: 仅序列=0.2, 有条件=0.5, 有循环=0.8, 条件+循环=1.0
        if has_condition and has_loop:
            flow_score = 1.0
        elif has_loop:
            flow_score = 0.8
        elif has_condition:
            flow_score = 0.5
        elif has_sequence:
            flow_score = 0.2
        else:
            flow_score = 0.0

        # 🔧 P41: 新的加权计算 (多样性为主)
        complexity_score = (
            diversity_score * 0.5 +    # 多样性权重提升到50%
            efficiency_score * 0.2 +   # 效率/惩罚重复 20%
            flow_score * 0.3           # 控制流 30%
        )

        details = {
            'unique_operators': list(unique_operators),
            'num_unique': num_unique,
            'total_operator_count': total_operator_count,
            'diversity_score': diversity_score,
            'repetition_ratio': repetition_ratio,
            'total_loops': total_loops,
            'efficiency_score': efficiency_score,
            'has_sequence': has_sequence,
            'has_condition': has_condition,
            'has_loop': has_loop,
            'flow_score': flow_score,
            'total_complexity': complexity_score
        }

        return complexity_score, details

    def calculate_total_reward(
        self,
        correctness_score: float,
        dsl_text: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算总奖励 = 正确性 × 0.6 + 复杂度 × 0.4

        Args:
            correctness_score: 正确性得分 (0-1)
            dsl_text: DSL文本

        Returns:
            (total_reward, details)
        """
        complexity_score, complexity_details = self.calculate_complexity_score(dsl_text)

        total_reward = (
            correctness_score * self.correctness_weight +
            complexity_score * self.complexity_weight
        )

        details = {
            'correctness_score': correctness_score,
            'complexity_score': complexity_score,
            'correctness_weight': self.correctness_weight,
            'complexity_weight': self.complexity_weight,
            'total_reward': total_reward,
            'complexity_details': complexity_details
        }

        return total_reward, details

    def extract_operators(self, workflow_code: str) -> List[str]:
        """提取 workflow 使用的 operators"""
        operators = []
        for op in self.known_operators:
            if re.search(rf'\b{op}\b', workflow_code):
                operators.append(op)
        return operators

    def calculate_group_rewards(
        self,
        workflows: List[str],
        problem_scores_per_workflow: List[List[Dict[str, Any]]],
        dsl_texts: Optional[List[str]] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        计算一组 workflow 的奖励

        新公式: Total_Reward = Correctness × 0.6 + Complexity × 0.4

        Args:
            workflows: K 个 workflow Python代码
            problem_scores_per_workflow: 每个 workflow 在每个问题上的得分
                [[{problem_0_score}, {problem_1_score}, ...], ...]
            dsl_texts: K 个 DSL 文本 (用于复杂度计算)

        Returns:
            (rewards, diagnostics)
            - rewards: K 个 workflow 的最终奖励
            - diagnostics: 调试信息
        """
        K = len(workflows)
        if K == 0:
            return [], {}

        # 如果没有提供DSL文本，使用workflow代码
        if dsl_texts is None:
            dsl_texts = workflows

        # 1. 计算每个 workflow 的正确性得分 (加权)
        correctness_scores = []
        for scores in problem_scores_per_workflow:
            cs = self.calculate_weighted_score(scores)
            correctness_scores.append(cs)

        # 2. 计算每个 workflow 的复杂度得分
        complexity_scores = []
        complexity_details_list = []
        for dsl in dsl_texts:
            cs, details = self.calculate_complexity_score(dsl)
            complexity_scores.append(cs)
            complexity_details_list.append(details)

        # 3. 计算总奖励 = 正确性 × 0.6 + 复杂度 × 0.4
        total_rewards = []
        for i in range(K):
            reward = (
                correctness_scores[i] * self.correctness_weight +
                complexity_scores[i] * self.complexity_weight
            )
            total_rewards.append(reward)

        # 4. Tie-breaker: 如果总分差距很小，用多样性打破平局
        score_range = max(total_rewards) - min(total_rewards)
        need_diversity_tiebreak = score_range < self.diversity_threshold

        if need_diversity_tiebreak:
            diversity_scores = [self.calculate_diversity_score(w) for w in workflows]
            for i in range(K):
                total_rewards[i] += self.diversity_weight * diversity_scores[i]
        else:
            diversity_scores = [0.0] * K

        # 5. 诊断信息
        diagnostics = {
            'correctness_scores': correctness_scores,
            'complexity_scores': complexity_scores,
            'complexity_details': complexity_details_list,
            'diversity_scores': diversity_scores,
            'score_range': score_range,
            'need_diversity_tiebreak': need_diversity_tiebreak,
            'total_rewards': total_rewards,
            'operators_per_workflow': [self.extract_operators(w) for w in workflows],
            'weights': {
                'correctness': self.correctness_weight,
                'complexity': self.complexity_weight
            }
        }

        if self.debug:
            print(f"\n🎯 GroupedReward 诊断:")
            print(f"  正确性分: {[f'{s:.3f}' for s in correctness_scores]}")
            print(f"  复杂度分: {[f'{s:.3f}' for s in complexity_scores]}")
            print(f"  总奖励: {[f'{r:.3f}' for r in total_rewards]}")
            print(f"  权重: 正确性={self.correctness_weight}, 复杂度={self.complexity_weight}")
            if need_diversity_tiebreak:
                print(f"  多样性分(tie-breaker): {[f'{s:.3f}' for s in diversity_scores]}")

        return total_rewards, diagnostics

    def compute_advantages(
        self,
        rewards: List[float],
        min_std: float = 0.01
    ) -> List[float]:
        """
        计算 GRPO 优势值

        Args:
            rewards: K 个 workflow 的奖励
            min_std: 最小标准差（防止除零）

        Returns:
            K 个优势值
        """
        if len(rewards) == 0:
            return []

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = max(math.sqrt(variance), min_std)

        advantages = [(r - mean_reward) / std for r in rewards]
        return advantages


class GroupedBatchProcessor:
    """
    分组批处理器 - 处理一个 batch 的问题组
    """

    def __init__(
        self,
        reward_calculator: GroupedRewardCalculator,
        base_reward_computer: Any  # 原始的 RewardComputer
    ):
        self.reward_calculator = reward_calculator
        self.base_reward_computer = base_reward_computer

    async def process_group(
        self,
        group: Dict[str, Any],
        workflows: List[str],
        executor: Any  # AFlowExecutor
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        处理一个问题组

        Args:
            group: 问题组数据
                {
                    'group_id': 'math_001',
                    'domain': 'math',
                    'problems': [{...}, {...}, {...}, {...}]
                }
            workflows: K 个 workflow 代码
            executor: AFlow 执行器

        Returns:
            (rewards, diagnostics)
        """
        problems = group['problems']
        K = len(workflows)

        # 每个 workflow 在每个问题上的得分
        problem_scores_per_workflow = [[] for _ in range(K)]

        # 遍历每个问题
        for problem in problems:
            # 遍历每个 workflow
            for i, workflow_code in enumerate(workflows):
                # 执行 workflow
                result = await executor.execute(
                    workflow_code=workflow_code,
                    problem=problem['question'],
                    ground_truth=problem['answer'],
                    domain=problem['domain'],
                    entry_point=problem.get('entry_point', ''),
                    test_cases=problem.get('test_cases', [])
                )

                # 计算正确性得分
                correctness = result.get('correctness_score', 0.0)

                problem_scores_per_workflow[i].append({
                    'problem_id': problem['id'],
                    'difficulty': problem['difficulty'],
                    'weight': problem['weight'],
                    'correctness': correctness,
                    'execution_time': result.get('execution_time', 0),
                    'success': result.get('success', False)
                })

        # 计算最终奖励
        rewards, diagnostics = self.reward_calculator.calculate_group_rewards(
            workflows=workflows,
            problem_scores_per_workflow=problem_scores_per_workflow
        )

        diagnostics['group_id'] = group['group_id']
        diagnostics['domain'] = group['domain']
        diagnostics['problem_scores'] = problem_scores_per_workflow

        return rewards, diagnostics


# 测试代码
if __name__ == "__main__":
    calc = GroupedRewardCalculator(debug=True)

    # 模拟两个 workflow
    workflows = [
        """class Workflow:
            def __init__(self):
                self.answer_generate = AnswerGenerate()
                self.review = Review()

            async def __call__(self, problem):
                ans = await self.answer_generate(problem)
                if ans:
                    review = await self.review(ans)
                return ans
        """,
        """class Workflow:
            def __init__(self):
                self.answer_generate = AnswerGenerate()
                self.programmer = Programmer()
                self.review = Review()
                self.revise = Revise()

            async def __call__(self, problem):
                ans = await self.answer_generate(problem)
                if not ans:
                    code = await self.programmer(problem)
                    ans = code
                review = await self.review(ans)
                if review.needs_revision:
                    ans = await self.revise(ans, review)
                return ans
        """
    ]

    # 模拟问题得分
    scores_w1 = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.4, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.0, 'problem_id': 'hard_1'},
    ]

    scores_w2 = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 0.7, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.7, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.4, 'problem_id': 'hard_1'},
    ]

    print("\n" + "="*60)
    print("测试 GroupedRewardCalculator")
    print("="*60)

    rewards, diag = calc.calculate_group_rewards(
        workflows=workflows,
        problem_scores_per_workflow=[scores_w1, scores_w2]
    )

    print(f"\n最终奖励: {rewards}")

    # 计算优势
    advantages = calc.compute_advantages(rewards)
    print(f"优势值: {advantages}")

    print("\n" + "="*60)
    print("测试平局情况（需要多样性打破平局）")
    print("="*60)

    # 两个 workflow 得分完全相同
    scores_tie = [
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_0'},
        {'difficulty': 'easy', 'weight': 0.15, 'correctness': 1.0, 'problem_id': 'easy_1'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.5, 'problem_id': 'hard_0'},
        {'difficulty': 'hard', 'weight': 0.35, 'correctness': 0.5, 'problem_id': 'hard_1'},
    ]

    rewards_tie, diag_tie = calc.calculate_group_rewards(
        workflows=workflows,
        problem_scores_per_workflow=[scores_tie, scores_tie]
    )

    print(f"\n最终奖励（有多样性加分）: {rewards_tie}")
    advantages_tie = calc.compute_advantages(rewards_tie)
    print(f"优势值（非零）: {advantages_tie}")
