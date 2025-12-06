#!/usr/bin/env python3
"""
GRPO训练器 - 在线学习模式的强化学习训练器
"""
import os

# P12修复: 禁用代理，确保LLM Judge可以直连localhost:8002
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

import gc
import torch
import torch.nn.functional as F
import asyncio
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json
import wandb  # ✨ 新增wandb集成

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data_manager import DataManager
from grouped_data_manager import GroupedDataManager  # P32: 分组数据管理器
from vllm_workflow_generator import VLLMWorkflowGenerator  # ✨ 使用新的生成器
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer
from grouped_reward import GroupedRewardCalculator  # P32: 分组奖励计算器（含复杂度）
from gpu_manager import GPUManager
from experience_buffer import ExperienceBuffer
from prompt_optimizer import PromptOptimizer
from operator_prompt_enhancer import OperatorPromptEnhancer
# P32: 删除WA-GRPO，使用新的复杂度奖励系统
# from wa_grpo import WAGRPOAdvantageComputer


class GRPOTrainer:
    """GRPO训练器：在线学习模式"""

    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Args:
            config_path: 训练配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("🚀 初始化GRPO训练器")
        print("=" * 60)

        # GPU管理（使用物理GPU ID）
        physical_gpus = self.config.get('physical_gpus', self.config['device_mapping'])
        self.gpu_manager = GPUManager(
            target_gpus=physical_gpus,
            protected_pids=self.config.get('protected_pids', []),
            auto_clean=False  # 禁用自动清理
        )

        # 跳过GPU环境验证，直接使用
        print(f"✅ 使用GPU {physical_gpus}（已禁用清理和验证）")

        # Temperature scheduling配置
        temp_config = self.config.get('temperature_schedule', {})
        self.temp_schedule = {
            'enabled': temp_config.get('enabled', True),
            'initial': temp_config.get('initial', 0.3),
            'final': temp_config.get('final', 0.8),
            'warmup_steps': temp_config.get('warmup_steps', 100)
        }
        print(f"\n🌡️  Temperature Scheduling:")
        print(f"  Enabled: {self.temp_schedule['enabled']}")
        if self.temp_schedule['enabled']:
            print(f"  Range: {self.temp_schedule['initial']} → {self.temp_schedule['final']}")
            print(f"  Warmup: {self.temp_schedule['warmup_steps']} steps")

        # ✨ 初始化wandb
        self._initialize_wandb()

        # 初始化组件
        self._initialize_components()

        print("=" * 60)
        print("✅ GRPO训练器初始化完成")
        print("=" * 60)

    def _initialize_wandb(self):
        """初始化wandb监控"""
        # 从配置或环境变量获取wandb设置
        wandb_config = self.config.get('wandb', {})

        # 设置API key(如果提供的话)
        wandb_api_key = wandb_config.get('api_key', 'b42ca0000cf06f97b05eba34f58823ad5f3122a4')

        # 尝试登录,如果失败则使用offline模式
        try:
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
                mode = "online"
            else:
                print("⚠️  wandb API key无效或未提供,使用offline模式")
                mode = "offline"
        except Exception as e:
            print(f"⚠️  wandb登录失败: {e}, 使用offline模式")
            mode = "offline"

        # 初始化wandb run
        wandb.init(
            project=wandb_config.get('project', 'aflow-roll-integration'),
            name=wandb_config.get('run_name', f"grpo-training-{time.strftime('%Y%m%d-%H%M%S')}"),
            mode=mode,  # online或offline
            config={
                # 训练配置
                "base_model": self.config['base_model'],
                "learning_rate": self.config['learning_rate'],
                "batch_size": self.config['rollout_batch_size'],
                "num_sequences": self.config['num_return_sequences_in_group'],
                "max_steps": self.config['max_steps'],
                "lora_rank": self.config['lora_rank'],
                "lora_alpha": self.config['lora_alpha'],
                # 数据配置
                "domain_ratios": self.config['domain_ratios'],
                # 奖励配置
                "reward_weights": self.config.get('reward_weights', {}),
            },
            tags=["grpo", "aflow", "roll", "workflow-generation"],
            notes="GRPO training with improved reward function (ROLL+AgentFlow design)"
        )

        print("\n✅ wandb初始化完成")
        print(f"  模式: {mode}")
        print(f"  项目: {wandb.run.project}")
        print(f"  Run名称: {wandb.run.name}")
        if mode == "online":
            print(f"  Run URL: {wandb.run.url}")
        else:
            print(f"  离线日志: wandb/offline-run-*")

    def _initialize_components(self):
        """初始化所有组件"""

        # P32: 检查是否启用分组训练模式
        grouped_config = self.config.get('grouped_training', {})
        self.use_grouped_training = grouped_config.get('enabled', False)

        # 1. 数据管理器
        print("\n📂 初始化数据管理器...")
        if self.use_grouped_training:
            # P32: 使用分组数据管理器
            print("  📦 使用分组训练模式 (P32)")
            self.grouped_data_manager = GroupedDataManager(
                data_dir=self.config['data_dir'] + "/grouped",
                groups_per_domain=grouped_config.get('groups_per_domain', 1),
                shuffle=True
            )
            self.grouped_data_manager.initialize()
            # 同时初始化普通数据管理器（用于向后兼容）
            self.data_manager = DataManager(
                data_dir=self.config['data_dir'],
                domain_ratios=self.config['domain_ratios']
            )
            self.data_manager.initialize()
        else:
            self.data_manager = DataManager(
                data_dir=self.config['data_dir'],
                domain_ratios=self.config['domain_ratios'],
                train_dataset=self.config.get('train_dataset'),  # P6: 支持配置数据集路径
                val_dataset=self.config.get('val_dataset'),
                test_dataset=self.config.get('test_dataset')
            )
            self.data_manager.initialize()
            self.grouped_data_manager = None

        # 2. RL模型（Qwen2.5-7B + LoRA）
        print("\n🤖 加载RL模型...")
        self._load_rl_model()

        # 3. RL工作流生成器（使用新的VLLMWorkflowGenerator）
        print("\n🔧 初始化工作流生成器（支持并行化）...")
        self.generator = VLLMWorkflowGenerator(
            model_name=self.config['base_model'],
            max_concurrent=self.config['num_return_sequences_in_group'],
            operator_descriptions_path=self.config.get('aflow_operator_descriptions_path'),
            use_vllm_api=False,  # 使用transformers模式
            device=f"cuda:{self.config['device_mapping'][0]}"
        )
        # 共享已加载的模型（避免重复加载）
        self.generator.model = self.model
        self.generator.tokenizer = self.tokenizer
        print(f"  ✅ 生成器已配置为并发模式（{self.config['num_return_sequences_in_group']} 并发）")

        # 4. ExperienceBuffer - 高质量样本管理（需先初始化，用于后续组件）
        print("\n📚 初始化ExperienceBuffer...")
        experience_config = self.config.get('experience_buffer', {})
        self.experience_buffer = ExperienceBuffer(
            buffer_size=experience_config.get('buffer_size', 100),
            reward_threshold=experience_config.get('reward_threshold', 8.0),
            persistence_dir=experience_config.get('persistence_dir', 'data/experience_buffer'),
            problem_types=["math", "code", "qa"]
        )
        print(f"  Buffer大小: {self.experience_buffer.buffer_size}")
        print(f"  奖励阈值: {self.experience_buffer.reward_threshold}")

        # 5. PromptOptimizer - Layer 1动态提示词优化
        print("\n✨ 初始化PromptOptimizer (Layer 1)...")
        prompt_config = self.config.get('prompt_optimizer', {})
        self.prompt_optimizer = PromptOptimizer()
        self.use_dynamic_prompts = prompt_config.get('enabled', True)
        print(f"  动态提示词: {'启用' if self.use_dynamic_prompts else '禁用'}")

        # 6. OperatorPromptEnhancer - Layer 2 operator提示词增强
        print("\n🔧 初始化OperatorPromptEnhancer (Layer 2)...")
        operator_config = self.config.get('operator_prompt_enhancer', {})
        self.operator_enhancer = OperatorPromptEnhancer(
            enable_enhancement=operator_config.get('enabled', True)
        )
        print(f"  Operator增强: {'启用' if self.operator_enhancer.enable_enhancement else '禁用'}")

        # 7. AFlow执行器（传入operator_enhancer）
        print("\n⚙️  初始化AFlow执行器...")
        timeout = self.config.get('execution_timeout', 180)  # 默认180秒
        # 🔧 P17修复: 传递正确的 llm_model_name 参数
        executor_model = self.config.get('aflow_executor_model', 'gpt-4o-mini')
        self.executor = AFlowExecutor(
            llm_config_path=self.config['aflow_config_path'],
            llm_model_name=executor_model,  # 🔧 P17: 使用配置中的模型名
            timeout=timeout,
            operator_enhancer=self.operator_enhancer  # 传递Layer 2增强器
        )
        print(f"  执行超时: {timeout}秒")

        # 8. 奖励计算器
        print("\n🎯 初始化奖励计算器...")
        # 🔧 P17修复: 从aflow_llm.yaml读取API key
        llm_judge_config = {
            "base_url": "https://api.openai.com/v1",
            "api_key": os.environ.get('OPENAI_API_KEY', ''),
            "model_name": "gpt-4o-mini"
        }
        # 尝试从aflow_llm.yaml读取API key
        try:
            import yaml
            aflow_config_path = self.config.get('aflow_config_path', 'config/aflow_llm.yaml')
            with open(aflow_config_path, 'r') as f:
                aflow_config = yaml.safe_load(f)
            if 'models' in aflow_config and 'gpt-4o-mini' in aflow_config['models']:
                api_key = aflow_config['models']['gpt-4o-mini'].get('api_key', '')
                if api_key and api_key != '${OPENAI_API_KEY}':
                    llm_judge_config['api_key'] = api_key
                    print(f"  ✅ 从 {aflow_config_path} 读取API key")
        except Exception as e:
            print(f"  ⚠️ 读取aflow配置失败: {e}")

        self.reward_computer = RewardComputer(
            reward_weights=self.config.get('reward_weights'),
            use_llm_judge=True,  # 启用LLM Judge (使用OpenAI API)
            llm_config=llm_judge_config,
            debug_logging=self.config.get('debug', False)  # P0修复: 传递debug设置
        )

        # P32: 分组奖励计算器（含复杂度奖励）
        if self.use_grouped_training:
            complexity_config = self.config.get('complexity_reward', {})
            self.grouped_reward_calculator = GroupedRewardCalculator(
                weight_easy=grouped_config.get('weight_easy', 0.3),
                weight_hard=grouped_config.get('weight_hard', 0.7),
                correctness_weight=complexity_config.get('correctness_weight', 0.6),
                complexity_weight=complexity_config.get('complexity_weight', 0.4),
                debug=self.config.get('debug', False)
            )
            print(f"\n🎯 初始化分组奖励计算器 (P32)...")
            print(f"  正确性权重: {complexity_config.get('correctness_weight', 0.6)}")
            print(f"  复杂度权重: {complexity_config.get('complexity_weight', 0.4)}")
        else:
            self.grouped_reward_calculator = None

        # 9. 优化器
        print("\n🔬 初始化优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # P1-3: Cosine学习率调度器
        print("\n📈 初始化Cosine学习率调度器 (P1-3)...")
        warmup_steps = self.config.get('warmup_steps', 50)
        max_steps = self.config['max_steps']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
        print(f"  Warmup步数: {warmup_steps}")
        print(f"  总训练步数: {max_steps}")
        print(f"  初始LR: {self.config['learning_rate']}")

        # P32: 删除WA-GRPO，使用新的复杂度奖励系统
        # 优势计算改为使用grouped_reward_calculator中的compute_advantages
        self.advantage_computer = None
        print("\n✅ 使用简化的GRPO优势计算（基于复杂度奖励）")

    def _load_rl_model(self):
        """加载RL模型（Qwen2.5-7B + LoRA）- 内存优化版"""
        device = f"cuda:{self.config['device_mapping'][0]}"

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16 if self.config.get('bf16') else torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )

        # 🔧 OOM修复: 启用梯度检查点
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            print("✅ 梯度检查点已启用 (节省显存)")

        # 应用LoRA
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config['lora_rank'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'].split(','),
                lora_dropout=self.config['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

            print(f"✅ LoRA应用完成")
            self.model.print_trainable_parameters()

    def get_current_temperature(self, step: int) -> float:
        """
        计算当前step的temperature

        策略: 线性从initial升至final
        - 早期: 低温度生成确定性workflow，建立baseline
        - 后期: 高温度探索多样性workflow

        Args:
            step: 当前训练步数

        Returns:
            当前的temperature值
        """
        if not self.temp_schedule['enabled']:
            return self.config['generation_config']['temperature']

        if step < self.temp_schedule['warmup_steps']:
            # Linear warmup
            progress = step / self.temp_schedule['warmup_steps']
            temp = (self.temp_schedule['initial'] +
                   progress * (self.temp_schedule['final'] - self.temp_schedule['initial']))
        else:
            temp = self.temp_schedule['final']

        return temp

    async def train_step(self, step: int) -> Dict:
        """
        单步GRPO训练（在线学习）

        Returns:
            metrics: 训练指标
        """
        import torch
        import gc

        # 显存监控开始
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9

        # 1. 采样batch
        # 🔥 P32修复: 使用分组数据管理器获取 3类×4题=12 问题
        if self.grouped_data_manager is not None:
            # 使用分组采样: 3 domain × 1 group × 4 problems = 12 problems
            groups = self.grouped_data_manager.sample_step_groups(split="train")
            batch = self.grouped_data_manager.flatten_groups_to_problems(groups)
            group_stats = self.grouped_data_manager.get_step_stats(groups)
            print(f"\n📦 P32分组采样: {group_stats['total_groups']}组×4题 = {len(batch)}问题")
            print(f"  领域分布: {group_stats['groups']}")
        else:
            # 后备：使用普通数据管理器
            batch = self.data_manager.sample_batch(
                batch_size=self.config['rollout_batch_size'],
                split="train"
            )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(batch)
        print(f"📦 Batch {step}: {len(batch)} 样本, 分布: {batch_stats}")

        # 获取当前temperature（动态调度）
        current_temp = self.get_current_temperature(step)
        print(f"🌡️  Temperature: {current_temp:.3f}")

        # 2. 为每个问题生成K个工作流（GRPO组）
        all_workflows = []
        all_problems = []
        all_answers = []
        all_rewards = []
        all_log_probs = []

        # ✨ 新增：准确率统计
        correctness_scores = []  # 存储所有正确性分数

        num_sequences = self.config['num_return_sequences_in_group']
        batch_size = len(batch)
        total_sequences = batch_size * num_sequences

        # 🚀🚀🚀 超级batch优化：一次性生成所有workflow（batch_size × num_sequences）
        print(f"\n🚀🚀🚀 超级batch推理: {batch_size}样本 × {num_sequences}序列 = {total_sequences}个workflow一次性GPU生成...")

        # 准备所有输入（4样本 × 6序列 = 24个prompt）
        all_problems = []
        all_types = []
        all_temps = []
        all_prompts = []
        sample_metadata = []  # 保存每个样本的元数据

        for sample_idx, sample in enumerate(batch):
            problem = sample['problem']
            problem_type = sample['problem_type']
            ground_truth = sample['ground_truth']

            # P0修复: 从meta中获取test_cases和entry_point
            meta = sample.get('meta', {})
            test_cases = meta.get('test_cases', sample.get('test', ''))
            # 优先从meta获取entry_point（HumanEval格式），否则从sample顶层获取
            entry_point = meta.get('entry_point', sample.get('entry_point', ''))

            # 如果没有entry_point，尝试从test_cases中提取（MBPP格式）
            if not entry_point and test_cases:
                import re
                match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test_cases)
                if match:
                    entry_point = match.group(1)

            # 保存样本元数据（用于后续执行）
            for seq_idx in range(num_sequences):
                sample_metadata.append({
                    'sample_idx': sample_idx,
                    'seq_idx': seq_idx,
                    'problem': problem,
                    'ground_truth': ground_truth,
                    'problem_type': problem_type,
                    'entry_point': entry_point,
                    'test': test_cases,
                    'source': sample.get('source', None),
                    'context': sample.get('context', [])  # P1修复: 传递context用于QA数据集
                })

                # 为每个序列添加输入
                all_problems.append(problem)
                all_types.append(problem_type)
                all_temps.append(current_temp)

                if self.use_dynamic_prompts:
                    custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                        problem=problem,
                        problem_type=problem_type
                    )
                    all_prompts.append(custom_prompt)
                else:
                    all_prompts.append(None)

        # 🚀 关键优化：一次性GPU batch生成所有workflow（24个）
        workflow_results = await self.generator.generate_workflows_batch(
            problems=all_problems,
            problem_types=all_types,
            temperatures=all_temps,
            custom_prompts=all_prompts if any(all_prompts) else None
        )

        print(f"✅ workflow生成完成，开始并行执行和奖励计算...")

        # 并行处理所有序列（执行+奖励）
        async def process_single_sequence_result(global_idx, workflow_result, metadata):
            """处理单个生成结果（执行+奖励）"""
            workflow_code = workflow_result['workflow_code']
            problem = metadata['problem']
            ground_truth = metadata['ground_truth']
            problem_type = metadata['problem_type']
            sample_idx = metadata['sample_idx']
            seq_idx = metadata['seq_idx']

            # P21修复: 提取DSL质量信息用于条件激活奖励
            dsl_info = workflow_result.get('metadata', {}).get('dsl_info', None)

            # 🔥 P23修复: 使用raw_text（原始模型输出）进行训练
            # 关键修复: 之前使用workflow_code（Python代码）训练，但prompt要求生成DSL
            # 这导致模型学到错误的输出格式，一次梯度更新后就崩溃
            # 现在使用raw_text（原始模型输出）训练，保持prompt和target格式一致
            raw_text = workflow_result.get('metadata', {}).get('raw_text', workflow_code)

            # 计算log概率（旧策略）- P23: 使用raw_text而不是workflow_code
            log_prob = await self._compute_log_prob(problem, raw_text, problem_type)

            # === 日志：打印QWEN生成的完整Workflow代码 ===
            print(f"\n===== [QWEN] Workflow 代码 S{sample_idx+1}-{seq_idx+1}/{num_sequences} =====\n{workflow_code}\n===== 代码结束 =====", flush=True)

            # P21修复: 打印DSL质量信息
            if dsl_info:
                print(f"  📊 DSL质量: fallback={dsl_info.get('is_fallback', True)}, "
                      f"ops={dsl_info.get('num_operators', 0)}, "
                      f"chain={dsl_info.get('has_chain', False)}, "
                      f"DSL={dsl_info.get('dsl_text', 'N/A')[:60]}", flush=True)

            # 执行工作流
            exec_metadata = {'success': False}
            try:
                print("  ▶️ 开始使用OSS执行Workflow...", flush=True)
                answer, cost, exec_metadata = await self.executor.execute_workflow(
                    workflow_code=workflow_code,
                    problem=problem,
                    problem_type=problem_type,
                    entry_point=metadata['entry_point'],
                    test=metadata['test'],
                    source=metadata.get('source', ''),
                    context=metadata.get('context', [])
                )

                # 🔥 P27修复: 简单奖励 - 直接使用正确性分数，去掉P21条件激活
                if exec_metadata['success']:
                    # 使用简单的正确性奖励（0-1之间）
                    reward = self.reward_computer.compute_reward(
                        problem=problem,
                        prediction=answer,
                        ground_truth=ground_truth,
                        problem_type=problem_type,
                        metadata=exec_metadata,
                        test=metadata['test'],
                        entry_point=metadata['entry_point'],
                        source=metadata['source']
                    )

                    # 日志：执行与评估详情
                    print(f"  🧪 执行成功 | cost: {cost:.4f} | 用时: {exec_metadata.get('execution_time', 'NA')}s", flush=True)
                    print(f"  预测答案: {str(answer)[:100]}", flush=True)
                    print(f"  标准答案: {str(ground_truth)[:100]}", flush=True)
                    print(f"  📊 P27简单奖励: {reward:.3f}", flush=True)

                    correctness = reward

                    # 🔥 P32修复: 添加复杂度奖励
                    if self.grouped_reward_calculator is not None:
                        total_reward, complexity_details = self.grouped_reward_calculator.calculate_total_reward(
                            correctness_score=correctness,
                            dsl_text=raw_text
                        )
                        complexity_score = complexity_details['complexity_score']
                        print(f"  🔧 P32复杂度: {complexity_score:.3f} (ops={complexity_details['complexity_details'].get('operator_count', 0)}, len={complexity_details['complexity_details'].get('dsl_length', 0)})", flush=True)
                        print(f"  🎯 P32总奖励: {total_reward:.3f} = {correctness:.3f}×0.6 + {complexity_score:.3f}×0.4", flush=True)
                        reward = total_reward

                    is_correct = correctness > 0.5
                    status_icon = "✅" if is_correct else "❌"

                    # 实时日志到 wandb
                    wandb.log({
                        f"sample/{problem_type}/correctness": correctness,
                        f"sample/{problem_type}/reward": reward,
                        f"sample/step": step,
                        f"sample/sample_id": sample_idx * num_sequences + seq_idx,
                    })

                    print(f"  [S{sample_idx+1}-{seq_idx+1}/{num_sequences}] {status_icon} 正确性: {correctness:.1f} | 预测: {str(answer)[:50]}")
                else:
                    # 执行失败：奖励为0
                    reward = 0.0
                    correctness = 0.0
                    print(f"  [S{sample_idx+1}-{seq_idx+1}/{num_sequences}] ❌ 执行失败 | 奖励: {reward:.3f}")

            except Exception as e:
                print(f"  [S{sample_idx+1}-{seq_idx+1}/{num_sequences}] ⚠️  错误: {type(e).__name__}: {e}")
                answer = None
                # P27修复: 异常时奖励为0（简化）
                reward = 0.0
                correctness = 0.0

            # P23修复: 返回raw_text用于训练（原始模型输出）
            return {
                'workflow_code': workflow_code,  # Python代码，用于执行
                'raw_text': raw_text,  # P23: 原始模型输出，用于训练
                'answer': answer,
                'reward': reward,
                'log_prob': log_prob,
                'correctness': correctness,
                'sample_idx': sample_idx,
                'seq_idx': seq_idx,
                'problem': problem,
                'ground_truth': ground_truth,
                'problem_type': problem_type,
                'metadata': metadata
            }

        # 🚀🚀 并发执行所有24个序列的执行和奖励计算
        tasks = [
            process_single_sequence_result(i, workflow_results[i], sample_metadata[i])
            for i in range(total_sequences)
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        print(f"✅ 所有workflow执行完成，开始整理结果...")

        # 初始化批量收集列表（用于WA-GRPO）
        batch_group_rewards = []
        batch_group_workflows = []
        batch_group_log_probs = []
        batch_group_correctness = []
        batch_group_answers = []
        batch_group_exec_metas = []  # WA-GRPO需要执行元信息

        # 按样本重新组织结果（用于GRPO组归一化）
        for sample_idx in range(batch_size):
            # 提取该样本的所有序列结果
            group_workflows = []
            group_answers = []
            group_rewards = []
            group_log_probs = []
            group_correctness = []
            group_exec_metas = []  # WA-GRPO需要

            # 从all_results中提取属于该样本的结果
            for global_idx in range(total_sequences):
                result = all_results[global_idx]
                # P9修复: 正确处理Exception对象
                if isinstance(result, Exception):
                    # Exception对象没有get方法，需要通过global_idx计算sample_idx
                    result_sample_idx = global_idx // num_sequences
                    result_seq_idx = global_idx % num_sequences
                    if result_sample_idx == sample_idx:
                        print(f"  ⚠️  样本{sample_idx+1}-序列{result_seq_idx+1} 异常: {type(result).__name__}: {str(result)[:100]}")
                        group_workflows.append("")
                        group_answers.append(None)
                        group_rewards.append(-10.0)
                        group_log_probs.append(0.0)
                        group_correctness.append(-10.0)
                        group_exec_metas.append({'success': False, 'error_type': type(result).__name__})
                    continue

                if result['sample_idx'] == sample_idx:
                    # P23修复: 使用raw_text（原始模型输出）用于训练
                    # workflow_code是Python代码（用于执行），raw_text是模型原始输出（用于训练）
                    group_workflows.append(result.get('raw_text', result['workflow_code']))
                    group_answers.append(result['answer'])
                    group_rewards.append(result['reward'])
                    group_log_probs.append(result['log_prob'])
                    group_correctness.append(result['correctness'])
                    # 提取执行元信息供WA-GRPO使用
                    exec_meta = result.get('metadata', {})
                    exec_meta['success'] = result.get('correctness', 0) > 0
                    exec_meta['execution_time'] = result.get('execution_time', 0.1)
                    group_exec_metas.append(exec_meta)

            correctness_scores.extend(group_correctness)

            # 收集到临时列表（稍后统一用WA-GRPO计算优势）
            batch_group_rewards.append(group_rewards)
            batch_group_workflows.append(group_workflows)
            batch_group_log_probs.append(group_log_probs)
            batch_group_correctness.append(group_correctness)
            batch_group_answers.append(group_answers)
            batch_group_exec_metas.append(group_exec_metas)

        # === P27: 简单GRPO组内归一化（去掉WA-GRPO） ===
        # 将所有组的数据展平
        all_rewards_flat = [r for group in batch_group_rewards for r in group]

        # 🔥 P27修复: 使用简单的组内归一化计算优势
        # 原因: WA-GRPO的tie-breaker和noise添加可能导致模型崩溃
        # 解决方案: 如果组内奖励相同，advantage全为0，不进行训练
        all_advantages = []
        zero_var_groups = 0
        for group_idx in range(batch_size):
            start_idx = group_idx * num_sequences
            end_idx = start_idx + num_sequences
            group_rewards = np.array(all_rewards_flat[start_idx:end_idx])

            # 组内归一化：(r - mean) / (std + eps)
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards)

            if group_std < 1e-8:
                # 零方差组：所有样本奖励相同，advantage全为0
                zero_var_groups += 1
                group_advantages = [0.0] * num_sequences
            else:
                group_advantages = [(r - group_mean) / group_std for r in group_rewards]

            all_advantages.extend(group_advantages)

        # 打印简单GRPO诊断信息
        print(f"\n🔧 P27 简单GRPO诊断:")
        print(f"  零方差组: {zero_var_groups}/{batch_size}")
        print(f"  平均奖励: {np.mean(all_rewards_flat):.3f}")
        print(f"  奖励范围: [{np.min(all_rewards_flat):.3f}, {np.max(all_rewards_flat):.3f}]")

        # 整理结果到全局列表
        for sample_idx in range(batch_size):
            # P9修复: 从batch获取problem，而不是从可能是Exception的result获取
            current_batch_sample = batch[sample_idx]
            start_idx = sample_idx * num_sequences
            end_idx = start_idx + num_sequences

            group_advantages = all_advantages[start_idx:end_idx]
            group_workflows = batch_group_workflows[sample_idx]
            group_answers = batch_group_answers[sample_idx]
            group_rewards = batch_group_rewards[sample_idx]
            group_log_probs = batch_group_log_probs[sample_idx]
            group_correctness = batch_group_correctness[sample_idx]

            # 💾 收集高质量样本到ExperienceBuffer
            for idx, (workflow, answer, reward) in enumerate(zip(group_workflows, group_answers, group_rewards)):
                if reward >= self.experience_buffer.reward_threshold:
                    result = all_results[sample_idx * num_sequences + idx]
                    # P9修复: 跳过Exception结果
                    if isinstance(result, Exception):
                        continue
                    exp_sample = {
                        'problem': result['problem'],
                        'workflow_code': workflow,
                        'answer': answer,
                        'ground_truth': result['ground_truth'],
                        'reward': reward,
                        'correctness_score': group_correctness[idx],
                        'metadata': {
                            'problem_type': result['problem_type'],
                            'step': step
                        },
                        'step': step
                    }
                    self.experience_buffer.add_sample(exp_sample, result['problem_type'])

            # 收集到全局列表（用于策略更新）
            all_workflows.extend(group_workflows)
            # P9修复: 使用current_batch_sample而不是result_sample
            all_problems.extend([current_batch_sample['problem']] * num_sequences)
            all_answers.extend(group_answers)
            all_rewards.extend(group_advantages)  # 使用WA-GRPO计算的优势
            all_log_probs.extend(group_log_probs)

        # 3. 策略梯度更新
        # 🔥 P28修复: 如果所有advantage都为0，跳过更新
        # 问题: 即使advantage=0，entropy_bonus仍会产生梯度，导致模型漂移
        # 解决: 检测全零advantage，跳过整个更新步骤
        all_adv_zero = all(abs(adv) < 1e-8 for adv in all_rewards)
        if all_adv_zero:
            print(f"\n⏭️  P28: 跳过更新 - 所有advantage为0，无学习信号")
            loss = 0.0
            kl_div = 0.0
        else:
            print(f"\n🔄 更新策略...")
            loss, kl_div = await self._update_policy(
                problems=all_problems,
                workflows=all_workflows,
                old_log_probs=all_log_probs,
                advantages=all_rewards,
                problem_types=[s['problem_type'] for s in batch for _ in range(num_sequences)]
            )

        # 4. 指标
        # ✨ 新增：计算准确率统计
        num_correct = sum(1 for score in correctness_scores if score >= 0.9) # 修改阈值为0.9适应二元奖励
        num_total = len(correctness_scores)
        accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

        # 计算问题类型分布的准确率
        problem_type_stats = {}
        for problem_type in ['math', 'code', 'qa']:
            type_scores = [s for s, p in zip(correctness_scores,
                          [s['problem_type'] for s in batch for _ in range(num_sequences)])
                          if p == problem_type]
            if type_scores:
                type_correct = sum(1 for s in type_scores if s >= 0.9) # 修改阈值
                type_accuracy = (type_correct / len(type_scores) * 100)
                type_avg = np.mean(type_scores)
                problem_type_stats[problem_type] = {
                    "accuracy": type_accuracy,
                    "avg_score": type_avg,
                    "count": len(type_scores)
                }

        metrics = {
            "step": step,
            "loss": loss,
            "kl_div": kl_div,
            "avg_reward": np.mean(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "num_samples": len(all_workflows),
            # ✨ 新增准确率指标
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "avg_correctness_score": avg_correctness
        }

        print(f"\n🎯 准确率统计: {num_correct}/{num_total} = {accuracy:.1f}% (平均正确性评分: {avg_correctness:.2f}/1.0)")

        # P1-1修复: 增强的分类准确率统计
        print(f"\n📊 问题类型分布 (P1增强):")
        for ptype, stats in problem_type_stats.items():
            count = stats['count']
            if count > 0:
                type_acc = stats['accuracy']
                type_avg = stats['avg_score']
                # 显示奖励分布
                print(f"  {ptype}: {type_acc:.1f}% 准确率 | 平均分: {type_avg:.2f} | 样本数: {count}")

        # ✨ 新增：子数据集（source）准确率统计
        source_stats = {}
        for score, sample in zip(correctness_scores, [s for s in batch for _ in range(num_sequences)]):
            source = sample.get('source', 'unknown')
            if source not in source_stats:
                source_stats[source] = {'scores': [], 'correct': 0, 'total': 0}
            source_stats[source]['scores'].append(score)
            source_stats[source]['total'] += 1
            if score >= 0.9:
                source_stats[source]['correct'] += 1

        print(f"\n📈 子数据集准确率:")
        for source in sorted(source_stats.keys()):
            stats = source_stats[source]
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            avg = np.mean(stats['scores']) if stats['scores'] else 0.0
            print(f"  {source:15s}: {stats['correct']:2d}/{stats['total']:2d} = {acc:5.1f}% | 平均分: {avg:.2f}")

        # P0-2统计: 全零优势组计数 + 详细Batch级别诊断
        zero_advantage_groups = 0
        all_correct_groups = 0  # 全对组
        all_wrong_groups = 0    # 全错组
        mixed_groups = 0        # 混合组（有对有错）
        group_reward_details = []  # 每组奖励详情

        for i in range(0, len(all_rewards), num_sequences):
            group_rewards = all_rewards[i:i+num_sequences]
            group_std = np.std(group_rewards)
            group_mean = np.mean(group_rewards)

            # 统计组类型
            if group_std < 1e-6:
                zero_advantage_groups += 1
                if group_mean > 0.5:  # 全对
                    all_correct_groups += 1
                elif group_mean < 0.1:  # 全错
                    all_wrong_groups += 1
            else:
                mixed_groups += 1

            group_reward_details.append({
                'mean': group_mean,
                'std': group_std,
                'min': min(group_rewards),
                'max': max(group_rewards),
                'rewards': group_rewards[:4]  # 只记录前4个
            })

        print(f"\n🔧 GRPO诊断 (Batch级别):")
        print(f"  全零方差组: {zero_advantage_groups}/{batch_size} ({zero_advantage_groups/batch_size*100:.1f}%)")
        print(f"    ├── 全对组: {all_correct_groups}")
        print(f"    ├── 全错组: {all_wrong_groups}")
        print(f"    └── 其他同分组: {zero_advantage_groups - all_correct_groups - all_wrong_groups}")
        print(f"  混合组(有差异): {mixed_groups}/{batch_size} ({mixed_groups/batch_size*100:.1f}%)")
        print(f"  Batch奖励: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.4f}")
        print(f"  奖励范围: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")

        # 如果全对或全错组占比超过50%，打印详细信息
        if (all_correct_groups + all_wrong_groups) / batch_size > 0.5:
            print(f"  ⚠️  警告: 高同质性Batch! 全对+全错占{(all_correct_groups + all_wrong_groups)/batch_size*100:.0f}%")
            print(f"  前3组奖励详情:")
            for idx, detail in enumerate(group_reward_details[:3]):
                print(f"    组{idx+1}: {detail['rewards']} (mean={detail['mean']:.2f}, std={detail['std']:.4f})")

        # ✨ 详细 wandb logging (实时仪表板) - P1增强版
        current_lr = self.scheduler.get_last_lr()[0]  # P1-3: 获取当前学习率
        wandb_log_data = {
            "train/loss": loss,
            "train/kl_div": kl_div,
            "train/avg_reward": np.mean(all_rewards),
            "train/max_reward": np.max(all_rewards),
            "train/min_reward": np.min(all_rewards),
            "train/accuracy": accuracy,
            "train/avg_correctness_score": avg_correctness,
            "train/num_correct": num_correct,
            "train/num_total": num_total,
            "train/temperature": current_temp,
            "train/learning_rate": current_lr,  # P1-3: 记录学习率
            "train/step": step,
            # P0-2: GRPO诊断指标
            "grpo/zero_advantage_groups": zero_advantage_groups,
            "grpo/zero_advantage_ratio": zero_advantage_groups / batch_size,
            "grpo/advantage_std": np.std(all_rewards),
            "grpo/advantage_min": min(all_rewards),
            "grpo/advantage_max": max(all_rewards),
        }

        # 添加问题类型的分布指标
        for ptype, stats in problem_type_stats.items():
            wandb_log_data[f"train/accuracy_{ptype}"] = stats['accuracy']
            wandb_log_data[f"train/avg_score_{ptype}"] = stats['avg_score']
            wandb_log_data[f"train/count_{ptype}"] = stats['count']

        # 清理张量列表,释放显存
        del all_workflows, all_problems, all_answers, all_rewards, all_log_probs, correctness_scores
        torch.cuda.empty_cache()
        gc.collect()

        # 显存监控结束
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"显存: 前={mem_before:.2f}GB, 后={mem_after:.2f}GB, 峰值={mem_peak:.2f}GB, 增长={(mem_after-mem_before):.3f}GB")

        # 记录到wandb
        wandb_log_data["memory/allocated_gb"] = mem_after
        wandb_log_data["memory/peak_gb"] = mem_peak
        wandb_log_data["memory/growth_gb"] = mem_after - mem_before

        wandb.log(wandb_log_data, step=step)

        return metrics

    async def _compute_log_prob(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（旧策略）

        🔥 P26修复: 只对response部分计算loss，与_compute_log_prob_trainable保持一致
        """

        self.model.eval()

        with torch.no_grad():
            # 构建prompt和response
            prompt = self.generator._build_generation_prompt(problem, problem_type)

            # 分别tokenize prompt和response
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
            response_ids = self.tokenizer(workflow_code, return_tensors="pt", add_special_tokens=False)["input_ids"]

            # 拼接为完整序列
            # 🔧 P31修复: 确保input_ids为Long类型，避免RuntimeError
            input_ids = torch.cat([prompt_ids, response_ids], dim=1).long().to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

            # 构建labels: prompt部分设为-100（忽略），只计算response部分的loss
            prompt_len = prompt_ids.shape[1]
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100  # 忽略prompt部分

            # 前向传播 - 只对response部分计算loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 负对数似然 -> log概率
            log_prob = -outputs.loss

            return log_prob.detach().cpu()

    async def _update_policy(
        self,
        problems: List[str],
        workflows: List[str],
        old_log_probs: List[torch.Tensor],
        advantages: List[float],
        problem_types: List[str]
    ) -> Tuple[float, float]:
        """更新策略（GRPO）- 内存优化版"""

        self.model.train()

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        # 梯度累积
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        # 🔧 OOM修复: 在策略更新前清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 🔧 OOM修复: 减小mini-batch大小，每次只处理1个样本
        mini_batch_size = 1  # 减小到1，确保不OOM

        # 🔥 P25修复: 修正梯度累积逻辑，避免OOM
        # 问题: 之前累积多个loss再backward，导致保留所有forward的计算图(8x内存)
        # 解决: 每个样本单独backward，梯度自动累积，只需1x内存

        for i, (problem, workflow, old_log_prob, advantage, problem_type) in enumerate(
            zip(problems, workflows, old_log_probs, advantages, problem_types)
        ):
            # 🔧 OOM修复: 每个样本处理前清理缓存
            if i % 3 == 0:
                torch.cuda.empty_cache()

            # 计算新log概率
            new_log_prob = await self._compute_log_prob_trainable(problem, workflow, problem_type)

            # 重要性采样比
            ratio = torch.exp(new_log_prob - old_log_prob.to(self.model.device))

            # 🔥 P22修复: DAPO非对称裁剪 (防止熵崩溃)
            clip_low = self.config.get('clip_range_low', 0.2)
            clip_high = self.config.get('clip_range_high', 0.28)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

            advantage_tensor = torch.tensor(advantage, device=self.model.device)
            policy_loss = -torch.min(
                ratio * advantage_tensor,
                clipped_ratio * advantage_tensor
            )

            # 🔥 P22修复: 熵奖励
            entropy_coef = self.config.get('entropy_coef', 0.01)
            entropy_bonus = -entropy_coef * new_log_prob

            # KL正则化
            if self.config.get('use_kl_loss'):
                kl_loss = self.config['kl_loss_coef'] * (new_log_prob - old_log_prob.to(self.model.device)).pow(2)
            else:
                kl_loss = torch.tensor(0.0, device=self.model.device)

            # 总损失 (P22: 添加熵奖励)
            loss = policy_loss + kl_loss - entropy_bonus

            # 🔥 P25修复: 每个样本单独backward，除以grad_accum_steps实现平均
            # 梯度会自动累积，不需要保留多个计算图
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.item()
            total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            num_updates += 1

            # 每grad_accum_steps个样本做一次优化器更新
            if (i + 1) % grad_accum_steps == 0 or i == len(workflows) - 1:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # 🔧 P25: 优化器步骤后清理显存
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)

        return avg_loss, avg_kl

    async def _compute_log_prob_trainable(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（新策略，可训练）

        🔥 P26修复: 只对response部分计算loss，忽略prompt
        之前的BUG: 对整个序列(prompt+response)计算loss
        - prompt ~500+ tokens, response ~5-10 tokens
        - loss主要由prompt决定，response梯度被稀释
        - 导致模型在一次梯度更新后崩溃

        修复: 使用labels=-100忽略prompt部分
        """

        # 构建prompt和response
        prompt = self.generator._build_generation_prompt(problem, problem_type)

        # 分别tokenize prompt和response
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        response_ids = self.tokenizer(workflow_code, return_tensors="pt", add_special_tokens=False)["input_ids"]

        # 拼接为完整序列
        # 🔧 P31修复: 确保input_ids为Long类型，避免RuntimeError
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).long().to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # 构建labels: prompt部分设为-100（忽略），只计算response部分的loss
        prompt_len = prompt_ids.shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # 忽略prompt部分

        # 前向传播 - 只对response部分计算loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 负对数似然 -> log概率
        log_prob = -outputs.loss

        return log_prob

    async def evaluate_on_val_set(self, num_samples: int = 50) -> Dict:
        """
        在验证集上评估模型性能

        Args:
            num_samples: 验证样本数量

        Returns:
            验证指标字典
        """
        print(f"\n{'='*60}")
        print(f"🧪 验证集评估 ({num_samples}个样本)")
        print(f"{'='*60}")

        # 采样测试集
        val_batch = self.data_manager.sample_batch(
            batch_size=num_samples,
            split="test"  # 使用测试集（87个样本）
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(val_batch)
        print(f"📦 验证集分布: {batch_stats}")

        # 评估每个样本
        correctness_scores = []
        total_cost = 0.0
        successful_executions = 0

        # 🚀🚀🚀 超级batch优化：批量生成所有workflow
        print(f"\n🚀 批量生成 {num_samples} 个workflow（超级batch推理）...")

        # 准备所有输入
        all_problems = []
        all_types = []
        all_temps = []
        all_prompts = []
        sample_metadata = []

        for sample in val_batch:
            problem = sample['problem']
            problem_type = sample['problem_type']
            ground_truth = sample['ground_truth']

            # 保存样本元数据
            # P0修复: 从meta中获取test_cases和entry_point
            meta = sample.get('meta', {})
            test_cases = meta.get('test_cases', sample.get('test', ''))
            # 优先从meta获取entry_point（HumanEval格式），否则从sample顶层获取
            entry_point = meta.get('entry_point', sample.get('entry_point', ''))

            # 如果没有entry_point，尝试从test_cases中提取（MBPP格式）
            if not entry_point and test_cases:
                import re
                match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test_cases)
                if match:
                    entry_point = match.group(1)

            sample_metadata.append({
                'problem': problem,
                'ground_truth': ground_truth,
                'problem_type': problem_type,
                'entry_point': entry_point,
                'test': test_cases,
                'source': sample.get('source', None),
                'context': sample.get('context', [])  # P1修复: 传递context用于QA数据集
            })

            # 准备生成输入
            all_problems.append(problem)
            all_types.append(problem_type)
            all_temps.append(self.config['generation_config']['temperature'])

            # 动态提示词
            if self.use_dynamic_prompts:
                custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                    problem=problem,
                    problem_type=problem_type
                )
                all_prompts.append(custom_prompt)
            else:
                all_prompts.append(None)

        # 🚀 关键：一次性GPU batch生成所有workflow
        workflow_results = await self.generator.generate_workflows_batch(
            problems=all_problems,
            problem_types=all_types,
            temperatures=all_temps,
            custom_prompts=all_prompts if any(all_prompts) else None
        )

        print(f"✅ workflow生成完成，开始并行执行和评估...")

        # 并行处理所有样本（执行+奖励）
        async def process_single_sample(idx, workflow_result, metadata):
            """处理单个样本"""
            workflow_code = workflow_result['workflow_code']
            problem = metadata['problem']
            ground_truth = metadata['ground_truth']
            problem_type = metadata['problem_type']

            try:
                # 执行workflow
                answer, cost, exec_metadata = await self.executor.execute_workflow(
                    workflow_code=workflow_code,
                    problem=problem,
                    problem_type=problem_type,
                    entry_point=metadata['entry_point'],
                    test=metadata['test'],
                    source=metadata.get('source', ''),
                    context=metadata.get('context', [])
                )

                # 计算正确性
                if exec_metadata['success']:
                    correctness = self.reward_computer.compute_reward(
                        problem=problem,
                        prediction=answer,
                        ground_truth=ground_truth,
                        problem_type=problem_type,
                        test=metadata['test'],
                        entry_point=metadata['entry_point'],
                        source=metadata['source']
                    )

                    is_correct = correctness > 0.5
                    status_icon = "✅" if is_correct else "❌"
                    if idx < 5:  # 只打印前5个样本
                        print(f"  {status_icon} [{idx+1}/{num_samples}] 正确性: {correctness:.1f}/1.0")

                    return {
                        'correctness': correctness,
                        'cost': cost,
                        'success': True
                    }
                else:
                    if idx < 5:
                        print(f"  ❌ [{idx+1}/{num_samples}] 执行失败")
                    return {'correctness': 0.0, 'cost': 0.0, 'success': False}

            except Exception as e:
                if idx < 5:
                    print(f"  ⚠️  [{idx+1}/{num_samples}] 错误: {type(e).__name__}")
                return {'correctness': 0.0, 'cost': 0.0, 'success': False}

        # 🚀🚀 并发执行所有样本
        tasks = [
            process_single_sample(i, workflow_results[i], sample_metadata[i])
            for i in range(num_samples)
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理结果
        for result in all_results:
            if isinstance(result, Exception):
                correctness_scores.append(0.0)
            else:
                correctness_scores.append(result['correctness'])
                if result['success']:
                    # 确保cost是float类型
                    cost_value = result.get('cost', 0.0)
                    if isinstance(cost_value, str):
                        try:
                            # 只转换纯数字字符串
                            if cost_value and cost_value.replace('.','',1).replace('-','',1).isdigit():
                                cost_value = float(cost_value)
                            else:
                                if idx < 5:  # 前5个样本打印详细信息
                                    print(f"  警告: cost包含非数字字符串 (idx={idx})")
                                    print(f"     cost内容预览: {str(cost_value)[:100]}...")
                                cost_value = 0.0
                        except (ValueError, AttributeError) as e:
                            if idx < 5:
                                print(f"  警告: cost转换失败 (idx={idx}): {type(e).__name__}")
                            cost_value = 0.0
                    elif not isinstance(cost_value, (int, float)):
                        if idx < 5:
                            print(f"  警告: cost类型异常 (idx={idx}): {type(cost_value).__name__}")
                        cost_value = 0.0

                    total_cost += cost_value
                    successful_executions += 1

        # 计算指标
        num_correct = sum(1 for score in correctness_scores if score >= 0.9)  # Binary reward: 0.9 threshold for 1.0 scores
        val_accuracy = (num_correct / num_samples * 100) if num_samples > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
        avg_cost = total_cost / successful_executions if successful_executions > 0 else 0.0
        success_rate = (successful_executions / num_samples * 100) if num_samples > 0 else 0.0

        metrics = {
            "val_accuracy": val_accuracy,
            "val_num_correct": num_correct,
            "val_num_total": num_samples,
            "val_avg_correctness": avg_correctness,
            "val_avg_cost": avg_cost,
            "val_success_rate": success_rate
        }

        print(f"\n📊 验证集结果:")
        print(f"  准确率: {num_correct}/{num_samples} = {val_accuracy:.1f}%")
        print(f"  平均正确性: {avg_correctness:.2f}/10.0")
        print(f"  执行成功率: {success_rate:.1f}%")
        print(f"  平均成本: ${avg_cost:.4f}")
        print(f"{'='*60}\n")

        return metrics

    async def _wait_for_gpu_memory(self, min_free_gb: float = 45, max_wait_seconds: int = 300):
        """
        🛡️ OOM保护: 等待GPU有足够空闲显存

        Args:
            min_free_gb: 最小空闲显存(GB)
            max_wait_seconds: 最大等待时间(秒)
        """
        import gc as gc_module

        wait_interval = 10  # 每10秒检查一次
        total_waited = 0

        while total_waited < max_wait_seconds:
            try:
                # 使用PyTorch直接检查当前CUDA设备的显存 (正确映射CUDA_VISIBLE_DEVICES)
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                free_gb = total_memory - reserved_memory  # 可用显存

                if free_gb >= min_free_gb:
                    if total_waited > 0:
                        print(f"✅ GPU显存充足 ({free_gb:.1f}GB空闲)，继续训练")
                    return  # 显存足够，继续训练
                else:
                    print(f"⏳ GPU显存不足 ({free_gb:.1f}GB < {min_free_gb}GB)，等待{wait_interval}s... (已等待{total_waited}s)")
                    # 主动清理显存
                    torch.cuda.empty_cache()
                    gc_module.collect()
            except Exception as e:
                print(f"⚠️ 检查GPU显存失败: {e}")

            await asyncio.sleep(wait_interval)
            total_waited += wait_interval

        print(f"⚠️ 等待GPU显存超时({max_wait_seconds}s)，尝试继续训练...")

    async def train(self):
        """完整训练循环"""
        print("\n" + "=" * 60)
        print("🎓 开始GRPO训练")
        print("=" * 60)

        max_steps = self.config['max_steps']
        save_every = self.config.get('save_every', 50)
        log_every = self.config.get('log_every', 5)
        eval_every = self.config.get('eval_every', 10)  # 每10步验证一次
        val_samples = self.config.get('val_samples', 50)  # 验证集样本数

        for step in range(1, max_steps + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 Step {step}/{max_steps}")
            print(f"{'=' * 60}")

            # 🛡️ OOM保护: 检查GPU显存，如果不足则等待
            await self._wait_for_gpu_memory(min_free_gb=5, max_wait_seconds=300)

            # 训练步骤 (带OOM重试)
            import gc as gc_module
            max_retries = 3
            for retry in range(max_retries):
                try:
                    metrics = await self.train_step(step)
                    break  # 成功则跳出重试循环
                except torch.cuda.OutOfMemoryError as e:
                    print(f"⚠️ OOM错误 (尝试 {retry+1}/{max_retries}): {e}")
                    torch.cuda.empty_cache()
                    gc_module.collect()
                    if retry < max_retries - 1:
                        wait_time = 30 * (retry + 1)  # 30s, 60s, 90s
                        print(f"   等待 {wait_time}s 后重试...")
                        await asyncio.sleep(wait_time)
                        await self._wait_for_gpu_memory(min_free_gb=8, max_wait_seconds=180)
                    else:
                        print(f"❌ OOM重试{max_retries}次后仍失败，跳过此step")
                        metrics = {'loss': 0.0, 'skipped': True}
            else:
                # 所有重试都失败
                continue

            # 日志
            if step % log_every == 0:
                print(f"\n📊 Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

                # 记录到wandb
                wandb.log(metrics, step=step)

            # 🧪 验证集评估（每N步）
            if step % eval_every == 0:
                val_metrics = await self.evaluate_on_val_set(num_samples=val_samples)

                # 合并验证指标到训练指标
                metrics.update(val_metrics)

                # 记录验证指标到wandb
                wandb.log(val_metrics, step=step)

                print(f"✅ 验证集评估完成 (Step {step})")

            # 保存检查点
            if step % save_every == 0:
                self.save_checkpoint(step)

        print("\n" + "=" * 60)
        print("✅ 训练完成")
        print("=" * 60)

    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output_dir']) / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存LoRA权重
        self.model.save_pretrained(checkpoint_dir)

        # 💾 保存ExperienceBuffer
        self.experience_buffer.save(step=step)

        # 📊 打印ExperienceBuffer统计信息
        buffer_stats = self.experience_buffer.get_stats()
        print(f"\n📚 ExperienceBuffer统计:")
        for problem_type, stats in buffer_stats.items():
            if stats['count'] > 0:
                print(f"  {problem_type}: {stats['count']}样本, "
                      f"平均奖励={stats['avg_reward']:.2f}, "
                      f"最高奖励={stats['max_reward']:.2f}, "
                      f"平均正确性={stats['avg_correctness']:.2f}")

        print(f"💾 检查点已保存: {checkpoint_dir}")


async def main():
    """主函数"""
    trainer = GRPOTrainer(config_path="config/training.yaml")
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
