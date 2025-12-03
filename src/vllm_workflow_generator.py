#!/usr/bin/env python3
"""
vLLM工作流生成器 - 使用vLLM API进行并发推理（Fallback: 使用transformers）
"""
import asyncio
import torch
from openai import AsyncOpenAI
from typing import Dict, List, Optional, Tuple
import json
import ast
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

class VLLMWorkflowGenerator:
    """使用vLLM API生成优化的工作流（支持并发）

    支持两种模式：
    1. vLLM API模式（推荐）：通过AsyncOpenAI客户端调用vLLM服务
    2. Transformers模式（Fallback）：直接使用transformers库
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003/v1",
        api_key: str = "EMPTY",
        model_name: str = "/home/yijia/verl-agent/models/qwen/Qwen2___5-7B-Instruct",
        max_concurrent: int = 6,
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None,
        use_vllm_api: bool = False,  # 默认使用transformers模式
        device: str = "cuda:0"
    ):
        """
        Args:
            base_url: vLLM服务器地址
            api_key: API密钥（vLLM不需要真实密钥）
            model_name: 模型名称/路径
            max_concurrent: 最大并发请求数
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
            use_vllm_api: 是否使用vLLM API（False则使用transformers）
            device: 设备（transformers模式）
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.config = config or {}
        self.use_vllm_api = use_vllm_api
        self.device = device

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        if use_vllm_api:
            # vLLM API模式
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=300.0,  # 5分钟超时
                max_retries=2
            )
            self.semaphore = asyncio.Semaphore(max_concurrent)
            print(f"✅ 初始化vLLM工作流生成器（API模式）")
            print(f"  服务器: {base_url}")
            print(f"  最大并发: {max_concurrent}")
        else:
            # Transformers模式（直接使用已加载的模型）
            self.model = None  # 将由外部设置（避免重复加载）
            self.tokenizer = None
            # ⚠️ 关键修复：使用锁保护GPU访问（同一时间只允许一个推理）
            self._generation_lock = asyncio.Lock()
            print(f"✅ 初始化workflow生成器（Transformers模式）")
            print(f"  模型: {model_name}")
            print(f"  设备: {device}")
            print(f"  ⚠️  GPU推理将串行执行（避免CUDA冲突）")

    def _load_operator_descriptions(self, descriptions_path: Optional[str]) -> Dict:
        """加载AFlow算子描述"""
        if descriptions_path and Path(descriptions_path).exists():
            with open(descriptions_path, 'r') as f:
                return json.load(f)

        # 默认算子描述 - AFlow标准10个算子
        return {
            "Custom": {
                "description": "Generates anything based on customized input and instruction.",
                "interface": "custom(input: str, instruction: str) -> dict with key 'response'"
            },
            "AnswerGenerate": {
                "description": "Generates step-by-step reasoning with thought process and final answer.",
                "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'"
            },
            "CustomCodeGenerate": {
                "description": "Generates code based on customized input and instruction.",
                "interface": "custom_code_generate(problem: str, entry_point: str, instruction: str) -> dict with key 'code'"
            },
            "Programmer": {
                "description": "Automatically writes and executes Python code, returns execution result.",
                "interface": "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'"
            },
            "Test": {
                "description": "Tests code with test cases, reflects on errors and revises.",
                "interface": "test(problem: str, solution: str, entry_point: str, test_loop: int = 3) -> dict with keys 'result' and 'solution'"
            },
            "Format": {
                "description": "Extracts concise answer from verbose solution.",
                "interface": "format(problem: str, solution: str) -> dict with key 'solution'"
            },
            "Review": {
                "description": "Reviews solution correctness using critical thinking.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' (bool) and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            },
            "MdEnsemble": {
                "description": "Majority voting ensemble - shuffles and votes multiple times (more robust than ScEnsemble).",
                "interface": "md_ensemble(solutions: List[str], problem: str) -> dict with key 'solution'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """构建生成提示词 - 开放式DSL组合格式

        让模型自由组合operators，而不是从预设选项中选择
        """
        prompt = f"""Design a workflow to solve this problem. Output a single-line DSL expression.

Available Operators:
- Custom: General reasoning, text generation. (input, instruction) -> response
- Programmer: Auto-execute Python code for calculations. (problem, analysis) -> code, output
- ScEnsemble: Vote on multiple solutions. (solutions[], problem) -> response
- Review: Check if solution is correct. (problem, solution) -> review_result, feedback
- Revise: Fix solution based on feedback. (problem, solution, feedback) -> solution

DSL Syntax:
- Single operator: Custom
- Chain (sequential): Custom -> Programmer -> Custom
- Parallel then merge: [Custom, Custom, Custom] -> ScEnsemble
- Conditional: Review ? Revise : done
- Loop: (Custom -> Review -> Revise) * 3

Examples:
- Simple QA: Custom
- Math calculation: Programmer
- Complex math: Programmer -> Custom
- Need multiple attempts: [Custom, Custom, Custom] -> ScEnsemble
- Self-correction: Custom -> Review -> Revise
- Code generation: Programmer -> Review ? Revise : done

Problem ({problem_type}): {problem}

DSL:"""
        return prompt

    async def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        生成单个工作流（异步）

        Returns:
            {
                "workflow_code": "Python代码",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """
        if self.use_vllm_api:
            return await self._generate_with_vllm_api(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )
        else:
            return await self._generate_with_transformers(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )

    async def _generate_with_vllm_api(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """使用vLLM API生成"""
        async with self.semaphore:  # 控制并发数
            try:
                # 构建提示词
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

                # 调用vLLM API
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a workflow generation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.get('top_p', 0.95),
                )

                # 提取生成的代码
                generated_text = response.choices[0].message.content
                workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "tokens": response.usage.total_tokens if response.usage else 0,
                        "model": self.model_name
                    }
                }

            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {}
                }

    async def _generate_with_transformers(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_new_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """使用transformers生成（使用锁保护GPU访问）"""
        # ⚠️ 关键：使用锁确保同一时间只有一个推理在执行
        async with self._generation_lock:
            loop = asyncio.get_event_loop()

            def _sync_generate():
                """同步生成函数（在线程池中执行）"""
                # 构建提示词
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.config.get('top_p', 0.95),
                        top_k=self.config.get('top_k', 50),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # 解码
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                return generated_text

            try:
                # 在默认executor中运行（CPU密集型操作）
                generated_text = await loop.run_in_executor(None, _sync_generate)

                # 解析输出
                workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problem,
                        "problem_type": problem_type,
                        "temperature": temperature
                    }
                }
            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {}
                }

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str]]:
        """解析生成的文本，提取并验证工作流代码

        支持开放式DSL格式：
        - 单一算子: Custom
        - 链式: Custom -> Programmer -> Custom
        - 并行: [Custom, Custom, Custom] -> ScEnsemble
        - 条件: Review ? Revise : done
        """
        import re

        # 🔧 首先尝试直接解析DSL（模型输出的第一行）
        text_clean = generated_text.strip()
        first_line = text_clean.split('\n')[0].strip()

        # 检查是否包含operator名称
        valid_ops = ['Custom', 'Programmer', 'ScEnsemble', 'Review', 'Revise', 'AnswerGenerate', 'CustomCodeGenerate', 'Test', 'Format', 'MdEnsemble']
        if any(op in first_line for op in valid_ops):
            # 清理DSL（移除可能的前缀如"DSL: "）
            dsl_text = re.sub(r'^[^A-Za-z\[]*', '', first_line)
            dsl_text = re.sub(r'[^A-Za-z\]>\-,\s\?\*\(\):]*$', '', dsl_text).strip()
            if dsl_text:
                print(f"  📝 检测到开放式DSL: {dsl_text}")
                generator = WorkflowCodeGenerator(problem_type)
                code, is_valid, error = generator.generate(dsl_text)
                if is_valid:
                    print(f"  ✅ DSL成功转换为代码")
                    return code, True, None
                else:
                    print(f"  ⚠️ DSL解析失败，尝试其他方法: {error}")

        # 🔧 尝试提取DSL格式 <workflow>...</workflow>
        workflow_match = re.search(r'<workflow>\s*([\s\S]*?)\s*(?:</workflow>|$)', generated_text)
        if workflow_match:
            dsl_text = workflow_match.group(1).strip()
            print(f"  📝 检测到XML DSL格式: {dsl_text}")
            generator = WorkflowCodeGenerator(problem_type)
            code, is_valid, error = generator.generate(dsl_text)
            if is_valid:
                print(f"  ✅ DSL成功转换为代码")
                return code, True, None
            else:
                print(f"  ⚠️ DSL解析失败: {error}")

        # 🔧 尝试逐行寻找有效DSL
        for line in text_clean.split('\n'):
            line = line.strip()
            if line and any(op in line for op in valid_ops):
                line = re.sub(r'^[^A-Za-z\[]*', '', line)
                line = re.sub(r'[^A-Za-z\]>\-,\s\?\*\(\):]*$', '', line)
                if line and '->' in line or '[' in line or line in valid_ops:
                    print(f"  📝 尝试行级DSL解析: {line}")
                    generator = WorkflowCodeGenerator(problem_type)
                    code, is_valid, error = generator.generate(line)
                    if is_valid:
                        print(f"  ✅ 行级DSL成功")
                        return code, True, None

        # 🔧 尝试提取旧XML格式 <graph>...</graph>
        graph_code, prompt_code = self._extract_xml_workflow(generated_text)
        if graph_code:
            print(f"  📝 检测到XML格式工作流")
            code = graph_code.strip()
            if prompt_code:
                prompt_custom_code = prompt_code.strip()
            else:
                prompt_custom_code = self._get_default_prompt_custom(problem_type)
        else:
            # 回退到默认workflow
            print(f"  ⚠️ 未检测到有效格式，使用默认workflow")
            return self._get_default_workflow(problem_type), False, "No valid format detected"

        if "TASK_PROMPT" not in code and prompt_custom_code:
            class_match = re.search(r'^class Workflow', code, re.MULTILINE)
            if class_match:
                code = prompt_custom_code + "\n\n" + code
            else:
                code = prompt_custom_code + "\n" + code

        code = self._validate_and_fix_workflow(code, problem_type)

        try:
            ast.parse(code)
            return code, True, None
        except SyntaxError as e:
            return self._get_default_workflow(problem_type), False, f"Syntax error: {str(e)}"

    def _extract_xml_workflow(self, text: str) -> Tuple[str, str]:
        """从XML格式提取graph和prompt代码

        Returns:
            (graph_code, prompt_code) - 如果未找到XML格式则返回空字符串
        """
        import re

        graph_code = ""
        prompt_code = ""

        # 尝试提取 <graph>...</graph>
        graph_match = re.search(r'<graph>\s*([\s\S]*?)\s*</graph>', text)
        if graph_match:
            graph_code = graph_match.group(1).strip()

        # 尝试提取 <prompt>...</prompt>
        prompt_match = re.search(r'<prompt>\s*([\s\S]*?)\s*</prompt>', text)
        if prompt_match:
            prompt_code = prompt_match.group(1).strip()

        return graph_code, prompt_code

    def _parse_legacy_format(self, generated_text: str, problem_type: str) -> Tuple[str, str]:
        """解析旧格式（Python代码块或直接class定义）"""
        import re

        # 提取代码块
        code_start = generated_text.find("```python")
        if code_start == -1:
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                return "", ""
            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)
            code = generated_text[code_start:code_end] if code_end != -1 else generated_text[code_start:]

        code = code.strip()

        # 解析并提取prompt_custom部分
        prompt_custom_start = code.find("# === PROMPT_CUSTOM START ===")
        prompt_custom_end = code.find("# === PROMPT_CUSTOM END ===")

        prompt_custom_code = ""
        if prompt_custom_start != -1 and prompt_custom_end != -1:
            end_line_end = code.find("\n", prompt_custom_end)
            if end_line_end == -1:
                end_line_end = len(code)
            prompt_custom_code = code[prompt_custom_start:end_line_end + 1]
            # 移除原位置的prompt_custom
            code = code[:prompt_custom_start] + code[end_line_end + 1:]
        else:
            # 尝试检测TASK_PROMPT变量定义
            task_prompt_match = re.search(
                r'^(TASK_PROMPT\s*=\s*(?:"""[\s\S]*?"""|\'\'\' [\s\S]*?\'\'\'))',
                code,
                re.MULTILINE
            )
            if task_prompt_match:
                prompt_custom_code = task_prompt_match.group(1)
            else:
                prompt_custom_code = self._get_default_prompt_custom(problem_type)

        return code.strip(), prompt_custom_code

    def _get_default_prompt_custom(self, problem_type: str) -> str:
        """获取默认的TASK_PROMPT"""
        if problem_type == "math":
            return '''TASK_PROMPT = """Solve this mathematical problem step by step.
Show your reasoning clearly and provide the final numerical answer.
Format: First explain your approach, then show calculations, finally state the answer."""'''
        elif problem_type == "code":
            return '''TASK_PROMPT = """Write a Python function to solve this problem.
Requirements:
1. The function should be efficient and handle edge cases
2. Include proper input validation
3. Return the correct type as specified"""'''
        else:
            return '''TASK_PROMPT = """Solve this problem carefully.
Provide a clear, structured answer with reasoning."""'''

    def _validate_and_fix_workflow(self, code: str, problem_type: str) -> str:
        """验证并自动修复workflow中缺失的operator初始化

        Args:
            code: 生成的workflow代码
            problem_type: 问题类型

        Returns:
            修复后的代码
        """
        import re

        # 1. 提取__init__中已初始化的operators
        initialized_ops = set()
        init_section = re.search(r'def __init__\([^)]+\):[\s\S]+?(?=\n    async def|\n    def|$)', code)
        if init_section:
            init_code = init_section.group(0)
            # 匹配 self.xxx = operator.XXX(self.llm)
            init_patterns = re.findall(r'self\.(\w+)\s*=\s*operator\.(\w+)\(', init_code)
            for attr_name, op_name in init_patterns:
                initialized_ops.add(attr_name)

        # 2. 提取__call__中使用的operators
        used_ops = set()
        call_section = re.search(r'async def __call__\([^)]+\):[\s\S]+', code)
        if call_section:
            call_code = call_section.group(0)
            # 匹配 await self.xxx(...)
            used_patterns = re.findall(r'await self\.(\w+)\(', call_code)
            for op_name in used_patterns:
                used_ops.add(op_name)

        # 3. 找出缺失的operators
        missing_ops = used_ops - initialized_ops

        if missing_ops:
            print(f"\n⚠️  检测到缺失的operator初始化: {missing_ops}")
            print(f"   已初始化: {initialized_ops}")
            print(f"   已使用: {used_ops}")

            # 4. 自动添加缺失的初始化代码
            # 找到 self.llm = create_llm_instance(...) 的位置
            llm_init_match = re.search(r'(\s+)(self\.llm = create_llm_instance\([^)]+\))', code)
            if llm_init_match:
                indent = llm_init_match.group(1)
                llm_init_line = llm_init_match.group(2)

                # 构建缺失的初始化代码
                missing_inits = []
                for op_name in sorted(missing_ops):
                    # 推断operator类名（首字母大写+驼峰命名）
                    # answer_generate -> AnswerGenerate
                    # review -> Review
                    op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))

                    # 检查是否是有效的operator（AFlow标准10个算子）
                    valid_operators = [
                        'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
                        'Programmer', 'Test', 'Format',
                        'Review', 'Revise', 'ScEnsemble', 'MdEnsemble'
                    ]
                    if op_class_name in valid_operators:
                        missing_inits.append(f"{indent}self.{op_name} = operator.{op_class_name}(self.llm)")

                if missing_inits:
                    # 在 self.llm = ... 之后插入
                    insert_code = '\n' + '\n'.join(missing_inits)
                    code = code.replace(llm_init_line, llm_init_line + insert_code)
                    print(f"✅ 自动添加了 {len(missing_inits)} 个缺失的operator初始化")

        return code

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流 - 包含TASK_PROMPT"""
        # 根据问题类型选择合适的默认prompt
        if problem_type == "math":
            task_prompt = '''"""Solve this mathematical problem step by step.
Show your complete reasoning process:
1. Identify what the problem is asking
2. List known information and variables
3. Apply relevant formulas or methods
4. Perform calculations carefully
5. State the final numerical answer clearly

IMPORTANT: Always verify your answer before providing it."""'''
        elif problem_type == "code":
            task_prompt = '''"""Write a Python function to solve this problem.
Requirements:
1. Handle all edge cases properly
2. Use efficient algorithms
3. Include proper input validation
4. Return the correct type as specified
5. Add brief comments for complex logic"""'''
        else:
            task_prompt = '''"""Solve this problem carefully and provide a clear answer.
Show your reasoning step by step."""'''

        return f"""# === PROMPT_CUSTOM START ===
TASK_PROMPT = {task_prompt}
# === PROMPT_CUSTOM END ===

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: str = "solve"):
        # entry_point used for code problems with Test operator
        solution = await self.custom(input=problem, instruction=TASK_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    async def generate_workflows_batch(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        批量并发生成工作流（优化版：使用GPU batch推理）

        Args:
            problems: 问题列表
            problem_types: 问题类型列表
            temperatures: 温度列表
            custom_prompts: 自定义提示词列表

        Returns:
            结果列表
        """
        if self.use_vllm_api:
            # vLLM API模式：并发调用
            tasks = []
            for i in range(len(problems)):
                task = self.generate_workflow(
                    problem=problems[i],
                    problem_type=problem_types[i],
                    temperature=temperatures[i],
                    custom_prompt=custom_prompts[i] if custom_prompts else None
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "workflow_code": "",
                        "valid": False,
                        "error": str(result),
                        "metadata": {}
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # Transformers模式：使用GPU batch推理（关键优化！）
            return await self._batch_generate_with_transformers(
                problems, problem_types, temperatures, custom_prompts
            )

    async def _batch_generate_with_transformers(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]]
    ) -> List[Dict]:
        """使用transformers批量生成（GPU batch推理，支持分批以降低显存）"""
        loop = asyncio.get_event_loop()

        # 🔧 显存优化：分批生成，每批最多8个序列
        MAX_BATCH_SIZE = 8  # 每批最多8个，降低显存峰值

        def _sync_batch_generate(batch_prompts, batch_temp):
            """同步批量生成函数（单批）"""
            # 批量tokenize（关键：padding对齐）
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,  # 对齐到最长序列
                truncation=True,
                max_length=3072
            ).to(self.device)

            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 2048),
                    temperature=batch_temp,
                    top_p=self.config.get('top_p', 0.95),
                    top_k=self.config.get('top_k', 50),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 批量解码
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # 🔧 显存优化：及时清理
            del inputs, outputs
            torch.cuda.empty_cache()

            return generated_texts

        try:
            # 构建所有prompts
            all_prompts = []
            for i in range(len(problems)):
                if custom_prompts and custom_prompts[i]:
                    prompt = custom_prompts[i]
                else:
                    prompt = self._build_generation_prompt(problems[i], problem_types[i])
                all_prompts.append(prompt)

            # 🔧 分批处理以降低显存峰值
            all_generated_texts = []
            for batch_start in range(0, len(all_prompts), MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, len(all_prompts))
                batch_prompts = all_prompts[batch_start:batch_end]
                batch_temp = temperatures[batch_start]  # 假设同批temperature相同

                print(f"  🔧 生成批次 {batch_start//MAX_BATCH_SIZE + 1}/{(len(all_prompts)-1)//MAX_BATCH_SIZE + 1} ({len(batch_prompts)}个序列)")

                # 在线程池执行单批推理
                batch_texts = await loop.run_in_executor(
                    None, _sync_batch_generate, batch_prompts, batch_temp
                )
                all_generated_texts.extend(batch_texts)

            # 解析所有结果
            results = []
            for i, generated_text in enumerate(all_generated_texts):
                workflow_code, is_valid, error = self._parse_workflow_code(
                    generated_text, problem_types[i]
                )
                results.append({
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problems[i],
                        "problem_type": problem_types[i],
                        "temperature": temperatures[i]
                    }
                })

            return results

        except Exception as e:
            # 出错时返回空结果
            return [{
                "workflow_code": "",
                "valid": False,
                "error": str(e),
                "metadata": {}
            } for _ in problems]


# ============================================================================
# DSL解析器和代码生成器 - 极简符号式工作流
# ============================================================================

class WorkflowDSLParser:
    """解析极简DSL符号格式

    支持的格式:
    - 顺序: "Programmer -> Custom"
    - 并行: "[Custom, Custom, Custom] -> ScEnsemble"
    - 混合: "Programmer -> [Custom, Custom] -> ScEnsemble"
    """

    # 有效的operator列表
    VALID_OPERATORS = {
        'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
        'Programmer', 'Test', 'Format',
        'Review', 'Revise', 'ScEnsemble', 'MdEnsemble'
    }

    # Operator输入输出类型定义（用于自动推断参数）
    OPERATOR_SIGNATURES = {
        'Custom': {
            'inputs': ['input', 'instruction'],
            'output': 'response',
            'output_type': 'str'
        },
        'CustomCodeGenerate': {
            'inputs': ['problem', 'entry_point', 'instruction'],
            'output': 'response',
            'output_type': 'str'
        },
        'Programmer': {
            'inputs': ['problem', 'analysis'],
            'output': 'output',  # 也有 'code'
            'output_type': 'str'
        },
        'ScEnsemble': {
            'inputs': ['solutions', 'problem'],
            'output': 'response',
            'output_type': 'str',
            'accepts_list': True  # 接受列表输入
        },
        'MdEnsemble': {
            'inputs': ['solutions', 'problem'],
            'output': 'solution',
            'output_type': 'str',
            'accepts_list': True
        },
        'Test': {
            'inputs': ['problem', 'solution', 'entry_point'],
            'output': 'solution',
            'output_type': 'str',
            'has_result': True  # 返回 result (bool) 和 solution
        },
        'Review': {
            'inputs': ['problem', 'solution'],
            'output': 'feedback',
            'output_type': 'str',
            'has_result': True  # 返回 review_result (bool) 和 feedback
        },
        'Revise': {
            'inputs': ['problem', 'solution', 'feedback'],
            'output': 'solution',
            'output_type': 'str'
        },
        'Format': {
            'inputs': ['problem', 'solution'],
            'output': 'solution',
            'output_type': 'str'
        },
        'AnswerGenerate': {
            'inputs': ['input'],
            'output': 'answer',  # 也有 'thought'
            'output_type': 'str'
        }
    }

    def __init__(self):
        pass

    def parse(self, dsl_text: str) -> dict:
        """解析DSL文本

        Args:
            dsl_text: DSL文本，如 "Programmer -> Custom" 或 "[Custom, Custom] -> ScEnsemble"

        Returns:
            {
                'valid': bool,
                'error': str or None,
                'stages': [  # 执行阶段列表
                    {
                        'type': 'single' | 'parallel',
                        'operators': ['Programmer'] | ['Custom', 'Custom', 'Custom'],
                    },
                    ...
                ]
            }
        """
        import re

        # 清理输入
        dsl_text = dsl_text.strip()

        # 移除可能的标签
        dsl_text = re.sub(r'</?workflow>', '', dsl_text).strip()

        if not dsl_text:
            return {'valid': False, 'error': '空的DSL', 'stages': []}

        # 🔧 预处理：处理条件语法 "Review ? Revise : done" -> "Review -> Revise"
        # 简化处理：取条件为真的分支
        cond_match = re.search(r'(\w+)\s*\?\s*(\w+)\s*:\s*(\w+)', dsl_text)
        if cond_match:
            condition_op, true_branch, false_branch = cond_match.groups()
            # 如果false_branch是done，取true_branch；否则都执行
            if false_branch.lower() == 'done':
                replacement = f"{condition_op} -> {true_branch}"
            else:
                replacement = f"{condition_op} -> {true_branch}"
            dsl_text = re.sub(r'\w+\s*\?\s*\w+\s*:\s*\w+', replacement, dsl_text)

        # 🔧 预处理：移除终止符 "-> done"
        dsl_text = re.sub(r'->\s*done\s*$', '', dsl_text, flags=re.IGNORECASE).strip()

        stages = []

        # 按 -> 分割
        parts = [p.strip() for p in dsl_text.split('->')]

        for part in parts:
            if not part:
                continue

            # 🔧 跳过done关键字
            if part.lower() == 'done':
                continue

            # 检查是否是并行格式 [Op1, Op2, ...]
            if part.startswith('[') and part.endswith(']'):
                # 并行阶段
                inner = part[1:-1].strip()
                operators = [op.strip() for op in inner.split(',')]

                # 验证每个operator
                for op in operators:
                    if op not in self.VALID_OPERATORS:
                        return {'valid': False, 'error': f'无效的operator: {op}', 'stages': []}

                stages.append({
                    'type': 'parallel',
                    'operators': operators
                })
            else:
                # 单个operator
                op = part.strip()
                if op not in self.VALID_OPERATORS:
                    return {'valid': False, 'error': f'无效的operator: {op}', 'stages': []}

                stages.append({
                    'type': 'single',
                    'operators': [op]
                })

        if not stages:
            return {'valid': False, 'error': '未找到有效的operator', 'stages': []}

        return {'valid': True, 'error': None, 'stages': stages}


class WorkflowCodeGenerator:
    """将解析后的DSL转换为可执行的Python Workflow代码"""

    def __init__(self, problem_type: str = 'math'):
        self.problem_type = problem_type
        self.parser = WorkflowDSLParser()

    def generate(self, dsl_text: str) -> Tuple[str, bool, Optional[str]]:
        """从DSL生成完整的Workflow代码

        Args:
            dsl_text: DSL文本

        Returns:
            (code, is_valid, error)
        """
        # 解析DSL
        parsed = self.parser.parse(dsl_text)

        if not parsed['valid']:
            return self._get_default_code(), False, parsed['error']

        stages = parsed['stages']

        # 收集所有需要的operators
        all_operators = set()
        for stage in stages:
            all_operators.update(stage['operators'])

        # 生成代码
        code = self._generate_workflow_code(stages, all_operators)

        # 验证语法
        try:
            ast.parse(code)
            return code, True, None
        except SyntaxError as e:
            return self._get_default_code(), False, f"语法错误: {e}"

    def _generate_workflow_code(self, stages: List[dict], all_operators: set) -> str:
        """生成Workflow类代码"""

        # 生成__init__中的operator初始化
        init_lines = []
        for op in sorted(all_operators):
            attr_name = self._to_snake_case(op)
            init_lines.append(f"        self.{attr_name} = operator.{op}(self.llm)")

        # 生成__call__中的执行逻辑
        call_lines = self._generate_call_body(stages)

        # 组装完整代码
        code = f'''class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
{chr(10).join(init_lines)}

    async def __call__(self, problem: str, entry_point: str = None):
        """
        Auto-generated workflow from DSL
        """
{chr(10).join(call_lines)}
'''
        return code

    def _generate_call_body(self, stages: List[dict]) -> List[str]:
        """生成__call__方法体"""
        lines = []
        prev_output = None  # 上一阶段的输出变量名
        prev_is_list = False  # 上一阶段是否是并行（输出列表）

        for i, stage in enumerate(stages):
            is_last = (i == len(stages) - 1)

            if stage['type'] == 'parallel':
                # 并行执行多个相同operator
                ops = stage['operators']
                op = ops[0]  # 假设并行时都是同一类型
                attr_name = self._to_snake_case(op)
                sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op, {})

                # 生成并行调用
                lines.append(f"        # 并行执行 {len(ops)} 个 {op}")
                lines.append(f"        import asyncio")

                # 构建参数
                if prev_output:
                    input_param = prev_output
                else:
                    input_param = 'problem'

                # 生成并行任务
                tasks = []
                for j in range(len(ops)):
                    param_str = self._build_params(op, input_param, is_first=(i == 0))
                    tasks.append(f"self.{attr_name}({param_str})")

                lines.append(f"        tasks = [{', '.join(tasks)}]")
                lines.append(f"        results_{i} = await asyncio.gather(*tasks)")
                lines.append(f"        solutions_{i} = [r['{sig.get('output', 'response')}'] for r in results_{i}]")

                prev_output = f"solutions_{i}"
                prev_is_list = True

            else:
                # 单个operator
                op = stage['operators'][0]
                attr_name = self._to_snake_case(op)
                sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op, {})

                # 构建参数
                if prev_is_list and sig.get('accepts_list'):
                    # 前一阶段是列表，当前operator接受列表（如ScEnsemble）
                    param_str = f"solutions={prev_output}, problem=problem"
                elif prev_output:
                    param_str = self._build_params(op, prev_output, is_first=False)
                else:
                    param_str = self._build_params(op, 'problem', is_first=True)

                lines.append(f"        result_{i} = await self.{attr_name}({param_str})")

                output_key = sig.get('output', 'response')
                prev_output = f"result_{i}['{output_key}']"
                prev_is_list = False

        # 最后返回
        lines.append(f"        return {prev_output}, self.llm.get_usage_summary()['total_cost']")

        return lines

    def _build_params(self, op: str, input_var: str, is_first: bool) -> str:
        """构建operator调用参数"""
        sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op, {})

        if op == 'Custom':
            return f"input={input_var}, instruction=''"
        elif op == 'CustomCodeGenerate':
            if is_first:
                return f"problem={input_var}, entry_point=entry_point or 'solve', instruction=''"
            else:
                return f"problem=problem, entry_point=entry_point or 'solve', instruction=''"
        elif op == 'Programmer':
            if is_first:
                return f"problem={input_var}, analysis='None'"
            else:
                return f"problem=problem, analysis={input_var}"
        elif op == 'Test':
            return f"problem=problem, solution={input_var}, entry_point=entry_point or 'solve'"
        elif op == 'Review':
            return f"problem=problem, solution={input_var}"
        elif op == 'Revise':
            return f"problem=problem, solution={input_var}, feedback=''"
        elif op == 'Format':
            return f"problem=problem, solution={input_var}"
        elif op == 'AnswerGenerate':
            return f"input={input_var}"
        elif op in ('ScEnsemble', 'MdEnsemble'):
            return f"solutions={input_var}, problem=problem"
        else:
            return f"input={input_var}, instruction=''"

    def _to_snake_case(self, name: str) -> str:
        """驼峰转下划线：CustomCodeGenerate -> custom_code_generate"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _get_default_code(self) -> str:
        """默认的简单Workflow"""
        return '''class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: str = None):
        result = await self.custom(input=problem, instruction="")
        return result['response'], self.llm.get_usage_summary()['total_cost']
'''
