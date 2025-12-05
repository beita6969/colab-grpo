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
        """构建生成提示词 - 基于业界最佳实践优化

        关键优化:
        1. XML标签分隔各部分，防止混淆
        2. 明确禁止约束，避免emoji/LaTeX
        3. 负面示例展示常见错误
        4. 单算子循环示例解决括号问题
        5. 结尾用<output>标签避免被误解为数学答案
        """
        prompt = f"""<task>
Generate a DSL expression for the workflow to solve this problem.
</task>

<operators>
Custom: General reasoning, text generation
Programmer: Write and execute Python code for calculations
ScEnsemble: Vote on multiple solutions to select best one
Review: Check if solution is correct, return feedback
Revise: Fix solution based on feedback
</operators>

<syntax>
Single: Custom
Chain: Custom -> Programmer -> Custom
Parallel: [Custom, Custom, Custom] -> ScEnsemble
Conditional: Review ? Revise : done
Loop (single operator): (Revise) * 3
Loop (chain): (Custom -> Review -> Revise) * 3
</syntax>

<examples>
Simple QA: Custom
Math calculation: Programmer
Complex reasoning: Programmer -> Custom
Multiple attempts: [Custom, Custom, Custom] -> ScEnsemble
Self-correction: Custom -> Review ? Revise : done
Iterative fix: Custom -> (Review -> Revise) * 2
</examples>

<constraints>
- Output ONLY the DSL expression, nothing else
- Use ONLY operators listed above: Custom, Programmer, ScEnsemble, Review, Revise
- NO emojis or special Unicode characters
- NO LaTeX formatting (no \\boxed{{}}, no $$, no \\text{{}})
- NO explanations before or after the DSL
- NO phrases like "The answer is" or "The workflow is"
- Single operator loop MUST use parentheses: (Custom) * 3, NOT Custom * 3
</constraints>

<wrong_outputs>
WRONG: chart_with_upwards_trend -> Review (emoji text not allowed)
WRONG: \\boxed{{Programmer -> Custom}} (LaTeX not allowed)
WRONG: Revise * 3 (missing parentheses, must be (Revise) * 3)
WRONG: The workflow is: Custom -> Review (no explanation allowed)
WRONG: Based on the problem, I suggest Custom (no preamble allowed)
</wrong_outputs>

<problem type="{problem_type}">
{problem}
</problem>

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
                # P21: 解包4元组，包含dsl_info
                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "tokens": response.usage.total_tokens if response.usage else 0,
                        "model": self.model_name,
                        "dsl_info": dsl_info  # P21: 添加DSL质量信息
                    }
                }

            except Exception as e:
                # P21: 异常情况也包含dsl_info
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {
                        "dsl_info": self._analyze_dsl_quality("", is_fallback=True)
                    }
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

                # 解析输出 - P21: 解包4元组，包含dsl_info
                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problem,
                        "problem_type": problem_type,
                        "temperature": temperature,
                        "dsl_info": dsl_info  # P21: 添加DSL质量信息
                    }
                }
            except Exception as e:
                # P21: 异常情况也包含dsl_info
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {
                        "dsl_info": self._analyze_dsl_quality("", is_fallback=True)
                    }
                }

    def _analyze_dsl_quality(self, dsl_text: str, is_fallback: bool = False) -> Dict:
        """
        P21修复: 分析DSL质量用于条件激活奖励

        基于Graph-R1论文的格式奖励设计：
        - is_fallback: 是否回退到默认workflow
        - num_operators: 总操作符数量
        - unique_operators: 唯一操作符集合
        - has_chain: 是否有链式结构 (->)
        - has_loop: 是否有循环结构 (*)
        - has_conditional: 是否有条件分支 (?)
        - has_parallel: 是否有并行结构 ([])
        - dsl_text: 原始DSL文本

        Returns:
            dsl_info dict with quality metrics
        """
        import re

        valid_ops = ['Custom', 'Programmer', 'ScEnsemble', 'Review', 'Revise',
                     'AnswerGenerate', 'CustomCodeGenerate', 'Test', 'Format', 'MdEnsemble']

        # 初始化默认值（fallback情况）
        dsl_info = {
            'is_fallback': is_fallback,
            'num_operators': 1 if is_fallback else 0,
            'unique_operators': ['Custom'] if is_fallback else [],
            'has_chain': False,
            'has_loop': False,
            'has_conditional': False,
            'has_parallel': False,
            'dsl_text': dsl_text if dsl_text else 'Custom (default fallback)',
            'dsl_quality_score': 0.0  # 将在reward_computer中计算
        }

        if is_fallback or not dsl_text:
            return dsl_info

        # 提取所有operator名称
        found_operators = []
        for op in valid_ops:
            # 使用word boundary匹配，避免部分匹配
            matches = re.findall(rf'\b{op}\b', dsl_text)
            found_operators.extend(matches)

        dsl_info['num_operators'] = len(found_operators)
        dsl_info['unique_operators'] = list(set(found_operators))

        # 检测结构特征
        dsl_info['has_chain'] = '->' in dsl_text
        dsl_info['has_loop'] = '*' in dsl_text
        dsl_info['has_conditional'] = '?' in dsl_text and ':' in dsl_text
        dsl_info['has_parallel'] = '[' in dsl_text and ']' in dsl_text

        return dsl_info

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str], Dict]:
        """解析生成的文本，提取并验证工作流代码

        P21修复: 返回4元组，包含dsl_info用于条件激活奖励

        支持开放式DSL格式：
        - 单一算子: Custom
        - 链式: Custom -> Programmer -> Custom
        - 并行: [Custom, Custom, Custom] -> ScEnsemble
        - 条件: Review ? Revise : done

        Returns:
            (workflow_code, is_valid, error, dsl_info)
        """
        import re

        # 🔧 预处理：清理XML标签和常见噪声
        text_clean = generated_text.strip()
        # 移除 </output> 等XML结束标签
        text_clean = re.sub(r'</?(output|dsl|workflow|answer)>', '', text_clean, flags=re.IGNORECASE)
        # 移除 ```dsl 等代码块标记
        text_clean = re.sub(r'```\w*', '', text_clean)
        text_clean = text_clean.strip()

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
                    dsl_info = self._analyze_dsl_quality(dsl_text, is_fallback=False)
                    return code, True, None, dsl_info
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
                dsl_info = self._analyze_dsl_quality(dsl_text, is_fallback=False)
                return code, True, None, dsl_info
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
                        dsl_info = self._analyze_dsl_quality(line, is_fallback=False)
                        return code, True, None, dsl_info

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
            # 回退到默认workflow - P21: 标记为fallback
            print(f"  ⚠️ 未检测到有效格式，使用默认workflow")
            dsl_info = self._analyze_dsl_quality("", is_fallback=True)
            return self._get_default_workflow(problem_type), False, "No valid format detected", dsl_info

        if "TASK_PROMPT" not in code and prompt_custom_code:
            class_match = re.search(r'^class Workflow', code, re.MULTILINE)
            if class_match:
                code = prompt_custom_code + "\n\n" + code
            else:
                code = prompt_custom_code + "\n" + code

        code = self._validate_and_fix_workflow(code, problem_type)

        try:
            ast.parse(code)
            # P21: XML格式的workflow，不是DSL格式，但仍然是有效解析
            dsl_info = self._analyze_dsl_quality("XML-format workflow", is_fallback=False)
            dsl_info['is_xml_format'] = True
            return code, True, None, dsl_info
        except SyntaxError as e:
            # P21: 语法错误回退到默认workflow
            dsl_info = self._analyze_dsl_quality("", is_fallback=True)
            return self._get_default_workflow(problem_type), False, f"Syntax error: {str(e)}", dsl_info

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

            # 解析所有结果 - P21修复: 解包4元组，包含dsl_info
            # P23修复: 添加raw_text存储原始模型输出，用于正确的训练目标
            results = []
            for i, generated_text in enumerate(all_generated_texts):
                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(
                    generated_text, problem_types[i]
                )
                results.append({
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problems[i],
                        "problem_type": problem_types[i],
                        "temperature": temperatures[i],
                        "dsl_info": dsl_info,  # P21: 添加DSL质量信息
                        "raw_text": generated_text  # P23: 原始模型输出（用于训练）
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

    # 🔧 P19修复: 常见operator名称幻觉的纠正映射
    # 模型可能生成的错误名称 -> 正确的operator名称
    OPERATOR_CORRECTIONS = {
        # 常见幻觉
        'Giver': 'Custom',           # "给出答案" 概念映射到通用推理
        'Generator': 'Custom',       # 生成器 -> 通用推理
        'Solver': 'Custom',          # 求解器 -> 通用推理
        'Thinker': 'Custom',         # 思考者 -> 通用推理
        'Reasoner': 'Custom',        # 推理者 -> 通用推理
        'Answer': 'Custom',          # 答案 -> 通用推理
        'Coder': 'Programmer',       # 编码器 -> 程序员
        'Code': 'Programmer',        # 代码 -> 程序员
        'Python': 'Programmer',      # Python -> 程序员
        'Execute': 'Programmer',     # 执行 -> 程序员
        'Calc': 'Programmer',        # 计算 -> 程序员
        'Calculator': 'Programmer',  # 计算器 -> 程序员
        'Check': 'Review',           # 检查 -> 审查
        'Verify': 'Review',          # 验证 -> 审查
        'Validate': 'Review',        # 校验 -> 审查
        'Fix': 'Revise',             # 修复 -> 修订
        'Correct': 'Revise',         # 纠正 -> 修订
        'Improve': 'Revise',         # 改进 -> 修订
        'Vote': 'ScEnsemble',        # 投票 -> 集成
        'Ensemble': 'ScEnsemble',    # 集成 -> ScEnsemble
        'Select': 'ScEnsemble',      # 选择 -> 集成
        # 截断/损坏的名称前缀映射
        'Cust': 'Custom',
        'Prog': 'Programmer',
        'Rev': 'Review',             # Rev可能是Review或Revise，默认Review
        'Sc': 'ScEnsemble',
        # 大小写变体
        'custom': 'Custom',
        'programmer': 'Programmer',
        'review': 'Review',
        'revise': 'Revise',
        'scensemble': 'ScEnsemble',
        'test': 'Test',
        'format': 'Format',
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

    def _correct_operator_name(self, op_name: str) -> str:
        """
        🔧 P19修复: 纠正无效的operator名称

        策略:
        1. 如果是有效operator，直接返回
        2. 检查是否在纠正映射中
        3. 尝试前缀匹配（处理截断的名称）
        4. 清理特殊字符后再次检查
        5. 最后回退到Custom

        Args:
            op_name: 原始operator名称

        Returns:
            纠正后的有效operator名称
        """
        # 1. 已经是有效的operator
        if op_name in self.VALID_OPERATORS:
            return op_name

        # 2. 清理特殊字符（如 G' -> G）
        cleaned = ''.join(c for c in op_name if c.isalpha())

        # 2.1 清理后是有效的
        if cleaned in self.VALID_OPERATORS:
            print(f"    🔧 P19: '{op_name}' -> '{cleaned}' (清理特殊字符)")
            return cleaned

        # 3. 检查纠正映射
        if op_name in self.OPERATOR_CORRECTIONS:
            corrected = self.OPERATOR_CORRECTIONS[op_name]
            print(f"    🔧 P19: '{op_name}' -> '{corrected}' (映射纠正)")
            return corrected

        if cleaned in self.OPERATOR_CORRECTIONS:
            corrected = self.OPERATOR_CORRECTIONS[cleaned]
            print(f"    🔧 P19: '{op_name}' -> '{corrected}' (清理后映射)")
            return corrected

        # 4. 尝试前缀匹配（至少2个字符）
        if len(cleaned) >= 2:
            for valid_op in self.VALID_OPERATORS:
                if valid_op.lower().startswith(cleaned.lower()):
                    print(f"    🔧 P19: '{op_name}' -> '{valid_op}' (前缀匹配)")
                    return valid_op

        # 5. 尝试包含匹配
        cleaned_lower = cleaned.lower()
        for valid_op in self.VALID_OPERATORS:
            if cleaned_lower in valid_op.lower() or valid_op.lower() in cleaned_lower:
                print(f"    🔧 P19: '{op_name}' -> '{valid_op}' (包含匹配)")
                return valid_op

        # 6. 最后回退到Custom（通用推理operator）
        print(f"    🔧 P19: '{op_name}' -> 'Custom' (默认回退)")
        return 'Custom'

    def _correct_dsl_operators(self, dsl_text: str) -> str:
        """
        🔧 P19修复: 在DSL文本中纠正所有operator名称

        Args:
            dsl_text: 原始DSL文本

        Returns:
            纠正后的DSL文本
        """
        import re

        # 找到所有可能是operator的单词（大写开头或全大写）
        # 但要保留DSL结构（->、?、:、[]、()、*）
        words = re.findall(r'\b([A-Z][a-zA-Z\']*)\b', dsl_text)

        corrections_made = []
        for word in set(words):  # 去重
            if word.lower() == 'done':  # 跳过done关键字
                continue
            corrected = self._correct_operator_name(word)
            if corrected != word:
                # 使用单词边界替换，避免部分匹配
                dsl_text = re.sub(r'\b' + re.escape(word) + r'\b', corrected, dsl_text)
                corrections_made.append(f"{word}->{corrected}")

        if corrections_made:
            print(f"    📝 P19 DSL纠正: {', '.join(corrections_made)}")

        return dsl_text

    def _clean_problem_content(self, dsl_text: str) -> str:
        """
        🔧 P20修复: 清理DSL开头混入的问题内容

        模型有时会将问题内容混入DSL输出，如:
        - "i)+3i(5-i) -> Programmer -> Custom"
        - "Final DSL: 5(3-i)+3i(5-i) -> Programmer"
        - "The answer is Programmer -> Custom"

        策略:
        1. 找到第一个有效operator的位置
        2. 检查operator之前的内容是否为有效DSL语法
        3. 如果不是，移除这些内容

        Args:
            dsl_text: 可能包含问题内容的DSL文本

        Returns:
            清理后的DSL文本
        """
        import re

        # 找到第一个有效operator的位置
        first_op_pos = len(dsl_text)
        first_op = None
        for op in self.VALID_OPERATORS:
            # 使用单词边界确保完整匹配
            match = re.search(r'\b' + op + r'\b', dsl_text)
            if match and match.start() < first_op_pos:
                first_op_pos = match.start()
                first_op = op

        if first_op is None:
            # 没有找到有效operator
            return dsl_text

        if first_op_pos == 0:
            # DSL以有效operator开头，无需清理
            return dsl_text

        # 检查operator之前的内容
        before_op = dsl_text[:first_op_pos].strip()

        # 有效的DSL前缀模式（应该只包含DSL语法元素）
        # 允许: [, (, 空格, 换行
        valid_prefix_pattern = r'^[\[\(\s\n]*$'

        if re.match(valid_prefix_pattern, before_op):
            # 前缀是有效的DSL语法
            return dsl_text

        # 前缀包含非DSL内容（如数学表达式、文本等）
        # 检查是否包含 "->" 分隔符
        if '->' in before_op:
            # 尝试找到最后一个 "->" 之后的有效DSL
            parts = dsl_text.split('->')
            for i, part in enumerate(parts):
                part_stripped = part.strip()
                # 检查这部分是否以有效operator开头
                for op in self.VALID_OPERATORS:
                    if part_stripped.startswith(op):
                        # 从这部分开始重建DSL
                        cleaned = ' -> '.join(parts[i:])
                        print(f"    🔧 P20: 清理问题内容: '{before_op}...' -> '{cleaned[:50]}...'")
                        return cleaned

        # 直接从第一个operator开始
        cleaned = dsl_text[first_op_pos:]
        print(f"    🔧 P20: 清理问题内容: '{before_op}' -> '{cleaned[:50]}...'")
        return cleaned

    def _expand_loops(self, dsl_text: str) -> str:
        """
        🔧 P15修复: 展开循环语法
        🔧 P18修复: 支持更多循环语法变体

        支持的语法:
        - (A) * N → A -> A -> ... (N次)
        - (A -> B) * N → A -> B -> A -> B -> ... (N次)
        - N * A → A -> A -> ... (N次) [P18新增]
        - A * → A -> A -> A (默认3次) [P18新增]

        Args:
            dsl_text: 原始DSL文本

        Returns:
            展开后的DSL文本
        """
        import re

        max_iterations = 10  # 防止无限循环

        # 🔧 P18修复: 先处理 "N * Operator" 格式 (如 "2 * Programmer")
        # 匹配: 数字 * 单词 (不在括号内)
        prefix_loop_pattern = r'(\d+)\s*\*\s*([A-Z][a-zA-Z]*)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(prefix_loop_pattern, dsl_text)
            if not match:
                break
            repeat_count = min(int(match.group(1)), 5)
            operator = match.group(2).strip()
            if operator in self.VALID_OPERATORS:
                expanded = ' -> '.join([operator] * repeat_count)
                dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            else:
                # 不是有效的operator，跳过
                break
            iteration += 1

        # 🔧 P18修复: 处理 "Operator *" 格式 (如 "Revise *", 默认重复3次)
        # 匹配: 单词 * (后面不跟数字)
        suffix_star_pattern = r'([A-Z][a-zA-Z]*)\s*\*(?!\s*\d)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(suffix_star_pattern, dsl_text)
            if not match:
                break
            operator = match.group(1).strip()
            if operator in self.VALID_OPERATORS:
                # 默认重复3次
                expanded = ' -> '.join([operator] * 3)
                dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            else:
                break
            iteration += 1

        # 🔧 P18修复增强: 处理 "(A)*" 格式 (括号内容后的*没有数字，默认3次)
        paren_star_pattern = r'\(([^()]+)\)\s*\*(?!\s*\d)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(paren_star_pattern, dsl_text)
            if not match:
                break
            inner_content = match.group(1).strip()
            # 默认重复3次
            expanded = ' -> '.join([inner_content] * 3)
            dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            iteration += 1

        # 原有逻辑: 匹配循环模式 (内容) * 数字
        # 支持: (Revise) * 3, (Custom -> Review -> Revise) * 2
        loop_pattern = r'\(([^()]+)\)\s*\*\s*(\d+)'

        iteration = 0
        while iteration < max_iterations:
            match = re.search(loop_pattern, dsl_text)
            if not match:
                break

            inner_content = match.group(1).strip()  # 括号内的内容
            repeat_count = int(match.group(2))      # 重复次数

            # 限制重复次数，避免生成过长的DSL
            repeat_count = min(repeat_count, 5)

            # 展开: 将内容重复N次，用 -> 连接
            expanded = ' -> '.join([inner_content] * repeat_count)

            # 替换原始的循环表达式
            dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]

            iteration += 1

        return dsl_text

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

        # 🔧 P15修复: 处理重复输出的情况（必须在XML清理之前）
        # 模型有时会输出多个DSL片段，用 </output> 分隔
        # 取第一个包含有效operator的片段
        if '</output>' in dsl_text or '<output>' in dsl_text:
            # 尝试按 </output> 或 <output> 分割，取第一个有效片段
            fragments = re.split(r'\s*</?\s*output\s*>\s*', dsl_text)
            for frag in fragments:
                frag = frag.strip()
                if frag and any(op in frag for op in self.VALID_OPERATORS):
                    dsl_text = frag
                    break

        # 🔧 P14修复: 更激进的清理，移除所有XML标签和噪声
        # 移除所有XML风格的标签 (包括 </output>, </dsl>, <workflow> 等)
        dsl_text = re.sub(r'</?[a-zA-Z_][a-zA-Z0-9_]*/?>', '', dsl_text)
        # 移除代码块标记
        dsl_text = re.sub(r'```\w*', '', dsl_text)
        # 移除可能的标签
        dsl_text = re.sub(r'</?workflow>', '', dsl_text).strip()

        if not dsl_text:
            return {'valid': False, 'error': '空的DSL', 'stages': []}

        # 🔧 P20修复: 清理DSL开头的问题内容
        # 模型有时会将问题内容混入DSL，如 "i)+3i(5-i) -> Programmer"
        # 需要找到第一个有效operator，并移除之前的非DSL内容
        dsl_text = self._clean_problem_content(dsl_text)

        if not dsl_text:
            return {'valid': False, 'error': '清理后DSL为空', 'stages': []}

        # 🔧 P19修复: 在循环展开之前先纠正operator名称
        # 这样可以修复 "Giver" -> "Custom", "G'" -> "Custom" 等幻觉
        dsl_text = self._correct_dsl_operators(dsl_text)

        # 🔧 P15修复: 循环展开预处理
        # 将 (A) * N 展开为 A -> A -> ... (N次)
        # 将 (A -> B) * N 展开为 A -> B -> A -> B -> ... (N次)
        dsl_text = self._expand_loops(dsl_text)

        # 🔧 P15修复: 早期噪声检测 - 如果DSL包含明显无效内容，直接拒绝
        # 检测是否包含有效的operator（至少一个）
        has_valid_op = any(op in dsl_text for op in self.VALID_OPERATORS)
        if not has_valid_op:
            return {'valid': False, 'error': '未包含有效的operator', 'stages': []}

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
                operators = []
                for op in inner.split(','):
                    op = op.strip()
                    # 🔧 P14修复: 清理operator名称中可能残留的噪声
                    op = re.sub(r'[<>/\s]+$', '', op)
                    op = re.sub(r'^[<>/\s]+', '', op)
                    op = op.strip()
                    operators.append(op)

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
                # 🔧 P14修复: 清理operator名称中可能残留的噪声
                op = re.sub(r'[<>/\s]+$', '', op)  # 移除结尾的 < > / 和空白
                op = re.sub(r'^[<>/\s]+', '', op)  # 移除开头的 < > / 和空白
                op = op.strip()
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
        """
        生成__call__方法体

        🔧 P20修复: 正确处理 Review -> Revise 序列
        - 跟踪 solution 变量（来自 Custom/Programmer 等）
        - 跟踪 feedback 变量（来自 Review）
        - Revise 同时使用 solution 和 feedback
        """
        lines = []
        prev_output = None  # 上一阶段的输出变量名
        prev_is_list = False  # 上一阶段是否是并行（输出列表）

        # 🔧 P20: 跟踪solution和feedback变量，用于Review->Revise序列
        last_solution_var = None  # 最近的solution输出（来自Custom/Programmer等）
        last_feedback_var = None  # 最近的feedback输出（来自Review）
        prev_op = None  # 上一个operator类型

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
                lines.append(f"        solutions_{i} = [r.get('{sig.get('output', 'response')}', r.get('response', str(r))) for r in results_{i}]")

                prev_output = f"solutions_{i}"
                prev_is_list = True
                # 🔧 P20: 并行阶段产生的是solution列表
                last_solution_var = f"solutions_{i}"
                prev_op = op

            else:
                # 单个operator
                op = stage['operators'][0]
                attr_name = self._to_snake_case(op)
                sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op, {})

                # 🔧 P20修复: 特殊处理 Review -> Revise 序列
                if op == 'Revise' and prev_op == 'Review' and last_solution_var and last_feedback_var:
                    # Revise需要原始solution和Review的feedback
                    param_str = f"problem=problem, solution={last_solution_var}, feedback={last_feedback_var}"
                elif prev_is_list and sig.get('accepts_list'):
                    # 前一阶段是列表，当前operator接受列表（如ScEnsemble）
                    param_str = f"solutions={prev_output}, problem=problem"
                elif prev_output:
                    param_str = self._build_params(op, prev_output, is_first=False)
                else:
                    param_str = self._build_params(op, 'problem', is_first=True)

                lines.append(f"        result_{i} = await self.{attr_name}({param_str})")

                # 🔧 P20: 使用.get()避免KeyError，并更新跟踪变量
                output_key = sig.get('output', 'response')
                # 使用更健壮的字典访问
                lines.append(f"        output_{i} = result_{i}.get('{output_key}', result_{i}.get('response', str(result_{i})))")
                prev_output = f"output_{i}"
                prev_is_list = False

                # 🔧 P20: 更新solution/feedback跟踪变量
                if op == 'Review':
                    # Review产生feedback，但保持上一个solution不变
                    last_feedback_var = f"output_{i}"
                elif op in ('Custom', 'Programmer', 'CustomCodeGenerate', 'Revise', 'Format', 'AnswerGenerate'):
                    # 这些operator产生solution/response类输出
                    last_solution_var = f"output_{i}"
                    last_feedback_var = None  # 清除feedback

                prev_op = op

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
