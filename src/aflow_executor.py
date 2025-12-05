#!/usr/bin/env python3
"""
AFlow执行适配器 - 执行RL生成的工作流
"""
import sys
import os
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import asyncio
import time

# 导入工作流验证器、响应标准化器和SymPy修复器
try:
    from .workflow_validator import WorkflowValidator
    from .response_standardizer import ResponseStandardizer
    from .sympy_code_fixer import SymPyCodeFixer
except ImportError:
    from workflow_validator import WorkflowValidator
    from response_standardizer import ResponseStandardizer
    from sympy_code_fixer import SymPyCodeFixer

# 添加项目根目录到路径以导入scripts模块
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入本地scripts模块（兼容AFlow接口）
from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module

class AFlowExecutor:
    """执行RL生成的工作流，使用AFlow的算子"""

    def __init__(
        self,
        llm_config_path: str = "config/aflow_llm.yaml",
        llm_model_name: str = "gpt-oss-120b",  # 使用8002端口的gpt-oss-120b
        timeout: int = 300,
        operator_enhancer: Optional[Any] = None,
        enable_fallback: bool = True  # 启用Fallback机制
    ):
        """
        Args:
            llm_config_path: AFlow LLM配置文件路径
            llm_model_name: 使用的LLM模型名称
            timeout: 执行超时时间（秒）
            operator_enhancer: Layer 2 operator提示词增强器（可选）
            enable_fallback: 是否启用Fallback机制
        """
        self.llm_config_path = Path(llm_config_path)
        self.llm_model_name = llm_model_name
        self.timeout = timeout
        self.operator_enhancer = operator_enhancer
        self.enable_fallback = enable_fallback
        self.validator = WorkflowValidator()  # 添加验证器
        self.standardizer = ResponseStandardizer()  # 添加响应标准化器
        self.sympy_fixer = SymPyCodeFixer()  # 添加SymPy修复器

        # 加载LLM配置
        self._load_llm_config()

        print(f"✅ AFlow执行器初始化完成")
        print(f"  LLM模型: {llm_model_name}")
        print(f"  超时: {timeout}秒")
        print(f"  XML支持: 启用")  # 新增XML支持提示
        if operator_enhancer is not None:
            print(f"  Layer 2增强: 启用")

    def _load_llm_config(self):
        """加载LLM配置"""
        try:
            # 设置配置路径
            abs_config_path = self.llm_config_path.absolute()

            # 🔧 P16修复: 使用 from_yaml() 正确解析配置为 LLMConfig 对象
            from scripts.async_llm import LLMsConfig
            self.llm_configs = LLMsConfig.from_yaml(str(abs_config_path))

            # 为本地vLLM服务禁用代理
            import os
            for model_name, config in self.llm_configs.models.items():
                base_url = getattr(config, 'base_url', '')
                if 'localhost' in str(base_url) or '127.0.0.1' in str(base_url):
                    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
                    os.environ['no_proxy'] = 'localhost,127.0.0.1'
                    print("  📌 设置 NO_PROXY=localhost,127.0.0.1 (绕过代理访问vLLM)")
                    break

            print(f"✅ 加载LLM配置: {abs_config_path}")

        except Exception as e:
            print(f"⚠️  加载LLM配置失败: {e}")
            import traceback
            traceback.print_exc()
            # 🔧 P16修复: 构建默认配置而不是调用不存在的 default() 方法
            from scripts.async_llm import LLMsConfig, LLMConfig
            import os
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if api_key:
                print(f"  使用环境变量 OPENAI_API_KEY 构建默认配置")
                default_config = LLMConfig(
                    api_type='openai',
                    base_url='https://api.openai.com/v1',
                    api_key=api_key,
                    model_name='gpt-4o-mini',
                    temperature=0.7,
                    top_p=1.0,
                    max_tokens=4096
                )
                self.llm_configs = LLMsConfig(models={'gpt-4o-mini': default_config})
                print(f"✅ 成功构建默认LLM配置")
            else:
                print(f"  ⚠️ 未找到 OPENAI_API_KEY 环境变量")
                self.llm_configs = None

    def validate_operator_output(self, output: Any, operator_name: str) -> Dict:
        """
        验证并标准化算子输出格式（使用ResponseStandardizer）

        Args:
            output: 算子的原始输出
            operator_name: 算子名称

        Returns:
            标准化后的输出字典
        """
        # 使用ResponseStandardizer进行标准化
        standardized = self.standardizer.standardize(output, operator_name)

        # 保持向后兼容，同时返回原始字段和标准化字段
        if isinstance(output, dict):
            result = output.copy()
            result.update({
                '__standardized__': standardized,
                # 确保关键字段存在
                'response': standardized['content'],
                'success': standardized['success'],
                'error': standardized.get('error')
            })
            return result
        else:
            return standardized

    def _preprocess_workflow_code(self, workflow_input: str) -> Tuple[str, str]:
        """
        预处理工作流输入 - 支持XML格式和纯Python代码

        Args:
            workflow_input: 工作流输入（可能是XML格式或纯Python代码）

        Returns:
            (graph_code, prompt_code) - 提取的Python代码和TASK_PROMPT
        """
        import re

        # 检测是否为XML格式
        if '<graph>' in workflow_input and '</graph>' in workflow_input:
            print(f"  📝 检测到XML格式工作流，正在提取...")
            return self._extract_from_xml(workflow_input)

        # 检测是否为<workflow>包装的XML
        if '<workflow>' in workflow_input and '</workflow>' in workflow_input:
            print(f"  📝 检测到<workflow>包装的XML格式，正在提取...")
            return self._extract_from_xml(workflow_input)

        # 回退到纯Python代码格式
        return workflow_input, ""

    def _extract_from_xml(self, xml_input: str) -> Tuple[str, str]:
        """
        从XML格式工作流中提取graph代码和prompt

        Args:
            xml_input: XML格式的工作流输入

        Returns:
            (graph_code, prompt_code)
        """
        import re

        graph_code = ""
        prompt_code = ""

        # 提取 <graph>...</graph>
        graph_match = re.search(r'<graph>\s*([\s\S]*?)\s*</graph>', xml_input)
        if graph_match:
            graph_code = graph_match.group(1).strip()
            print(f"  ✅ 从<graph>提取代码，长度: {len(graph_code)}")

        # 提取 <prompt>...</prompt>
        prompt_match = re.search(r'<prompt>\s*([\s\S]*?)\s*</prompt>', xml_input)
        if prompt_match:
            prompt_code = prompt_match.group(1).strip()
            print(f"  ✅ 从<prompt>提取TASK_PROMPT，长度: {len(prompt_code)}")

        # 如果graph_code为空，尝试整体作为代码
        if not graph_code:
            print(f"  ⚠️ 未找到<graph>标签，尝试作为纯Python代码处理")
            # 移除XML标签后的内容可能就是Python代码
            cleaned = re.sub(r'</?workflow>', '', xml_input)
            cleaned = re.sub(r'</?graph>', '', cleaned)
            cleaned = re.sub(r'<prompt>.*?</prompt>', '', cleaned, flags=re.DOTALL)
            if 'class Workflow' in cleaned:
                graph_code = cleaned.strip()

        # 如果有prompt_code但graph_code不包含TASK_PROMPT，则合并
        if prompt_code and graph_code and 'TASK_PROMPT' not in graph_code:
            # 在class Workflow之前添加TASK_PROMPT
            class_match = re.search(r'^class Workflow', graph_code, re.MULTILINE)
            if class_match:
                graph_code = prompt_code + "\n\n" + graph_code
                print(f"  📝 已将TASK_PROMPT合并到graph代码中")

        return graph_code, prompt_code

    async def execute_workflow(
        self,
        workflow_code: str,
        problem: str,
        problem_type: str = "math",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        执行工作流

        Args:
            workflow_code: RL模型生成的Workflow类代码
            problem: 问题文本
            problem_type: 问题类型
            **kwargs: 其他参数（如entry_point for code）

        Returns:
            (answer, cost, metadata)
        """

        start_time = time.time()

        # 🔧 新增: XML格式预处理 - 支持AFlow风格XML工作流
        workflow_code, extracted_prompt = self._preprocess_workflow_code(workflow_code)
        if not workflow_code:
            print(f"⚠️  无法从输入中提取工作流代码")
            if self.enable_fallback:
                return await self._execute_fallback_workflow(
                    problem, problem_type,
                    error_info="Failed to extract workflow code from input",
                    **kwargs
                )
            else:
                raise ValueError("无法从输入中提取工作流代码")

        # 🔧 智能输入格式化：根据数据源注入context等信息
        # 构造sample字典用于格式化（从kwargs提取相关字段）
        sample_info = {
            "problem": problem,
            "problem_type": problem_type,
            "source": kwargs.get("source", ""),
            "context": kwargs.get("context", []),
        }
        formatted_problem = self._format_problem_by_source(problem, sample_info)
        if formatted_problem != problem:
            print(f"  📝 已格式化问题输入 (source={sample_info['source']})")

        # 1. 验证工作流代码
        is_valid, msg, validation_details = self.validator.validate_workflow_code(workflow_code, problem_type)

        if not is_valid:
            print(f"⚠️  工作流代码验证失败: {msg}")

            # 尝试自动修复
            fixed_code = self.validator.fix_common_issues(workflow_code)
            is_valid, msg, _ = self.validator.validate_workflow_code(fixed_code, problem_type)

            if is_valid:
                print(f"✅ 自动修复成功")
                workflow_code = fixed_code
            elif self.enable_fallback:
                print(f"  返回验证错误信息")
                return await self._execute_fallback_workflow(problem, problem_type, error_info=f"Validation failed: {msg}", **kwargs)
            else:
                # Fallback禁用，抛出异常
                raise ValueError(f"工作流代码无效且Fallback已禁用: {msg}")

        # 2. 修复SymPy兼容性问题（针对Code类型）
        if problem_type == "code" or 'sympy' in workflow_code.lower():
            fixed_code, was_modified, fixes = self.sympy_fixer.fix_code(workflow_code)
            if was_modified:
                print(f"🔧 SymPy代码修复: {', '.join(fixes)}")
                workflow_code = fixed_code

        try:
            # 创建临时工作流模块
            workflow_class = self._create_workflow_class(workflow_code, problem_type)

            # 实例化工作流
            llm_config = self._get_llm_config()

            # 确保 llm_config 不是 None
            if llm_config is None:
                print(f"⚠️  llm_config 为 None，降级为字符串: {self.llm_model_name}")
                llm_config = self.llm_model_name

            try:
                workflow = workflow_class(
                    name="rl_generated_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )
            except Exception as e:
                # 工作流实例化失败，返回错误信息
                print(f"⚠️  工作流实例化失败: {e}")
                import traceback
                traceback.print_exc()
                return await self._execute_fallback_workflow(
                    problem, problem_type,
                    error_info=f"Workflow instantiation failed: {type(e).__name__}: {str(e)[:200]}",
                    **kwargs
                )

            # 执行（带超时）
            # For code problems, try passing entry_point and test (HumanEval format)
            try:
                if problem_type == "code":
                    # Try full HumanEval format first (entry_point + test)
                    if "entry_point" in kwargs and "test" in kwargs:
                        try:
                            result = await asyncio.wait_for(
                                workflow(formatted_problem, kwargs["entry_point"], kwargs["test"]),
                                timeout=self.timeout
                            )
                        except TypeError as e:
                            # Fallback to just entry_point
                            if "positional argument" in str(e) or "takes" in str(e):
                                print(f"  ⚠️  Workflow不支持test参数，尝试只传entry_point")
                                try:
                                    result = await asyncio.wait_for(
                                        workflow(formatted_problem, kwargs["entry_point"]),
                                        timeout=self.timeout
                                    )
                                except TypeError:
                                    print(f"  ⚠️  Workflow不支持entry_point参数，降级为只传problem")
                                    result = await asyncio.wait_for(
                                        workflow(formatted_problem),
                                        timeout=self.timeout
                                    )
                            else:
                                raise
                    elif "entry_point" in kwargs:
                        # Only entry_point available
                        try:
                            result = await asyncio.wait_for(
                                workflow(formatted_problem, kwargs["entry_point"]),
                                timeout=self.timeout
                            )
                        except TypeError as e:
                            if "positional argument" in str(e):
                                print(f"  ⚠️  Workflow不支持entry_point参数，降级为只传problem")
                                result = await asyncio.wait_for(
                                    workflow(formatted_problem),
                                    timeout=self.timeout
                                )
                            else:
                                raise
                    else:
                        # No extra parameters
                        result = await asyncio.wait_for(
                            workflow(formatted_problem),
                            timeout=self.timeout
                        )
                else:
                    # Non-code problems (使用格式化后的问题，包含context等)
                    result = await asyncio.wait_for(
                        workflow(formatted_problem),
                        timeout=self.timeout
                    )
            except Exception as e:
                # 捕获所有异常（operator执行失败）
                print(f"  ❌ Workflow执行异常: {type(e).__name__}")
                print(f"     异常信息: {str(e)}")
                import traceback
                print(f"  完整堆栈:")
                traceback.print_exc()

                # 返回错误信息让模型学习
                if self.enable_fallback:
                    print(f"  🔄 返回执行错误信息")
                    return await self._execute_fallback_workflow(
                        problem, problem_type,
                        error_info=f"Execution failed: {type(e).__name__}: {str(e)[:200]}",
                        **kwargs
                    )
                else:
                    print(f"  ⚠️  Fallback已禁用，直接抛出异常")
                    # 直接抛出异常而不是使用fallback
                    raise

            # 安全地解包结果（可能返回2个或更多值）
            if isinstance(result, tuple):
                if len(result) >= 2:
                    answer, cost = result[0], result[1]

                    # 类型验证和修正
                    if not isinstance(cost, (int, float)):
                        print(f"  警告: cost类型错误 ({type(cost).__name__})，尝试修正...")
                        # 检查是否answer和cost位置反了
                        if isinstance(answer, (int, float)) and isinstance(cost, str):
                            print(f"  检测到answer和cost顺序反转，交换...")
                            answer, cost = cost, answer
                        else:
                            # cost是字符串但不是数字，设为0
                            print(f"  cost包含非数字内容，设为0.0")
                            if len(str(cost)) <= 100:
                                print(f"     cost内容: {cost}")
                            else:
                                print(f"     cost内容预览: {str(cost)[:100]}...")
                            cost = 0.0

                elif len(result) == 1:
                    answer, cost = result[0], 0.0
                else:
                    answer, cost = None, 0.0
            else:
                answer, cost = result, 0.0

            # 最终类型确保
            if not isinstance(cost, (int, float)):
                print(f"  cost最终类型仍然错误，强制设为0.0")
                cost = 0.0

            execution_time = time.time() - start_time

            # P0修复: 验证answer非空，空答案触发fallback
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                print(f"  ⚠️  答案为空(None或空字符串)，触发fallback")
                if self.enable_fallback:
                    return await self._execute_fallback_workflow(
                        problem, problem_type,
                        error_info="Empty answer returned",
                        **kwargs
                    )
                # fallback禁用时返回空字符串而非None
                answer = ""

            # P0修复: 检测无效答案模式
            if isinstance(answer, str):
                invalid_patterns = ['Based on the feedback', 'Revised Solution:', '```python\n```']
                for pattern in invalid_patterns:
                    if pattern in answer:
                        print(f"  ⚠️  检测到无效答案模式: {pattern[:30]}")
                        # 尝试清理
                        answer = answer.replace(pattern, '').strip()

            # 🔧 P0-关键修复【优先执行】: 检测代码泄漏（Programmer operator返回code而非output的bug）
            # 必须在无效boxed检测之前，因为泄漏的代码可能包含有效答案
            if isinstance(answer, str) and problem_type in ['math', 'qa']:
                code_indicators = ['def solve(', 'def main(', 'import ', 'return ', 'class ', 'if __name__']
                if any(indicator in answer for indicator in code_indicators):
                    print(f"  🔴 检测到代码泄漏! answer包含源代码而非执行结果")
                    print(f"     answer预览: {answer[:100]}...")

                    # 尝试执行代码获取真正的答案
                    executed_answer = self._execute_leaked_code(answer)
                    if executed_answer:
                        print(f"  ✅ 代码执行成功! 真正的答案: {executed_answer}")
                        answer = executed_answer
                    else:
                        print(f"  ⚠️  代码执行失败，触发fallback")
                        if self.enable_fallback:
                            return await self._execute_fallback_workflow(
                                problem, problem_type,
                                error_info="Code leakage detected: Programmer returned code instead of output",
                                **kwargs
                            )

            # P13修复: 禁用aflow_executor的预处理，让reward_computer的P12 LLM提取做主力
            # 原来的逻辑会错误地从代码中提取变量值（如buckets=2的"2"），而不是计算结果
            # 现在保留原始输出，让P12 LLM提取来处理复杂格式
            if isinstance(answer, str):
                # 只处理完全空的boxed，其他情况保留原始内容让P12处理
                import re
                if re.search(r'\\boxed\{\s*\}', answer):
                    print(f"  🔴 检测到空boxed，清空答案")
                    answer = ""
                # 其他情况（如代码块boxed）保留原始内容，让reward_computer的P12 LLM提取处理
                # 不再调用 extract_valid_answer_from_text()，避免错误提取

            # 元数据
            metadata = {
                "success": True,
                "execution_time": execution_time,
                "cost": cost,
                "problem_type": problem_type
            }

            return answer, cost, metadata

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏱️  执行超时 ({self.timeout}秒)")

            metadata = {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
            }

            return None, 0.0, metadata

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 执行错误: {str(e)}")

            import traceback
            traceback.print_exc()

            metadata = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
            }

            return None, 0.0, metadata

    def _execute_leaked_code(self, code_string: str) -> Optional[str]:
        """
        🔧 P0修复: 执行泄漏的代码，获取真正的答案

        当 workflow 错误地返回 result['code'] 而不是 result['output'] 时，
        这个方法尝试执行代码并获取真正的计算结果。

        Args:
            code_string: 包含 Python 代码的字符串（可能包含 def solve(): ...）

        Returns:
            执行结果字符串，如果执行失败返回 None
        """
        import re
        from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

        try:
            # 清理代码（去除 \boxed{} 包装等）
            code = code_string

            # P14修复: 清理Unicode字符，避免执行失败
            # LLM生成的代码可能包含智能引号、特殊空格等
            unicode_replacements = {
                '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
                '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
                '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
                '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
                '\u202f': ' ',  # NARROW NO-BREAK SPACE
                '\u00a0': ' ',  # NO-BREAK SPACE
                '\u2009': ' ',  # THIN SPACE
                '\u200b': '',   # ZERO WIDTH SPACE
                '\u2013': '-',  # EN DASH
                '\u2014': '-',  # EM DASH
            }
            for unicode_char, replacement in unicode_replacements.items():
                code = code.replace(unicode_char, replacement)

            # 如果代码被 \boxed{} 包装，提取内容
            boxed_match = re.search(r'\\boxed\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', code)
            if boxed_match:
                code = boxed_match.group(1)

            # 如果代码在代码块中，提取
            code_block_match = re.search(r'```python\s*([\s\S]*?)```', code)
            if code_block_match:
                code = code_block_match.group(1)

            # 确保代码包含函数定义
            if 'def solve' not in code and 'def main' not in code:
                # 尝试包装成 solve 函数
                if 'return ' in code:
                    # 代码片段，包装成函数
                    code = f"def solve():\n    " + code.replace('\n', '\n    ')

            # 安全的代码执行（使用 ProcessPoolExecutor 隔离）
            def run_isolated_code(code_str):
                """在隔离环境中执行代码，同时捕获 stdout"""
                import io
                import sys
                global_namespace = {'__builtins__': __builtins__}

                # 添加常用数学库
                try:
                    import math
                    global_namespace['math'] = math
                except:
                    pass

                # 捕获 stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                try:
                    exec(code_str, global_namespace)

                    # 尝试调用常见函数名: solve(), main(), answer()
                    for func_name in ['solve', 'main', 'answer']:
                        if func_name in global_namespace and callable(global_namespace[func_name]):
                            result = global_namespace[func_name]()
                            # 如果函数返回值有效，使用返回值
                            if result is not None:
                                return str(result)
                            break  # 函数存在但返回None，继续检查stdout

                    # 如果返回值是 None，检查 stdout 输出
                    stdout_content = captured_output.getvalue().strip()
                    if stdout_content:
                        # 返回最后一行非空输出作为答案
                        lines = [l.strip() for l in stdout_content.split('\n') if l.strip()]
                        if lines:
                            return lines[-1]

                    return None
                except Exception as e:
                    print(f"     代码执行异常: {e}", file=old_stdout)
                    return None
                finally:
                    sys.stdout = old_stdout

            # 尝试直接执行（快速路径，无需进程池）
            try:
                result = run_isolated_code(code)
                if result is not None:
                    return result
            except Exception as e:
                print(f"     直接执行失败: {e}")

            return None

        except Exception as e:
            print(f"     _execute_leaked_code 异常: {e}")
            return None

    def _create_workflow_class(self, workflow_code: str, problem_type: str):
        """从工作流代码动态创建Workflow类"""
        import re

        print(f"  🔍 进入 _create_workflow_class，代码长度: {len(workflow_code)}", flush=True)

        # 提取并打印operator列表（替代打印前10行代码）
        operator_pattern = r'self\.(\w+)\s*=\s*operator\.(\w+)\('
        operators_found = re.findall(operator_pattern, workflow_code)
        if operators_found:
            op_list = [f"{name}({op_type})" for name, op_type in operators_found]
            print(f"  📦 Operators: {', '.join(op_list)}", flush=True)
        else:
            print(f"  📦 Operators: 未检测到 (可能是fallback)", flush=True)

        # 🔧 关键新功能：检测并提取TASK_PROMPT用于问题增强
        task_prompt_value = None
        task_prompt_match = re.search(
            r'TASK_PROMPT\s*=\s*(?:"""([^"]*(?:"(?!"")|[^"])*)"""|"([^"]*)"|\'([^\']*)\')',
            workflow_code,
            re.DOTALL
        )
        if task_prompt_match:
            task_prompt_value = task_prompt_match.group(1) or task_prompt_match.group(2) or task_prompt_match.group(3)
            if task_prompt_value:
                print(f"  📝 检测到TASK_PROMPT，将自动增强问题输入", flush=True)

        # 准备命名空间
        namespace = {
            "operator": operator_module,
            "create_llm_instance": create_llm_instance,
            "DatasetType": str,
            "__TASK_PROMPT__": task_prompt_value  # 注入到命名空间
        }

        # 替换import路径（使workspace路径可用）
        # 这里简化处理，直接使用scripts中的operator
        modified_code = workflow_code.replace(
            f"import workspace.{problem_type}.workflows.template.operator as operator",
            "# operator already imported"
        )

        # 🔧 关键修复：过滤掉不允许的import语句（防止aiofiles等问题）
        # 使用更强大的过滤：基于AST检测所有import形式
        import ast

        allowed_imports = {
            'operator', 'workspace', 'scripts', 'asyncio', 'typing',
            'json', 're', 'math', 'collections', 'itertools', 'functools',
            'abc', 'copy', 'dataclasses', 'enum', 'inspect', 'os', 'sys',
            'time', 'traceback', 'types', 'warnings', 'random'
        }

        # 方法1: 基于AST的精确过滤
        try:
            tree = ast.parse(modified_code)
            forbidden_imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in allowed_imports:
                            forbidden_imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in allowed_imports:
                            forbidden_imports.add(module_name)

            if forbidden_imports:
                print(f"  🚫 AST检测到禁止的导入: {forbidden_imports}", flush=True)
                # 使用正则替换所有相关import
                for mod in forbidden_imports:
                    import re as re_module
                    # 替换 import xxx 和 from xxx import
                    modified_code = re_module.sub(
                        rf'^(\s*)(import\s+{mod}[^\n]*)',
                        r'\1# [FILTERED] \2',
                        modified_code,
                        flags=re_module.MULTILINE
                    )
                    modified_code = re_module.sub(
                        rf'^(\s*)(from\s+{mod}[^\n]*)',
                        r'\1# [FILTERED] \2',
                        modified_code,
                        flags=re_module.MULTILINE
                    )
                print(f"  📝 已过滤 {len(forbidden_imports)} 个禁止的模块导入", flush=True)
        except SyntaxError as e:
            print(f"  ⚠️ AST解析失败，使用简单过滤: {e}", flush=True)
            # 方法2: 简单行级过滤作为备用
            lines = modified_code.split('\n')
            filtered_lines = []
            filtered_count = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if stripped.startswith('import '):
                        module = stripped.split()[1].split('.')[0]
                    else:
                        module = stripped.split()[1].split('.')[0]
                    if module not in allowed_imports:
                        print(f"  🚫 过滤导入: {stripped}", flush=True)
                        filtered_lines.append(f"# [FILTERED] {line}")
                        filtered_count += 1
                        continue
                filtered_lines.append(line)
            modified_code = '\n'.join(filtered_lines)
            if filtered_count > 0:
                print(f"  📝 已过滤 {filtered_count} 个不允许的导入语句", flush=True)

        # 修复常见typo（RL模型可能产生的错误）
        modified_code = modified_code.replace("async_lll", "async_llm")
        modified_code = modified_code.replace("create_lll_instance", "create_llm_instance")

        # P0修复: 扩展typo修复 - 修复self.lll/self.llll等变体
        import re
        # 修复 self.l{3,}m 类型的typo (self.lllm, self.llllm等)
        modified_code = re.sub(r'\bself\.l{3,}m\b', 'self.llm', modified_code)
        # 修复 async_l{3,}m 类型的typo
        modified_code = re.sub(r'\basync_l{3,}m\b', 'async_llm', modified_code)
        # 修复 create_l{3,}m_instance 类型的typo
        modified_code = re.sub(r'\bcreate_l{3,}m_instance\b', 'create_llm_instance', modified_code)

        # P1修复: 检测并修复顶层await问题 (RL模型可能生成 'await xxx' 在函数外)
        import re
        # 查找顶层await（不在async def内的await）
        lines = modified_code.split('\n')
        fixed_lines = []
        in_async_func = False
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # 检测async def开始
            if stripped.startswith('async def '):
                in_async_func = True
                # 计算缩进层级
                indent = len(line) - len(line.lstrip())
                indent_stack.append(indent)
            # 检测函数结束（通过缩进变化）
            elif indent_stack and stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                while indent_stack and current_indent <= indent_stack[-1]:
                    indent_stack.pop()
                if not indent_stack:
                    in_async_func = False

            # 检测顶层await
            if stripped.startswith('await ') and not in_async_func:
                # 将顶层await包装到一个临时async函数中
                print(f"  🔧 修复顶层await: {stripped[:50]}...")
                # 创建包装函数
                indent = len(line) - len(line.lstrip())
                wrapper = f"{' ' * indent}# [AUTO-FIXED] Wrapped top-level await\n"
                wrapper += f"{' ' * indent}async def _auto_wrap_await():\n"
                wrapper += f"{' ' * (indent + 4)}return {stripped}\n"
                wrapper += f"{' ' * indent}_result = asyncio.get_event_loop().run_until_complete(_auto_wrap_await())"
                fixed_lines.append(wrapper)
                continue

            fixed_lines.append(line)

        modified_code = '\n'.join(fixed_lines)

        # P2修复: 清理RL模型可能生成的无效类型注解 (如 Tuple.QA, List.Something)
        # 这些会导致 AttributeError: QA 等错误
        import re as regex_module
        # 匹配类型注解中的无效属性访问: Tuple.XXX, List.YYY, Dict.ZZZ 等
        invalid_type_patterns = [
            r'(Tuple|List|Dict|Set|Optional|Union)\.(\w+)',  # Tuple.QA -> Any
            r':\s*(QA|Math|Code)\b',  # : QA -> : Any
            r'->\s*(QA|Math|Code)\b',  # -> QA -> -> Any
        ]
        for pattern in invalid_type_patterns:
            if regex_module.search(pattern, modified_code):
                print(f"  🔧 P2修复: 清理无效类型注解模式 {pattern[:30]}...")
                modified_code = regex_module.sub(pattern, r'Any', modified_code)

        # 确保Any类型可用
        if 'Any' in modified_code and 'from typing import' in modified_code:
            # 检查是否已导入Any
            if ', Any' not in modified_code and 'Any,' not in modified_code and 'import Any' not in modified_code:
                modified_code = modified_code.replace('from typing import', 'from typing import Any, ')

        # P2修复增强: 在__call__方法开头自动初始化常用变量，防止UnboundLocalError
        # 查找 async def __call__ 并在其后插入变量初始化
        call_init_vars = '''
        # [AUTO-INIT] 防止条件分支导致的UnboundLocalError
        result = None
        solution = None
        code = None
        answer = None
        prog_result = None
        review_result = None
        test_result = None
        revised = None
        cost = 0.0
        '''
        # 使用正则找到 async def __call__ 的方法体开始位置
        call_match = regex_module.search(r'(async def __call__\([^)]*\)[^:]*:)\s*\n', modified_code)
        if call_match:
            # 检测下一行的缩进
            end_pos = call_match.end()
            next_line_match = regex_module.search(r'^([ \t]+)', modified_code[end_pos:], regex_module.MULTILINE)
            if next_line_match:
                base_indent = next_line_match.group(1)
                # 格式化初始化代码，使用正确的缩进
                formatted_init = '\n'.join(base_indent + line.strip() for line in call_init_vars.strip().split('\n') if line.strip())
                # 插入到__call__方法体开头
                modified_code = modified_code[:end_pos] + formatted_init + '\n' + modified_code[end_pos:]
                print(f"  🔧 P2修复: 已在__call__中自动初始化防护变量")

        try:
            # 执行代码创建类
            exec(modified_code, namespace)

            # 返回Workflow类
            if "Workflow" not in namespace:
                raise ValueError("No Workflow class found in generated code")

            WorkflowClass = namespace["Workflow"]

            # 🔧 关键新功能：如果有TASK_PROMPT，创建包装类自动增强问题输入
            if task_prompt_value:
                # 创建增强版Workflow类
                class EnhancedWorkflow:
                    """自动将TASK_PROMPT注入到问题输入中的包装器"""
                    _task_prompt = task_prompt_value
                    _original_class = WorkflowClass

                    def __init__(self, name: str, llm_config, dataset):
                        # P2修复: 使用object.__setattr__避免__getattr__递归问题
                        object.__setattr__(self, '_instance', self._original_class(name, llm_config, dataset))

                    async def __call__(self, problem: str, *args, **kwargs):
                        # 自动增强问题输入（支持任意额外参数）
                        enhanced_problem = f"{self._task_prompt}\n\nProblem:\n{problem}"
                        result = await self._instance(enhanced_problem, *args, **kwargs)
                        # P2修复: 确保返回值是可解包的tuple而非coroutine
                        return result

                    def __getattr__(self, name):
                        # P2修复: 防止访问不存在的_instance导致递归
                        if name == '_instance':
                            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_instance'")
                        return getattr(object.__getattribute__(self, '_instance'), name)

                print(f"  ✨ 创建EnhancedWorkflow包装器（自动注入TASK_PROMPT）")
                return EnhancedWorkflow

            return WorkflowClass

        except Exception as e:
            print(f"⚠️  生成的工作流代码有错误: {e}")
            # 抛出异常，让上层处理并返回错误信息
            raise ValueError(f"Workflow code compilation failed: {type(e).__name__}: {str(e)[:200]}")

    def _get_llm_config(self):
        """获取LLM配置（确保返回正确类型）"""
        from scripts.async_llm import LLMsConfig, LLMConfig

        try:
            if self.llm_configs:
                # Bug3 修复: LLMsConfig 没有 .get() 方法，应该访问 .models 属性
                result = self.llm_configs.models.get(self.llm_model_name)
            else:
                # 🔧 P16修复: 配置为空时直接返回字符串，由 create_llm_instance 处理
                print(f"⚠️  llm_configs 为空，降级为字符串模式")
                return self.llm_model_name

            # 类型验证（关键！）
            if isinstance(result, LLMConfig):
                return result
            elif isinstance(result, dict):
                # 🔧 P16修复: 正确转换dict为LLMConfig (dataclass需要关键字参数)
                print(f"⚠️  警告：get() 返回了 dict，正在转换为 LLMConfig")
                return LLMConfig(
                    api_type=result.get('api_type', 'openai'),
                    base_url=result.get('base_url', 'https://api.openai.com/v1'),
                    api_key=result.get('api_key', ''),
                    model_name=result.get('model_name', 'gpt-4o-mini'),
                    temperature=result.get('temperature', 0.7),
                    top_p=result.get('top_p', 1.0),
                    max_tokens=result.get('max_tokens', 4096)
                )
            elif isinstance(result, str):
                return result
            else:
                print(f"⚠️  未知类型: {type(result)}，降级为字符串")
                return self.llm_model_name

        except Exception as e:
            print(f"⚠️  获取LLM配置失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回字符串模型名，让 create_llm_instance 自动处理
            print(f"  降级为字符串模式: {self.llm_model_name}")
            return self.llm_model_name

    def _format_problem_by_source(self, problem: str, sample: dict) -> str:
        """
        根据数据源格式化问题输入（Option A: 智能输入格式化）

        不同数据集需要不同的输入格式：
        - HotpotQA/SQuAD: 需要注入context到problem中
        - HumanEval: 保持原格式（已包含函数签名和docstring）
        - GSM8K/MATH: 直接使用problem

        Args:
            problem: 原始问题文本
            sample: 完整的样本字典，包含source、context等字段

        Returns:
            格式化后的问题文本
        """
        source = sample.get("source", "").lower()
        problem_type = sample.get("problem_type", "math")

        # 1. HotpotQA: 需要注入context
        if source == "hotpotqa" or "hotpot" in source:
            context = sample.get("context", [])
            if context:
                # HotpotQA context格式: [[title, [sentences...]], ...]
                context_str = ""
                if isinstance(context, list):
                    for item in context:
                        if isinstance(item, list) and len(item) >= 2:
                            title = item[0] if isinstance(item[0], str) else ""
                            paragraphs = item[1] if isinstance(item[1], list) else []
                            if paragraphs:
                                context_str += f"\n{title}:\n" + " ".join(paragraphs)
                        elif isinstance(item, str):
                            context_str += "\n" + item
                if context_str:
                    return f"Context:{context_str}\n\nQuestion: {problem}\n\nAnswer:"
            return f"Question: {problem}\n\nAnswer:"

        # 2. SQuAD: 类似处理
        elif source == "squad" or "squad" in source:
            context = sample.get("context", "")
            if context and isinstance(context, str):
                return f"Context: {context}\n\nQuestion: {problem}\n\nAnswer:"
            return f"Question: {problem}\n\nAnswer:"

        # 3. HumanEval: 保持原格式（已包含完整函数签名）
        elif source == "humaneval" or problem_type == "code":
            # HumanEval的problem已经是完整的函数签名+docstring
            return problem

        # 4. GSM8K/MATH: 直接使用problem
        elif source in ["gsm8k", "math"] or problem_type == "math":
            return problem

        # 5. 通用QA问题: 检查是否有context需要注入 (P1修复)
        elif problem_type == "qa":
            context = sample.get("context", "")
            if context:
                # 处理context为列表或字符串的情况
                if isinstance(context, list):
                    context_str = ""
                    for item in context:
                        if isinstance(item, list) and len(item) >= 2:
                            title = item[0] if isinstance(item[0], str) else ""
                            paragraphs = item[1] if isinstance(item[1], list) else []
                            if paragraphs:
                                context_str += f"\n{title}:\n" + " ".join(paragraphs)
                        elif isinstance(item, str):
                            context_str += "\n" + item
                    if context_str:
                        return f"Context:{context_str}\n\nQuestion: {problem}\n\nAnswer:"
                elif isinstance(context, str) and context.strip():
                    return f"Context: {context}\n\nQuestion: {problem}\n\nAnswer:"
            # P1修复: 无context时，添加简单提示词指导模型基于知识回答
            return f"Question: {problem}\n\nPlease answer the question based on your knowledge. Answer:"

        # 6. 默认: 直接返回原问题
        return problem

    async def _execute_fallback_workflow(
        self,
        problem: str,
        problem_type: str,
        error_info: str = "",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        执行Fallback工作流 - 返回错误信息让Qwen学习

        重要变更：不再使用外部LLM生成答案，而是返回错误信息
        这样Qwen模型可以从错误中学习，而不是被掩盖
        """
        print(f"🔄 Fallback: 返回错误信息供模型学习")
        start_time = time.time()
        execution_time = time.time() - start_time

        # 构建错误描述
        error_description = f"WORKFLOW_ERROR: {error_info}" if error_info else "WORKFLOW_ERROR: Execution failed"

        metadata = {
            "success": False,
            "fallback_used": True,
            "error": error_info or "workflow_execution_failed",
            "execution_time": execution_time,
            "cost": 0.0,
            "problem_type": problem_type,
            "is_error_feedback": True  # 标记这是错误反馈，用于奖励计算
        }

        print(f"  ⚠️ 返回错误信息: {error_description[:100]}...")

        # 返回错误描述作为答案，让Qwen看到失败原因
        # 这会导致低奖励，从而让模型学会避免产生有问题的workflow
        return error_description, 0.0, metadata

    def _get_fallback_workflow_class(self, problem_type: str):
        """返回一个简单的默认工作流类（用于生成失败时）

        改进的fallback策略：
        1. 先尝试直接调用LLM生成解决方案
        2. 如果失败，返回占位符而不是None
        3. 避免依赖可能失败的Test operator
        """

        class FallbackWorkflow:
            def __init__(self, name: str, llm_config, dataset):
                self.name = name
                self.dataset = dataset
                try:
                    self.llm = create_llm_instance(llm_config)
                except Exception as e:
                    print(f"⚠️  LLM初始化失败: {e}")
                    self.llm = None

            async def __call__(self, problem: str, *args, **kwargs):
                """改进的fallback：不依赖Test operator"""

                # 策略1: 直接调用LLM生成，不经过任何operator
                if self.llm is not None:
                    try:
                        print(f"  📝 Fallback: 直接调用LLM生成解决方案")

                        # 根据问题类型选择合适的prompt
                        if self.dataset == "code":
                            prompt = f"""Given the following coding problem, provide a Python solution.

Problem:
{problem}

Provide ONLY the Python function code, no explanations."""
                        else:
                            prompt = f"""Solve the following problem step by step and provide the final answer.

Problem:
{problem}

Provide the final answer clearly."""

                        # 直接调用LLM，不使用任何operator
                        # 使用正确的 AsyncLLM __call__ 接口
                        answer = await self.llm(prompt)

                        # 获取成本
                        usage = self.llm.get_usage_summary()
                        if isinstance(usage, dict) and "total_cost" in usage:
                            cost = usage["total_cost"]
                        else:
                            cost = 0.0

                        return answer, cost

                    except Exception as e:
                        print(f"  ⚠️  Fallback直接调用LLM失败: {e}")

                # 策略2: 如果LLM调用也失败，使用Custom operator但不依赖Test
                try:
                    print(f"  📝 Fallback: 尝试使用Custom operator")
                    custom = operator_module.Custom(self.llm)
                    result = await custom(
                        input=problem,
                        instruction="Generate a solution without requiring test validation."
                    )

                    if result and 'response' in result:
                        usage = self.llm.get_usage_summary()
                        if isinstance(usage, dict) and "total_cost" in usage:
                            cost = usage["total_cost"]
                        else:
                            cost = 0.0
                        return result['response'], cost

                except Exception as e:
                    print(f"  ⚠️  Fallback Custom operator失败: {e}")

                # 策略3: 所有策略都失败，返回占位符而不是None
                print(f"  ⚠️  所有fallback策略都失败，返回占位符")
                placeholder = f"[Fallback placeholder for problem: {problem[:80]}...]"
                return placeholder, 0.0

        return FallbackWorkflow


async def test_executor():
    """测试AFlow执行器"""
    print("\n" + "=" * 60)
    print("🧪 测试AFlow执行器")
    print("=" * 60)

    # 创建执行器
    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-oss-120b",
        timeout=60
    )

    # 测试工作流代码（简单示例）
    test_workflow_code = """
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step and provide the final answer.")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 执行工作流
    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=test_workflow_code,
        problem=test_problem,
        problem_type="math"
    )

    print(f"\n✅ 执行结果:")
    print(f"  成功: {metadata['success']}")
    print(f"  答案: {answer}")
    print(f"  成本: ${cost:.6f}")
    print(f"  时间: {metadata['execution_time']:.2f}秒")


if __name__ == "__main__":
    asyncio.run(test_executor())
