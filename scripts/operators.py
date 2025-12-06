#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Desc: AFlow-compatible operators module
# Provides workflow operators with Formatter abstraction and ProcessPoolExecutor
# 🔧 P42修复: 添加详细调试日志，打印operator输入输出和LLM调用

import asyncio
import concurrent.futures
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import logging

from tenacity import retry, stop_after_attempt, wait_fixed

# 🔧 P42: 配置operator调试日志
OPERATOR_DEBUG = True  # 设置为True启用详细日志
logger = logging.getLogger("operators")
if OPERATOR_DEBUG:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] %(message)s')

from scripts.formatter import (
    BaseFormatter,
    FormatError,
    XmlFormatter,
    TextFormatter,
    CodeFormatter
)
from scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
)
from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
)
from scripts.utils.sanitize import sanitize, DISALLOWED_IMPORTS


class Operator:
    """Base class for all operators with Formatter support"""

    def __init__(self, llm, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Core method to call LLM with optional formatting

        🔧 P42修复: 添加详细调试日志
        """
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)

        # 🔧 P42: 打印LLM输入
        if OPERATOR_DEBUG:
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.debug(f"[{self.name}] LLM输入 (mode={mode}):")
            logger.debug(f"  prompt前500字符: {prompt_preview}")

        try:
            if formatter:
                # Use formatter for structured responses
                response = await self.llm.call_with_format(prompt, formatter)
            else:
                # Direct call without formatting
                response = await self.llm(prompt)

            # 🔧 P42: 打印LLM输出
            if OPERATOR_DEBUG:
                response_str = str(response)
                response_preview = response_str[:500] + "..." if len(response_str) > 500 else response_str
                logger.debug(f"[{self.name}] LLM输出:")
                logger.debug(f"  response前500字符: {response_preview}")

            # Normalize response format
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            if OPERATOR_DEBUG:
                logger.error(f"[{self.name}] FormatError: {str(e)}")
            return {"error": str(e)}

    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None


class Custom(Operator):
    """Custom operator - most flexible, generates anything based on instruction"""

    def __init__(self, llm, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input: str, instruction: str) -> Dict[str, str]:
        # 🔧 P42: 打印operator输入
        if OPERATOR_DEBUG:
            input_preview = input[:200] + "..." if len(input) > 200 else input
            logger.debug(f"[Custom] 调用输入:")
            logger.debug(f"  input: {input_preview}")
            logger.debug(f"  instruction: {instruction[:100] if instruction else 'None'}")

        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")

        # 🔧 P42: 打印operator输出
        if OPERATOR_DEBUG:
            resp_str = str(response.get('response', ''))[:200]
            logger.debug(f"[Custom] 调用输出: {resp_str}")

        return response


class AnswerGenerate(Operator):
    """Generates step-by-step reasoning with thought and final answer"""

    def __init__(self, llm, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Dict[str, str]:
        # 🔧 P42: 打印operator输入
        if OPERATOR_DEBUG:
            input_preview = input[:200] + "..." if len(input) > 200 else input
            logger.debug(f"[AnswerGenerate] 调用输入: {input_preview}")

        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")

        # 🔧 P42: 打印operator输出
        if OPERATOR_DEBUG:
            answer = str(response.get('answer', ''))[:200]
            thought = str(response.get('thought', ''))[:100]
            logger.debug(f"[AnswerGenerate] 调用输出: thought={thought}, answer={answer}")

        return response


class CustomCodeGenerate(Operator):
    """Generates code based on customized input and instruction"""

    def __init__(self, llm, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, entry_point: str, instruction: str) -> Dict[str, str]:
        prompt = instruction + problem
        response = await self._fill_node(
            GenerateOp, prompt, mode="code_fill", function_name=entry_point
        )
        return response


class ScEnsemble(Operator):
    """
    Self-Consistency Ensemble - selects most consistent solution
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    """

    def __init__(self, llm, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str) -> Dict[str, str]:
        if not solutions:
            return {"response": ""}
        if len(solutions) == 1:
            return {"response": solutions[0]}

        # Create answer mapping (A, B, C, ...)
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "A")
        answer = answer.strip().upper()

        # Get first letter if multiple characters
        if len(answer) > 0:
            answer = answer[0]

        if answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}
        return {"response": solutions[0]}  # Fallback to first


def run_code(code: str) -> Tuple[str, str]:
    """Execute Python code in isolated context (called in separate process)"""
    try:
        global_namespace = {}

        # Check for prohibited imports
        for lib in DISALLOWED_IMPORTS:
            if f"import {lib}" in code or f"from {lib}" in code:
                return "Error", f"Prohibited import: {lib}"

        # Execute the code
        exec(code, global_namespace)

        # Look for solve function
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            result = global_namespace["solve"]()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


class Programmer(Operator):
    """
    Programmer operator - generates and executes Python code
    Uses ProcessPoolExecutor for safe code execution
    """

    def __init__(self, llm, name: str = "Programmer"):
        super().__init__(llm, name)
        # Create process pool for isolated code execution
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        """Ensure process pool is cleaned up"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)

    async def exec_code(self, code: str, timeout: int = 30) -> Tuple[str, str]:
        """Execute code asynchronously with timeout"""
        loop = asyncio.get_running_loop()

        try:
            future = loop.run_in_executor(self.process_pool, run_code, code)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            future.cancel()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # Recreate broken pool
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, please retry"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem: str, analysis: str, feedback: str, mode: str):
        """Generate code using LLM"""
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(
            CodeGenerateOp, prompt, mode, function_name="solve"
        )
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None") -> Dict[str, str]:
        """Generate and execute code with retry logic"""
        # 🔧 P42: 打印operator输入
        if OPERATOR_DEBUG:
            problem_preview = problem[:200] + "..." if len(problem) > 200 else problem
            logger.debug(f"[Programmer] 调用输入:")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  analysis: {analysis[:100] if analysis else 'None'}")

        code = None
        output = None
        feedback = ""

        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code") or code_response.get("response")

            if not code:
                if OPERATOR_DEBUG:
                    logger.debug(f"[Programmer] 调用输出: No code generated")
                return {"code": "", "output": "No code generated"}

            status, output = await self.exec_code(code)

            if status == "Success":
                # 🔧 P42: 打印operator输出
                if OPERATOR_DEBUG:
                    output_preview = str(output)[:200]
                    logger.debug(f"[Programmer] 调用输出成功: {output_preview}")
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )

        # 🔧 P42: 打印最终输出
        if OPERATOR_DEBUG:
            logger.debug(f"[Programmer] 调用输出(3次失败): output={str(output)[:200]}")

        return {"code": code, "output": output}


class Test(Operator):
    """Test operator - tests code with test cases and reflects on errors"""

    def __init__(self, llm, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution: str, test_code: str) -> str:
        """Execute solution with test code"""
        try:
            full_code = f"{solution}\n\n{test_code}"
            global_namespace = {}
            exec(full_code, global_namespace)
            return "no error"
        except AssertionError as e:
            return f"AssertionError: {str(e)}"
        except Exception as e:
            return f"ExecutionError: {str(e)}"

    async def __call__(
        self,
        problem: str,
        solution: str,
        entry_point: str,
        test_loop: int = 3
    ) -> Dict[str, Any]:
        """Test solution and reflect/revise if needed"""
        # Simple test - try to execute the solution
        test_code = f"# Testing {entry_point}\nresult = {entry_point}()"

        for _ in range(test_loop):
            result = self.exec_code(solution, test_code)

            if result == "no error":
                return {"result": True, "solution": solution}

            # Reflect and revise
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass=f"executed unsuccessfully, error: {result}",
                test_fail=result,
            )
            response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
            solution = response.get("response", solution)

        # Final check
        result = self.exec_code(solution, test_code)
        return {
            "result": result == "no error",
            "solution": solution
        }


class Format(Operator):
    """Format operator - extracts concise answer from solution"""

    def __init__(self, llm, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, solution: str, mode: str = None) -> Dict[str, str]:
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    """Review operator - reviews solution correctness using critical thinking"""

    def __init__(self, llm, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, solution: str, mode: str = None) -> Dict[str, Any]:
        # 🔧 P42: 打印operator输入
        if OPERATOR_DEBUG:
            problem_preview = problem[:150] + "..." if len(problem) > 150 else problem
            solution_preview = solution[:150] + "..." if len(solution) > 150 else solution
            logger.debug(f"[Review] 调用输入:")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  solution: {solution_preview}")

        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")

        # Handle boolean parsing from XML
        review_result = response.get("review_result", False)
        if isinstance(review_result, str):
            review_result = review_result.lower() in ("true", "yes", "1")

        result = {
            "review_result": review_result,
            "feedback": response.get("feedback", "")
        }

        # 🔧 P42: 打印operator输出
        if OPERATOR_DEBUG:
            logger.debug(f"[Review] 调用输出: result={review_result}, feedback={result['feedback'][:100]}")

        return result


class Revise(Operator):
    """Revise operator - revises solution based on feedback"""

    def __init__(self, llm, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(
        self,
        problem: str,
        solution: str,
        feedback: str,
        mode: str = None
    ) -> Dict[str, str]:
        # 🔧 P42: 打印operator输入
        if OPERATOR_DEBUG:
            problem_preview = problem[:100] + "..." if len(problem) > 100 else problem
            solution_preview = solution[:100] + "..." if len(solution) > 100 else solution
            feedback_preview = feedback[:100] + "..." if len(feedback) > 100 else feedback
            logger.debug(f"[Revise] 调用输入:")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  solution: {solution_preview}")
            logger.debug(f"  feedback: {feedback_preview}")

        prompt = REVISE_PROMPT.format(
            problem=problem,
            solution=solution,
            feedback=feedback
        )
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")

        # 🔧 P42: 打印operator输出
        if OPERATOR_DEBUG:
            sol = str(response.get('solution', response.get('response', '')))[:200]
            logger.debug(f"[Revise] 调用输出: {sol}")

        return response


class MdEnsemble(Operator):
    """
    Majority voting ensemble - shuffles and votes multiple times
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning?
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Shuffle solutions and create mapping"""
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {
            chr(65 + i): solutions.index(sol)
            for i, sol in enumerate(shuffled_solutions)
        }
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None) -> Dict[str, str]:
        if not solutions:
            return {"solution": ""}
        if len(solutions) == 1:
            return {"solution": solutions[0]}

        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if len(answer) > 0:
                answer = answer[0]

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        if not all_responses:
            return {"solution": solutions[0]}

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        return {"solution": solutions[most_frequent_index]}


# Export all operators
__all__ = [
    'Operator',
    'Custom',
    'AnswerGenerate',
    'CustomCodeGenerate',
    'ScEnsemble',
    'Programmer',
    'Test',
    'Format',
    'Review',
    'Revise',
    'MdEnsemble',
    'run_code'
]
