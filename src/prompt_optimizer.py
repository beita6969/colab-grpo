#!/usr/bin/env python3
"""
提示词优化器 - Layer 1: Workflow生成提示词动态优化
"""
from typing import Dict, List, Optional


class PromptOptimizer:
    """
    动态提示词优化器

    功能：
    1. 提供完整7个operator模板
    2. Few-shot示例学习（从experience_buffer检索）
    3. 问题类型自适应指导
    4. 动态组合生成最优提示词
    """

    def __init__(self, experience_buffer=None):
        """
        初始化提示词优化器

        Args:
            experience_buffer: ExperienceBuffer实例，用于few-shot学习
        """
        self.experience_buffer = experience_buffer
        self.operator_templates = self._load_operator_templates()
        self.type_guidance = self._load_type_guidance()

    def build_dynamic_prompt(
        self,
        problem: str,
        problem_type: str,
        use_few_shot: bool = False,
        few_shot_k: int = 2,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        构建动态优化的提示词
        
        Args:
            problem: 问题文本
            problem_type: 问题类型 (math/code/qa)
            use_few_shot: 是否使用few-shot示例
            few_shot_k: few-shot示例数量
            similarity_threshold: 相似度阈值
            
        Returns:
            优化后的完整提示词
        """
        # 1. 基础模板（完整7算子）
        base_template = self._get_full_operator_template()

        # 2. Few-shot示例（可选）
        few_shot_section = ""
        if use_few_shot and self.experience_buffer is not None:
            few_shot_examples = self.experience_buffer.retrieve_top_k(
                problem=problem,
                problem_type=problem_type,
                k=few_shot_k,
                similarity_threshold=similarity_threshold
            )
            if len(few_shot_examples) > 0:
                few_shot_section = self._format_few_shot_examples(few_shot_examples)

        # 3. 类型自适应指导
        # 🔧 FIX: 确保正确路由problem_type，不要让Code任务使用Math/QA的\boxed{}格式
        type_guidance_section = self.type_guidance.get(
            problem_type,
            ""  # 默认使用空字符串，避免错误路由
        )
        # 如果问题类型不在预定义中，使用通用指导
        if not type_guidance_section and problem_type not in self.type_guidance:
            type_guidance_section = f"""✅ {problem_type.upper()} Problem: Provide accurate solution based on problem requirements."""

        # 4. 根据问题类型确定workflow签名
        if problem_type == "code":
            call_signature = "async def __call__(self, problem: str, entry_point: str, test: str):"
            call_comment = """# Solve: {problem}
        # entry_point: The function name to test (HumanEval format)
        # test: The test code containing check() function (HumanEval format)
        # MUST return (solution, cost) tuple
        # Example: return code, self.llm.get_usage_summary()["total_cost"]"""
        else:
            call_signature = "async def __call__(self, problem: str):"
            call_comment = """# Solve: {problem}
        # MUST return (solution, cost) tuple
        # Safe access: return solution.get('response', ''), self.llm.get_usage_summary().get("total_cost", 0.0)"""

        # 4. 组合提示词 - 🔧 修复: 添加XML格式要求
        prompt = f"""Generate a Python Workflow class to solve the problem.

## 🚨 CRITICAL: OUTPUT FORMAT
You MUST output your workflow in XML format with <graph> and <prompt> tags:

<workflow>
<graph>
[Your Python Workflow class code here]
</graph>
<prompt>
[Your custom TASK_PROMPT here]
</prompt>
</workflow>

DO NOT:
- Directly answer the problem
- Output explanations without the XML tags
- Skip the <graph> or <prompt> sections

IMPORTANT: First, ANALYZE the problem's difficulty and complexity.
- Simple problems -> Use direct operators (AnswerGenerate, Programmer).
- Complex problems -> Use robust workflows (Review, Revise, ScEnsemble).
- YOU decide the best strategy. Do not over-engineer simple tasks.

🚨 CRITICAL RULES FOR OPERATOR INITIALIZATION AND CALLS:

1️⃣ OPERATOR CLASS NAMES (PascalCase - VERY IMPORTANT):
   ✅ CORRECT: self.custom = operator.Custom(self.llm)
   ✅ CORRECT: self.answer_generate = operator.AnswerGenerate(self.llm)
   ✅ CORRECT: self.test = operator.Test(self.llm)
   ❌ WRONG: self.custom = operator.custom(self.llm)
   ❌ WRONG: self.answer_generate = operator.answer_generate(self.llm)

98|⚡ PERFORMANCE CRITICAL - AVOID REDUNDANT CALLS:
   ✅ CORRECT: Cache operator results and reuse them
   result = await self.answer_generate(input=problem)
   answer = result.get('answer', '')
   # Use 'answer' variable multiple times - DO NOT call again!

   ❌ WRONG: Calling the same operator multiple times with same inputs
   result1 = await self.answer_generate(input=problem)  # First call
   # ... some code ...
   result2 = await self.answer_generate(input=problem)  # Redundant! Wastes time!

2️⃣ OPERATOR CALL RULES:
   ├─ EVERY operator call MUST include ALL required parameters
   ├─ DO NOT skip any required parameters - this will cause TypeError
   ├─ Use the exact parameter names shown in the Interface
   └─ Follow the Example call format exactly

3️⃣ Example INCORRECT calls (WILL FAIL):
   ❌ await self.test(problem=problem)  # Missing solution and entry_point!
   ❌ await self.review(solution=code)  # Missing problem!
   ❌ self.custom = operator.custom(self.llm)  # Wrong case!

4️⃣ Example CORRECT calls (WILL WORK):
   ✅ self.custom = operator.Custom(self.llm)  # Correct case!
   ✅ await self.test(problem=problem, solution=solution, entry_point=entry_point)
   ✅ await self.review(problem=problem, solution=code)

5️⃣ 🔴🔴🔴 VARIABLE INITIALIZATION - MANDATORY FIRST LINE IN __call__ 🔴🔴🔴
   THIS IS THE #1 CAUSE OF WORKFLOW FAILURES! READ CAREFULLY!

   RULE: At the START of __call__, you MUST initialize ALL result variables:

   async def __call__(self, problem, ...):
       # ⬇️⬇️⬇️ MANDATORY: Initialize ALL variables FIRST ⬇️⬇️⬇️
       solution = ""           # For math/qa problems
       code = ""               # For code problems
       revised_code = code     # If using Revise operator
       final_answer = ""       # If storing final result
       review_result = None    # If using Review operator
       # ⬆️⬆️⬆️ Add ALL variables you will use BEFORE any logic ⬆️⬆️⬆️

       # Now your actual logic...
       result = await self.answer_generate(input=problem)
       solution = result.get('answer', solution)  # Use initialized default

       if needs_revision:  # This is now SAFE
           revised = await self.revise(...)
           revised_code = revised.get('solution', revised_code)  # Safe fallback

       return revised_code, cost  # ALWAYS works, never UnboundLocalError!

   ❌ FATAL ERROR - Will crash:
   async def __call__(self, problem):
       if condition:  # If False, revised_code is NEVER defined!
           revised_code = await self.revise(...)
       return revised_code  # 💥 UnboundLocalError!

   ✅ CORRECT PATTERN - Always works:
   async def __call__(self, problem):
       revised_code = ""  # 👈 INITIALIZE FIRST!
       if condition:
           revised_code = await self.revise(...)
       return revised_code  # ✅ Always defined

{base_template}

{type_guidance_section}

{few_shot_section}

Template (complete the __call__ method):

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators you need (ONLY the ones you will use):
        # self.custom = operator.Custom(self.llm)
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.sc_ensemble = operator.ScEnsemble(self.llm)
        # self.test = operator.Test(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)

    {call_signature}
        {call_comment}
        pass
"""

        return prompt

    def _get_full_operator_template(self) -> str:
        """
        返回完整operator列表定义（基于operator_templates）

        Returns:
            完整operator列表说明
        """
        lines = ["Available Operators (use intelligently based on problem type):\n"]

        for i, (op_name, op_info) in enumerate(self.operator_templates.items(), 1):
            lines.append(f"{i}. {op_name}")
            lines.append(f"   Description: {op_info['description']}")
            lines.append(f"   Interface: {op_info['interface']}")
            lines.append(f"   Returns: {op_info['returns']}")
            if 'required_params' in op_info:
                lines.append(f"   ⚠️ REQUIRED params: {', '.join(op_info['required_params'])}")
            if 'example_call' in op_info:
                lines.append(f"   Example call: {op_info['example_call']}")
            if 'note' in op_info:
                lines.append(f"   Note: {op_info['note']}")
            lines.append("")

        return "\n".join(lines)

    def _load_operator_templates(self) -> Dict:
        """
        加载operator模板定义
        
        Returns:
            operator模板字典
        """
        return {
            "Custom": {
                "description": "Most flexible, for any custom task",
                "interface": "Custom(input: str, instruction: str)",
                "returns": "{'response': str}"
            },
            "AnswerGenerate": {
                "description": "Step-by-step reasoning. Best for standard logical problems.",
                "interface": "AnswerGenerate(input: str)",
                "returns": "{'thought': str, 'answer': str}",
                "example_call": "ans_result = await self.answer_generate(input=problem)\nanswer = ans_result.get('answer', '')  # Extract 'answer' not 'thought'",
                "required_params": ["input"],
                "note": "Returns dict with 'thought' (reasoning) and 'answer' (final result) - use 'answer' for final output"
            },
            "Programmer": {
                "description": "Auto-generate and execute Python code. Essential for CODE problems and calculation-heavy tasks.",
                "interface": "Programmer(problem: str, analysis: str)",
                "returns": "{'code': str (SOURCE CODE - for Code problems only), 'output': str (EXECUTION RESULT - ALWAYS USE for Math/QA!)}",
                "example_call": "prog_result = await self.programmer(problem=problem, analysis='Analyze and solve')\n# ⚠️ CRITICAL: Use 'output' for MATH/QA answer, NOT 'code'!\nexecuted_answer = prog_result.get('output', '')  # THE COMPUTED RESULT: '42', '3.14'\n# Only use 'code' for Code problems where you need source code",
                "note": "Returns 'output' (execution result) and 'code' (source). For MATH/QA: ALWAYS use output! For CODE: use code."
            },
            "ScEnsemble": {
                "description": "Self-consistency ensemble. Use for complex reasoning where single attempt is unreliable.",
                "interface": "ScEnsemble(solutions: List[str], problem: str)",
                "returns": "{'response': str}"
            },
            "Test": {
                "description": "Test the solution with test cases. CRITICAL for CODE problems. DO NOT use for QA.",
                "interface": "Test(problem: str, solution: str, entry_point: str)",
                "returns": "{'result': bool, 'solution': str}",
                "note": "For HumanEval format - automatically extracts test cases using entry_point",
                "example_call": "result = await self.test(problem=problem, solution=solution, entry_point=entry_point)",
                "required_params": ["problem", "solution", "entry_point"]
            },
            "Review": {
                "description": "Review and verify solution. Use to check quality or catch errors in complex tasks.",
                "interface": "Review(problem: str, solution: str)",
                "returns": "{'review_result': str, 'feedback': str}",
                "example_call": "review_result = await self.review(problem=problem, solution=code)\nfeedback = review_result.get('feedback', review_result.get('review_result', 'No feedback'))  # Handle multiple formats",
                "required_params": ["problem", "solution"],
                "note": "May return 'feedback' OR 'review_result' key - use nested .get() for safety"
            },
            "Revise": {
                "description": "Revise based on feedback. Use AFTER Review to fix issues.",
                "interface": "Revise(problem: str, solution: str, feedback: str)",
                "returns": "{'solution': str}",
                "example_call": "revised = await self.revise(problem=problem, solution=code, feedback=feedback)\nrevised_code = revised.get('solution', code)  # Extract 'solution' with fallback",
                "required_params": ["problem", "solution", "feedback"],
                "note": "Returns dict with 'solution' key - always use .get() to avoid KeyError"
            },
            "MdEnsemble": {
                "description": "Majority voting ensemble (more robust than ScEnsemble). Shuffles and votes multiple times.",
                "interface": "MdEnsemble(solutions: List[str], problem: str)",
                "returns": "{'solution': str}",
                "example_call": "result = await self.md_ensemble(solutions=[sol1, sol2, sol3], problem=problem)\nbest = result.get('solution', '')",
                "required_params": ["solutions", "problem"],
                "note": "Better for noisy/uncertain problems - uses multiple voting rounds"
            },
            "Decompose": {
                "description": "Break complex problem into simpler sub-problems. Best for multi-step reasoning.",
                "interface": "Decompose(problem: str)",
                "returns": "{'subproblems': List[str], 'reasoning': str, 'count': int}",
                "example_call": "result = await self.decompose(problem=problem)\nsubproblems = result.get('subproblems', [problem])",
                "required_params": ["problem"],
                "note": "Useful for complex math or multi-part questions"
            },
            "Verify": {
                "description": "Verify answer correctness and provide corrections if invalid.",
                "interface": "Verify(problem: str, answer: str)",
                "returns": "{'is_valid': bool, 'issues': List[str], 'corrected_answer': str}",
                "example_call": "result = await self.verify(problem=problem, answer=solution)\nif not result.get('is_valid', True):\n    solution = result.get('corrected_answer', solution)",
                "required_params": ["problem", "answer"],
                "note": "Double-check answers before returning - catches calculation errors"
            }
        }

    def _load_type_guidance(self) -> Dict:
        """
        加载问题类型自适应指导
        
        Returns:
            类型指导字典
        """
        return {
            "math": """
✅ MATH Problem Requirements:
- Return final answer in \\boxed{} notation
- STRATEGY:
  * For standard calculations -> Use Programmer
  * For logical reasoning -> Use AnswerGenerate
  * For complex/ambiguous problems -> Use Review/Revise
- IMPORTANT: If using Test operator, MUST call with ALL parameters:
    result = await self.test(problem=problem, solution=solution, entry_point="solve")

📝 RECOMMENDED PATTERN (for complex problems):
1. Initial solution
   ans = await self.answer_generate(input=problem)
   solution = ans.get('answer', '')
2. Review once
   review = await self.review(problem=problem, solution=solution)
   feedback = review.get('feedback', review.get('review_result', ''))
3. If feedback indicates error -> Revise once
   if feedback:
       revised = await self.revise(problem=problem, solution=solution, feedback=feedback)
       solution = revised.get('solution', solution)
4. Return final answer in \\boxed{<answer>} format (string)
""",
            "code": """
🚨🚨🚨 CRITICAL - CODE WORKFLOW SIGNATURE 🚨🚨🚨
Your __call__ method MUST accept exactly these parameters:
  async def __call__(self, problem: str, entry_point: str, test: str):

  - problem: The problem description
  - entry_point: Function name to test (e.g., "has_close_elements")
  - test: HumanEval test code containing check() function

⚠️⚠️⚠️ TEST OPERATOR CRITICAL REQUIREMENTS ⚠️⚠️⚠️
The Test operator MUST be called with ALL 3 PARAMETERS or it will fail with TypeError:

✅ CORRECT (WILL WORK):
  result = await self.test(problem=problem, solution=code, entry_point=entry_point)

❌ WRONG (WILL FAIL WITH TypeError):
  await self.test(problem=problem)  # Missing solution and entry_point!
  await self.test(solution=code)    # Missing problem and entry_point!
  await self.test(problem=problem, solution=code)  # Missing entry_point!

📝 RESPONSE FORMAT - USE .get() FOR SAFETY:
```python
# ✅ ALWAYS use .get() with defaults to avoid KeyError
result = await self.some_operator(...)

# Safe access patterns:
content = result.get('response', result.get('answer', ''))
code = result.get('code', '')
feedback = result.get('feedback', result.get('review_result', 'No feedback'))
success = result.get('success', result.get('result', False))

# For nested access:
standardized = result.get('__standardized__', {})
if standardized:
    content = standardized.get('content', '')
    success = standardized.get('success', False)
```
```python
async def __call__(self, problem: str, entry_point: str, test: str):
    # Step 1: Generate code ONCE - cache result
    prog_result = await self.programmer(problem=problem, analysis="Analyze and solve")
    code = prog_result.get('code', '')

    # Step 2: Test with ALL 3 PARAMETERS
    test_result = await self.test(problem=problem, solution=code, entry_point=entry_point)

    # Step 3: If failed, review ONCE and revise ONCE
    if not test_result.get('result', False):
        review_result = await self.review(problem=problem, solution=code)
        feedback = review_result.get('feedback', review_result.get('review_result', 'Review completed'))
        
        revised = await self.revise(problem=problem, solution=code, feedback=feedback)
        final_code = revised.get('solution', code)
        
        # Optional: Test revised code (remove if time is critical)
        # final_test = await self.test(problem=problem, solution=final_code, entry_point=entry_point)
        
        return final_code, self.llm.get_usage_summary()["total_cost"]

    return code, self.llm.get_usage_summary()["total_cost"]
```

✅ CODE Problem Requirements:
- Your solution MUST pass the test cases
- ALWAYS use Programmer to generate code
- ALWAYS test with Test operator (with ALL 3 parameters)
- Use Review and Revise if tests fail
- NEVER call Test with only problem parameter - it WILL FAIL with TypeError
- Use Programmer to generate code, Test to verify it
- Generated code must be self-contained and executable
""",
            "qa": """
✅ QA Problem Requirements:
- Provide accurate and concise answers
- STRATEGY:
  * For simple questions -> Use AnswerGenerate directly
  * For complex/reasoning questions -> Use AnswerGenerate then Review
- IMPORTANT: DO NOT call Test operator for QA problems (they don't have test cases)
- If using Test operator with custom test, MUST call with ALL parameters:
    result = await self.test(problem=problem, solution=answer, entry_point="execute")
- Choose operators based on problem complexity

📝 RECOMMENDED PATTERN (for complex questions):
1. Initial answer
   ans = await self.answer_generate(input=problem)
   answer = ans.get('answer', '')
2. Review once
   review = await self.review(problem=problem, solution=answer)
   feedback = review.get('feedback', review.get('review_result', ''))
3. If feedback indicates correction -> Revise once
   if feedback:
       revised = await self.revise(problem=problem, solution=answer, feedback=feedback)
       answer = revised.get('solution', answer)
4. Return concise final answer (string only, no code, no \\boxed{})
"""
        }

    def _format_few_shot_examples(self, examples: List[Dict]) -> str:
        """
        格式化few-shot示例

        Args:
            examples: 样本列表

        Returns:
            格式化的few-shot示例文本
        """
        if len(examples) == 0:
            return ""

        few_shot_text = "\n" + "="*70 + "\n"
        few_shot_text += "📚 HIGH-QUALITY WORKFLOW EXAMPLES (Learn from these successful cases!)\n"
        few_shot_text += "="*70 + "\n\n"

        for i, example in enumerate(examples, 1):
            problem = example.get('problem', '')[:150]
            workflow_code = example.get('workflow_code', '')
            reward = example.get('reward', 0)

            # Extract key workflow pattern
            workflow_snippet = self._extract_workflow_snippet(workflow_code)

            few_shot_text += f"Example {i} (Reward: {reward:.1f}/10.0):\n"
            few_shot_text += f"Problem: {problem}...\n"
            few_shot_text += f"Workflow Pattern:\n{workflow_snippet}\n"
            few_shot_text += f"Result: ✅ Correct\n\n"

        few_shot_text += "="*70 + "\n"
        few_shot_text += "💡 Learn from these patterns and adapt them to your problem!\n"
        few_shot_text += "="*70 + "\n\n"

        return few_shot_text

    def _extract_workflow_snippet(self, workflow_code: str) -> str:
        """Extract key workflow pattern from full code"""
        if not workflow_code:
            return "(No workflow code available)"

        lines = workflow_code.split('\n')
        snippet_lines = []
        in_call_method = False

        for line in lines:
            if 'def __call__' in line:
                in_call_method = True
            if in_call_method:
                # Extract operator calls
                if 'await self.' in line or 'return' in line:
                    snippet_lines.append(line.strip())
                if len(snippet_lines) >= 8:  # Limit to 8 key lines
                    break

        if snippet_lines:
            return '\n'.join(snippet_lines)
        return workflow_code[:300] + "..."

    def get_operator_count(self, workflow_code: str) -> Dict[str, int]:
        """
        统计workflow中使用的operator数量

        Args:
            workflow_code: workflow代码

        Returns:
            operator使用计数字典
        """
        operator_count = {name: 0 for name in self.operator_templates.keys()}

        for op_name in self.operator_templates.keys():
            # 检查初始化和调用
            if f'operator.{op_name}' in workflow_code or \
               f'self.{op_name.lower()}' in workflow_code:
                operator_count[op_name] += 1

        return operator_count
