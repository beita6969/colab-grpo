#!/usr/bin/env python3
"""
Response Standardizer - 统一所有算子的返回值格式
"""
from typing import Dict, Any, Optional


class ResponseStandardizer:
    """
    强制标准化所有算子的返回值

    标准格式:
    {
        'success': bool,
        'content': str,  # 主要内容
        'metadata': dict,  # 额外信息
        'error': Optional[str]
    }
    """

    @staticmethod
    def standardize(raw_response: Any, operator_type: str) -> Dict[str, Any]:
        """
        标准化算子返回值

        Args:
            raw_response: 算子的原始返回值
            operator_type: 算子类型

        Returns:
            标准化后的字典
        """
        # 如果是None或空，返回失败
        if raw_response is None:
            return {
                'success': False,
                'content': '',
                'metadata': {},
                'error': 'Operator returned None'
            }

        # 如果已经是字符串，直接包装
        if isinstance(raw_response, str):
            return {
                'success': True,
                'content': raw_response,
                'metadata': {},
                'error': None
            }

        # 如果不是字典，转换为字符串
        if not isinstance(raw_response, dict):
            return {
                'success': True,
                'content': str(raw_response),
                'metadata': {},
                'error': None
            }

        # 根据算子类型标准化
        if operator_type == 'Custom':
            return ResponseStandardizer._standardize_custom(raw_response)
        elif operator_type == 'AnswerGenerate':
            return ResponseStandardizer._standardize_answer_generate(raw_response)
        elif operator_type == 'Programmer':
            return ResponseStandardizer._standardize_programmer(raw_response)
        elif operator_type == 'Test':
            return ResponseStandardizer._standardize_test(raw_response)
        elif operator_type == 'Review':
            return ResponseStandardizer._standardize_review(raw_response)
        elif operator_type == 'Revise':
            return ResponseStandardizer._standardize_revise(raw_response)
        elif operator_type == 'ScEnsemble':
            return ResponseStandardizer._standardize_ensemble(raw_response)
        elif operator_type == 'Format':
            return ResponseStandardizer._standardize_format(raw_response)
        else:
            # 未知类型，尝试通用处理
            return ResponseStandardizer._standardize_generic(raw_response)

    @staticmethod
    def _standardize_custom(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('answer', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_answer_generate(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('answer', ''),
            'metadata': {
                'thought': resp.get('thought', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_programmer(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('output', ''),
            'metadata': {
                'code': resp.get('code', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_test(resp: Dict) -> Dict:
        return {
            'success': resp.get('result', False),
            'content': resp.get('solution', ''),
            'metadata': {
                'test_result': resp.get('result', False),
                'original': resp
            },
            'error': None if resp.get('result', False) else 'Test failed'
        }

    @staticmethod
    def _standardize_review(resp: Dict) -> Dict:
        # 处理多种可能的返回格式
        feedback = (
            resp.get('feedback') or
            resp.get('review_result') or
            resp.get('review') or
            'Review completed'
        )

        review_result = resp.get('review_result', True)
        if isinstance(review_result, str):
            review_result = 'pass' in review_result.lower() or 'success' in review_result.lower()

        return {
            'success': True,
            'content': feedback,
            'metadata': {
                'review_result': review_result,
                'feedback': feedback,
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_revise(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('solution', resp.get('code', '')),
            'metadata': {
                'solution': resp.get('solution', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_ensemble(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('solution', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_format(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('formatted', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_generic(resp: Dict) -> Dict:
        """通用标准化 - 尝试找到最可能的内容字段"""
        # 尝试常见字段名
        # 🔧 P30修复: 将 'output' 移到 'code' 之前
        # 对于Programmer算子，'output' 是执行结果，'code' 是源代码
        # Math/QA任务应该优先使用执行结果而非源代码
        content_fields = ['response', 'answer', 'solution', 'output', 'result', 'code']
        content = ''

        for field in content_fields:
            if field in resp:
                content = resp[field]
                break

        # 如果都没有，转换整个字典为字符串
        if not content:
            content = str(resp)

        return {
            'success': True,
            'content': content,
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def safe_get(data: Dict, *keys, default='') -> Any:
        """
        安全地从嵌套字典获取值

        Args:
            data: 源字典
            *keys: 按优先级排列的键
            default: 默认值

        Returns:
            找到的第一个非空值，或默认值
        """
        for key in keys:
            value = data.get(key)
            if value is not None and value != '':
                return value
        return default
