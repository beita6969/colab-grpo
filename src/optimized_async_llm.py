#!/usr/bin/env python3
"""
优化版异步LLM客户端 - Plan A + Plan B 实现

Plan A: httpx连接池优化
- 自定义httpx客户端配置
- 增加max_connections到50
- 保持长连接复用

Plan B: 批量API支持
- 并发执行多个prompt
- 使用asyncio.gather和信号量控制并发
"""
import asyncio
from typing import List, Optional, Dict, Any, Tuple
import httpx
from openai import AsyncOpenAI

# 全局连接池配置
_GLOBAL_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_GLOBAL_SEMAPHORE: Optional[asyncio.Semaphore] = None


def get_optimized_http_client(max_connections: int = 50) -> httpx.AsyncClient:
    """
    获取优化的httpx客户端（单例模式，避免重复创建）

    Plan A 核心：配置高并发连接池

    Args:
        max_connections: 最大连接数（默认50，vLLM通常能处理）

    Returns:
        配置好的httpx.AsyncClient
    """
    global _GLOBAL_HTTP_CLIENT

    if _GLOBAL_HTTP_CLIENT is None:
        # 创建带连接池的httpx客户端
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=30.0  # 30秒保活
        )

        timeout = httpx.Timeout(
            connect=10.0,      # 连接超时
            read=300.0,        # 读取超时（LLM生成可能较长）
            write=30.0,        # 写入超时
            pool=10.0          # 等待连接池超时
        )

        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,  # 启用HTTP/2提升并发性能
        )

        print(f"✅ 创建优化的HTTP连接池: max_connections={max_connections}")

    return _GLOBAL_HTTP_CLIENT


def get_concurrency_semaphore(max_concurrent: int = 20) -> asyncio.Semaphore:
    """
    获取并发控制信号量

    Args:
        max_concurrent: 最大并发请求数

    Returns:
        asyncio.Semaphore
    """
    global _GLOBAL_SEMAPHORE

    if _GLOBAL_SEMAPHORE is None:
        _GLOBAL_SEMAPHORE = asyncio.Semaphore(max_concurrent)
        print(f"✅ 创建并发控制信号量: max_concurrent={max_concurrent}")

    return _GLOBAL_SEMAPHORE


class OptimizedAsyncLLM:
    """
    优化版异步LLM客户端

    特性:
    - Plan A: httpx连接池，支持高并发HTTP请求
    - Plan B: 批量API，一次性处理多个prompt
    - 兼容原AsyncLLM接口
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_connections: int = 50,
        max_concurrent: int = 20,
        system_msg: Optional[str] = None
    ):
        """
        初始化优化版LLM客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 生成温度
            top_p: Top-p采样
            max_connections: 最大HTTP连接数 (Plan A)
            max_concurrent: 最大并发请求数 (Plan B)
            system_msg: 系统消息
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.system_msg = system_msg

        # Plan A: 使用优化的httpx客户端
        http_client = get_optimized_http_client(max_connections)

        # 创建AsyncOpenAI客户端，注入自定义httpx客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=2,
        )

        # Plan B: 并发控制
        self.semaphore = get_concurrency_semaphore(max_concurrent)

        # Token统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    async def __call__(self, prompt: str) -> str:
        """
        单个prompt调用（兼容原接口）

        Args:
            prompt: 用户提示词

        Returns:
            LLM响应文本
        """
        async with self.semaphore:  # 并发控制
            messages = []
            if self.system_msg:
                messages.append({"role": "system", "content": self.system_msg})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # 统计token
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            self.total_calls += 1

            return response.choices[0].message.content

    async def aask(self, msg: str, system_msgs: list = None) -> str:
        """兼容MetaGPT风格的aask接口"""
        original_sys_msg = self.system_msg
        if system_msgs:
            self.system_msg = system_msgs[0] if isinstance(system_msgs, list) else system_msgs

        try:
            return await self.__call__(msg)
        finally:
            self.system_msg = original_sys_msg

    async def batch_call(
        self,
        prompts: List[str],
        return_exceptions: bool = True
    ) -> List[Tuple[bool, Any]]:
        """
        Plan B: 批量调用多个prompt

        使用asyncio.gather并发执行，大幅提升吞吐量

        Args:
            prompts: 多个prompt列表
            return_exceptions: 是否返回异常而不是抛出

        Returns:
            List of (success: bool, result_or_error: Any)
        """
        async def safe_call(prompt: str) -> Tuple[bool, Any]:
            try:
                result = await self.__call__(prompt)
                return (True, result)
            except Exception as e:
                if return_exceptions:
                    return (False, str(e))
                raise

        # 并发执行所有prompts
        results = await asyncio.gather(
            *[safe_call(p) for p in prompts],
            return_exceptions=return_exceptions
        )

        # 处理gather返回的异常对象
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append((False, str(r)))
            else:
                processed.append(r)

        return processed

    async def batch_call_with_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        return_exceptions: bool = True
    ) -> List[Tuple[bool, Any]]:
        """
        Plan B扩展: 批量调用完整消息格式

        Args:
            messages_list: 多个消息列表
            return_exceptions: 是否返回异常

        Returns:
            List of (success, result_or_error)
        """
        async def call_with_messages(messages: List[Dict[str, str]]) -> Tuple[bool, Any]:
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )

                    if response.usage:
                        self.total_input_tokens += response.usage.prompt_tokens
                        self.total_output_tokens += response.usage.completion_tokens
                    self.total_calls += 1

                    return (True, response.choices[0].message.content)
            except Exception as e:
                if return_exceptions:
                    return (False, str(e))
                raise

        results = await asyncio.gather(
            *[call_with_messages(m) for m in messages_list],
            return_exceptions=return_exceptions
        )

        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append((False, str(r)))
            else:
                processed.append(r)

        return processed

    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_cost": 0.0  # vLLM本地部署无成本
        }


def create_optimized_llm_instance(
    config: Dict[str, Any],
    max_connections: int = 50,
    max_concurrent: int = 20
) -> OptimizedAsyncLLM:
    """
    创建优化版LLM实例的工厂函数

    Args:
        config: LLM配置字典，包含:
            - api_key: API密钥
            - base_url: API基础URL
            - model_name: 模型名称
            - temperature: 生成温度 (可选)
            - top_p: Top-p采样 (可选)
        max_connections: Plan A - 最大HTTP连接数
        max_concurrent: Plan B - 最大并发请求数

    Returns:
        OptimizedAsyncLLM实例
    """
    return OptimizedAsyncLLM(
        api_key=config.get("api_key", "dummy"),
        base_url=config.get("base_url", "http://localhost:8002/v1"),
        model=config.get("model_name", config.get("model", "gpt-oss-120b")),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 1.0),
        max_connections=max_connections,
        max_concurrent=max_concurrent,
    )


async def cleanup_global_resources():
    """清理全局资源（程序结束时调用）"""
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is not None:
        await _GLOBAL_HTTP_CLIENT.aclose()
        _GLOBAL_HTTP_CLIENT = None
        print("✅ 清理HTTP连接池完成")


# ============================================================
# 测试代码
# ============================================================

async def test_optimized_llm():
    """测试优化版LLM客户端"""
    print("\n" + "=" * 60)
    print("🧪 测试优化版LLM客户端")
    print("=" * 60)

    # 创建实例
    config = {
        "api_key": "dummy",
        "base_url": "http://localhost:8002/v1",
        "model_name": "/home/yijia/lhy/openai/gpt-oss-120b",
        "temperature": 0.7,
    }

    llm = create_optimized_llm_instance(
        config,
        max_connections=50,
        max_concurrent=20
    )

    # 测试1: 单个调用
    print("\n📝 测试单个调用...")
    try:
        result = await llm("What is 2 + 2?")
        print(f"  结果: {result[:100]}...")
    except Exception as e:
        print(f"  ❌ 错误: {e}")

    # 测试2: 批量调用
    print("\n📝 测试批量调用 (3个prompt)...")
    prompts = [
        "What is 1 + 1?",
        "What is the capital of France?",
        "Write a haiku about coding."
    ]

    import time
    start = time.time()
    try:
        results = await llm.batch_call(prompts)
        elapsed = time.time() - start

        print(f"  耗时: {elapsed:.2f}秒")
        for i, (success, result) in enumerate(results):
            status = "✅" if success else "❌"
            preview = str(result)[:50] if result else "N/A"
            print(f"  {status} Prompt {i+1}: {preview}...")
    except Exception as e:
        print(f"  ❌ 批量调用错误: {e}")

    # 统计
    print(f"\n📊 使用统计:")
    usage = llm.get_usage_summary()
    print(f"  总调用: {usage['total_calls']}")
    print(f"  总Token: {usage['total_tokens']}")

    # 清理
    await cleanup_global_resources()


if __name__ == "__main__":
    asyncio.run(test_optimized_llm())
