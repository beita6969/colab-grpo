#!/usr/bin/env python3
"""
训练入口 - 启动GRPO训练
"""
import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime

# P12修复: 禁用代理，确保LLM Judge可以直连localhost:8002
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 添加src到路径
sys.path.insert(0, 'src')


def setup_logging(log_dir: str = "logs"):
    """设置日志系统 - 同时输出到文件和终端"""
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除已有handlers
    root_logger.handlers.clear()

    # 文件handler - 记录所有日志
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 终端handler - 同时输出到终端
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 创建latest符号链接
    latest_link = os.path.join(log_dir, "latest_training.log")
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    elif os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.basename(log_file), latest_link)

    print(f"\n📝 日志文件: {log_file}")
    print(f"📝 最新日志链接: {latest_link}\n")

    return log_file


class TeeOutput:
    """同时输出到多个流的类"""
    def __init__(self, *streams):
        self.streams = streams
        self._original = streams[0] if streams else sys.stdout

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        """检查是否是终端"""
        return hasattr(self._original, 'isatty') and self._original.isatty()

    def fileno(self):
        """返回文件描述符"""
        return self._original.fileno() if hasattr(self._original, 'fileno') else -1


from grpo_trainer import GRPOTrainer


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AFlow + ROLL GRPO训练")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='日志目录'
    )
    args = parser.parse_args()

    # 设置日志系统
    log_file = setup_logging(args.log_dir)

    # 打开日志文件用于Tee输出
    log_file_handle = open(log_file, 'a', encoding='utf-8')

    # 重定向stdout和stderr到同时输出到终端和文件
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file_handle)
    sys.stderr = TeeOutput(original_stderr, log_file_handle)

    try:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     AFlow + ROLL 深度融合 - GRPO在线学习                    ║
║                                                              ║
║     基于Qwen2.5-7B的工作流优化                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # 创建训练器
        trainer = GRPOTrainer(config_path=args.config)

        # 开始训练
        await trainer.train()
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        print(f"\n📝 训练日志已保存到: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
