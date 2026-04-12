# -*- coding: utf-8 -*-
"""通知抽象层：统一接口 send_report(markdown, image_paths?, file_paths?)

使用：
    from notifier import get_notifier
    n = get_notifier()         # 读环境变量 NOTIFIER_TYPE，默认 'file'
    n.send_report(md_content)

新增通道只需在本包内新增 xxx.py 实现 Notifier 接口，并在 get_notifier 注册。
"""
import os

from .base import Notifier
from .noop import NoopNotifier
from .file import FileNotifier

__all__ = ['Notifier', 'NoopNotifier', 'FileNotifier', 'get_notifier']


def get_notifier(kind: str = None) -> Notifier:
    """工厂函数。kind 不给则读 env NOTIFIER_TYPE，默认 'file'。"""
    kind = kind or os.environ.get('NOTIFIER_TYPE', 'file')
    kind = kind.lower().strip()
    if kind == 'noop':
        return NoopNotifier()
    if kind == 'file':
        return FileNotifier()
    raise ValueError(f"未知的 notifier 类型: {kind!r}（支持: noop / file）")
