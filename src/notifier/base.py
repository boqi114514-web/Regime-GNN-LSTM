# -*- coding: utf-8 -*-
"""Notifier 基类：所有通道的统一接口"""
from abc import ABC, abstractmethod
from typing import List, Optional


class Notifier(ABC):
    """发送周报/错误日志的抽象通道。

    实现类需实现 send_report；返回 bool 表示是否成功。
    image_paths 与 file_paths 为预留参数，给未来 Telegram/邮件等通道用。
    """

    @abstractmethod
    def send_report(
        self,
        markdown: str,
        image_paths: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> bool:
        raise NotImplementedError
