# -*- coding: utf-8 -*-
"""NoopNotifier：什么都不发，只打 log。用于调试或禁用通知时。"""
from typing import List, Optional

from .base import Notifier


class NoopNotifier(Notifier):
    def send_report(
        self,
        markdown: str,
        image_paths: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> bool:
        n_img = len(image_paths) if image_paths else 0
        n_file = len(file_paths) if file_paths else 0
        print(f"[notifier/noop] 跳过发送 · md={len(markdown)}字 · imgs={n_img} · files={n_file}")
        return True
