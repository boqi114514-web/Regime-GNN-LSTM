# -*- coding: utf-8 -*-
"""FileNotifier：把 markdown 写到 reports/latest.md，并按 ISO 周归档。"""
import os
import shutil
from datetime import datetime
from typing import List, Optional

from .base import Notifier
from config import REPORTS_DIR


class FileNotifier(Notifier):
    """本地文件通道。

    - `latest.md` 永远是最新一次
    - `YYYY-Www.md` 按 ISO 周归档
    - 附图/附件若有会拷贝到 `reports/attachments/YYYY-Www/`
    """

    def __init__(self, archive: bool = True):
        self.archive = archive

    def send_report(
        self,
        markdown: str,
        image_paths: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> bool:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        latest_path = os.path.join(REPORTS_DIR, 'latest.md')
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        if self.archive:
            now = datetime.now()
            iso_year, iso_week, _ = now.isocalendar()
            week_tag = f'{iso_year}-W{iso_week:02d}'
            arch_path = os.path.join(REPORTS_DIR, f'{week_tag}.md')
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            # 附图/附件归档
            if image_paths or file_paths:
                att_dir = os.path.join(REPORTS_DIR, 'attachments', week_tag)
                os.makedirs(att_dir, exist_ok=True)
                for p in (image_paths or []) + (file_paths or []):
                    if os.path.exists(p):
                        shutil.copy2(p, os.path.join(att_dir, os.path.basename(p)))

        print(f"[notifier/file] 写入 {latest_path}")
        return True
