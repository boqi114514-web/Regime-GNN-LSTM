# -*- coding: utf-8 -*-
"""live/：实盘化核心

- state.py      状态持久化（JSON）
- predict.py    加载 current 模型 → 输出最新月份信号
- monitor.py    对比上期 → 生成周报 → 调 notifier
- train.py      quarterly 全量 / monthly 微调入口
- scheduler.py  本地调度器骨架
"""
