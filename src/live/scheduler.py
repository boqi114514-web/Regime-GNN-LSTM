# -*- coding: utf-8 -*-
"""本地调度器（Phase 5）

任务：
  - 周日 20:00     → weekly_job   update → monitor（推理 + 周报）
  - 每日 16:30     → daily_job    update → 月末/季末判定 → 触发 train

交易日判定：live.trade_cal（tushare trade_cal + 本地缓存）。
重叠处理：季末与月末重合时只跑 quarterly。
异常处理：任务内部的失败写到 reports/error_*.log，不中断主循环。

依赖：
    pip install schedule

运行：
    python -m live.scheduler
    python -m live.scheduler --once weekly    # 立即跑一次 weekly_job 后退出
    python -m live.scheduler --once daily     # 立即跑一次 daily_job 后退出
"""
import argparse
import os
import sys
import time
import traceback
from datetime import date, datetime

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import REPORTS_DIR
from data_pipeline import update as data_update
from live import monitor, train, trade_cal


# ---------- 错误日志 ----------

def _log_error(job_name: str, tb: str) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{job_name}.log'
    path = os.path.join(REPORTS_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tb)
    print(f'[错误] {job_name} 失败，日志 → {path}')


def _safe(job_name: str, fn, *args, **kwargs):
    """隔离单个子步骤的异常：失败时写日志、返回 None，不抛。"""
    try:
        return fn(*args, **kwargs)
    except Exception:
        _log_error(job_name, traceback.format_exc())
        return None


# ---------- 任务定义 ----------

def weekly_job() -> None:
    """周日 20:00：数据增量 → 推理 → 周报"""
    now = datetime.now()
    print(f'\n[{now:%Y-%m-%d %H:%M:%S}] weekly_job 开始')

    # 1. 数据增量（失败不中断，monitor 会用现有数据）
    _safe('weekly_job.update', data_update.run)

    # 2. 推理 + 周报（monitor 内部会调 predict）
    _safe('weekly_job.monitor', monitor.run)

    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] weekly_job 完成')


def daily_job() -> None:
    """每日 16:30：数据增量 → 如果是月末/季末最后一个交易日，触发 train"""
    today = date.today()
    now = datetime.now()
    print(f'\n[{now:%Y-%m-%d %H:%M:%S}] daily_job 检查 {today}')

    # 1. 非交易日直接跳过（连 update 都不跑）
    if not trade_cal.is_trading_day(today):
        print('  → 非交易日，跳过')
        return

    # 2. 训练判定
    is_q_end = trade_cal.is_last_trading_day_of_quarter(today)
    is_m_end = trade_cal.is_last_trading_day_of_month(today)

    # 月末/季末：全量数据更新（含 processed），然后触发训练
    # 普通交易日：只更新 raw 数据
    if is_q_end or is_m_end:
        _safe('daily_job.update_full', data_update.run, processed=True)
    else:
        _safe('daily_job.update', data_update.run)

    if is_q_end:
        print(f'  → 季末最后交易日，触发 quarterly 全量重训')
        _safe('daily_job.train_quarterly', train.train, 'quarterly')
    elif is_m_end:
        print(f'  → 月末最后交易日，触发 monthly 微调')
        _safe('daily_job.train_monthly', train.train, 'monthly')
    else:
        print('  → 非月末/季末，跳过训练')

    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] daily_job 完成')


# ---------- 入口 ----------

def _run_once(which: str) -> None:
    if which == 'weekly':
        weekly_job()
    elif which == 'daily':
        daily_job()
    else:
        print(f'未知的 --once 参数: {which}')
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description='Regime-GNN-LSTM 实盘调度器')
    parser.add_argument('--once', choices=['weekly', 'daily'], default=None,
                        help='立即跑一次指定任务后退出（调试用）')
    args = parser.parse_args()

    if args.once:
        _run_once(args.once)
        return

    try:
        import schedule
    except ImportError:
        print('缺少依赖：pip install schedule')
        sys.exit(1)

    schedule.every().sunday.at('20:00').do(weekly_job)
    schedule.every().day.at('16:30').do(daily_job)

    print('=' * 60)
    print('  Regime-GNN-LSTM 实盘调度器')
    print('=' * 60)
    print('  · 周日 20:00  → weekly_job  (update + 推理 + 周报)')
    print('  · 每日 16:30  → daily_job   (update + 月末/季末训练)')
    print('  交易日判定：tushare trade_cal (cached)')
    print('  Ctrl+C 退出')
    print('=' * 60)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print('\n调度器退出')


if __name__ == '__main__':
    main()
