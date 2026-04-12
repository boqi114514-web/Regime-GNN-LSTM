# -*- coding: utf-8 -*-
"""周报生成：对比上期预测，输出 markdown 并通过 notifier 发送。

执行顺序（被 scheduler 周日 20:00 调用）：
  1. predict.infer_latest()  取最新月份 Top-K
  2. 读 state.last_holdings  拿上次对比基准
  3. 生成 markdown
  4. notifier.send_report   （默认写到 reports/latest.md）
  5. 回写 state.last_holdings
"""
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_PROCESSED, LOCAL_DATA_RAW
from live import predict, state
from notifier import get_notifier


# ---------- 数据源健康度 ----------

def _max_month_of(df: pd.DataFrame) -> Optional[str]:
    """取 DataFrame 的"最新月份"字符串（适配 date 列 or year+quarter 列）"""
    if 'date' in df.columns:
        d = pd.to_datetime(df['date']).max()
        return d.strftime('%Y-%m')
    if 'year' in df.columns and 'quarter' in df.columns:
        y, q = int(df['year'].max()), int(df[df['year'] == df['year'].max()]['quarter'].max())
        return f'{y}-Q{q}'
    return None


def _data_source_status() -> list:
    """扫一遍 data/raw 和 data/processed 的关键数据文件，返回 [(name, latest, path), ...]"""
    targets = [
        ('raw: sw_industry', os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv')),
        ('raw: csi300',      os.path.join(LOCAL_DATA_RAW, 'ts_csi300_monthly.csv')),
        ('raw: macro',       os.path.join(LOCAL_DATA_RAW, 'ts_macro_factors.csv')),
        ('proc: prosperity', os.path.join(LOCAL_DATA_PROCESSED, 'prosperity_indicators_clean.pkl')),
        ('proc: pv_factors', os.path.join(LOCAL_DATA_PROCESSED, 'price_volume_factors.pkl')),
        ('proc: pattern',    os.path.join(LOCAL_DATA_PROCESSED, 'pattern_factors.pkl')),
    ]
    out = []
    for name, path in targets:
        if not os.path.exists(path):
            out.append((name, '缺失', path))
            continue
        try:
            df = pd.read_pickle(path) if path.endswith('.pkl') else pd.read_csv(path)
            # csi300 的 date 是 yyyymmdd 整数
            if 'date' in df.columns and df['date'].dtype.kind in ('i', 'O'):
                try:
                    df = df.assign(date=pd.to_datetime(df['date'].astype(str), format='%Y%m%d'))
                except (ValueError, TypeError):
                    df = df.assign(date=pd.to_datetime(df['date']))
            latest = _max_month_of(df) or '?'
        except Exception as e:
            latest = f'读取失败: {type(e).__name__}'
        out.append((name, latest, path))
    return out


REGIME_NAMES = {-1: '未知', 0: '衰退', 1: '复苏', 2: '扩张', 3: '过热'}


# ---------- 对比工具 ----------

def _diff_holdings(current: list, previous: Optional[list]) -> dict:
    cur = set(current)
    prev = set(previous) if previous else set()
    kept = cur & prev
    return {
        'entered': sorted(cur - prev),
        'exited': sorted(prev - cur),
        'kept': sorted(kept),
        'turnover': (1 - len(kept) / max(len(cur), 1)) if prev else 1.0,
    }


def _rank_delta_map(current_all: list, previous_all: Optional[list]) -> dict:
    prev_rank = {x['ts_code']: x['rank'] for x in (previous_all or [])}
    return {
        item['ts_code']: (prev_rank[item['ts_code']] - item['rank']) if item['ts_code'] in prev_rank else None
        for item in current_all
    }


def _arrow(delta: Optional[int]) -> str:
    if delta is None:
        return '新进'
    if delta == 0:
        return '↔'
    if delta > 0:
        return f'↑{delta}'
    return f'↓{-delta}'


# ---------- 报告生成 ----------

def generate_report(current: dict, previous: Optional[dict]) -> str:
    now = datetime.now()
    iso_year, iso_week, _ = now.isocalendar()

    prev_top = previous['top_k'] if previous else None
    prev_all = previous.get('all_ranked') if previous else None
    prev_as_of = previous.get('as_of') if previous else None

    diff = _diff_holdings(current['top_k'], prev_top)
    deltas = _rank_delta_map(current['all_ranked'], prev_all)

    lines = []
    # --- 头部 ---
    lines.append(f'# 行业轮动周报 · {now.strftime("%Y-%m-%d")}(周日生成)')
    lines.append('')
    lines.append(f'**生成时间**：{now.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**ISO 周**：{iso_year}-W{iso_week:02d}')
    lines.append(f'**数据月份**：{current["as_of"]}')
    if current.get('model_label'):
        lines.append(f'**模型版本**：{current["model_label"]}')
    if current.get('regime') is not None:
        lines.append(f'**HMM regime**：{REGIME_NAMES.get(current["regime"], current["regime"])}')
    if current.get('w_gnn') is not None and current.get('w_lstm') is not None:
        lines.append(f'**集成权重**：GNN={current["w_gnn"]:.2f} / LSTM-B={current["w_lstm"]:.2f}')
    lines.append('')

    # --- Top-K ---
    lines.append(f'## 🎯 Top-{len(current["top_k"])} 推荐行业')
    lines.append('')
    lines.append('| 排名 | 行业代码 | 集成得分 | 对比上次 |')
    lines.append('|------|---------|---------|----------|')
    for i, code in enumerate(current['top_k'], 1):
        s = current['scores'][code]
        # score = N - rank 加权，越大越好；fallback 回 ensemble（负原始值）
        score_val = s.get('score', s.get('ensemble', 0.0))
        lines.append(f'| {i} | {code} | {score_val:.2f} | {_arrow(deltas.get(code))} |')
    lines.append('')

    # --- 持仓变动 ---
    lines.append('## 📊 持仓变动')
    lines.append('')
    if not previous:
        lines.append('- 首次生成周报，无对比基准')
    else:
        lines.append(f'- 上次对比基准：{prev_as_of}')
        lines.append(f'- 换手率：**{diff["turnover"]:.0%}**（保留 {len(diff["kept"])}/{len(current["top_k"])}）')
        if diff['entered']:
            lines.append(f'- 新进入：{", ".join(diff["entered"])}')
        if diff['exited']:
            lines.append(f'- 被踢出：{", ".join(diff["exited"])}')
        if not diff['entered'] and not diff['exited']:
            lines.append('- 本周 Top-K 与上期完全一致 ✅')
    lines.append('')

    # --- 模型健康度 ---
    last_run = state.get_last_run()
    lines.append('## 📈 模型健康度')
    lines.append('')
    lines.append(f'- 推理数据月份（predictions_ensemble 末行）：{current["as_of"]}')

    # 数据源逐项
    lines.append('- 数据源最新月份：')
    sources = _data_source_status()
    predict_month = pd.to_datetime(current['as_of']).strftime('%Y-%m') if current.get('as_of') else None
    for name, latest, _path in sources:
        tag = ''
        if predict_month and latest not in ('缺失',) and not latest.startswith('读取失败'):
            if 'Q' not in latest and latest < predict_month:
                tag = ' ⚠️ 落后于推理'
            elif 'Q' not in latest and latest > predict_month:
                tag = ' ✅ 比推理新（下次重训会吸收）'
        lines.append(f'  - `{name}` → {latest}{tag}')

    if last_run.get('last_quarterly_train'):
        lines.append(f'- 上次季末重训：{last_run["last_quarterly_train"]}  (label={last_run.get("quarterly_label","?")})')
    if last_run.get('last_monthly_train'):
        lines.append(f'- 上次月末微调：{last_run["last_monthly_train"]}  (label={last_run.get("monthly_label","?")})')
    lines.append('')

    # --- 执行建议 ---
    lines.append('## 🔧 执行建议')
    lines.append('')
    if not previous:
        lines.append('- 首次发报，建议周一开盘后 30 分钟内按 Top-K 建仓。')
    elif diff['turnover'] > 0.01:
        lines.append('- 建议周一开盘后 30 分钟内完成调仓。')
    else:
        lines.append('- 当前持仓与上期一致，**无需调仓**。')
    lines.append('')

    lines.append('---')
    lines.append('*自动生成 by live/monitor.py · 周日晚 20:00 出报 · 人工复核后周一执行*')

    return '\n'.join(lines)


# ---------- 主流程 ----------

def run() -> str:
    """推理 → 生成报告 → 发送 → 更新 state，返回 markdown 文本"""
    current = predict.infer_latest()
    previous = state.get_last_holdings()

    report = generate_report(current, previous)

    notifier = get_notifier()
    notifier.send_report(report)

    # 更新 last_holdings
    state.set_last_holdings({
        'as_of': current['as_of'],
        'generated_at': current['generated_at'],
        'model_label': current.get('model_label'),
        'top_k': current['top_k'],
        'all_ranked': current['all_ranked'],
        'scores_ensemble': {k: v['ensemble'] for k, v in current['scores'].items()},
    })
    state.update_last_run(last_monitor=datetime.now().isoformat(timespec='seconds'))
    return report


def main():
    print('=' * 60)
    print('  live/monitor: 生成周报')
    print('=' * 60)
    run()
    print('\n完成 → reports/latest.md')


if __name__ == '__main__':
    main()
