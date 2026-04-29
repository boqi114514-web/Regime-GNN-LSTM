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

_SW_NAMES = {
    '801010.SI':'农林牧渔','801020.SI':'采掘','801030.SI':'化工',
    '801040.SI':'钢铁','801050.SI':'有色金属','801080.SI':'电子',
    '801110.SI':'家用电器','801120.SI':'食品饮料','801130.SI':'纺织服饰',
    '801140.SI':'轻工制造','801150.SI':'医药生物','801160.SI':'公用事业',
    '801170.SI':'交通运输','801180.SI':'房地产','801200.SI':'商贸零售',
    '801210.SI':'社会服务','801230.SI':'综合','801710.SI':'建筑材料',
    '801720.SI':'建筑装饰','801730.SI':'电力设备','801740.SI':'国防军工',
    '801750.SI':'计算机','801760.SI':'传媒','801770.SI':'通信',
    '801780.SI':'银行','801790.SI':'非银金融','801880.SI':'汽车',
    '801890.SI':'机械设备','801950.SI':'煤炭','801960.SI':'石油石化',
    '801970.SI':'环保','801980.SI':'美容护理',
}


def _topk_table_lines(cur: dict, deltas: dict) -> list:
    lines = []
    lines.append('| 排名 | 行业代码 | 行业名称 | 得分 | 对比上次 |')
    lines.append('|------|---------|---------|------|----------|')
    for i, code in enumerate(cur['top_k'], 1):
        s = cur['scores'][code]
        score_val = s.get('score', s.get('ensemble', 0.0))
        lines.append(f'| {i} | {code} | {_SW_NAMES.get(code, code)} | {score_val:.2f} | {_arrow(deltas.get(code))} |')
    return lines


def _stock_table_lines(stock_df: pd.DataFrame) -> list:
    lines = []
    lines.append('| 行业 | 代码 | 名称 | β | 动量 | 综合分 |')
    lines.append('|------|------|------|---|------|--------|')
    for ind in stock_df['ind_code'].unique():
        sub = stock_df[stock_df['ind_code'] == ind].sort_values('rank_in_ind')
        ind_label = sub['ind_name'].iloc[0] if 'ind_name' in sub.columns else ind
        for _, r in sub.iterrows():
            name_str = r.get('name', r['stock_code'])
            mom_str  = f"{r['momentum']*100:+.1f}%" if pd.notna(r.get('momentum')) else '-'
            comp_str = f"{r['composite']:.3f}"      if pd.notna(r.get('composite')) else '-'
            beta_str = f"{r['beta']:.2f}"           if pd.notna(r.get('beta'))      else '-'
            lines.append(f'| {ind_label} | {r["stock_code"]} | {name_str} | {beta_str} | {mom_str} | {comp_str} |')
    return lines


def _diff_summary_line(diff_r: dict, diff_e: dict, k: int) -> str:
    """持仓变动对比摘要"""
    tr = diff_r['turnover']
    te = diff_e['turnover'] if diff_e else None
    if te is None:
        return f'Regime 换手率 {tr:.0%}'
    if tr == 0 and te == 0:
        return '两模型持仓均与上期完全一致'
    parts = []
    if tr > 0:
        parts.append(f'Regime 换手 {tr:.0%}（新进 {len(diff_r["entered"])} 踢出 {len(diff_r["exited"])}）')
    else:
        parts.append('Regime 无变化')
    if te > 0:
        parts.append(f'等权 换手 {te:.0%}（新进 {len(diff_e["entered"])} 踢出 {len(diff_e["exited"])}）')
    else:
        parts.append('等权 无变化')
    return '、'.join(parts)


def generate_report(current: dict, previous: Optional[dict],
                    current_equal: Optional[dict] = None) -> str:
    now = datetime.now()
    iso_year, iso_week, _ = now.isocalendar()
    K = len(current['top_k'])

    prev_top = previous['top_k'] if previous else None
    prev_all = previous.get('all_ranked') if previous else None
    prev_as_of = previous.get('as_of') if previous else None

    diff_r = _diff_holdings(current['top_k'], prev_top)
    deltas_r = _rank_delta_map(current['all_ranked'], prev_all)

    has_eq = current_equal is not None
    if has_eq:
        diff_e   = _diff_holdings(current_equal['top_k'], prev_top)
        deltas_e = _rank_delta_map(current_equal['all_ranked'], prev_all)
    else:
        diff_e = deltas_e = None

    lines = []

    # ── 头部 ──────────────────────────────────────────────────
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
        lines.append(f'**Regime集成权重**：GNN={current["w_gnn"]:.2f} / LSTM-B={current["w_lstm"]:.2f}')
    lines.append('')

    # ── 1. Top-K 推荐行业 ─────────────────────────────────────
    lines.append(f'## 🎯 Top-{K} 推荐行业')
    lines.append('')
    lines.append('### Regime 集成')
    lines.append('')
    lines += _topk_table_lines(current, deltas_r)
    lines.append('')
    if has_eq:
        lines.append('### 等权集成')
        lines.append('')
        lines += _topk_table_lines(current_equal, deltas_e)
        lines.append('')
        # 对比摘要
        regime_set  = set(current['top_k'])
        equal_set   = set(current_equal['top_k'])
        consensus   = regime_set & equal_set
        only_regime = regime_set - equal_set
        only_equal  = equal_set  - regime_set
        summary = []
        if consensus:
            summary.append('共识：' + '、'.join(_SW_NAMES.get(c, c) for c in sorted(consensus)))
        if only_regime:
            summary.append('仅 Regime：' + '、'.join(_SW_NAMES.get(c, c) for c in sorted(only_regime)))
        if only_equal:
            summary.append('仅等权：' + '、'.join(_SW_NAMES.get(c, c) for c in sorted(only_equal)))
        lines.append('> ' + '　'.join(summary))
        lines.append('')

    # ── 2. 持仓变动 ───────────────────────────────────────────
    lines.append('## 📊 持仓变动')
    lines.append('')

    def _holdings_block(diff: dict, label: str) -> list:
        bl = [f'### {label}', '']
        if not previous:
            bl.append('- 首次生成周报，无对比基准')
        else:
            bl.append(f'- 上次对比基准：{prev_as_of}')
            bl.append(f'- 换手率：**{diff["turnover"]:.0%}**（保留 {len(diff["kept"])}/{K}）')
            if diff['entered']:
                bl.append(f'- 新进入：{", ".join(_SW_NAMES.get(c,c) for c in diff["entered"])}')
            if diff['exited']:
                bl.append(f'- 被踢出：{", ".join(_SW_NAMES.get(c,c) for c in diff["exited"])}')
            if not diff['entered'] and not diff['exited']:
                bl.append('- 本周 Top-K 与上期完全一致 ✅')
        bl.append('')
        return bl

    lines += _holdings_block(diff_r, 'Regime 集成')
    if has_eq:
        lines += _holdings_block(diff_e, '等权集成')
        lines.append('> ' + _diff_summary_line(diff_r, diff_e, K))
        lines.append('')

    # ── 3. 模型健康度（共用）─────────────────────────────────
    last_run = state.get_last_run()
    lines.append('## 📈 模型健康度')
    lines.append('')
    lines.append(f'- 推理数据月份：{current["as_of"]}')
    lines.append('- 数据源最新月份：')
    sources = _data_source_status()
    predict_month = pd.to_datetime(current['as_of']).strftime('%Y-%m') if current.get('as_of') else None
    for name, src_latest, _path in sources:
        tag_str = ''
        if predict_month and src_latest not in ('缺失',) and not src_latest.startswith('读取失败'):
            if 'Q' not in src_latest and src_latest < predict_month:
                tag_str = ' ⚠️ 落后于推理'
            elif 'Q' not in src_latest and src_latest > predict_month:
                tag_str = ' ✅ 比推理新（下次重训会吸收）'
        lines.append(f'  - `{name}` → {src_latest}{tag_str}')
    if last_run.get('last_quarterly_train'):
        lines.append(f'- 上次季末重训：{last_run["last_quarterly_train"]}  (label={last_run.get("quarterly_label","?")})')
    if last_run.get('last_monthly_train'):
        lines.append(f'- 上次月末微调：{last_run["last_monthly_train"]}  (label={last_run.get("monthly_label","?")})')
    lines.append('')

    # ── 4. 执行建议 ───────────────────────────────────────────
    lines.append('## 🔧 执行建议')
    lines.append('')

    def _advice(diff: dict, label: str) -> list:
        bl = [f'### {label}', '']
        if not previous:
            bl.append('- 首次发报，建议周一开盘后 30 分钟内按 Top-K 建仓。')
        elif diff['turnover'] > 0.01:
            bl.append('- 建议周一开盘后 30 分钟内完成调仓。')
        else:
            bl.append('- 当前持仓与上期一致，**无需调仓**。')
        bl.append('')
        return bl

    lines += _advice(diff_r, 'Regime 集成')
    if has_eq:
        lines += _advice(diff_e, '等权集成')
        agree = (diff_r['turnover'] > 0.01) == (diff_e['turnover'] > 0.01)
        lines.append('> ' + ('两模型建议一致' if agree else '两模型建议不同，等权出现换手信号'))
        lines.append('')

    # ── 5. 个股持仓 ───────────────────────────────────────────
    stock_r = current.get('stock_holdings')
    stock_e = current_equal.get('stock_holdings') if has_eq else None
    has_stock_r = stock_r is not None and not stock_r.empty
    has_stock_e = stock_e is not None and not stock_e.empty

    if has_stock_r or has_stock_e:
        lines.append('## 📋 个股持仓（选股层）')
        lines.append('')

        if has_stock_r:
            lines.append(f'### Regime 集成（{len(stock_r)} 只）')
            lines.append('')
            lines += _stock_table_lines(stock_r)
            lines.append('')

        if has_stock_e:
            lines.append(f'### 等权集成（{len(stock_e)} 只）')
            lines.append('')
            lines += _stock_table_lines(stock_e)
            lines.append('')

        if has_stock_r and has_stock_e:
            r_codes = set(stock_r['stock_code'])
            e_codes = set(stock_e['stock_code'])
            overlap = r_codes & e_codes
            lines.append(f'> 共同持仓 {len(overlap)} 只，Regime 独有 {len(r_codes-e_codes)} 只，等权独有 {len(e_codes-r_codes)} 只')
            lines.append('')

    lines.append('---')
    lines.append('*自动生成 by live/monitor.py · 周日晚 20:00 出报 · 人工复核后周一执行*')

    return '\n'.join(lines)


# ---------- 主流程 ----------

def run() -> str:
    """推理 → 选股 → 生成报告 → 发送 → 更新 state，返回 markdown 文本"""
    # 增量推理：若 pv_factors 有新月份，追加预测行（无需重训）
    try:
        from live import infer_incremental
        infer_incremental.run()
    except Exception as e:
        print(f'  [增量推理] 跳过（{type(e).__name__}: {e}）')

    current = predict.infer_latest(mode='regime')
    previous = state.get_last_holdings()

    # 等权集成（失败不中断周报）
    try:
        current_equal = predict.infer_latest(mode='equal')
    except Exception as e:
        print(f'  [等权推理] 跳过（{type(e).__name__}: {e}）')
        current_equal = None

    # 更新个股日线（取最新数据用于选股层 beta/动量计算）
    try:
        from data_pipeline import download as _dl
        _dl.update_stock_daily()
    except Exception as e:
        print(f'  [日线更新] 跳过（{type(e).__name__}: {e}）')

    # 选股层：regime 分支
    try:
        import s4_beta_selection as s4
        current['stock_holdings'] = s4.run_live(force_refresh_latest=True)
    except Exception as e:
        print(f'  [选股 regime] 跳过（{type(e).__name__}: {e}）')
        current['stock_holdings'] = None

    # 选股层：等权分支
    if current_equal is not None:
        try:
            current_equal['stock_holdings'] = s4.run_live(
                pred_pkl='predictions_ensemble_equal.pkl',
                ckpt_suffix='_equal',
                force_refresh_latest=True,
            )
        except Exception as e:
            print(f'  [选股 equal] 跳过（{type(e).__name__}: {e}）')
            current_equal['stock_holdings'] = None

    report = generate_report(current, previous, current_equal)

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
