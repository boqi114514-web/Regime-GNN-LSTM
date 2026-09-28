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

_ETF_MAPPING_PATH = os.path.join(LOCAL_DATA_RAW, 'etf_sw_mapping_v2.csv')


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


# ---------- ETF 执行载体 ----------

def _load_etf_mapping() -> Optional[pd.DataFrame]:
    if not os.path.exists(_ETF_MAPPING_PATH):
        return None
    try:
        return pd.read_csv(_ETF_MAPPING_PATH, encoding='utf-8-sig')
    except Exception:
        return None


_ETF_R2_WARN = 0.85   # 低于此 R² 标注 ⚠️，提示代理质量偏低


def _best_etf_for_industry(sw_code: str, mapping: pd.DataFrame) -> Optional[dict]:
    """取该行业 R² 最高的 ETF"""
    sub = mapping[mapping['sw_code'] == sw_code].sort_values('r2', ascending=False)
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {
        'code': row['etf_code'],
        'name': str(row.get('etf_name', '') or ''),
        'r2':   float(row['r2']),
        'beta': float(row['beta']),
        'aum':  row.get('aum_亿'),
    }


def _etf_section_lines(top_k_codes: list, mapping: Optional[pd.DataFrame]) -> list:
    """生成 ETF 执行载体章节（不区分板块，R²<阈值加警告）"""
    if mapping is None or mapping.empty:
        return ['## 🏦 ETF 执行载体', '', '> ETF 映射表未找到，请先运行 `data_pipeline.etf_mapping_v2`', '']

    lines = ['## 🏦 ETF 执行载体', '']
    lines.append('| 行业 | ETF代码 | ETF名称 | R² | β | 规模(亿) |')
    lines.append('|------|---------|--------|-----|---|---------|')

    for code in top_k_codes:
        ind_name = _SW_NAMES.get(code, code)
        etf = _best_etf_for_industry(code, mapping)
        if etf is None:
            lines.append(f'| {ind_name} | — | 无合适ETF | — | — | — |')
            continue
        aum_str  = f'{etf["aum"]:.1f}' if pd.notna(etf.get('aum')) else 'N/A'
        warn     = ' ⚠️' if etf['r2'] < _ETF_R2_WARN else ''
        name_str = etf['name'][:14] + warn
        lines.append(
            f'| {ind_name} | {etf["code"]} | {name_str} '
            f'| {etf["r2"]:.3f} | {etf["beta"]:.2f} | {aum_str} |'
        )

    lines.append('')
    lines.append(f'> ⚠️ R²<{_ETF_R2_WARN} 表示该 ETF 对行业的统计拟合偏低，慎用作执行载体')
    lines.append('')
    return lines


# ---------- 股票板块判断 ----------

_RESTRICT_THRESHOLD = 3   # 受限股票超过此数时展示主板备选表


def _stock_board(stock_code: str) -> tuple:
    """返回 (板块标签, is_restricted)"""
    code = str(stock_code or '')
    if code.startswith('688'):
        return '科创', True
    if code.startswith('300') or code.startswith('301'):
        return '创业', True
    return '主板', False


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


def _stock_table_lines(stock_df: pd.DataFrame, main_board_only: bool = False) -> list:
    lines = []
    lines.append('| 行业 | 代码 | 名称 | 板块 | β | 动量 | 综合分 |')
    lines.append('|------|------|------|------|---|------|--------|')
    for ind in stock_df['ind_code'].unique():
        sub = stock_df[stock_df['ind_code'] == ind].sort_values('rank_in_ind')
        ind_label = sub['ind_name'].iloc[0] if 'ind_name' in sub.columns else ind
        for _, r in sub.iterrows():
            board, restricted = _stock_board(r['stock_code'])
            if main_board_only and restricted:
                continue
            name_str  = r.get('name', r['stock_code'])
            mom_str   = f"{r['momentum']*100:+.1f}%" if pd.notna(r.get('momentum')) else '-'
            comp_str  = f"{r['composite']:.3f}"      if pd.notna(r.get('composite')) else '-'
            beta_str  = f"{r['beta']:.2f}"           if pd.notna(r.get('beta'))      else '-'
            lines.append(
                f'| {ind_label} | {r["stock_code"]} | {name_str} '
                f'| {board} | {beta_str} | {mom_str} | {comp_str} |'
            )
    return lines


def _stock_section_lines(stock_df: pd.DataFrame, label: str) -> list:
    """带板块标注 + 主板备选的个股持仓章节"""
    n_total = len(stock_df)
    restricted_mask = stock_df['stock_code'].apply(lambda c: _stock_board(c)[1])
    n_restricted = restricted_mask.sum()

    lines = [f'### {label}（{n_total} 只）', '']
    lines += _stock_table_lines(stock_df)
    lines.append('')

    if n_restricted > 0:
        boards = stock_df.loc[restricted_mask, 'stock_code'].apply(
            lambda c: _stock_board(c)[0]
        )
        kechuang = (boards == '科创').sum()
        chuangye  = (boards == '创业').sum()
        parts = []
        if kechuang:
            parts.append(f'{kechuang} 只科创板')
        if chuangye:
            parts.append(f'{chuangye} 只创业板')
        lines.append(f'> 含 {"、".join(parts)}（需对应交易权限）')
        lines.append('')

        if n_restricted >= _RESTRICT_THRESHOLD:
            main_lines = _stock_table_lines(stock_df, main_board_only=True)
            n_main = sum(1 for l in main_lines if l.startswith('|') and '---' not in l)
            if n_main > 0:
                lines.append(f'**主板备选（{n_main} 只，剔除科创/创业）：**')
                lines.append('')
                lines += main_lines
                lines.append('')

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
                    current_equal: Optional[dict] = None,
                    etf_mapping: Optional[pd.DataFrame] = None,
                    primary_mode: str = 'fixed_46') -> str:
    now = datetime.now()
    iso_year, iso_week, _ = now.isocalendar()
    K = len(current['top_k'])

    # 主输出标签随分支模式变化：regime 分支(main/fix)标 "Regime 集成"，
    # equal 分支(refactor)标 "等权集成"
    is_regime = (primary_mode == 'regime')
    primary_label = ('Regime 集成' if is_regime else
                     f'固定权重 GNN:LSTM = {primary_mode[-2]}:{primary_mode[-1]}'
                     if primary_mode.startswith('fixed_') else '等权集成')

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
    lines.append(f'# 行业轮动周报 · {now.strftime("%Y-%m-%d")}生成')
    lines.append('')
    lines.append(f'**生成时间**：{now.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**ISO 周**：{iso_year}-W{iso_week:02d}')
    lines.append(f'**数据月份**：{current["as_of"]}')
    if current.get('model_label'):
        lines.append(f'**模型版本**：{current["model_label"]}')
    if is_regime and current.get('regime') is not None:
        lines.append(f'**HMM regime**：{REGIME_NAMES.get(current["regime"], current["regime"])}')
    if is_regime and current.get('w_gnn') is not None and current.get('w_lstm') is not None:
        lines.append(f'**Regime集成权重**：GNN={current["w_gnn"]:.2f} / LSTM-B={current["w_lstm"]:.2f}')
    if primary_mode.startswith('fixed_') and current.get('w_gnn') is not None:
        lines.append(f'**固定集成权重**：GNN={current["w_gnn"]:.2f} / LSTM-B={current["w_lstm"]:.2f}')
    lines.append('')

    # ── 1. Top-K 推荐行业 ─────────────────────────────────────
    lines.append(f'## 🎯 Top-{K} 推荐行业')
    lines.append('')
    lines.append(f'### {primary_label}')
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

    # ── 2. ETF 执行载体 ───────────────────────────────────────
    # 合并两模式 top_k（去重，保持顺序：regime 优先，然后补等权独有）
    combined_top_k = list(current['top_k'])
    if has_eq:
        for c in current_equal['top_k']:
            if c not in combined_top_k:
                combined_top_k.append(c)
    lines += _etf_section_lines(combined_top_k, etf_mapping)

    # ── 3. 持仓变动 ───────────────────────────────────────────
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

    lines += _holdings_block(diff_r, primary_label)
    if has_eq:
        lines += _holdings_block(diff_e, '等权集成')
        lines.append('> ' + _diff_summary_line(diff_r, diff_e, K))
        lines.append('')

    # ── 4. 模型健康度（共用）─────────────────────────────────
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

    # ── 5. 执行建议 ───────────────────────────────────────────
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

    lines += _advice(diff_r, primary_label)
    if has_eq:
        lines += _advice(diff_e, '等权集成')
        agree = (diff_r['turnover'] > 0.01) == (diff_e['turnover'] > 0.01)
        lines.append('> ' + ('两模型建议一致' if agree else '两模型建议不同，等权出现换手信号'))
        lines.append('')

    # ── 6. 个股持仓 ───────────────────────────────────────────
    stock_r    = current.get('stock_holdings')
    stock_r_mb = current.get('stock_holdings_mainboard')
    stock_e    = current_equal.get('stock_holdings')    if has_eq else None
    stock_e_mb = current_equal.get('stock_holdings_mainboard') if has_eq else None
    has_stock_r    = stock_r    is not None and not stock_r.empty
    has_stock_r_mb = stock_r_mb is not None and not stock_r_mb.empty
    has_stock_e    = stock_e    is not None and not stock_e.empty
    has_stock_e_mb = stock_e_mb is not None and not stock_e_mb.empty

    if has_stock_r or has_stock_e:
        lines.append('## 📋 个股持仓（选股层）')
        lines.append('')

        if has_stock_r:
            lines += _stock_section_lines(stock_r, primary_label)

        if has_stock_e:
            lines += _stock_section_lines(stock_e, '等权集成')

        if has_stock_r and has_stock_e:
            r_codes = set(stock_r['stock_code'])
            e_codes = set(stock_e['stock_code'])
            overlap = r_codes & e_codes
            lines.append(f'> 共同持仓 {len(overlap)} 只，{primary_label} 独有 {len(r_codes-e_codes)} 只，等权独有 {len(e_codes-r_codes)} 只')
            lines.append('')

    if has_stock_r_mb or has_stock_e_mb:
        lines.append('## 📋 主板持仓（保证主板，每行业满额）')
        lines.append('')

        if has_stock_r_mb:
            lines += _stock_section_lines(stock_r_mb, f'{primary_label}·主板')

        if has_stock_e_mb:
            lines += _stock_section_lines(stock_e_mb, '等权集成·主板')

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

    # 主输出模式随分支：main/fix='regime'，refactor='equal'
    try:
        from config import ENSEMBLE_MODE as _ENS_MODE
    except ImportError:
        _ENS_MODE = 'fixed_46'
    primary_mode = _ENS_MODE if _ENS_MODE in ('fixed_46', 'fixed_55', 'fixed_64',
                                              'regime', 'equal') else 'fixed_46'
    is_regime    = (primary_mode == 'regime')
    primary_pkl  = (f'predictions_ensemble_{primary_mode[-2:]}.pkl'
                    if primary_mode.startswith('fixed_') else
                    'predictions_ensemble.pkl' if is_regime else
                    'predictions_ensemble_equal.pkl')
    print(f'  [模式] 主输出={primary_mode}  预测文件={primary_pkl}')

    current = predict.infer_latest(mode=primary_mode)
    previous = state.get_last_holdings()

    # 仅 regime 分支才额外出等权对比（refactor 本身即等权，无需重复）
    current_equal = None
    if is_regime:
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

    # 选股层：主输出（常规=全市场含双创；主板=强制主板满额）
    try:
        import s4_beta_selection as s4
        current['stock_holdings'] = s4.run_live(
            pred_pkl=primary_pkl, force_refresh_latest=True,
            main_board_only=False)
    except Exception as e:
        print(f'  [选股 主输出] 跳过（{type(e).__name__}: {e}）')
        current['stock_holdings'] = None

    try:
        current['stock_holdings_mainboard'] = s4.run_live(
            pred_pkl=primary_pkl,
            force_refresh_latest=True,
            main_board_only=True,
            ckpt_suffix='_mb',
        )
    except Exception as e:
        print(f'  [选股 主输出 主板] 跳过（{type(e).__name__}: {e}）')
        current['stock_holdings_mainboard'] = None

    # 选股层：等权对比分支（仅 regime 模式）
    if current_equal is not None:
        try:
            current_equal['stock_holdings'] = s4.run_live(
                pred_pkl='predictions_ensemble_equal.pkl',
                ckpt_suffix='_equal',
                force_refresh_latest=True,
                main_board_only=False,
            )
        except Exception as e:
            print(f'  [选股 equal] 跳过（{type(e).__name__}: {e}）')
            current_equal['stock_holdings'] = None

        try:
            current_equal['stock_holdings_mainboard'] = s4.run_live(
                pred_pkl='predictions_ensemble_equal.pkl',
                ckpt_suffix='_equal_mb',
                force_refresh_latest=True,
                main_board_only=True,
            )
        except Exception as e:
            print(f'  [选股 equal 主板] 跳过（{type(e).__name__}: {e}）')
            current_equal['stock_holdings_mainboard'] = None

    etf_mapping = _load_etf_mapping()
    report = generate_report(current, previous, current_equal,
                             etf_mapping=etf_mapping, primary_mode=primary_mode)

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
