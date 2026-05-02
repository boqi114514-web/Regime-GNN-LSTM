# -*- coding: utf-8 -*-
"""ETF执行回测：用实际ETF月度收益替代行业指数，评估策略可执行性

对比五条曲线：
  1. 等权基准      —— 31个申万一级行业指数等权，每月再平衡
  2. 轮动-指数     —— Top-K行业指数（s3原始结果，理想情况）
  3. 轮动-ETF      —— Top-K行业最优ETF；ETF无数据时退回行业指数
  4. 轮动-ETF高质量—— R²≥阈值用ETF，否则退回行业指数
  5. 轮动-等权ETF  —— 等权集成版本的ETF执行

输入：
  models/current/predictions_ensemble.pkl        (Regime集成预测)
  models/current/predictions_ensemble_equal.pkl  (等权集成预测)
  data/raw/etf_sw_mapping_v2.csv
  data/raw/_cache_v2_etf_prices.pkl
  data/raw/ts_sw_industry_monthly.csv

输出：
  results/etf_backtest_results.png
  results/etf_backtest_summary.csv

用法：
    python -m s6_etf_execution_backtest
    python -m s6_etf_execution_backtest --topk 3
    python -m s6_etf_execution_backtest --r2-threshold 0.80
"""
import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import MODELS_CURRENT_DIR, LOCAL_DATA_RAW, OUTPUT_DIR, TOP_K

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR     = os.path.join(os.path.dirname(_SRC_DIR), 'results')
ETF_MAPPING_PATH = os.path.join(LOCAL_DATA_RAW, 'etf_sw_mapping_v2.csv')
ETF_PRICES_PATH  = os.path.join(LOCAL_DATA_RAW, '_cache_v2_etf_prices.pkl')
SW_MONTHLY_PATH  = os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv')
RF_ANNUAL        = 0.015   # 年化无风险利率


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_sw_returns() -> pd.DataFrame:
    """申万行业月度收益率，period 索引，列为 ts_code"""
    df = pd.read_csv(SW_MONTHLY_PATH)
    df['date'] = pd.to_datetime(df['date'])
    pivot = df.pivot_table(index='date', columns='ts_code', values='pct_chg', aggfunc='last') / 100
    sw_cols = [c for c in pivot.columns if str(c).startswith('8') and str(c).endswith('.SI')]
    pivot = pivot[sw_cols].sort_index()
    pivot.index = pivot.index.to_period('M')
    return pivot


def load_etf_returns() -> pd.DataFrame:
    """ETF月度收益率，period 索引"""
    prices = pd.read_pickle(ETF_PRICES_PATH)
    prices.index = pd.to_datetime(prices.index).to_period('M')
    prices = prices.sort_index()
    returns = prices.pct_change(fill_method=None)
    return returns


def load_ensemble(fname: str) -> pd.DataFrame:
    """加载 predictions_ensemble*.pkl，返回 (period, ts_code) 索引的 pred_ensemble"""
    path = os.path.join(MODELS_CURRENT_DIR, fname)
    df = pd.read_pickle(path)
    df['date'] = pd.to_datetime(df['date'])
    df['period'] = df['date'].dt.to_period('M')
    return df


def load_mapping() -> pd.DataFrame:
    return pd.read_csv(ETF_MAPPING_PATH, encoding='utf-8-sig')


# ─────────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────────

def build_etf_return_map(mapping: pd.DataFrame, etf_returns: pd.DataFrame) -> dict:
    """
    预计算每个行业→该行业所有候选ETF的月度收益序列，按 R² 排序。
    返回 {sw_code: [(r2, etf_code, ret_series), ...]}
    """
    etf_map = {}
    for sw_code, grp in mapping.groupby('sw_code'):
        grp = grp.sort_values('r2', ascending=False)
        candidates = []
        for _, row in grp.iterrows():
            code = row['etf_code']
            r2   = float(row['r2'])
            if code in etf_returns.columns:
                candidates.append((r2, code, etf_returns[code]))
        etf_map[sw_code] = candidates
    return etf_map


def _get_etf_ret(sw_code: str, period, etf_map: dict,
                 sw_returns: pd.DataFrame, r2_threshold: float = 0.0) -> float:
    """
    取该行业在 period 月的 ETF 收益。
    - 按 R² 降序遍历候选ETF，找第一个在该月有收益数据且 R² ≥ threshold 的
    - 找不到则退回行业指数
    """
    if sw_code in etf_map:
        for r2, code, ret_series in etf_map[sw_code]:
            if r2 < r2_threshold:
                break  # 候选已按R²排序，后面的也不满足
            if period in ret_series.index:
                v = ret_series[period]
                if pd.notna(v):
                    return float(v)
    # 退回行业指数
    if sw_code in sw_returns.columns and period in sw_returns.index:
        v = sw_returns.loc[period, sw_code]
        if pd.notna(v):
            return float(v)
    return np.nan


def run_backtest(ens: pd.DataFrame, sw_returns: pd.DataFrame,
                 etf_map: dict, top_k: int, r2_threshold: float) -> dict:
    """
    对每个月份计算三种组合的月度收益：
      - index    : Top-K行业指数等权
      - etf_all  : Top-K行业ETF等权（无R²门槛，但无数据退回指数）
      - etf_hq   : Top-K行业ETF等权（R²≥threshold用ETF，否则退回指数）
    返回 {strategy: pd.Series(period → monthly_return)}
    """
    periods = sorted(ens['period'].unique())
    results = {'index': {}, 'etf_all': {}, 'etf_hq': {}}
    coverage = {}   # {period: n_etf_used / top_k}

    for period in periods:
        month_data = ens[ens['period'] == period]
        if month_data.empty:
            continue
        top = month_data.nlargest(top_k, 'pred_ensemble')['ts_code'].tolist()

        # 三种收益计算
        idx_rets, etf_all_rets, etf_hq_rets = [], [], []
        n_etf_used = 0

        for sw_code in top:
            # 行业指数
            if sw_code in sw_returns.columns and period in sw_returns.index:
                idx_r = sw_returns.loc[period, sw_code]
                if pd.notna(idx_r):
                    idx_rets.append(float(idx_r))

            # ETF（无门槛）
            etf_r = _get_etf_ret(sw_code, period, etf_map, sw_returns, r2_threshold=0.0)
            etf_all_rets.append(etf_r)

            # ETF（高质量门槛）
            etf_hq_r = _get_etf_ret(sw_code, period, etf_map, sw_returns, r2_threshold=r2_threshold)
            etf_hq_rets.append(etf_hq_r)

            # 统计真实用了ETF而非退回指数的比例
            if sw_code in etf_map:
                for r2, code, ret_series in etf_map[sw_code]:
                    if period in ret_series.index and pd.notna(ret_series[period]):
                        n_etf_used += 1
                        break

        results['index'][period]   = np.nanmean(idx_rets)   if idx_rets   else np.nan
        results['etf_all'][period] = np.nanmean([r for r in etf_all_rets if pd.notna(r)]) if etf_all_rets else np.nan
        results['etf_hq'][period]  = np.nanmean([r for r in etf_hq_rets  if pd.notna(r)]) if etf_hq_rets  else np.nan
        coverage[period] = n_etf_used / top_k

    coverage_rate = np.mean(list(coverage.values()))
    print(f'  ETF 数据覆盖率（Top-K 有 ETF 数据的月份比例）：{coverage_rate:.1%}')

    return {k: pd.Series(v).sort_index() for k, v in results.items()}


def build_benchmark(sw_returns: pd.DataFrame, periods) -> pd.Series:
    """等权基准：31行业指数每月等权"""
    bench = {}
    for p in periods:
        if p in sw_returns.index:
            bench[p] = sw_returns.loc[p].mean()
    return pd.Series(bench).sort_index()


# ─────────────────────────────────────────────
# 绩效指标
# ─────────────────────────────────────────────

def _metrics(ret_series: pd.Series, label: str) -> dict:
    ret = ret_series.dropna()
    if len(ret) < 3:
        return {'策略': label, '年化收益': 'N/A', 'Sharpe': 'N/A', '最大回撤': 'N/A', '月数': len(ret)}

    n_months = len(ret)
    ann_ret  = (1 + ret).prod() ** (12 / n_months) - 1
    ann_vol  = ret.std() * np.sqrt(12)
    rf_m     = RF_ANNUAL / 12
    sharpe   = (ret.mean() - rf_m) / ret.std() * np.sqrt(12) if ret.std() > 0 else np.nan
    nav      = (1 + ret).cumprod()
    peak     = nav.cummax()
    drawdown = (nav - peak) / peak
    max_dd   = drawdown.min()
    return {
        '策略': label,
        '年化收益': f'{ann_ret:.1%}',
        'Sharpe':  f'{sharpe:.3f}',
        '最大回撤': f'{max_dd:.1%}',
        '月数':    n_months,
    }


def nav_from_returns(ret: pd.Series) -> pd.Series:
    return (1 + ret.fillna(0)).cumprod()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main(top_k: int = TOP_K, r2_threshold: float = 0.85):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('加载数据...')
    sw_returns = load_sw_returns()
    etf_returns = load_etf_returns()
    mapping = load_mapping()

    ens_regime = load_ensemble('predictions_ensemble.pkl')
    ens_equal  = load_ensemble('predictions_ensemble_equal.pkl')

    print('构建ETF收益映射...')
    etf_map = build_etf_return_map(mapping, etf_returns)

    print(f'\n回测参数: Top-{top_k}  R²阈值={r2_threshold}')
    print('=' * 60)

    print('\n[Regime集成]')
    res_r = run_backtest(ens_regime, sw_returns, etf_map, top_k, r2_threshold)

    print('\n[等权集成]')
    res_e = run_backtest(ens_equal, sw_returns, etf_map, top_k, r2_threshold)

    # 公共时段
    all_periods = sorted(set(res_r['index'].index) | set(res_e['index'].index))
    bench = build_benchmark(sw_returns, all_periods)

    # NAV 序列
    strategies = {
        '等权基准':         bench,
        'Regime-指数':      res_r['index'],
        'Regime-ETF':       res_r['etf_all'],
        f'Regime-ETF(R²≥{r2_threshold})': res_r['etf_hq'],
        '等权集成-ETF':     res_e['etf_all'],
    }

    # 绩效表
    print('\n' + '=' * 60)
    rows = []
    for label, ret_s in strategies.items():
        rows.append(_metrics(ret_s, label))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, 'etf_backtest_summary.csv'),
                   index=False, encoding='utf-8-sig')

    # ── 绘图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'ETF执行回测  Top-{top_k}  (Regime集成 vs 等权集成 vs 行业指数)',
                 fontsize=14, fontweight='bold')

    ax_nav, ax_dd = axes

    colors = {
        '等权基准':         '#888888',
        'Regime-指数':      '#2166ac',
        'Regime-ETF':       '#d73027',
        f'Regime-ETF(R²≥{r2_threshold})': '#f46d43',
        '等权集成-ETF':     '#1a9850',
    }
    styles = {
        '等权基准':         '--',
        'Regime-指数':      '-',
        'Regime-ETF':       '-',
        f'Regime-ETF(R²≥{r2_threshold})': '-.',
        '等权集成-ETF':     '-',
    }

    all_navs = {}
    for label, ret_s in strategies.items():
        nav = nav_from_returns(ret_s.dropna())
        all_navs[label] = nav
        # x轴转为datetime便于matplotlib
        x = nav.index.to_timestamp()
        ax_nav.plot(x, nav.values, label=label,
                    color=colors[label], linestyle=styles[label],
                    linewidth=1.8 if label != '等权基准' else 1.2)

    ax_nav.set_ylabel('净值（起始=1）', fontsize=11)
    ax_nav.legend(fontsize=9, loc='upper left')
    ax_nav.grid(alpha=0.3)
    ax_nav.set_title('')

    # 回撤（Regime-ETF）
    nav_r = all_navs['Regime-ETF'].dropna()
    peak_r = nav_r.cummax()
    dd_r = (nav_r - peak_r) / peak_r
    ax_dd.fill_between(dd_r.index.to_timestamp(), dd_r.values, 0,
                       alpha=0.5, color='#d73027', label='Regime-ETF 回撤')
    nav_e = all_navs['等权集成-ETF'].dropna()
    peak_e = nav_e.cummax()
    dd_e = (nav_e - peak_e) / peak_e
    ax_dd.fill_between(dd_e.index.to_timestamp(), dd_e.values, 0,
                       alpha=0.35, color='#1a9850', label='等权集成-ETF 回撤')
    ax_dd.set_ylabel('回撤', fontsize=11)
    ax_dd.legend(fontsize=9)
    ax_dd.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'etf_backtest_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n图表已保存: {out_path}')

    # 每行业ETF覆盖率报告
    print('\n── 各行业 ETF 覆盖情况 ──')
    _SW_NAMES = {
        '801010.SI':'农林牧渔','801030.SI':'基础化工','801040.SI':'钢铁',
        '801050.SI':'有色金属','801080.SI':'电子','801110.SI':'家用电器',
        '801120.SI':'食品饮料','801130.SI':'纺织服饰','801140.SI':'轻工制造',
        '801150.SI':'医药生物','801160.SI':'公用事业','801170.SI':'交通运输',
        '801180.SI':'房地产','801200.SI':'商贸零售','801210.SI':'社会服务',
        '801230.SI':'综合','801710.SI':'建筑材料','801720.SI':'建筑装饰',
        '801730.SI':'电力设备','801740.SI':'国防军工','801750.SI':'计算机',
        '801760.SI':'传媒','801770.SI':'通信','801780.SI':'银行',
        '801790.SI':'非银金融','801880.SI':'汽车','801890.SI':'机械设备',
        '801950.SI':'煤炭','801960.SI':'石油石化','801970.SI':'环保',
        '801980.SI':'美容护理',
    }
    top1 = mapping[mapping['rank'] == 1].copy()
    for _, row in top1.sort_values('r2', ascending=False).iterrows():
        sw = row['sw_code']
        etf_code = row['etf_code']
        r2 = row['r2']
        # 计算该ETF在回测期有多少月有数据
        n_avail = 0
        if etf_code in etf_returns.columns:
            n_avail = etf_returns[etf_code].loc[
                etf_returns.index.isin(all_periods)
            ].notna().sum()
        n_total = len(all_periods)
        warn = ' ⚠️' if r2 < 0.85 else ''
        name = str(row.get('etf_name') or '')[:16]
        ind  = _SW_NAMES.get(sw, sw)
        print(f'  {ind:<8} → {etf_code} {name:<16} R²={r2:.3f}{warn}  数据{n_avail}/{n_total}月')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=TOP_K)
    parser.add_argument('--r2-threshold', type=float, default=0.85)
    args = parser.parse_args()
    main(top_k=args.topk, r2_threshold=args.r2_threshold)
