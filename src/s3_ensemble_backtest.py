# -*- coding: utf-8 -*-
"""
Step 3：集成 + 回测

功能：
  1. 加载 GNN 和 LSTM-B 的预测
  2. 自适应权重集成（按近期 IC 加权）
  3. Top-K 纯多头策略回测
  4. 绩效评估 + 可视化
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
import os
import hashlib
import json

from config import *

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  集成方法
# ============================================================

def adaptive_ensemble(gnn_df, lstm_df, ic_lookback=12):
    """
    自适应权重集成：按最近 ic_lookback 个月的 IC 分配权重

    两个分支的预测转为排名后加权平均
    """
    # 合并两个分支
    merged = pd.merge(
        gnn_df[['ts_code', 'date', 'actual_ret', 'pred_gnn']],
        lstm_df[['ts_code', 'date', 'pred_lstm_b']],
        on=['ts_code', 'date'],
        how='inner'
    )

    if merged.empty:
        print("  警告：两个分支没有重叠的预测日期")
        return None

    # 对每个月：计算排名 → 加权平均
    dates = sorted(merged['date'].unique())
    results = []

    # 滚动 IC 记录
    gnn_ics = []
    lstm_ics = []

    for dt in dates:
        m_data = merged[merged['date'] == dt].copy()
        if len(m_data) < 5:
            continue

        # 转为排名分数（1=最好）
        m_data['rank_gnn'] = m_data['pred_gnn'].rank(ascending=False)
        m_data['rank_lstm'] = m_data['pred_lstm_b'].rank(ascending=False)

        # 计算当月两个分支的 IC（用于下一期权重）
        ic_gnn, _ = spearmanr(m_data['pred_gnn'], m_data['actual_ret'])
        ic_lstm, _ = spearmanr(m_data['pred_lstm_b'], m_data['actual_ret'])
        gnn_ics.append(ic_gnn if np.isfinite(ic_gnn) else 0)
        lstm_ics.append(ic_lstm if np.isfinite(ic_lstm) else 0)

        # 用过去 ic_lookback 个月的平均 IC 作为权重
        if len(gnn_ics) > 1:
            recent_gnn_ic = np.mean(gnn_ics[-ic_lookback-1:-1])  # 不含当月
            recent_lstm_ic = np.mean(lstm_ics[-ic_lookback-1:-1])
        else:
            recent_gnn_ic = 0.5
            recent_lstm_ic = 0.5

        # 权重 = max(0, IC) 归一化
        w_gnn = max(0, recent_gnn_ic)
        w_lstm = max(0, recent_lstm_ic)
        w_sum = w_gnn + w_lstm
        if w_sum < 1e-8:
            w_gnn, w_lstm = 0.5, 0.5
        else:
            w_gnn /= w_sum
            w_lstm /= w_sum

        # 加权排名
        w_lstm = max(MIN_LSTM_WEIGHT, w_lstm)
        w_gnn = 1.0 - w_lstm
        m_data['rank_ensemble'] = w_gnn * m_data['rank_gnn'] + w_lstm * m_data['rank_lstm']
        m_data['pred_ensemble'] = -m_data['rank_ensemble']  # 排名越小分数越高
        m_data['w_gnn'] = w_gnn
        m_data['w_lstm'] = w_lstm

        results.append(m_data)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def regime_ensemble(gnn_df, lstm_df, regime_df, ic_lookback=12):
    """
    Regime 条件集成：按成交额状态分别学习 GNN/LSTM-B 的历史 IC，
    在不同 regime 下给不同权重
    """
    merged = pd.merge(
        gnn_df[['ts_code', 'date', 'actual_ret', 'pred_gnn']],
        lstm_df[['ts_code', 'date', 'pred_lstm_b']],
        on=['ts_code', 'date'], how='inner'
    )
    if regime_df is not None:
        merged = pd.merge(merged, regime_df[['date', 'regime']],
                           on='date', how='left')
        merged['regime'] = merged['regime'].fillna(-1).astype(int)
    else:
        merged['regime'] = -1

    dates = sorted(merged['date'].unique())
    results = []

    # 按 regime 分别记录 IC 历史
    regime_gnn_ics = {r: [] for r in range(-1, HMM_N_STATES)}
    regime_lstm_ics = {r: [] for r in range(-1, HMM_N_STATES)}

    for dt in dates:
        m_data = merged[merged['date'] == dt].copy()
        if len(m_data) < 5:
            continue

        regime = m_data['regime'].iloc[0]

        m_data['rank_gnn'] = m_data['pred_gnn'].rank(ascending=False)
        m_data['rank_lstm'] = m_data['pred_lstm_b'].rank(ascending=False)

        # 用同一 regime 下的历史 IC 做权重
        past_gnn = regime_gnn_ics[regime][-ic_lookback:]
        past_lstm = regime_lstm_ics[regime][-ic_lookback:]

        if past_gnn:
            w_gnn = max(0, np.mean(past_gnn))
        else:
            w_gnn = 0.5
        if past_lstm:
            w_lstm = max(0, np.mean(past_lstm))
        else:
            w_lstm = 0.5

        w_sum = w_gnn + w_lstm
        if w_sum < 1e-8:
            w_gnn, w_lstm = 0.5, 0.5
        else:
            w_gnn /= w_sum
            w_lstm /= w_sum

        w_lstm = max(MIN_LSTM_WEIGHT, w_lstm)
        w_gnn = 1.0 - w_lstm
        m_data['rank_ensemble'] = w_gnn * m_data['rank_gnn'] + w_lstm * m_data['rank_lstm']
        m_data['pred_ensemble'] = -m_data['rank_ensemble']
        m_data['w_gnn'] = w_gnn
        m_data['w_lstm'] = w_lstm
        m_data['regime'] = regime

        results.append(m_data)

        # 本信号月的下月收益须等到下月结束才可观察，因此只供后续月份使用。
        labeled = m_data.dropna(subset=['actual_ret'])
        if len(labeled) >= 5:
            ic_gnn, _ = spearmanr(labeled['pred_gnn'], labeled['actual_ret'])
            ic_lstm, _ = spearmanr(labeled['pred_lstm_b'], labeled['actual_ret'])
            if np.isfinite(ic_gnn):
                regime_gnn_ics[regime].append(ic_gnn)
            if np.isfinite(ic_lstm):
                regime_lstm_ics[regime].append(ic_lstm)

    return pd.concat(results, ignore_index=True) if results else None


def simple_rank_ensemble(gnn_df, lstm_df, w_gnn=0.4, w_lstm=0.6):
    """固定权重排名集成"""
    if w_gnn < 0 or w_lstm < 0 or not np.isclose(w_gnn + w_lstm, 1.0):
        raise ValueError('固定权重必须非负且合计为 1')
    merged = pd.merge(
        gnn_df[['ts_code', 'date', 'actual_ret', 'pred_gnn']],
        lstm_df[['ts_code', 'date', 'pred_lstm_b']],
        on=['ts_code', 'date'],
        how='inner', validate='one_to_one'
    )

    if merged.empty:
        return None

    results = []
    for dt in merged['date'].unique():
        m_data = merged[merged['date'] == dt].copy()
        if len(m_data) < 5:
            continue

        m_data['rank_gnn'] = m_data['pred_gnn'].rank(ascending=False)
        m_data['rank_lstm'] = m_data['pred_lstm_b'].rank(ascending=False)
        m_data['rank_ensemble'] = w_gnn * m_data['rank_gnn'] + w_lstm * m_data['rank_lstm']
        m_data['pred_ensemble'] = -m_data['rank_ensemble']
        m_data['w_gnn'] = w_gnn
        m_data['w_lstm'] = w_lstm
        results.append(m_data)

    return pd.concat(results, ignore_index=True) if results else None


def validate_backtest_inputs(gnn_df, lstm_df, mkt):
    """Reject skipped signal months, duplicate predictions and stale labels."""
    market = mkt.sort_values(['ts_code', 'date']).copy()
    market['fwd_ret'] = market.groupby('ts_code')['ret'].shift(-1)
    expected_labels = market[['ts_code', 'date', 'fwd_ret']]
    last_market_month = market['date'].max().to_period('M')
    if last_market_month >= pd.Timestamp.today().to_period('M'):
        raise ValueError(f'行情包含尚未结束的月份 {last_market_month}，拒绝当作完整月回测')

    for name, frame, score in (('GNN', gnn_df, 'pred_gnn'),
                               ('LSTM-B', lstm_df, 'pred_lstm_b')):
        if frame.duplicated(['ts_code', 'date']).any():
            raise ValueError(f'{name} 预测存在重复的行业+信号月')
        periods = pd.PeriodIndex(pd.to_datetime(frame['date']).dt.to_period('M').unique())
        if periods.empty or periods.max() != last_market_month:
            raise ValueError(f'{name} 预测未覆盖最新行情月 {last_market_month}')
        missing = pd.period_range(periods.min(), periods.max(), freq='M').difference(periods)
        if len(missing):
            raise ValueError(f'{name} 预测缺失月份: {list(map(str, missing))}')
        if frame[score].isna().any():
            raise ValueError(f'{name} 预测分数含缺失值')
        checked = frame.merge(expected_labels, on=['ts_code', 'date'], how='left', validate='one_to_one')
        both = checked['actual_ret'].notna() & checked['fwd_ret'].notna()
        if both.any() and not np.allclose(checked.loc[both, 'actual_ret'],
                                          checked.loc[both, 'fwd_ret'], atol=1e-6):
            raise ValueError(f'{name} 标签并非信号月的下一月收益')
        if (checked['actual_ret'].notna() != checked['fwd_ret'].notna()).any():
            raise ValueError(f'{name} 标签缺失状态与下一月行情不一致')

    start = max(gnn_df['date'].min(), lstm_df['date'].min()).to_period('M')
    expected = pd.period_range(start, last_market_month, freq='M')
    gnn_months = set(pd.to_datetime(gnn_df['date']).dt.to_period('M'))
    lstm_months = set(pd.to_datetime(lstm_df['date']).dt.to_period('M'))
    missing_common = [str(month) for month in expected
                      if month not in gnn_months or month not in lstm_months]
    if missing_common:
        raise ValueError(f'集成预测缺失月份: {missing_common}')

    for name, frame in (('GNN', gnn_df), ('LSTM-B', lstm_df)):
        counts = frame.assign(month=pd.to_datetime(frame['date']).dt.to_period('M'))
        counts = counts.groupby('month')['actual_ret'].count()
        missing_labels = [str(month) for month in expected[:-1]
                          if counts.get(month, 0) < TOP_K]
        if missing_labels:
            raise ValueError(f'{name} 已完成持有期缺少足够收益标签: {missing_labels}')
    return expected


# ============================================================
#  回测引擎
# ============================================================

def run_topk_strategy(pred_df, pred_col, k=TOP_K, inertia=0.0):
    """
    Top-K 纯多头策略

    参数：
        pred_df: 预测 DataFrame (ts_code, date, actual_ret, pred_col)
        pred_col: 用于排名的列名
        k: 买入行业数
        inertia: 持仓惯性（对上期持仓加分）

    返回：
        port_rets: Series (date → return)
        holdings_log: DataFrame
    """
    dates = sorted(pred_df['date'].unique())
    port_rets = []
    prev_holdings = set()
    holdings_log = []

    for dt in dates:
        m_data = pred_df[pred_df['date'] == dt].copy()
        has_realized_label = m_data['actual_ret'].notna().any()
        m_data = m_data.dropna(subset=[pred_col, 'actual_ret'])
        if len(m_data) < k:
            if has_realized_label:
                raise ValueError(f'{pd.Timestamp(dt):%Y-%m} 仅有 {len(m_data)} 个可用行业，不足 Top-{k}')
            continue

        # 惯性加分
        if inertia > 0 and prev_holdings:
            score_std = m_data[pred_col].std()
            bonus = inertia * score_std
            m_data.loc[m_data['ts_code'].isin(prev_holdings), pred_col] += bonus

        m_data = m_data.sort_values(pred_col, ascending=False)
        top_k = m_data.head(k)

        port_ret = top_k['actual_ret'].mean()
        port_rets.append({'date': dt, 'ret': port_ret})

        current_holdings = set(top_k['ts_code'].tolist())
        turnover = 1 - len(current_holdings & prev_holdings) / k if prev_holdings else 1.0

        holdings_log.append({
            'date': dt,
            'holdings': list(current_holdings),
            'turnover': turnover,
            'port_ret': port_ret,
        })
        prev_holdings = current_holdings

    if not port_rets:
        return pd.Series(dtype=float), pd.DataFrame()

    port_df = pd.DataFrame(port_rets)
    port_df['date'] = pd.to_datetime(port_df['date'])
    return port_df.set_index('date')['ret'], pd.DataFrame(holdings_log)


def calc_metrics(rets, rf=RF_ANNUAL):
    """计算策略绩效"""
    rets = rets.dropna()
    if len(rets) == 0:
        return {}

    n = len(rets)
    n_years = n / 12

    nav = (1 + rets).cumprod()
    total = nav.iloc[-1] - 1
    ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    dd = (nav / nav.cummax() - 1)
    max_dd = dd.min()
    win_rate = (rets > 0).mean()

    return {
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'n_months': n,
    }


def calc_rank_ic(pred_df, pred_col):
    """计算月度 Rank IC 序列"""
    ics = []
    for dt in sorted(pred_df['date'].unique()):
        m = pred_df[pred_df['date'] == dt]
        if len(m) >= 10:
            ic, _ = spearmanr(m[pred_col], m['actual_ret'])
            if np.isfinite(ic):
                ics.append({'date': dt, 'ic': ic})
    return pd.DataFrame(ics)


# ============================================================
#  可视化
# ============================================================

def plot_results(nav_dict, ic_dict, metrics_dict, output_path):
    """绘制回测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 净值曲线
    ax = axes[0, 0]
    colors = {'GNN': '#2196F3', 'LSTM-B': '#FF9800', '集成': '#4CAF50',
              '等权基准': '#9E9E9E', '自适应集成': '#E91E63', 'Regime集成': '#9C27B0'}
    for name, nav in nav_dict.items():
        c = colors.get(name, '#333333')
        lw = 2.0 if name in ['集成', '自适应集成', 'Regime集成'] else 1.2
        ls = '--' if name == '等权基准' else '-'
        ax.plot(nav.index, nav.values, label=name, color=c, linewidth=lw, linestyle=ls)
    ax.set_title('净值曲线')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 月度 IC 柱状图
    ax = axes[0, 1]
    best_ic_name = None
    for name, ic_df in ic_dict.items():
        if ic_df is not None and len(ic_df) > 0:
            if best_ic_name is None:
                best_ic_name = name
            mean_ic = ic_df['ic'].mean()
            ax.bar(range(len(ic_df)), ic_df['ic'].values, alpha=0.4, label=f'{name} (IC={mean_ic:.3f})')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('月度 Rank IC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 绩效对比表
    ax = axes[1, 0]
    ax.axis('off')
    if metrics_dict:
        table_data = []
        col_labels = ['策略', '年化', '夏普', '回撤', '胜率']
        for name, m in metrics_dict.items():
            table_data.append([
                name,
                f"{m.get('annual_return', 0):.1%}",
                f"{m.get('sharpe_ratio', 0):.3f}",
                f"{m.get('max_drawdown', 0):.1%}",
                f"{m.get('win_rate', 0):.1%}",
            ])
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        ax.set_title('绩效对比', pad=20)

    # 4. 超额收益
    ax = axes[1, 1]
    if '等权基准' in nav_dict:
        bench_nav = nav_dict['等权基准']
        for name, nav in nav_dict.items():
            if name == '等权基准':
                continue
            aligned = pd.DataFrame({'s': nav, 'b': bench_nav}).dropna()
            if len(aligned) > 0:
                excess_nav = (1 + aligned['s'] - aligned['b']).cumprod() if False else \
                    aligned['s'] / aligned['b']  # 相对净值
                c = colors.get(name, '#333333')
                ax.plot(excess_nav.index, excess_nav.values, label=name, color=c)
        ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title('相对基准净值')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表保存至: {output_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Step 3：集成 + 回测")
    print("=" * 60)

    # 加载预测
    gnn_path = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')
    lstm_path = os.path.join(OUTPUT_DIR, 'predictions_lstm_b.pkl')

    has_gnn = os.path.exists(gnn_path)
    has_lstm = os.path.exists(lstm_path)

    if not has_gnn and not has_lstm:
        print("错误：没有找到任何预测文件！请先运行 s1 和 s2。")
        return

    gnn_df = pd.read_pickle(gnn_path) if has_gnn else None
    lstm_df = pd.read_pickle(lstm_path) if has_lstm else None

    if has_gnn:
        gnn_df['date'] = pd.to_datetime(gnn_df['date'])
        print(f"  GNN 预测: {len(gnn_df)} 条, "
              f"{gnn_df['date'].min().strftime('%Y-%m')} ~ {gnn_df['date'].max().strftime('%Y-%m')}")
    if has_lstm:
        lstm_df['date'] = pd.to_datetime(lstm_df['date'])
        print(f"  LSTM-B 预测: {len(lstm_df)} 条, "
              f"{lstm_df['date'].min().strftime('%Y-%m')} ~ {lstm_df['date'].max().strftime('%Y-%m')}")

    # 加载行情（用于基准）
    mkt = load_industry_monthly()

    if has_gnn and has_lstm:
        coverage = validate_backtest_inputs(gnn_df, lstm_df, mkt)
        print(f'  输入审计通过：{coverage[0]} ~ {coverage[-1]}，{len(coverage)} 个连续信号月')

    nav_dict = {}
    returns_dict = {}
    ic_dict = {}
    metrics_dict = {}

    # ---- 基准 ----
    bench_source = gnn_df if has_gnn else lstm_df
    bench_dates = bench_source['date'].unique()
    mkt = mkt.sort_values(['ts_code', 'date']).copy()
    mkt['fwd_ret'] = mkt.groupby('ts_code')['ret'].shift(-1)
    bench_rets = mkt[mkt['date'].isin(bench_dates)].groupby('date')['fwd_ret'].mean()
    bench_rets = bench_rets.sort_index()
    bench_nav = (1 + bench_rets).cumprod()
    bench_metrics = calc_metrics(bench_rets)
    nav_dict['等权基准'] = bench_nav
    returns_dict['等权基准'] = bench_rets
    metrics_dict['等权基准'] = bench_metrics

    # ---- GNN 单独 ----
    if has_gnn:
        print("\n回测 GNN 分支...")
        gnn_rets, gnn_holdings = run_topk_strategy(gnn_df, 'pred_gnn', inertia=0.2)
        if len(gnn_rets) > 0:
            nav_dict['GNN'] = (1 + gnn_rets).cumprod()
            returns_dict['GNN'] = gnn_rets
            metrics_dict['GNN'] = calc_metrics(gnn_rets)
            ic_dict['GNN'] = calc_rank_ic(gnn_df, 'pred_gnn')
            avg_turnover = gnn_holdings['turnover'].mean() if 'turnover' in gnn_holdings else 0
            print(f"  年化: {metrics_dict['GNN']['annual_return']:.1%}, "
                  f"夏普: {metrics_dict['GNN']['sharpe_ratio']:.3f}, "
                  f"回撤: {metrics_dict['GNN']['max_drawdown']:.1%}, "
                  f"换手: {avg_turnover:.1%}")

    # ---- LSTM-B 单独 ----
    if has_lstm:
        print("\n回测 LSTM-B 分支...")
        lstm_rets, lstm_holdings = run_topk_strategy(lstm_df, 'pred_lstm_b', inertia=0.2)
        if len(lstm_rets) > 0:
            nav_dict['LSTM-B'] = (1 + lstm_rets).cumprod()
            returns_dict['LSTM-B'] = lstm_rets
            metrics_dict['LSTM-B'] = calc_metrics(lstm_rets)
            ic_dict['LSTM-B'] = calc_rank_ic(lstm_df, 'pred_lstm_b')
            avg_turnover = lstm_holdings['turnover'].mean() if 'turnover' in lstm_holdings else 0
            print(f"  年化: {metrics_dict['LSTM-B']['annual_return']:.1%}, "
                  f"夏普: {metrics_dict['LSTM-B']['sharpe_ratio']:.3f}, "
                  f"回撤: {metrics_dict['LSTM-B']['max_drawdown']:.1%}, "
                  f"换手: {avg_turnover:.1%}")

    # ---- 三组固定权重：同一批预测、同一持有月，避免状态权重过拟合。----
    if has_gnn and has_lstm:
        for code, (w_gnn, w_lstm) in FIXED_WEIGHT_VARIANTS.items():
            label = f'固定{code}'
            ens = simple_rank_ensemble(gnn_df, lstm_df, w_gnn, w_lstm)
            if ens is None:
                continue
            path = os.path.join(OUTPUT_DIR, f'predictions_ensemble_{code}.pkl')
            ens.to_pickle(path)
            if code == '46':
                ens.to_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl'))
            if code == '55':
                ens.to_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble_equal.pkl'))
            returns, _ = run_topk_strategy(ens, 'pred_ensemble', inertia=0.2)
            if returns.empty:
                continue
            nav_dict[label] = (1 + returns).cumprod()
            returns_dict[label] = returns
            metrics_dict[label] = calc_metrics(returns)
            ic_dict[label] = calc_rank_ic(ens, 'pred_ensemble')
            print(f"  {label} (GNN={w_gnn:.0%}, LSTM={w_lstm:.0%}): "
                  f"年化={metrics_dict[label]['annual_return']:.1%}, "
                  f"夏普={metrics_dict[label]['sharpe_ratio']:.3f}, "
                  f"回撤={metrics_dict[label]['max_drawdown']:.1%}")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  绩效汇总（Top-{TOP_K} 纯多头）")
    print(f"{'='*60}")
    for name, m in metrics_dict.items():
        matched_benchmark = calc_metrics(bench_rets.reindex(nav_dict[name].index))
        excess = m.get('annual_return', 0) - matched_benchmark.get('annual_return', 0)
        print(f"  {name:12s}  年化={m.get('annual_return',0):7.1%}  "
              f"夏普={m.get('sharpe_ratio',0):6.3f}  "
              f"回撤={m.get('max_drawdown',0):7.1%}  "
              f"胜率={m.get('win_rate',0):5.1%}  "
              f"超额={excess:+.1%}")

    # IC 汇总
    print(f"\n  {'分支':12s}  {'IC':>8s}  {'ICIR':>8s}  {'IC>0':>8s}")
    for name, ic_df in ic_dict.items():
        if ic_df is not None and len(ic_df) > 0:
            mic = ic_df['ic'].mean()
            sicir = mic / (ic_df['ic'].std() + 1e-8)
            pos = (ic_df['ic'] > 0).mean()
            print(f"  {name:12s}  {mic:8.4f}  {sicir:8.4f}  {pos:8.1%}")

    # ---- 保存 ----
    # 每条收益明确标记信号月和下一持有月，避免把信号年误当收益年。
    monthly_returns = pd.DataFrame(returns_dict).sort_index().dropna(how='all')
    monthly_returns.index.name = 'signal_date'
    monthly_returns.insert(0, 'holding_month', monthly_returns.index + pd.offsets.MonthEnd(1))
    monthly_returns.to_csv(os.path.join(OUTPUT_DIR, 'backtest_monthly_returns.csv'),
                           encoding='utf-8-sig')

    # 净值
    nav_df = pd.DataFrame(nav_dict).dropna(how='all')
    nav_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_nav.csv'), encoding='utf-8-sig')

    # 绩效
    summary_rows = []
    for name, metrics in metrics_dict.items():
        matched_benchmark = calc_metrics(bench_rets.reindex(nav_dict[name].index))
        summary_rows.append({'strategy': name, **metrics,
                             'annual_excess_vs_matched_benchmark':
                             metrics.get('annual_return', np.nan) -
                             matched_benchmark.get('annual_return', np.nan)})
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'backtest_summary.csv'),
        index=False, encoding='utf-8-sig')

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    provenance_files = [gnn_path, lstm_path,
                        gnn_path.replace('.pkl', '.csv'),
                        lstm_path.replace('.pkl', '.csv'),
                        __file__, os.path.join(PROJECT_DIR, 'src', 'config.py'),
                        os.path.join(PROJECT_DIR, 'src', 'data_pipeline', 'industry_monthly.py')]
    if LEVEL == 'l1':
        provenance_files.append(os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv'))
    manifest = {
        'strategy': 'fixed_weights_no_hmm',
        'signal_start': str(coverage[0]) if has_gnn and has_lstm else None,
        'signal_end': str(coverage[-1]) if has_gnn and has_lstm else None,
        'realized_holding_end': str(monthly_returns.loc[
            monthly_returns.get('固定46', pd.Series(dtype=float)).notna(),
            'holding_month'].max().to_period('M')) if '固定46' in monthly_returns else None,
        'top_k': TOP_K,
        'inertia': 0.2,
        'weights': FIXED_WEIGHT_VARIANTS,
        'sha256': {os.path.relpath(path, PROJECT_DIR): sha256(path)
                   for path in provenance_files if os.path.exists(path)},
    }
    with open(os.path.join(OUTPUT_DIR, 'backtest_manifest.json'), 'w', encoding='utf-8') as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)

    # 图表
    plot_results(nav_dict, ic_dict, metrics_dict,
                 os.path.join(OUTPUT_DIR, 'backtest_results.png'))

    print(f"\n回测完成！结果保存至 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
