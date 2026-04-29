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
    Regime 条件集成：按宏观状态分别学习 GNN/LSTM-B 的历史 IC，
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
    regime_gnn_ics = {r: [] for r in range(-1, 4)}
    regime_lstm_ics = {r: [] for r in range(-1, 4)}

    for dt in dates:
        m_data = merged[merged['date'] == dt].copy()
        if len(m_data) < 5:
            continue

        regime = m_data['regime'].iloc[0]

        m_data['rank_gnn'] = m_data['pred_gnn'].rank(ascending=False)
        m_data['rank_lstm'] = m_data['pred_lstm_b'].rank(ascending=False)

        # 计算当月 IC
        ic_gnn, _ = spearmanr(m_data['pred_gnn'], m_data['actual_ret'])
        ic_lstm, _ = spearmanr(m_data['pred_lstm_b'], m_data['actual_ret'])
        ic_gnn = ic_gnn if np.isfinite(ic_gnn) else 0
        ic_lstm = ic_lstm if np.isfinite(ic_lstm) else 0

        for r in range(-1, 4):
            regime_gnn_ics[r].append(ic_gnn if regime == r else None)
            regime_lstm_ics[r].append(ic_lstm if regime == r else None)

        # 用同一 regime 下的历史 IC 做权重
        past_gnn = [x for x in regime_gnn_ics[regime][:-1] if x is not None][-ic_lookback:]
        past_lstm = [x for x in regime_lstm_ics[regime][:-1] if x is not None][-ic_lookback:]

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

        m_data['rank_ensemble'] = w_gnn * m_data['rank_gnn'] + w_lstm * m_data['rank_lstm']
        m_data['pred_ensemble'] = -m_data['rank_ensemble']
        m_data['w_gnn'] = w_gnn
        m_data['w_lstm'] = w_lstm
        m_data['regime'] = regime

        results.append(m_data)

    return pd.concat(results, ignore_index=True) if results else None


def simple_rank_ensemble(gnn_df, lstm_df, w_gnn=0.5, w_lstm=0.5):
    """固定权重排名集成"""
    merged = pd.merge(
        gnn_df[['ts_code', 'date', 'actual_ret', 'pred_gnn']],
        lstm_df[['ts_code', 'date', 'pred_lstm_b']],
        on=['ts_code', 'date'],
        how='inner'
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
        results.append(m_data)

    return pd.concat(results, ignore_index=True) if results else None


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
        m_data = m_data.dropna(subset=[pred_col, 'actual_ret'])
        if len(m_data) < k:
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
    regime_path = os.path.join(OUTPUT_DIR, 'regime_labels.pkl')

    has_gnn = os.path.exists(gnn_path)
    has_lstm = os.path.exists(lstm_path)
    has_regime = os.path.exists(regime_path)

    if not has_gnn and not has_lstm:
        print("错误：没有找到任何预测文件！请先运行 s1 和 s2。")
        return

    gnn_df = pd.read_pickle(gnn_path) if has_gnn else None
    lstm_df = pd.read_pickle(lstm_path) if has_lstm else None
    regime_df = pd.read_pickle(regime_path) if has_regime else None

    if has_gnn:
        gnn_df['date'] = pd.to_datetime(gnn_df['date'])
        print(f"  GNN 预测: {len(gnn_df)} 条, "
              f"{gnn_df['date'].min().strftime('%Y-%m')} ~ {gnn_df['date'].max().strftime('%Y-%m')}")
    if has_lstm:
        lstm_df['date'] = pd.to_datetime(lstm_df['date'])
        print(f"  LSTM-B 预测: {len(lstm_df)} 条, "
              f"{lstm_df['date'].min().strftime('%Y-%m')} ~ {lstm_df['date'].max().strftime('%Y-%m')}")
    if has_regime:
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        print(f"  Regime: {len(regime_df)} 月")

    # 加载行情（用于基准）
    mkt = load_industry_monthly()

    nav_dict = {}
    ic_dict = {}
    metrics_dict = {}

    # ---- 基准 ----
    bench_source = gnn_df if has_gnn else lstm_df
    bench_dates = bench_source['date'].unique()
    bench_rets = mkt[mkt['date'].isin(bench_dates)].groupby('date')['ret'].mean()
    bench_rets = bench_rets.sort_index()
    bench_nav = (1 + bench_rets).cumprod()
    bench_metrics = calc_metrics(bench_rets)
    nav_dict['等权基准'] = bench_nav
    metrics_dict['等权基准'] = bench_metrics

    # ---- GNN 单独 ----
    if has_gnn:
        print("\n回测 GNN 分支...")
        gnn_rets, gnn_holdings = run_topk_strategy(gnn_df, 'pred_gnn', inertia=0.2)
        if len(gnn_rets) > 0:
            nav_dict['GNN'] = (1 + gnn_rets).cumprod()
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
            metrics_dict['LSTM-B'] = calc_metrics(lstm_rets)
            ic_dict['LSTM-B'] = calc_rank_ic(lstm_df, 'pred_lstm_b')
            avg_turnover = lstm_holdings['turnover'].mean() if 'turnover' in lstm_holdings else 0
            print(f"  年化: {metrics_dict['LSTM-B']['annual_return']:.1%}, "
                  f"夏普: {metrics_dict['LSTM-B']['sharpe_ratio']:.3f}, "
                  f"回撤: {metrics_dict['LSTM-B']['max_drawdown']:.1%}, "
                  f"换手: {avg_turnover:.1%}")

    # ---- 集成（两个分支都有时）----
    if has_gnn and has_lstm:
        # 固定等权集成
        print("\n回测等权集成...")
        equal_df = simple_rank_ensemble(gnn_df, lstm_df, 0.5, 0.5)
        if equal_df is not None:
            eq_rets, eq_holdings = run_topk_strategy(equal_df, 'pred_ensemble', inertia=0.2)
            if len(eq_rets) > 0:
                nav_dict['集成'] = (1 + eq_rets).cumprod()
                metrics_dict['集成'] = calc_metrics(eq_rets)
                ic_dict['集成'] = calc_rank_ic(equal_df, 'pred_ensemble')
                print(f"  年化: {metrics_dict['集成']['annual_return']:.1%}, "
                      f"夏普: {metrics_dict['集成']['sharpe_ratio']:.3f}, "
                      f"回撤: {metrics_dict['集成']['max_drawdown']:.1%}")

        # 自适应集成
        print("\n回测自适应集成...")
        adaptive_df = adaptive_ensemble(gnn_df, lstm_df, ic_lookback=12)
        if adaptive_df is not None:
            ad_rets, ad_holdings = run_topk_strategy(adaptive_df, 'pred_ensemble', inertia=0.2)
            if len(ad_rets) > 0:
                nav_dict['自适应集成'] = (1 + ad_rets).cumprod()
                metrics_dict['自适应集成'] = calc_metrics(ad_rets)
                ic_dict['自适应集成'] = calc_rank_ic(adaptive_df, 'pred_ensemble')
                print(f"  年化: {metrics_dict['自适应集成']['annual_return']:.1%}, "
                      f"夏普: {metrics_dict['自适应集成']['sharpe_ratio']:.3f}, "
                      f"回撤: {metrics_dict['自适应集成']['max_drawdown']:.1%}")

        # Regime 条件集成
        if has_regime:
            print("\n回测 Regime 条件集成...")
            regime_ens_df = regime_ensemble(gnn_df, lstm_df, regime_df, ic_lookback=12)
            if regime_ens_df is not None:
                re_rets, re_holdings = run_topk_strategy(regime_ens_df, 'pred_ensemble', inertia=0.2)
                if len(re_rets) > 0:
                    nav_dict['Regime集成'] = (1 + re_rets).cumprod()
                    metrics_dict['Regime集成'] = calc_metrics(re_rets)
                    ic_dict['Regime集成'] = calc_rank_ic(regime_ens_df, 'pred_ensemble')
                    print(f"  年化: {metrics_dict['Regime集成']['annual_return']:.1%}, "
                          f"夏普: {metrics_dict['Regime集成']['sharpe_ratio']:.3f}, "
                          f"回撤: {metrics_dict['Regime集成']['max_drawdown']:.1%}")

                    # 保存 Regime集成
                    regime_ens_df.to_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl'))

                    # 分 regime 绩效
                    regime_names = {-1: '未知', 0: '衰退', 1: '复苏', 2: '扩张', 3: '过热'}
                    print(f"\n  分 Regime 绩效:")
                    for r in sorted(regime_ens_df['regime'].unique()):
                        r_data = regime_ens_df[regime_ens_df['regime'] == r]
                        r_rets, _ = run_topk_strategy(r_data, 'pred_ensemble')
                        if len(r_rets) > 0:
                            r_metrics = calc_metrics(r_rets)
                            name = regime_names.get(r, f'R{r}')
                            print(f"    {name}({r}): 年化={r_metrics.get('annual_return',0):.1%}, "
                                  f"夏普={r_metrics.get('sharpe_ratio',0):.3f}, "
                                  f"月数={r_metrics.get('n_months',0)}")

                    # 打印权重变化
                    w_log = regime_ens_df.groupby('date')[['w_gnn', 'w_lstm', 'regime']].first()
                    print(f"\n  权重变化 (最近5期):")
                    for dt, row in w_log.tail(5).iterrows():
                        r_name = regime_names.get(int(row['regime']), '?')
                        print(f"    {pd.Timestamp(dt).strftime('%Y-%m')} [{r_name}]: "
                              f"GNN={row['w_gnn']:.2f}, LSTM={row['w_lstm']:.2f}")
        else:
            # 没有 regime 时，保存自适应集成
            if adaptive_df is not None:
                adaptive_df.to_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl'))

        # 保存等权集成（供 ENSEMBLE_MODE='equal' 使用）
        if equal_df is not None:
            equal_df.to_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble_equal.pkl'))

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  绩效汇总（Top-{TOP_K} 纯多头）")
    print(f"{'='*60}")
    for name, m in metrics_dict.items():
        excess = m.get('annual_return', 0) - bench_metrics.get('annual_return', 0)
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
    # 净值
    nav_df = pd.DataFrame(nav_dict)
    nav_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_nav.csv'), encoding='utf-8-sig')

    # 绩效
    summary_rows = [{'strategy': k, **v} for k, v in metrics_dict.items()]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'backtest_summary.csv'),
        index=False, encoding='utf-8-sig')

    # 图表
    plot_results(nav_dict, ic_dict, metrics_dict,
                 os.path.join(OUTPUT_DIR, 'backtest_results.png'))

    print(f"\n回测完成！结果保存至 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
