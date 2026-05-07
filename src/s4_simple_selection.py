# -*- coding: utf-8 -*-
"""
Step 4 (简单版): 行业内动量选股

思路：行业轮动选出 Top-K 行业后，在每个行业内选过去N个月涨幅最大的个股。
不做 beta 估计，纯动量 + 流动性过滤，作为选股基线。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from config import OUTPUT_DIR, TOP_K, RF_ANNUAL, ARIMAX_PROJECT

# ==================== 参数 ====================
TOPN_PER_IND = 10          # 每个行业选 N 只
MOM_LOOKBACK = 60          # 动量回看天数（约3个月）
ADV_MIN = 20000            # 最低日均成交额（千元）
ADV_WINDOW = 20
COMMISSION_RATE = 0.0003
SLIPPAGE_BPS = 5
HOLDING_INERTIA = 0.3      # 持仓惯性

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EXISTING_DATA_DIR = os.path.join(ARIMAX_PROJECT, r"数据\已有数据")
STOCK_DATA_PATH = r"D:\desktop\有意思的事情\量化\项目\天风选股模型\数据\full_market_data_v18.pkl"
SW_MEMBERS_PATH = os.path.join(EXISTING_DATA_DIR, 'ts_sw_members.csv')
SW_EXCLUDE = ['801780.SI', '801790.SI']


# ============================================================
#  数据加载（复用 beta 版的缓存）
# ============================================================

def load_stock_daily():
    """加载个股日线，返回按 code 分组的 dict 加速查询"""
    cache_path = os.path.join(OUTPUT_DIR, '_cache_stock_daily.pkl')
    if os.path.exists(cache_path):
        print("  加载个股日线缓存...")
        df = pd.read_pickle(cache_path)
    else:
        print("  加载个股日线原始数据...")
        data = pd.read_pickle(STOCK_DATA_PATH)
        df = data['df_stock'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        df['ret'] = df.groupby('code')['close'].pct_change()
        df = df[['date', 'code', 'close', 'ret', 'amount']].copy()
        df.to_pickle(cache_path)
        print(f"    {df['code'].nunique()} 只股票, 缓存已保存")

    # 按 code 分组建索引，查询 O(1)
    print("  构建股票索引...")
    stock_dict = {code: grp.set_index('date').sort_index()
                  for code, grp in df.groupby('code')}
    print(f"    {len(stock_dict)} 只股票已索引")
    return df, stock_dict


def load_stock_industry_map():
    sw = pd.read_csv(SW_MEMBERS_PATH)
    cur = sw[sw['is_new'] == 'Y'].copy()
    cur['stock_code'] = cur['ts_code'].str[:6]
    stock_to_ind = dict(zip(cur['stock_code'], cur['l1_code']))
    ind_to_name = dict(zip(cur['l1_code'], cur['l1_name']))
    print(f"  行业映射: {len(stock_to_ind)} 只 -> {len(ind_to_name)} 个行业")
    return stock_to_ind, ind_to_name


# ============================================================
#  选股：行业内动量 Top-N
# ============================================================

def select_stocks_for_month(pred_month, top_industries, stock_dict,
                            stock_to_ind, ind_scores, prev_holdings=None):
    cutoff = pred_month - pd.Timedelta(days=1)
    lookback_start = cutoff - pd.Timedelta(days=int(MOM_LOOKBACK * 1.6))

    all_selected = []

    for ind_code in top_industries:
        ind_stocks = [s for s, i in stock_to_ind.items() if i == ind_code]
        if not ind_stocks:
            continue

        records = []
        for stock_code in ind_stocks:
            if stock_code not in stock_dict:
                continue
            s_data = stock_dict[stock_code].loc[lookback_start:cutoff]

            if len(s_data) < MOM_LOOKBACK // 2:
                continue

            # ADV 过滤
            adv = s_data['amount'].tail(ADV_WINDOW).mean()
            if adv < ADV_MIN:
                continue

            # 涨跌停过滤
            last_ret = s_data['ret'].iloc[-1]
            if abs(last_ret) >= 0.095:
                continue

            # 动量：过去 N 天累计收益
            mom = (1 + s_data['ret'].tail(MOM_LOOKBACK).fillna(0)).prod() - 1

            records.append({
                'stock_code': stock_code,
                'ind_code': ind_code,
                'momentum': mom,
                'adv': adv,
            })

        if not records:
            continue

        mom_df = pd.DataFrame(records)

        # 持仓惯性
        if prev_holdings and HOLDING_INERTIA > 0:
            mom_std = mom_df['momentum'].std()
            if mom_std > 0:
                bonus = HOLDING_INERTIA * mom_std
                mom_df.loc[mom_df['stock_code'].isin(prev_holdings), 'momentum'] += bonus

        # 取动量最高的 Top-N
        mom_df = mom_df.sort_values('momentum', ascending=False).head(TOPN_PER_IND)
        mom_df['month'] = pred_month
        mom_df['ind_score'] = ind_scores.get(ind_code, 0)
        mom_df['rank_in_ind'] = range(1, len(mom_df) + 1)

        all_selected.append(mom_df)

    if not all_selected:
        return pd.DataFrame()
    return pd.concat(all_selected, ignore_index=True)


# ============================================================
#  回测
# ============================================================

def backtest_stock_portfolio(monthly_selections, stock_dict):
    results = []
    prev_holdings = set()

    for month in sorted(monthly_selections['month'].unique()):
        sel = monthly_selections[monthly_selections['month'] == month]
        if sel.empty:
            continue

        stocks = sel['stock_code'].tolist()
        month_end = month + pd.offsets.MonthEnd(0)
        next_month_end = (month + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)

        stock_rets = []
        for code in stocks:
            if code not in stock_dict:
                continue
            # stock_dict values are indexed by date, use slice (exclusive start via index trick)
            s_all = stock_dict[code]
            s_data = s_all[(s_all.index > month_end) & (s_all.index <= next_month_end)]
            if len(s_data) > 0:
                monthly_ret = (1 + s_data['ret'].fillna(0)).prod() - 1
                stock_rets.append(monthly_ret)

        if not stock_rets:
            continue

        port_ret = np.mean(stock_rets)
        current_holdings = set(stocks)
        turnover = 1 - len(current_holdings & prev_holdings) / max(len(current_holdings), 1) \
            if prev_holdings else 1.0
        cost = turnover * (2 * COMMISSION_RATE + 2 * SLIPPAGE_BPS / 10000)

        results.append({
            'date': next_month_end,
            'ret_gross': port_ret,
            'ret_net': port_ret - cost,
            'n_stocks': len(stocks),
            'n_industries': sel['ind_code'].nunique(),
            'turnover': turnover,
            'cost': cost,
        })
        prev_holdings = current_holdings

    return pd.DataFrame(results)


def calc_metrics(rets, rf=RF_ANNUAL):
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
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    win_rate = (rets > 0).mean()
    return {
        'annual_return': ann_ret, 'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe, 'max_drawdown': max_dd,
        'win_rate': win_rate, 'n_months': n,
    }


def plot_results(bt_results, ind_nav, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if not bt_results.empty:
        nav_net = (1 + bt_results.set_index('date')['ret_net']).cumprod()
        nav_gross = (1 + bt_results.set_index('date')['ret_gross']).cumprod()
        ax.plot(nav_net.index, nav_net.values, label='动量选股(扣费)', color='#E91E63', linewidth=2)
        ax.plot(nav_gross.index, nav_gross.values, label='动量选股(毛)', color='#FF9800', linewidth=1.5, linestyle='--')
    if ind_nav is not None:
        ax.plot(ind_nav.index, ind_nav.values, label='行业轮动', color='#2196F3', linewidth=1.5)
    ax.set_title('净值对比: 动量选股 vs 行业轮动')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['ret_net'].values, alpha=0.6, color='#4CAF50')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=bt_results['ret_net'].mean(), color='red', linewidth=1, linestyle='--',
                    label=f"均值={bt_results['ret_net'].mean():.2%}")
    ax.set_title('月度收益')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['turnover'].values, alpha=0.6, color='#9C27B0')
        ax.axhline(y=bt_results['turnover'].mean(), color='red', linewidth=1, linestyle='--',
                    label=f"平均换手={bt_results['turnover'].mean():.1%}")
    ax.set_title('月度换手率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    if not bt_results.empty:
        m_net = calc_metrics(bt_results.set_index('date')['ret_net'])
        m_gross = calc_metrics(bt_results.set_index('date')['ret_gross'])
        table_data.append(['动量选股(扣费)', f"{m_net.get('annual_return',0):.1%}",
                           f"{m_net.get('sharpe_ratio',0):.3f}",
                           f"{m_net.get('max_drawdown',0):.1%}",
                           f"{m_net.get('win_rate',0):.1%}"])
        table_data.append(['动量选股(毛)', f"{m_gross.get('annual_return',0):.1%}",
                           f"{m_gross.get('sharpe_ratio',0):.3f}",
                           f"{m_gross.get('max_drawdown',0):.1%}",
                           f"{m_gross.get('win_rate',0):.1%}"])
    if ind_nav is not None:
        ind_rets = ind_nav.pct_change().dropna()
        m_ind = calc_metrics(ind_rets)
        table_data.append(['行业轮动', f"{m_ind.get('annual_return',0):.1%}",
                           f"{m_ind.get('sharpe_ratio',0):.3f}",
                           f"{m_ind.get('max_drawdown',0):.1%}",
                           f"{m_ind.get('win_rate',0):.1%}"])
    if table_data:
        table = ax.table(cellText=table_data,
                         colLabels=['策略', '年化', '夏普', '回撤', '胜率'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
    ax.set_title('绩效对比', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表保存至: {output_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Step 4 (简单版): 行业内动量选股")
    print("=" * 60)

    # 加载预测
    ensemble_path = os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl')
    gnn_path = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')

    if os.path.exists(ensemble_path):
        pred_df = pd.read_pickle(ensemble_path)
        pred_col = 'pred_ensemble'
        print(f"  使用集成预测")
    elif os.path.exists(gnn_path):
        pred_df = pd.read_pickle(gnn_path)
        pred_col = 'pred_gnn'
        print(f"  使用 GNN 预测")
    else:
        print("错误: 没有预测文件!")
        return

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    print(f"  预测: {len(pred_df)} 条, "
          f"{pred_df['date'].min().strftime('%Y-%m')} ~ "
          f"{pred_df['date'].max().strftime('%Y-%m')}")

    # 行业级净值
    nav_path = os.path.join(OUTPUT_DIR, 'backtest_nav.csv')
    ind_nav = None
    if os.path.exists(nav_path):
        nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
        for col in ['自适应集成', '集成', 'GNN']:
            if col in nav_df.columns:
                ind_nav = nav_df[col].dropna()
                print(f"  行业级对比: {col}")
                break

    # 加载数据
    stock_to_ind, ind_to_name = load_stock_industry_map()
    df_stock, stock_dict = load_stock_daily()

    # 断点续传
    ckpt_path = os.path.join(OUTPUT_DIR, '_ckpt_simple.pkl')
    all_selections = []
    prev_holdings = set()
    done_months = set()

    if os.path.exists(ckpt_path):
        ckpt = pd.read_pickle(ckpt_path)
        all_selections = ckpt.get('selections', [])
        prev_holdings = ckpt.get('prev_holdings', set())
        done_months = ckpt.get('done_months', set())
        print(f"  断点续传: 已完成 {len(done_months)} 个月")

    # 逐月选股
    pred_months = sorted(pred_df['date'].unique())
    remaining = [m for m in pred_months if m not in done_months]
    print(f"\n  开始逐月选股 (剩余 {len(remaining)}/{len(pred_months)} 个月)...")

    for i, month in enumerate(remaining):
        m_pred = pred_df[pred_df['date'] == month].copy()
        m_pred = m_pred.sort_values(pred_col, ascending=False)
        top_k = m_pred.head(TOP_K)
        top_industries = top_k['ts_code'].tolist()
        ind_scores = dict(zip(top_k['ts_code'], top_k[pred_col]))

        selected = select_stocks_for_month(
            pred_month=pd.Timestamp(month),
            top_industries=top_industries,
            stock_dict=stock_dict,
            stock_to_ind=stock_to_ind,
            ind_scores=ind_scores,
            prev_holdings=prev_holdings,
        )

        if not selected.empty:
            all_selections.append(selected)
            prev_holdings = set(selected['stock_code'].tolist())
        else:
            prev_holdings = set()

        done_months.add(month)

        # 每5个月保存一次checkpoint
        if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
            pd.to_pickle({'selections': all_selections,
                          'prev_holdings': prev_holdings,
                          'done_months': done_months}, ckpt_path)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            n_sel = len(selected) if not selected.empty else 0
            ind_names = [ind_to_name.get(c, c) for c in top_industries[:3]]
            print(f"    [{len(done_months)}/{len(pred_months)}] "
                  f"{pd.Timestamp(month).strftime('%Y-%m')}: "
                  f"{n_sel} 只, "
                  f"Top: {', '.join(ind_names[:3])}... ({elapsed:.0f}s)")

    # 清理checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    if not all_selections:
        print("错误: 无选股结果!")
        return

    monthly_selections = pd.concat(all_selections, ignore_index=True)
    print(f"\n  选股完成: {len(monthly_selections)} 条, "
          f"{monthly_selections['month'].nunique()} 个月")

    # 回测
    print("\n  回测...")
    bt_results = backtest_stock_portfolio(monthly_selections, stock_dict)

    if bt_results.empty:
        print("错误: 回测无结果!")
        return

    # 绩效
    print(f"\n{'='*60}")
    print(f"  动量选股绩效")
    print(f"{'='*60}")

    m_net = calc_metrics(bt_results.set_index('date')['ret_net'])
    m_gross = calc_metrics(bt_results.set_index('date')['ret_gross'])

    print(f"  动量选股(扣费): 年化={m_net.get('annual_return',0):.1%}, "
          f"夏普={m_net.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_net.get('max_drawdown',0):.1%}, "
          f"胜率={m_net.get('win_rate',0):.1%}")
    print(f"  动量选股(毛):   年化={m_gross.get('annual_return',0):.1%}, "
          f"夏普={m_gross.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_gross.get('max_drawdown',0):.1%}, "
          f"胜率={m_gross.get('win_rate',0):.1%}")

    print(f"\n  平均持股: {bt_results['n_stocks'].mean():.0f} 只/月")
    print(f"  平均换手: {bt_results['turnover'].mean():.1%}")

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s")

    # 保存
    monthly_selections.to_pickle(os.path.join(OUTPUT_DIR, 'stock_selections_simple.pkl'))
    bt_results.to_csv(os.path.join(OUTPUT_DIR, 'stock_backtest_simple.csv'),
                      index=False, encoding='utf-8-sig')
    plot_results(bt_results, ind_nav,
                 os.path.join(OUTPUT_DIR, 'stock_backtest_simple.png'))
    print(f"  结果保存至 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
