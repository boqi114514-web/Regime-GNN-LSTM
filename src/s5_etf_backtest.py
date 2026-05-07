# -*- coding: utf-8 -*-
"""
Step 5: ETF实盘回测

思路：
  1. 从tushare获取所有行业ETF基本信息
  2. 建立申万一级行业 → 可交易ETF的映射（手动+自动匹配）
  3. 下载ETF日线行情
  4. 用行业轮动模型的Top-K预测，直接买对应ETF
  5. 与理想行业轮动回测对比

核心价值：验证行业轮动策略的ETF可执行性
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import io
import time
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from config import OUTPUT_DIR, TOP_K, RF_ANNUAL
from data_pipeline.tushare_config import get_pro as _get_pro

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

pro = _get_pro()

# ==================== 申万行业 → ETF 映射 ====================
# ⚠️ 已知问题（2026-04-04 审计）：
#   1. 多个ETF跟踪标的与申万行业严重错配：
#      - 801080 电子 → 159813 实际是半导体ETF（国证半导体芯片指数），不覆盖整个电子行业
#      - 801160 公用事业 → 159625 实际是绿色电力ETF（国证绿色电力指数），不等于公用事业
#      - 801760 传媒 → 516620 实际是影视ETF（中证影视主题指数），只是传媒子行业
#      - 801890 机械设备 → 159542 实际是工程机械ETF，只是子行业且2024年6月才上市
#   2. 中等偏差：
#      - 801720 建筑装饰 → 516950 中证基建指数，比建筑装饰宽泛
#      - 801730 电力设备 → 516160 中证新能源指数，有重叠但不完全一致
#      - 801150 医药生物 → 512010 沪深300医药卫生，只取300成分股覆盖不全
#   3. 18/25个ETF在回测起始(2019-01)后才上市，前期可交易ETF严重不足
#   整个ETF回测模块结果不可信，需要重新整理映射后才能使用。
#
SW_TO_ETF = {
    '801010.SI': {'etf': '159825.SZ', 'name': '农业ETF'},
    '801030.SI': {'etf': '516220.SH', 'name': '化工ETF'},
    '801040.SI': {'etf': '515210.SH', 'name': '钢铁ETF'},
    '801050.SI': {'etf': '512400.SH', 'name': '有色金属ETF'},
    '801080.SI': {'etf': '159813.SZ', 'name': '半导体ETF'},          # ⚠️ 错配：跟踪半导体≠电子
    '801110.SI': {'etf': '159996.SZ', 'name': '家电ETF'},
    '801120.SI': {'etf': '515170.SH', 'name': '食品饮料ETF'},
    '801130.SI': {'etf': None,        'name': '纺织服饰（无ETF）'},
    '801140.SI': {'etf': None,        'name': '轻工制造（无ETF）'},
    '801150.SI': {'etf': '512010.SH', 'name': '医药ETF'},            # ⚠️ 偏差：仅沪深300医药
    '801160.SI': {'etf': '159625.SZ', 'name': '绿色电力ETF'},        # ⚠️ 错配：绿色电力≠公用事业
    '801170.SI': {'etf': '561320.SH', 'name': '交通运输ETF'},
    '801180.SI': {'etf': '512200.SH', 'name': '房地产ETF'},
    '801200.SI': {'etf': None,        'name': '商贸零售（无ETF）'},
    '801210.SI': {'etf': None,        'name': '社会服务（无ETF）'},
    '801230.SI': {'etf': None,        'name': '综合（无ETF）'},
    '801710.SI': {'etf': '159745.SZ', 'name': '建材ETF'},
    '801720.SI': {'etf': '516950.SH', 'name': '基建ETF'},            # ⚠️ 偏差：中证基建≠建筑装饰
    '801730.SI': {'etf': '516160.SH', 'name': '新能源ETF'},          # ⚠️ 偏差：新能源≈电力设备但不完全一致
    '801740.SI': {'etf': '512810.SH', 'name': '军工ETF'},
    '801750.SI': {'etf': '512720.SH', 'name': '计算机ETF'},
    '801760.SI': {'etf': '516620.SH', 'name': '影视ETF'},            # ⚠️ 错配：影视≠传媒
    '801770.SI': {'etf': '515880.SH', 'name': '通信ETF'},
    '801780.SI': {'etf': '512800.SH', 'name': '银行ETF'},
    '801790.SI': {'etf': '512070.SH', 'name': '非银ETF'},
    '801880.SI': {'etf': '516110.SH', 'name': '汽车ETF'},
    '801890.SI': {'etf': '159542.SZ', 'name': '工程机械ETF'},        # ⚠️ 错配：工程机械≠机械设备，且2024-06才上市
    '801950.SI': {'etf': '515220.SH', 'name': '煤炭ETF'},
    '801960.SI': {'etf': '159731.SZ', 'name': '石化ETF'},
    '801970.SI': {'etf': '512580.SH', 'name': '环保ETF'},
    '801980.SI': {'etf': None,        'name': '美容护理（无ETF）'},
}


# ============================================================
#  数据下载
# ============================================================

def download_etf_daily(etf_codes, start_date='20170101', end_date=None):
    """批量下载ETF日线行情（支持增量日期更新）"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y%m%d')

    cache_path = os.path.join(OUTPUT_DIR, '_cache_etf_daily.pkl')
    existing = None
    incremental_start = start_date

    if os.path.exists(cache_path):
        print("  加载ETF日线缓存...")
        existing = pd.read_pickle(cache_path)
        cached_codes = set(existing['ts_code'].unique())
        missing_codes = [c for c in etf_codes if c not in cached_codes]

        # 检查日期是否滞后（允许 5 天宽限）
        cache_max = existing['trade_date'].max()
        end_ts = pd.to_datetime(end_date, format='%Y%m%d')
        is_stale = (end_ts - cache_max).days > 5

        if not missing_codes and not is_stale:
            print(f"    {len(cached_codes)} 只ETF已缓存, 最新 {cache_max.date()}")
            return existing

        if is_stale:
            # 增量：对已有+新增 ETF 从缓存末日起拉新数据
            incremental_start = (cache_max + pd.Timedelta(days=1)).strftime('%Y%m%d')
            download_codes = list(cached_codes | set(etf_codes))
            print(f"    数据截至 {cache_max.date()}, 增量补充至 {end_date} ({len(download_codes)} 只ETF)")
        else:
            download_codes = missing_codes
            print(f"    需补充下载 {len(missing_codes)} 只新ETF")
    else:
        download_codes = etf_codes

    all_dfs = []
    for i, code in enumerate(download_codes):
        try:
            df = pro.fund_daily(ts_code=code, start_date=incremental_start, end_date=end_date,
                                fields='ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount')
            if df is not None and not df.empty:
                all_dfs.append(df)
                print(f"    [{i+1}/{len(download_codes)}] {code}: +{len(df)} 条")
            else:
                print(f"    [{i+1}/{len(download_codes)}] {code}: 无新数据")
            time.sleep(0.3)
        except Exception as e:
            print(f"    [{i+1}/{len(download_codes)}] {code}: 错误 {e}")
            time.sleep(1)

    if all_dfs:
        new_df = pd.concat(all_dfs, ignore_index=True)
        if existing is not None:
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df['trade_date'] = pd.to_datetime(new_df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        new_df = (new_df.drop_duplicates(subset=['ts_code', 'trade_date'])
                  .sort_values(['ts_code', 'trade_date']).reset_index(drop=True))
        new_df.to_pickle(cache_path)
        print(f"    共 {new_df['ts_code'].nunique()} 只ETF, 最新 {new_df['trade_date'].max().date()}, 已缓存")
        return new_df

    return existing if existing is not None else pd.DataFrame()


def validate_etf_mapping():
    """验证ETF映射：检查每个ETF的上市时间和规模"""
    print("\n验证ETF映射...")
    etf_codes = [v['etf'] for v in SW_TO_ETF.values() if v['etf'] is not None]

    # 获取ETF基本信息
    etf_info = {}
    for code in etf_codes:
        try:
            df = pro.etf_basic(ts_code=code,
                               fields='ts_code,extname,index_code,index_name,list_date,list_status')
            if df is not None and not df.empty:
                etf_info[code] = df.iloc[0]
            time.sleep(0.2)
        except:
            pass

    print(f"\n  {'申万行业':<12s} {'ETF代码':<12s} {'ETF名称':<20s} {'上市日期':<12s} {'状态'}")
    print("  " + "-" * 70)

    valid_count = 0
    for sw_code, info in SW_TO_ETF.items():
        etf_code = info['etf']
        etf_name = info['name']
        if etf_code is None:
            print(f"  {sw_code:<12s} {'---':<12s} {etf_name:<20s} {'---':<12s} 无可用ETF")
            continue

        if etf_code in etf_info:
            ei = etf_info[etf_code]
            list_date = ei.get('list_date', '?')
            status = ei.get('list_status', '?')
            idx_name = ei.get('index_name', '?')
            print(f"  {sw_code:<12s} {etf_code:<12s} {etf_name:<20s} {list_date:<12s} {status}  跟踪:{idx_name}")
            valid_count += 1
        else:
            print(f"  {sw_code:<12s} {etf_code:<12s} {etf_name:<20s} {'查询失败':<12s}")

    print(f"\n  有效ETF映射: {valid_count}/{len(SW_TO_ETF)}")
    return etf_info


# ============================================================
#  ETF回测
# ============================================================

def etf_monthly_returns(etf_daily, etf_code):
    """计算单只ETF的月度收益率"""
    df = etf_daily[etf_daily['ts_code'] == etf_code].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df = df.set_index('trade_date').sort_index()
    # 月末收盘价
    monthly = df['close'].resample('ME').last().dropna()
    rets = monthly.pct_change().dropna()
    return rets


def run_etf_backtest(pred_df, pred_col, etf_daily, k=TOP_K, inertia=0.2):
    """
    用行业轮动预测直接买ETF

    对于没有ETF映射的行业，跳过并从下一个行业补位
    """
    dates = sorted(pred_df['date'].unique())
    results = []
    prev_holdings = set()

    # 预计算所有ETF月度收益
    etf_monthly = {}
    for sw_code, info in SW_TO_ETF.items():
        if info['etf'] is not None:
            rets = etf_monthly_returns(etf_daily, info['etf'])
            if not rets.empty:
                etf_monthly[sw_code] = rets

    available_industries = set(etf_monthly.keys())
    print(f"  有月度收益数据的行业ETF: {len(available_industries)}")

    for dt in dates:
        m_data = pred_df[pred_df['date'] == dt].copy()
        m_data = m_data.dropna(subset=[pred_col])
        m_data = m_data.sort_values(pred_col, ascending=False)

        # 从排名最高的行业开始，跳过没有ETF的，直到选够k个
        selected = []
        for _, row in m_data.iterrows():
            ind = row['ts_code']
            if ind in available_industries:
                selected.append(ind)
            if len(selected) >= k:
                break

        if not selected:
            continue

        # 计算组合收益：找到最接近的月末日期
        dt_ts = pd.Timestamp(dt)
        month_rets = []
        for ind in selected:
            ret_series = etf_monthly[ind]
            # 找最接近dt的月份
            closest_dates = ret_series.index[ret_series.index >= dt_ts - pd.Timedelta(days=5)]
            if len(closest_dates) > 0:
                closest = closest_dates[0]
                month_rets.append(ret_series[closest])

        if not month_rets:
            continue

        port_ret = np.mean(month_rets)
        current_holdings = set(selected)
        turnover = 1 - len(current_holdings & prev_holdings) / max(len(current_holdings), 1) \
            if prev_holdings else 1.0

        # ETF交易成本更低
        cost = turnover * 0.0003  # ETF佣金约万3，无印花税

        results.append({
            'date': dt,
            'ret_gross': port_ret,
            'ret_net': port_ret - cost,
            'n_etfs': len(month_rets),
            'industries': selected,
            'turnover': turnover,
            'cost': cost,
        })
        prev_holdings = current_holdings

    return pd.DataFrame(results)


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
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    win_rate = (rets > 0).mean()
    return {
        'annual_return': ann_ret, 'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe, 'max_drawdown': max_dd,
        'win_rate': win_rate, 'n_months': n,
    }


def plot_etf_results(etf_bt, ind_nav, output_path):
    """绘制ETF回测 vs 行业轮动对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 净值对比
    ax = axes[0, 0]
    if not etf_bt.empty:
        etf_nav = (1 + etf_bt.set_index('date')['ret_net']).cumprod()
        etf_nav_gross = (1 + etf_bt.set_index('date')['ret_gross']).cumprod()
        ax.plot(etf_nav.index, etf_nav.values,
                label='ETF策略(扣费)', color='#E91E63', linewidth=2)
        ax.plot(etf_nav_gross.index, etf_nav_gross.values,
                label='ETF策略(毛)', color='#FF9800', linewidth=1.5, linestyle='--')
    if ind_nav is not None:
        ax.plot(ind_nav.index, ind_nav.values,
                label='行业轮动(理想)', color='#2196F3', linewidth=1.5)
    ax.set_title('净值对比: ETF实盘 vs 行业轮动理想')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 月度收益
    ax = axes[0, 1]
    if not etf_bt.empty:
        ax.bar(range(len(etf_bt)), etf_bt['ret_net'].values,
               alpha=0.6, color='#4CAF50')
        ax.axhline(y=0, color='black', linewidth=0.5)
        mean_ret = etf_bt['ret_net'].mean()
        ax.axhline(y=mean_ret, color='red', linewidth=1, linestyle='--',
                    label=f'均值={mean_ret:.2%}')
    ax.set_title('ETF策略月度收益')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 月度换手
    ax = axes[1, 0]
    if not etf_bt.empty:
        ax.bar(range(len(etf_bt)), etf_bt['turnover'].values,
               alpha=0.6, color='#9C27B0')
        avg_turn = etf_bt['turnover'].mean()
        ax.axhline(y=avg_turn, color='red', linewidth=1, linestyle='--',
                    label=f'平均换手={avg_turn:.1%}')
    ax.set_title('月度换手率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 绩效表
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    if not etf_bt.empty:
        m_net = calc_metrics(etf_bt.set_index('date')['ret_net'])
        m_gross = calc_metrics(etf_bt.set_index('date')['ret_gross'])
        table_data.append(['ETF(扣费)', f"{m_net.get('annual_return',0):.1%}",
                           f"{m_net.get('sharpe_ratio',0):.3f}",
                           f"{m_net.get('max_drawdown',0):.1%}",
                           f"{m_net.get('win_rate',0):.1%}"])
        table_data.append(['ETF(毛)', f"{m_gross.get('annual_return',0):.1%}",
                           f"{m_gross.get('sharpe_ratio',0):.3f}",
                           f"{m_gross.get('max_drawdown',0):.1%}",
                           f"{m_gross.get('win_rate',0):.1%}"])
    if ind_nav is not None:
        ind_rets = ind_nav.pct_change().dropna()
        m_ind = calc_metrics(ind_rets)
        table_data.append(['行业轮动(理想)', f"{m_ind.get('annual_return',0):.1%}",
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
    print("  Step 5: ETF实盘回测")
    print("=" * 60)

    # ---- 1. 验证ETF映射 ----
    etf_info = validate_etf_mapping()

    # ---- 2. 下载ETF日线 ----
    print("\n下载ETF日线行情...")
    etf_codes = [v['etf'] for v in SW_TO_ETF.values() if v['etf'] is not None]
    etf_daily = download_etf_daily(etf_codes, start_date='20170101', end_date='20260401')

    if etf_daily.empty:
        print("错误: 无ETF数据!")
        return

    # 统计ETF覆盖情况
    print(f"\n  ETF数据覆盖:")
    for sw_code, info in SW_TO_ETF.items():
        if info['etf'] is not None:
            etf_data = etf_daily[etf_daily['ts_code'] == info['etf']]
            if not etf_data.empty:
                min_d = etf_data['trade_date'].min().strftime('%Y-%m-%d')
                max_d = etf_data['trade_date'].max().strftime('%Y-%m-%d')
                print(f"    {sw_code} → {info['etf']} ({info['name']}): {min_d} ~ {max_d}, {len(etf_data)}条")

    # ---- 3. 加载行业轮动预测 ----
    ensemble_path = os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl')
    gnn_path = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')

    if os.path.exists(ensemble_path):
        pred_df = pd.read_pickle(ensemble_path)
        pred_col = 'pred_ensemble'
        print(f"\n  使用集成预测")
    elif os.path.exists(gnn_path):
        pred_df = pd.read_pickle(gnn_path)
        pred_col = 'pred_gnn'
        print(f"\n  使用GNN预测")
    else:
        print("错误: 没有找到预测文件!")
        return

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    print(f"  预测: {pred_df['date'].min().strftime('%Y-%m')} ~ "
          f"{pred_df['date'].max().strftime('%Y-%m')}, {len(pred_df)}条")

    # ---- 4. 行业级净值（对比用）----
    nav_path = os.path.join(OUTPUT_DIR, 'backtest_nav.csv')
    ind_nav = None
    if os.path.exists(nav_path):
        nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
        for col in ['自适应集成', '集成', 'GNN']:
            if col in nav_df.columns:
                ind_nav = nav_df[col].dropna()
                print(f"  行业级对比基准: {col}")
                break

    # ---- 5. ETF回测 ----
    print("\n开始ETF回测...")
    etf_bt = run_etf_backtest(pred_df, pred_col, etf_daily, k=TOP_K, inertia=0.2)

    if etf_bt.empty:
        print("错误: ETF回测无结果!")
        return

    # ---- 6. 绩效 ----
    print(f"\n{'='*60}")
    print(f"  ETF实盘回测绩效 (Top-{TOP_K})")
    print(f"{'='*60}")

    m_net = calc_metrics(etf_bt.set_index('date')['ret_net'])
    m_gross = calc_metrics(etf_bt.set_index('date')['ret_gross'])

    print(f"  ETF(扣费): 年化={m_net.get('annual_return',0):.1%}, "
          f"夏普={m_net.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_net.get('max_drawdown',0):.1%}, "
          f"胜率={m_net.get('win_rate',0):.1%}")
    print(f"  ETF(毛):   年化={m_gross.get('annual_return',0):.1%}, "
          f"夏普={m_gross.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_gross.get('max_drawdown',0):.1%}, "
          f"胜率={m_gross.get('win_rate',0):.1%}")

    if ind_nav is not None:
        ind_rets = ind_nav.pct_change().dropna()
        m_ind = calc_metrics(ind_rets)
        print(f"\n  行业轮动(理想): 年化={m_ind.get('annual_return',0):.1%}, "
              f"夏普={m_ind.get('sharpe_ratio',0):.3f}")
        gap = m_net.get('annual_return', 0) - m_ind.get('annual_return', 0)
        print(f"  ETF vs 理想 年化差距: {gap:+.1%}")

    print(f"\n  平均持有ETF: {etf_bt['n_etfs'].mean():.1f} 只/月")
    print(f"  平均换手: {etf_bt['turnover'].mean():.1%}")

    # 哪些行业被选中最多
    all_ind = []
    for _, row in etf_bt.iterrows():
        all_ind.extend(row['industries'])
    ind_counts = pd.Series(all_ind).value_counts().head(10)
    print(f"\n  最常选中的行业:")
    from config import SW_MEMBERS_PATH as _SW_PATH
    sw = pd.read_csv(_SW_PATH)
    cur = sw[sw['is_new'] == 'Y']
    ind_to_name = dict(zip(cur['l1_code'], cur['l1_name']))
    for ind, cnt in ind_counts.items():
        name = ind_to_name.get(ind, ind)
        pct = cnt / len(etf_bt) * 100
        print(f"    {name}: {cnt}次 ({pct:.0f}%)")

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s")

    # ---- 7. 保存 ----
    etf_bt.to_csv(os.path.join(OUTPUT_DIR, 'etf_backtest.csv'),
                  index=False, encoding='utf-8-sig')
    plot_etf_results(etf_bt, ind_nav,
                     os.path.join(OUTPUT_DIR, 'etf_backtest_results.png'))
    print(f"  结果保存至 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
