# -*- coding: utf-8 -*-
"""
Step 4: 选股 v4

vs v3 唯一改动：
  quality子因子（ROE/现金流/毛利率）各自在行业内rank(pct=True)后取均值合成
  → 解决v3直接取原始值均值的量纲混乱问题

其余全部保留v3原值（惯性1.0，换仓缓冲0.15，三因子权重0.35/0.35/0.30）

保留：
  - 三因子打分：beta(0.35) + 动量(0.35) + 质量(0.30)
  - 卡尔曼beta（基本面驱动先验）
  - 每行业5只，Top-5行业共25只
  - ADV/涨跌停/beta稳定性过滤
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
import time

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from config import OUTPUT_DIR, TOP_K, RF_ANNUAL, ARIMAX_PROJECT

# ==================== 参数 ====================
TOPN_PER_IND = 5           # 每个行业选 N 只
BETA_LOOKBACK = 252        # beta 估计回看交易日
BETA_MIN_OBS = 120         # 最少日度观测
MOM_LOOKBACK = 120         # 动量回看天数
COMMISSION_RATE = 0.0003
SLIPPAGE_BPS = 5
ADV_MIN = 20000            # 最低日均成交额（千元）
ADV_WINDOW = 20
HOLDING_INERTIA = 1.0      # 持仓惯性：恢复v3原值，换手控制靠这个
REPLACE_THRESHOLD = 0.15   # 换仓缓冲：恢复v3原值

# 打分权重：保留v3的三因子结构
W_BETA = 0.35
W_MOM  = 0.35
W_QUAL = 0.30

# 卡尔曼滤波参数
KF_GAMMA = 0.95
KF_Q_BETA = 1e-5
KF_Q_ALPHA = 1e-6
KF_R = 0.005

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据路径 ====================
RAW_DATA_DIR = os.path.join(ARIMAX_PROJECT, r"数据\原始数据")
EXISTING_DATA_DIR = os.path.join(ARIMAX_PROJECT, r"数据\已有数据")
STOCK_DATA_PATH = r"D:\desktop\有意思的事情\量化\项目\天风选股模型\数据\full_market_data_v18.pkl"
SW_MEMBERS_PATH = os.path.join(EXISTING_DATA_DIR, 'ts_sw_members.csv')
SW_EXCLUDE = ['801780.SI', '801790.SI']

# ==================== 基本面因子（用于卡尔曼先验） ====================
FACTOR_DIRECTIONS = {
    'oper_leverage':    +1,
    'fin_leverage':     +1,
    'roe':              -1,
    'roa':              -1,
    'gross_margin':     -1,
    'cashflow_quality': -1,
    'revenue_yoy':      +1,
    'n_income_yoy':     +1,
    'roe_stability':    +1,
}

# 质量子因子（用于合成quality得分，越大越好）
QUALITY_FACTORS = ['roe', 'cashflow_quality', 'gross_margin']


# ============================================================
#  数据加载（复用v3缓存）
# ============================================================

def load_stock_daily():
    cache_path = os.path.join(OUTPUT_DIR, '_cache_stock_daily.pkl')
    if os.path.exists(cache_path):
        print("  加载个股日线缓存...")
        df = pd.read_pickle(cache_path)
        print(f"    {df['code'].nunique()} 只股票")
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

    print("  构建股票索引...")
    stock_dict = {code: grp.set_index('date').sort_index()
                  for code, grp in df.groupby('code')}
    print(f"    {len(stock_dict)} 只股票已索引")
    return df, stock_dict


def load_industry_daily(df_stock, stock_to_ind):
    cache_path = os.path.join(OUTPUT_DIR, '_cache_ind_daily.pkl')
    if os.path.exists(cache_path):
        print("  加载行业日度缓存...")
        ind_daily = pd.read_pickle(cache_path)
    else:
        print("  计算行业日度收益率...")
        df = df_stock[['date', 'code', 'ret']].copy()
        df['ind_code'] = df['code'].map(stock_to_ind)
        df = df.dropna(subset=['ind_code', 'ret'])
        ind_daily = df.groupby(['date', 'ind_code'])['ret'].mean().reset_index()
        ind_daily.columns = ['date', 'ind_code', 'ind_ret']
        ind_daily.to_pickle(cache_path)

    print("  构建行业索引...")
    ind_dict = {code: grp.set_index('date')['ind_ret'].sort_index()
                for code, grp in ind_daily.groupby('ind_code')}
    print(f"    {len(ind_dict)} 个行业已索引")
    return ind_daily, ind_dict


def load_stock_industry_map():
    sw = pd.read_csv(SW_MEMBERS_PATH)
    cur = sw[sw['is_new'] == 'Y'].copy()
    cur['stock_code'] = cur['ts_code'].str[:6]
    stock_to_ind = dict(zip(cur['stock_code'], cur['l1_code']))
    ind_to_name = dict(zip(cur['l1_code'], cur['l1_name']))
    print(f"  行业映射: {len(stock_to_ind)} 只股票 -> {len(ind_to_name)} 个行业")
    return stock_to_ind, ind_to_name


def load_fundamental_features():
    cache_path = os.path.join(OUTPUT_DIR, '_cache_fund_monthly.pkl')
    if os.path.exists(cache_path):
        print("  加载基本面缓存...")
        fund_dict = pd.read_pickle(cache_path)
        print(f"    {len(fund_dict)} 只股票")
        return fund_dict

    print("  加载个股基本面数据...")
    income = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'raw_income.pkl'))
    balance = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'raw_balancesheet.pkl'))
    cashflow = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'raw_cashflow.pkl'))

    for df in [income, balance, cashflow]:
        df['report_type'] = df['report_type'].astype(str).str.strip()
        df['comp_type'] = df['comp_type'].astype(str).str.strip()

    income = income[(income['report_type'] == '1') & (income['comp_type'] == '1')].copy()
    balance = balance[(balance['report_type'] == '1') & (balance['comp_type'] == '1')].copy()
    cashflow = cashflow[(cashflow['report_type'] == '1') & (cashflow['comp_type'] == '1')].copy()

    for df in [income, balance, cashflow]:
        df['ann_date'] = df['ann_date'].fillna(df['f_ann_date'])
        df.sort_values(['ts_code', 'end_date', 'ann_date'],
                       ascending=[True, True, False], inplace=True)
        df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first', inplace=True)
        df['stock_code'] = df['ts_code'].str[:6]

    def add_quarter_info(df):
        df['end_date_str'] = df['end_date'].astype(str)
        df['year'] = df['end_date_str'].str[:4].astype(int)
        df['qtr_month'] = df['end_date_str'].str[4:6].astype(int)
        df = df[df['qtr_month'].isin([3, 6, 9, 12])].copy()
        df['quarter'] = df['qtr_month'].map({3: 1, 6: 2, 9: 3, 12: 4})
        return df

    income = add_quarter_info(income)
    balance = add_quarter_info(balance)
    cashflow = add_quarter_info(cashflow)

    merge_keys = ['stock_code', 'year', 'quarter']
    merged = income[merge_keys + ['revenue', 'total_cogs', 'oper_cost',
                                   'n_income', 'n_income_attr_p', 'total_profit',
                                   'ebit', 'int_exp']].merge(
        balance[merge_keys + ['total_assets', 'total_hldr_eqy_exc_min_int',
                               'total_liab', 'total_cur_assets', 'total_cur_liab']],
        on=merge_keys, how='inner'
    ).merge(
        cashflow[merge_keys + ['n_cashflow_act']],
        on=merge_keys, how='left'
    )

    eps = 1e-8
    equity = merged['total_hldr_eqy_exc_min_int'].replace(0, np.nan)
    revenue = merged['revenue'].replace(0, np.nan)
    assets = merged['total_assets'].replace(0, np.nan)

    merged['oper_leverage'] = merged['total_cogs'] / (revenue + eps)
    merged['fin_leverage'] = merged['total_liab'] / (equity + eps)
    merged['roe'] = merged['n_income'] / (equity + eps)
    merged['roa'] = merged['n_income'] / (assets + eps)
    merged['gross_margin'] = (merged['revenue'] - merged['oper_cost']) / (revenue + eps)
    merged['cashflow_quality'] = merged['n_cashflow_act'] / (merged['n_income'].replace(0, np.nan) + eps)
    merged = merged.sort_values(['stock_code', 'year', 'quarter']).reset_index(drop=True)
    for col in ['revenue', 'n_income']:
        merged[f'{col}_yoy'] = merged.groupby('stock_code')[col].pct_change(4)
    merged['roe_stability'] = merged.groupby('stock_code')['roe'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )

    factor_cols = list(FACTOR_DIRECTIONS.keys())
    for col in factor_cols:
        lower = merged[col].quantile(0.01)
        upper = merged[col].quantile(0.99)
        merged[col] = merged[col].clip(lower, upper)
        merged[col] = merged[col].fillna(0.0)

    records = []
    for _, row in merged.iterrows():
        q, y = int(row['quarter']), int(row['year'])
        if q == 1:
            months = [(y, 4), (y, 5), (y, 6)]
        elif q == 2:
            months = [(y, 7), (y, 8), (y, 9)]
        elif q == 3:
            months = [(y, 10), (y, 11), (y, 12)]
        elif q == 4:
            months = [(y + 1, 1), (y + 1, 2), (y + 1, 3)]
        else:
            continue
        for (my, mm) in months:
            rec = {'stock_code': row['stock_code'], 'year': my, 'month': mm}
            for col in factor_cols:
                rec[col] = row[col]
            records.append(rec)

    fund_df = pd.DataFrame(records)
    fund_dict = {}
    for code, grp in fund_df.groupby('stock_code'):
        fund_dict[code] = grp.reset_index(drop=True)

    pd.to_pickle(fund_dict, cache_path)
    print(f"    {len(fund_dict)} 只股票, 缓存已保存")
    return fund_dict


# ============================================================
#  卡尔曼beta（和v3一样）
# ============================================================

def compute_fund_score(stock_code, year, month, fund_dict):
    if stock_code not in fund_dict:
        return 0.0
    df = fund_dict[stock_code]
    row = df[(df['year'] == year) & (df['month'] == month)]
    if row.empty:
        return 0.0
    row = row.iloc[0]
    score = 0.0
    n = 0
    for col, direction in FACTOR_DIRECTIONS.items():
        val = row[col]
        if np.isfinite(val):
            score += direction * val
            n += 1
    return score / max(n, 1)


def kalman_beta_daily(stock_rets, ind_rets, fund_score=0.0,
                      gamma=KF_GAMMA, Q_beta=KF_Q_BETA,
                      Q_alpha=KF_Q_ALPHA, R_noise=KF_R):
    T = len(stock_rets)
    if T < 30:
        return np.nan, np.nan

    x = np.array([0.0, 1.0])
    P = np.eye(2) * 0.5
    Q_mat = np.diag([Q_alpha, Q_beta])
    beta_arr = np.zeros(T)

    for t in range(T):
        rs, ri = stock_rets[t], ind_rets[t]
        if np.isnan(rs) or np.isnan(ri):
            beta_arr[t] = x[1]
            continue
        beta_prior = gamma * x[1] + (1 - gamma) * (1.0 + fund_score)
        x_pred = np.array([x[0], beta_prior])
        P_pred = P + Q_mat
        H = np.array([1.0, ri])
        y_pred = H @ x_pred
        S = H @ P_pred @ H.T + R_noise
        K = P_pred @ H.T / S
        x = x_pred + K * (rs - y_pred)
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
        beta_arr[t] = x[1]

    beta_last = beta_arr[-1]
    beta_std = np.std(beta_arr[-min(60, T):])
    return beta_last, beta_std


# ============================================================
#  选股核心：v4
# ============================================================

def get_quality_raw(stock_code, year, month, fund_dict):
    """获取质量子因子原始值（用于截面rank）"""
    result = {}
    if stock_code not in fund_dict:
        return result
    df = fund_dict[stock_code]
    row = df[(df['year'] == year) & (df['month'] == month)]
    if row.empty:
        return result
    row = row.iloc[0]
    for qf in QUALITY_FACTORS:
        v = row.get(qf, np.nan)
        if np.isfinite(v):
            result[qf] = v
    return result


def select_stocks_for_month(pred_month, top_industries, stock_dict, ind_dict,
                            stock_to_ind, fund_dict, ind_scores,
                            prev_holdings=None):
    cutoff = pred_month - pd.Timedelta(days=1)
    lookback_start = cutoff - pd.Timedelta(days=int(BETA_LOOKBACK * 1.6))
    year = pred_month.year
    month = pred_month.month

    all_selected = []

    for ind_code in top_industries:
        if ind_code not in ind_dict:
            continue
        ind_ret_df = ind_dict[ind_code].loc[lookback_start:cutoff]
        if len(ind_ret_df) < BETA_MIN_OBS:
            continue

        ind_stocks = [s for s, i in stock_to_ind.items() if i == ind_code]
        if not ind_stocks:
            continue

        records = []
        for stock_code in ind_stocks:
            if stock_code not in stock_dict:
                continue
            s_data = stock_dict[stock_code].loc[lookback_start:cutoff][['ret', 'amount']]
            if len(s_data) < BETA_MIN_OBS:
                continue

            aligned = pd.DataFrame({
                'stock_ret': s_data['ret'],
                'ind_ret': ind_ret_df,
                'amount': s_data['amount'],
            }).dropna()

            if len(aligned) < BETA_MIN_OBS:
                continue

            # ADV过滤
            adv = aligned['amount'].tail(ADV_WINDOW).mean()
            if adv < ADV_MIN:
                continue

            # 涨跌停过滤
            last_ret = aligned['stock_ret'].iloc[-1]
            if abs(last_ret) >= 0.095:
                continue

            # 基本面得分（用于卡尔曼先验）
            fs = compute_fund_score(stock_code, year, month, fund_dict)

            # 卡尔曼beta
            beta_last, beta_std = kalman_beta_daily(
                aligned['stock_ret'].values,
                aligned['ind_ret'].values,
                fund_score=fs,
            )
            if not np.isfinite(beta_last) or not np.isfinite(beta_std):
                continue

            # 动量
            mom = (1 + s_data['ret'].tail(MOM_LOOKBACK).fillna(0)).prod() - 1

            # 质量子因子原始值
            qual_raw = get_quality_raw(stock_code, year, month, fund_dict)

            rec = {
                'stock_code': stock_code,
                'ind_code': ind_code,
                'beta': beta_last,
                'beta_std': beta_std,
                'momentum': mom,
                'fund_score': fs,
                'nobs': len(aligned),
                'adv': adv,
            }
            for qf in QUALITY_FACTORS:
                rec[qf] = qual_raw.get(qf, np.nan)
            records.append(rec)

        if not records:
            continue

        cand_df = pd.DataFrame(records)

        # beta > 0
        cand_df = cand_df[cand_df['beta'] > 0]

        # beta稳定性过滤：去掉波动最大的25%
        if len(cand_df) > 4:
            std_threshold = cand_df['beta_std'].quantile(0.75)
            cand_df = cand_df[cand_df['beta_std'] <= std_threshold]

        if len(cand_df) < 2:
            continue

        # === v4改动1：质量子因子各自rank后合成（解决量纲问题） ===
        cand_df['beta_rank'] = cand_df['beta'].rank(pct=True)
        cand_df['mom_rank'] = cand_df['momentum'].rank(pct=True)

        qual_ranks = []
        for qf in QUALITY_FACTORS:
            col = f'{qf}_rank'
            cand_df[col] = cand_df[qf].rank(pct=True)
            # NaN的rank填0.5（中性）
            cand_df[col] = cand_df[col].fillna(0.5)
            qual_ranks.append(col)
        cand_df['quality_rank'] = cand_df[qual_ranks].mean(axis=1)

        cand_df['composite'] = (W_BETA * cand_df['beta_rank'] +
                                W_MOM  * cand_df['mom_rank'] +
                                W_QUAL * cand_df['quality_rank'])

        # 持仓惯性（恢复v3原值）
        if prev_holdings and HOLDING_INERTIA > 0:
            comp_std = cand_df['composite'].std()
            if comp_std > 0:
                bonus = HOLDING_INERTIA * comp_std
                cand_df.loc[cand_df['stock_code'].isin(prev_holdings), 'composite'] += bonus

        # 换仓缓冲（恢复v3逻辑）
        cand_df = cand_df.sort_values('composite', ascending=False)
        held_in_ind = [s for s in cand_df['stock_code'] if s in (prev_holdings or set())]
        new_candidates = cand_df[~cand_df['stock_code'].isin(prev_holdings or set())]

        selected_codes = []
        for code in held_in_ind:
            if len(selected_codes) >= TOPN_PER_IND:
                break
            row_score = cand_df.loc[cand_df['stock_code'] == code, 'composite'].iloc[0]
            if row_score >= cand_df['composite'].quantile(0.25):
                selected_codes.append(code)

        if len(selected_codes) < TOPN_PER_IND and not new_candidates.empty:
            held_min_score = (cand_df.loc[cand_df['stock_code'].isin(selected_codes), 'composite'].min()
                              if selected_codes else -np.inf)
            threshold = held_min_score + REPLACE_THRESHOLD * cand_df['composite'].std() if selected_codes else -np.inf
            for _, row in new_candidates.iterrows():
                if len(selected_codes) >= TOPN_PER_IND:
                    break
                if row['composite'] >= threshold or not selected_codes:
                    selected_codes.append(row['stock_code'])

        if len(selected_codes) < TOPN_PER_IND:
            for _, row in cand_df.iterrows():
                if row['stock_code'] not in selected_codes:
                    selected_codes.append(row['stock_code'])
                if len(selected_codes) >= TOPN_PER_IND:
                    break

        cand_df = cand_df[cand_df['stock_code'].isin(selected_codes)]
        cand_df['month'] = pred_month
        cand_df['ind_score'] = ind_scores.get(ind_code, 0)
        cand_df['rank_in_ind'] = range(1, len(cand_df) + 1)
        # 补一个quality列方便统计
        cand_df['quality'] = cand_df['quality_rank']

        all_selected.append(cand_df)

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
        port_ret_net = port_ret - cost

        results.append({
            'date': next_month_end,
            'ret_gross': port_ret,
            'ret_net': port_ret_net,
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


def plot_stock_results(bt_results, bt_v3, ind_nav, output_path):
    """绘制v4 vs v3 vs 行业轮动对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 净值对比
    ax = axes[0, 0]
    if not bt_results.empty:
        nav_v4 = (1 + bt_results.set_index('date')['ret_net']).cumprod()
        ax.plot(nav_v4.index, nav_v4.values,
                label='v4选股(扣费)', color='#E91E63', linewidth=2)
    if bt_v3 is not None and not bt_v3.empty:
        nav_v3 = (1 + bt_v3.set_index('date')['ret_net']).cumprod()
        ax.plot(nav_v3.index, nav_v3.values,
                label='v3选股(扣费)', color='#FF9800', linewidth=1.5, linestyle='--')
    if ind_nav is not None:
        ax.plot(ind_nav.index, ind_nav.values,
                label='行业轮动', color='#2196F3', linewidth=1.5)
    ax.set_title('净值对比: v4 vs v3 vs 行业轮动')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 月度收益
    ax = axes[0, 1]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['ret_net'].values,
               alpha=0.6, color='#4CAF50')
        ax.axhline(y=0, color='black', linewidth=0.5)
        mean_ret = bt_results['ret_net'].mean()
        ax.axhline(y=mean_ret, color='red', linewidth=1, linestyle='--',
                    label=f'均值={mean_ret:.2%}')
    ax.set_title('v4月度收益')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 换手率
    ax = axes[1, 0]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['turnover'].values,
               alpha=0.6, color='#9C27B0')
        avg_turn = bt_results['turnover'].mean()
        ax.axhline(y=avg_turn, color='red', linewidth=1, linestyle='--',
                    label=f'平均换手={avg_turn:.1%}')
    ax.set_title('月度换手率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 绩效表
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    if not bt_results.empty:
        m = calc_metrics(bt_results.set_index('date')['ret_net'])
        table_data.append(['v4选股(扣费)', f"{m.get('annual_return',0):.1%}",
                           f"{m.get('sharpe_ratio',0):.3f}",
                           f"{m.get('max_drawdown',0):.1%}",
                           f"{m.get('win_rate',0):.1%}"])
        m2 = calc_metrics(bt_results.set_index('date')['ret_gross'])
        table_data.append(['v4选股(毛)', f"{m2.get('annual_return',0):.1%}",
                           f"{m2.get('sharpe_ratio',0):.3f}",
                           f"{m2.get('max_drawdown',0):.1%}",
                           f"{m2.get('win_rate',0):.1%}"])
    if bt_v3 is not None and not bt_v3.empty:
        m3 = calc_metrics(bt_v3.set_index('date')['ret_net'])
        table_data.append(['v3选股(扣费)', f"{m3.get('annual_return',0):.1%}",
                           f"{m3.get('sharpe_ratio',0):.3f}",
                           f"{m3.get('max_drawdown',0):.1%}",
                           f"{m3.get('win_rate',0):.1%}"])
    if ind_nav is not None:
        ind_rets = ind_nav.pct_change().dropna()
        mi = calc_metrics(ind_rets)
        table_data.append(['行业轮动', f"{mi.get('annual_return',0):.1%}",
                           f"{mi.get('sharpe_ratio',0):.3f}",
                           f"{mi.get('max_drawdown',0):.1%}",
                           f"{mi.get('win_rate',0):.1%}"])
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
    print("  Step 4: 选股 v4 (三因子+quality截面rank)")
    print("=" * 60)

    # ---- 加载预测 ----
    ensemble_path = os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl')
    gnn_path = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')

    if os.path.exists(ensemble_path):
        pred_df = pd.read_pickle(ensemble_path)
        pred_col = 'pred_ensemble'
        print(f"  使用集成预测")
    elif os.path.exists(gnn_path):
        pred_df = pd.read_pickle(gnn_path)
        pred_col = 'pred_gnn'
        print(f"  使用GNN预测")
    else:
        print("错误: 没有预测文件!")
        return

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    print(f"  预测: {len(pred_df)} 条, "
          f"{pred_df['date'].min().strftime('%Y-%m')} ~ "
          f"{pred_df['date'].max().strftime('%Y-%m')}")

    # ---- 行业级净值 ----
    nav_path = os.path.join(OUTPUT_DIR, 'backtest_nav.csv')
    ind_nav = None
    if os.path.exists(nav_path):
        nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
        for col in ['自适应集成', '集成', 'GNN']:
            if col in nav_df.columns:
                ind_nav = nav_df[col].dropna()
                print(f"  行业级对比: {col}")
                break

    # ---- v3结果（对比用）----
    v3_path = os.path.join(OUTPUT_DIR, 'stock_backtest.csv')
    bt_v3 = None
    if os.path.exists(v3_path):
        bt_v3 = pd.read_csv(v3_path)
        bt_v3['date'] = pd.to_datetime(bt_v3['date'])
        print(f"  v3对比数据已加载")

    # ---- 加载数据 ----
    stock_to_ind, ind_to_name = load_stock_industry_map()
    df_stock, stock_dict = load_stock_daily()
    ind_daily, ind_dict = load_industry_daily(df_stock, stock_to_ind)
    fund_dict = load_fundamental_features()

    # ---- 断点续传 ----
    ckpt_path = os.path.join(OUTPUT_DIR, '_ckpt_v4.pkl')
    all_selections = []
    prev_holdings = set()
    done_months = set()

    if os.path.exists(ckpt_path):
        ckpt = pd.read_pickle(ckpt_path)
        all_selections = ckpt.get('selections', [])
        prev_holdings = ckpt.get('prev_holdings', set())
        done_months = ckpt.get('done_months', set())
        print(f"  断点续传: 已完成 {len(done_months)} 个月")

    # ---- 逐月选股 ----
    pred_months = sorted(pred_df['date'].unique())
    remaining = [m for m in pred_months if m not in done_months]
    print(f"\n  开始选股 v4 (剩余 {len(remaining)}/{len(pred_months)} 个月)...")

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
            ind_dict=ind_dict,
            stock_to_ind=stock_to_ind,
            fund_dict=fund_dict,
            ind_scores=ind_scores,
            prev_holdings=prev_holdings,
        )

        if not selected.empty:
            all_selections.append(selected)
            prev_holdings = set(selected['stock_code'].tolist())
        else:
            prev_holdings = set()

        done_months.add(month)

        if (i + 1) % 3 == 0 or (i + 1) == len(remaining):
            pd.to_pickle({'selections': all_selections,
                          'prev_holdings': prev_holdings,
                          'done_months': done_months}, ckpt_path)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            n_sel = len(selected) if not selected.empty else 0
            ind_names = [ind_to_name.get(c, c) for c in top_industries[:3]]
            print(f"    [{len(done_months)}/{len(pred_months)}] "
                  f"{pd.Timestamp(month).strftime('%Y-%m')}: "
                  f"选出 {n_sel} 只, Top: {', '.join(ind_names[:3])}... "
                  f"({elapsed:.0f}s)")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    if not all_selections:
        print("错误: 所有月份均无选股结果!")
        return

    monthly_selections = pd.concat(all_selections, ignore_index=True)
    print(f"\n  选股完成: {len(monthly_selections)} 条, "
          f"覆盖 {monthly_selections['month'].nunique()} 个月")

    # ---- 回测 ----
    print("\n  回测 v4 组合...")
    bt_results = backtest_stock_portfolio(monthly_selections, stock_dict)

    if bt_results.empty:
        print("错误: 回测无结果!")
        return

    # ---- 绩效 ----
    print(f"\n{'='*60}")
    print(f"  选股 v4 绩效")
    print(f"{'='*60}")

    m_net = calc_metrics(bt_results.set_index('date')['ret_net'])
    m_gross = calc_metrics(bt_results.set_index('date')['ret_gross'])

    print(f"  v4(扣费): 年化={m_net.get('annual_return',0):.1%}, "
          f"夏普={m_net.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_net.get('max_drawdown',0):.1%}, "
          f"胜率={m_net.get('win_rate',0):.1%}")
    print(f"  v4(毛):   年化={m_gross.get('annual_return',0):.1%}, "
          f"夏普={m_gross.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_gross.get('max_drawdown',0):.1%}, "
          f"胜率={m_gross.get('win_rate',0):.1%}")

    if bt_v3 is not None:
        m_v3 = calc_metrics(bt_v3.set_index('date')['ret_net'])
        gap = m_net.get('annual_return', 0) - m_v3.get('annual_return', 0)
        print(f"\n  v3(扣费): 年化={m_v3.get('annual_return',0):.1%}, "
              f"夏普={m_v3.get('sharpe_ratio',0):.3f}")
        print(f"  v4 vs v3: {gap:+.1%}")

    print(f"\n  平均持股: {bt_results['n_stocks'].mean():.0f} 只/月")
    print(f"  平均换手: {bt_results['turnover'].mean():.1%}")
    print(f"  平均成本: {bt_results['cost'].mean():.4%}/月")

    print(f"\n  因子统计:")
    print(f"    beta均值: {monthly_selections['beta'].mean():.3f}")
    print(f"    动量均值: {monthly_selections['momentum'].mean():.3f}")

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed/60:.1f} 分钟")

    # ---- 保存 ----
    monthly_selections.to_pickle(os.path.join(OUTPUT_DIR, 'stock_selections_v4.pkl'))
    bt_results.to_csv(os.path.join(OUTPUT_DIR, 'stock_backtest_v4.csv'),
                      index=False, encoding='utf-8-sig')
    plot_stock_results(bt_results, bt_v3, ind_nav,
                       os.path.join(OUTPUT_DIR, 'stock_backtest_v4_results.png'))
    print(f"  结果保存至 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
