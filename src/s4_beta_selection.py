# -*- coding: utf-8 -*-
"""
Step 4: 多因子复合选股（v3）

改进点（vs v2）：
  1. 缩小选股范围：每行业 10→4 只，减少噪声
  2. 多因子复合：beta(0.4) + 动量(0.3) + 质量(0.3) 行业内rank加权
  3. 加大持仓惯性：0.3→0.6，降低换手率

方法论（天风研报）：
  状态方程: beta_t = gamma * beta_{t-1} + (1-gamma) * f(fundamental) + eps
  观测方程: R_stock = alpha + beta * R_industry + eta

数据源：
  - 个股财务三表: ARIMAX项目/数据/原始数据/raw_*.pkl
  - 行业映射: ts_sw_members.csv
  - 个股日线: 天风选股模型/数据/full_market_data_v18.pkl
  - 行业月度行情: ts_sw_industry_monthly.csv
  - 行业轮动预测: s3 集成输出
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
import time

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from config import OUTPUT_DIR, TOP_K, RF_ANNUAL, SW_MEMBERS_PATH as _CFG_SW_MEMBERS, \
    LOCAL_DATA_RAW, STOCK_DAILY_PATH

# ARIMAX_PROJECT is optional: only needed when fundamental cache is missing
try:
    from config import ARIMAX_PROJECT as _ARIMAX_PROJECT
    RAW_DATA_DIR      = os.path.join(_ARIMAX_PROJECT, r"数据\原始数据")
    EXISTING_DATA_DIR = os.path.join(_ARIMAX_PROJECT, r"数据\已有数据")
    SW_MEMBERS_PATH   = os.path.join(EXISTING_DATA_DIR, 'ts_sw_members.csv')
except ImportError:
    _ARIMAX_PROJECT   = None
    RAW_DATA_DIR      = None
    EXISTING_DATA_DIR = None
    SW_MEMBERS_PATH   = _CFG_SW_MEMBERS

# ==================== 参数 ====================
TOPN_PER_IND = 5           # 每个行业选 N 只（25只总持仓，兼顾集中与分散）
BETA_LOOKBACK = 252        # beta 估计回看交易日（约1年）
BETA_MIN_OBS = 120         # 最少日度观测
MOM_LOOKBACK = 120         # 动量回看天数（约6个月，更稳定的信号）
COMMISSION_RATE = 0.0003   # 佣金率
SLIPPAGE_BPS = 5           # 滑点（基点）
ADV_MIN = 20000            # 最低日均成交额（千元，约2000万）
ADV_WINDOW = 20            # ADV 计算窗口
HOLDING_INERTIA = 1.0      # 持仓惯性：强偏好保留现有持仓
REPLACE_THRESHOLD = 0.15   # 换仓缓冲：新股必须比旧股高出此比例才替换

# 多因子复合权重（beta + 动量 + 质量）
W_BETA = 0.35              # beta 权重
W_MOM  = 0.35              # 动量权重
W_QUAL = 0.30              # 质量因子权重（ROE + 现金流 + 毛利率）

# 卡尔曼滤波参数
KF_GAMMA = 0.95            # beta 日度惯性（日频用更高惯性）
KF_Q_BETA = 1e-5           # beta 过程噪声（日频更小）
KF_Q_ALPHA = 1e-6          # alpha 过程噪声
KF_R = 0.005               # 观测噪声

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据路径 ====================
STOCK_DATA_PATH = r"D:\desktop\有意思的事情\量化\项目\天风选股模型\数据\full_market_data_v18.pkl"
SW_EXCLUDE = ['801780.SI', '801790.SI']

# ==================== 基本面因子定义 ====================
# (因子名, 方向) : +1 表示该因子越大 beta 越高, -1 表示越大 beta 越低
FACTOR_DIRECTIONS = {
    'oper_leverage':    +1,   # 高经营杠杆 → 高 beta
    'fin_leverage':     +1,   # 高财务杠杆 → 高 beta
    'roe':              -1,   # 高盈利能力 → 经营稳定 → 低 beta
    'roa':              -1,   # 同上
    'gross_margin':     -1,   # 高毛利率 → 护城河 → 低 beta
    'cashflow_quality': -1,   # 高现金流质量 → 稳定 → 低 beta
    'revenue_yoy':      +1,   # 高成长 → 高 beta
    'n_income_yoy':     +1,   # 高利润增速 → 高 beta
    'roe_stability':    +1,   # ROE 标准差大 → 盈利不稳 → 高 beta
}


# ============================================================
#  数据加载
# ============================================================

def load_stock_daily():
    """加载个股日线数据（带缓存），返回 (df, stock_dict) 加速查询"""
    cache_path = os.path.join(OUTPUT_DIR, '_cache_stock_daily.pkl')

    # 若 data/raw/stock_daily.pkl 比缓存更新，自动重建缓存
    if os.path.exists(cache_path) and os.path.exists(STOCK_DAILY_PATH):
        if os.path.getmtime(STOCK_DAILY_PATH) > os.path.getmtime(cache_path):
            print("  检测到个股日线原始数据已更新，重建缓存...")
            os.remove(cache_path)

    if os.path.exists(cache_path):
        print("  加载个股日线缓存...")
        df = pd.read_pickle(cache_path)
        print(f"    {df['code'].nunique()} 只股票, "
              f"{df['date'].min().strftime('%Y-%m-%d')} ~ "
              f"{df['date'].max().strftime('%Y-%m-%d')}")
    else:
        # 优先用 data/raw/stock_daily.pkl，兜底用旧版路径
        raw_src = STOCK_DAILY_PATH if os.path.exists(STOCK_DAILY_PATH) else STOCK_DATA_PATH
        print(f"  加载个股日线原始数据（{raw_src}）...")
        import pickle as _pkl
        with open(raw_src, 'rb') as _f:
            _raw = _pkl.load(_f)
        df = _raw['df_stock'].copy()
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


def load_industry_daily(df_stock, stock_to_ind):
    """从个股日线计算行业日度收益率（等权），返回 (df, ind_dict) 加速查询"""
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
        print(f"    {ind_daily['ind_code'].nunique()} 个行业, {len(ind_daily)} 条")

    # 按行业分组建索引
    print("  构建行业索引...")
    ind_dict = {code: grp.set_index('date')['ind_ret'].sort_index()
                for code, grp in ind_daily.groupby('ind_code')}
    print(f"    {len(ind_dict)} 个行业已索引")
    return ind_daily, ind_dict


def load_stock_industry_map():
    """加载股票 -> 申万一级行业映射"""
    sw = pd.read_csv(SW_MEMBERS_PATH)
    cur = sw[sw['is_new'] == 'Y'].copy()
    cur['stock_code'] = cur['ts_code'].str[:6]
    stock_to_ind = dict(zip(cur['stock_code'], cur['l1_code']))
    ind_to_name = dict(zip(cur['l1_code'], cur['l1_name']))
    print(f"  行业映射: {len(stock_to_ind)} 只股票 -> {len(ind_to_name)} 个行业")
    return stock_to_ind, ind_to_name


def load_fundamental_features():
    """
    从个股三表提取基本面特征，季频 -> 月频映射（滞后一季度）
    返回 dict: {stock_code: DataFrame(year, month, factor1, ...)}  便于快速查询
    """
    print("  加载个股基本面数据...")

    cache_path = os.path.join(OUTPUT_DIR, '_cache_fund_monthly.pkl')

    # 优先用本项目 data/raw/ 的财报；若财报比缓存更新则自动重建
    _fund_raw_dir = LOCAL_DATA_RAW if os.path.exists(
        os.path.join(LOCAL_DATA_RAW, 'raw_income.pkl')) else RAW_DATA_DIR

    if os.path.exists(cache_path) and _fund_raw_dir:
        _income_src = os.path.join(_fund_raw_dir, 'raw_income.pkl')
        if os.path.exists(_income_src) and \
                os.path.getmtime(_income_src) > os.path.getmtime(cache_path):
            print("    检测到财报数据已更新，重建基本面缓存...")
            os.remove(cache_path)

    if os.path.exists(cache_path):
        print("    加载基本面缓存...")
        fund_dict = pd.read_pickle(cache_path)
        print(f"    {len(fund_dict)} 只股票")
        return fund_dict

    if not _fund_raw_dir:
        raise FileNotFoundError("财报原始数据不可用：RAW_DATA_DIR 未配置且 data/raw/ 无财报文件")

    income   = pd.read_pickle(os.path.join(_fund_raw_dir, 'raw_income.pkl'))
    balance  = pd.read_pickle(os.path.join(_fund_raw_dir, 'raw_balancesheet.pkl'))
    cashflow = pd.read_pickle(os.path.join(_fund_raw_dir, 'raw_cashflow.pkl'))

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

    # Winsorize
    for col in factor_cols:
        lower = merged[col].quantile(0.01)
        upper = merged[col].quantile(0.99)
        merged[col] = merged[col].clip(lower, upper)
        merged[col] = merged[col].fillna(0.0)

    # 季频 -> 月频，按股票分组存入 dict
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

    # 按股票分组存入 dict，加速查询
    fund_dict = {}
    for code, grp in fund_df.groupby('stock_code'):
        fund_dict[code] = grp.reset_index(drop=True)

    pd.to_pickle(fund_dict, cache_path)
    print(f"    基本面因子: {factor_cols}")
    print(f"    {len(fund_dict)} 只股票, 缓存已保存")
    return fund_dict


# ============================================================
#  基本面驱动的卡尔曼 beta 估计（日频）
# ============================================================

def compute_fund_score(stock_code, year, month, fund_dict):
    """
    计算某只股票某月的基本面综合得分
    按 FACTOR_DIRECTIONS 的方向加权
    若精确月份无数据，回退到最近12个月内最新可用财报（防止财报发布滞后导致全零）
    """
    if stock_code not in fund_dict:
        return 0.0

    df = fund_dict[stock_code]
    row = df[(df['year'] == year) & (df['month'] == month)]
    if row.empty:
        # fallback: most recent data within past 12 months
        target_ym = year * 100 + month
        df2 = df.copy()
        df2['_ym'] = df2['year'] * 100 + df2['month']
        past = df2[df2['_ym'] < target_ym].sort_values('_ym', ascending=False)
        if past.empty:
            return 0.0
        latest_ym = past['_ym'].iloc[0]
        ly, lm = divmod(latest_ym, 100)
        gap_months = (year - ly) * 12 + (month - lm)
        if gap_months > 12:
            return 0.0
        row = past.iloc[0]
    else:
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
    """
    日频卡尔曼 beta 估计

    fund_score 是该股票当期的基本面综合得分（标量），
    作为 beta 的先验锚定：beta_prior = gamma * beta + (1-gamma) * (1 + fund_score)
    """
    T = len(stock_rets)
    if T < 30:
        return np.nan, np.nan

    x = np.array([0.0, 1.0])  # [alpha, beta]
    P = np.eye(2) * 0.5
    Q_mat = np.diag([Q_alpha, Q_beta])

    beta_arr = np.zeros(T)

    for t in range(T):
        rs, ri = stock_rets[t], ind_rets[t]
        if np.isnan(rs) or np.isnan(ri):
            beta_arr[t] = x[1]
            continue

        # 预测步
        beta_prior = gamma * x[1] + (1 - gamma) * (1.0 + fund_score)
        x_pred = np.array([x[0], beta_prior])
        P_pred = P + Q_mat

        # 更新步
        H = np.array([1.0, ri])
        y_pred = H @ x_pred
        S = H @ P_pred @ H.T + R_noise
        K = P_pred @ H.T / S
        x = x_pred + K * (rs - y_pred)
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        beta_arr[t] = x[1]

    # 返回最终 beta 和近期稳定性（最近60天标准差）
    beta_last = beta_arr[-1]
    beta_std = np.std(beta_arr[-min(60, T):])
    return beta_last, beta_std


# ============================================================
#  选股核心逻辑
# ============================================================

def select_stocks_for_month(pred_month, top_industries, stock_dict, ind_dict,
                            stock_to_ind, fund_dict, ind_scores,
                            prev_holdings=None):
    """
    对单个月份，在 Top-K 行业内用基本面卡尔曼 beta 选股（日频）
    """
    cutoff = pred_month - pd.Timedelta(days=1)
    lookback_start = cutoff - pd.Timedelta(days=int(BETA_LOOKBACK * 1.6))

    year = pred_month.year
    month = pred_month.month

    all_selected = []

    for ind_code in top_industries:
        # 行业日度收益率
        if ind_code not in ind_dict:
            continue
        ind_ret_df = ind_dict[ind_code].loc[lookback_start:cutoff]

        if len(ind_ret_df) < BETA_MIN_OBS:
            continue

        # 该行业成分股
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

            # 对齐日期
            aligned = pd.DataFrame({
                'stock_ret': s_data['ret'],
                'ind_ret': ind_ret_df,
                'amount': s_data['amount'],
            }).dropna()

            if len(aligned) < BETA_MIN_OBS:
                continue

            # ADV 过滤
            adv = aligned['amount'].tail(ADV_WINDOW).mean()
            if adv < ADV_MIN:
                continue

            # 涨跌停过滤
            last_ret = aligned['stock_ret'].iloc[-1]
            if abs(last_ret) >= 0.095:
                continue

            # 基本面得分
            fs = compute_fund_score(stock_code, year, month, fund_dict)

            # 日频卡尔曼 beta
            beta_last, beta_std = kalman_beta_daily(
                aligned['stock_ret'].values,
                aligned['ind_ret'].values,
                fund_score=fs,
            )

            if not np.isfinite(beta_last) or not np.isfinite(beta_std):
                continue

            # 动量：过去 MOM_LOOKBACK 天累计收益
            mom = (1 + s_data['ret'].tail(MOM_LOOKBACK).fillna(0)).prod() - 1

            # 质量因子：ROE + 现金流质量 + 毛利率（从fund_dict取，fallback到最近12个月）
            qual = 0.0
            if stock_code in fund_dict:
                fdf = fund_dict[stock_code]
                frow = fdf[(fdf['year'] == year) & (fdf['month'] == month)]
                if frow.empty:
                    fdf2 = fdf.copy()
                    fdf2['_ym'] = fdf2['year'] * 100 + fdf2['month']
                    past = fdf2[fdf2['_ym'] < year * 100 + month].sort_values('_ym', ascending=False)
                    if not past.empty:
                        ly, lm = divmod(past['_ym'].iloc[0], 100)
                        if (year - ly) * 12 + (month - lm) <= 12:
                            frow = past.head(1)
                if not frow.empty:
                    frow = frow.iloc[0]
                    q_vals = []
                    for qf in ['roe', 'cashflow_quality', 'gross_margin']:
                        v = frow.get(qf, 0.0)
                        if np.isfinite(v):
                            q_vals.append(v)
                    if q_vals:
                        qual = np.mean(q_vals)

            records.append({
                'stock_code': stock_code,
                'ind_code': ind_code,
                'beta': beta_last,
                'beta_std': beta_std,
                'momentum': mom,
                'quality': qual,
                'fund_score': fs,
                'nobs': len(aligned),
                'adv': adv,
            })

        if not records:
            continue

        cand_df = pd.DataFrame(records)

        # beta > 0 过滤
        cand_df = cand_df[cand_df['beta'] > 0]

        # beta 稳定性过滤：去掉波动最大的 25%
        if len(cand_df) > 4:
            std_threshold = cand_df['beta_std'].quantile(0.75)
            cand_df = cand_df[cand_df['beta_std'] <= std_threshold]

        if cand_df.empty or len(cand_df) < 2:
            continue

        # 多因子复合打分：beta + 动量 + 质量 → 行业内 rank 标准化后加权
        for col in ['beta', 'momentum', 'quality']:
            r = cand_df[col].rank(pct=True)
            cand_df[f'{col}_rank'] = r

        cand_df['composite'] = (W_BETA * cand_df['beta_rank'] +
                                W_MOM  * cand_df['momentum_rank'] +
                                W_QUAL * cand_df['quality_rank'])

        # 持仓惯性 + 换仓缓冲
        if prev_holdings and HOLDING_INERTIA > 0:
            comp_std = cand_df['composite'].std()
            if comp_std > 0:
                bonus = HOLDING_INERTIA * comp_std
                cand_df.loc[cand_df['stock_code'].isin(prev_holdings), 'composite'] += bonus

        # 带缓冲的选股：现有持仓优先保留，新股必须显著更优才替换
        cand_df = cand_df.sort_values('composite', ascending=False)
        held_in_ind = [s for s in cand_df['stock_code'] if s in (prev_holdings or set())]
        new_candidates = cand_df[~cand_df['stock_code'].isin(prev_holdings or set())]

        selected_codes = []
        # 先保留还在候选池中的持仓股（不超过 TOPN）
        for code in held_in_ind:
            if len(selected_codes) >= TOPN_PER_IND:
                break
            row_score = cand_df.loc[cand_df['stock_code'] == code, 'composite'].iloc[0]
            # 持仓股只要不在最差 25% 就保留
            if row_score >= cand_df['composite'].quantile(0.25):
                selected_codes.append(code)

        # 用新候选补满剩余名额，但需超过持仓股最低分 + 阈值
        if len(selected_codes) < TOPN_PER_IND and not new_candidates.empty:
            held_min_score = (cand_df.loc[cand_df['stock_code'].isin(selected_codes), 'composite'].min()
                              if selected_codes else -np.inf)
            threshold = held_min_score + REPLACE_THRESHOLD * cand_df['composite'].std() if selected_codes else -np.inf
            for _, row in new_candidates.iterrows():
                if len(selected_codes) >= TOPN_PER_IND:
                    break
                if row['composite'] >= threshold or not selected_codes:
                    selected_codes.append(row['stock_code'])

        # 如果还没满（持仓股太少且新股不够好），放宽直接取top
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

        all_selected.append(cand_df)

    if not all_selected:
        return pd.DataFrame()

    return pd.concat(all_selected, ignore_index=True)


# ============================================================
#  回测
# ============================================================

def backtest_stock_portfolio(monthly_selections, stock_dict):
    """回测个股组合：每月初等权建仓，持有一个月"""
    results = []
    prev_holdings = set()

    for month in sorted(monthly_selections['month'].unique()):
        sel = monthly_selections[monthly_selections['month'] == month]
        if sel.empty:
            continue

        stocks = sel['stock_code'].tolist()

        # 当月收益：从 month 到 month+1个月
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


def plot_stock_results(bt_results, ind_nav, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if not bt_results.empty:
        stock_nav = (1 + bt_results.set_index('date')['ret_net']).cumprod()
        stock_nav_gross = (1 + bt_results.set_index('date')['ret_gross']).cumprod()
        ax.plot(stock_nav.index, stock_nav.values,
                label='beta选股(扣费)', color='#E91E63', linewidth=2)
        ax.plot(stock_nav_gross.index, stock_nav_gross.values,
                label='beta选股(毛)', color='#FF9800', linewidth=1.5, linestyle='--')
    if ind_nav is not None:
        ax.plot(ind_nav.index, ind_nav.values,
                label='行业轮动', color='#2196F3', linewidth=1.5)
    ax.set_title('净值对比: beta选股 vs 行业轮动')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['ret_net'].values,
               alpha=0.6, color='#4CAF50')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=bt_results['ret_net'].mean(), color='red',
                    linewidth=1, linestyle='--',
                    label=f"均值={bt_results['ret_net'].mean():.2%}")
    ax.set_title('月度收益')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not bt_results.empty:
        ax.bar(range(len(bt_results)), bt_results['turnover'].values,
               alpha=0.6, color='#9C27B0')
        ax.axhline(y=bt_results['turnover'].mean(), color='red',
                    linewidth=1, linestyle='--',
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
        table_data.append(['beta选股(扣费)', f"{m_net.get('annual_return',0):.1%}",
                           f"{m_net.get('sharpe_ratio',0):.3f}",
                           f"{m_net.get('max_drawdown',0):.1%}",
                           f"{m_net.get('win_rate',0):.1%}"])
        table_data.append(['beta选股(毛)', f"{m_gross.get('annual_return',0):.1%}",
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
    print("  Step 4: 多因子复合选股 (v3: beta+动量+质量)")
    print("=" * 60)

    # ---- 加载集成预测 ----
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
        print("错误: 没有找到预测文件!")
        return

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    print(f"  预测: {len(pred_df)} 条, "
          f"{pred_df['date'].min().strftime('%Y-%m')} ~ "
          f"{pred_df['date'].max().strftime('%Y-%m')}")

    # ---- 行业级净值（对比用）----
    nav_path = os.path.join(OUTPUT_DIR, 'backtest_nav.csv')
    ind_nav = None
    if os.path.exists(nav_path):
        nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
        for col in ['自适应集成', '集成', 'GNN']:
            if col in nav_df.columns:
                ind_nav = nav_df[col].dropna()
                print(f"  行业级对比基准: {col}")
                break

    # ---- 加载数据 ----
    stock_to_ind, ind_to_name = load_stock_industry_map()
    df_stock, stock_dict = load_stock_daily()
    ind_daily, ind_dict = load_industry_daily(df_stock, stock_to_ind)
    fund_dict = load_fundamental_features()

    # ---- 断点续传 ----
    ckpt_path = os.path.join(OUTPUT_DIR, '_ckpt_beta.pkl')
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

        # 每3个月保存一次checkpoint（beta版每月更慢，更频繁保存）
        if (i + 1) % 3 == 0 or (i + 1) == len(remaining):
            pd.to_pickle({'selections': all_selections,
                          'prev_holdings': prev_holdings,
                          'done_months': done_months}, ckpt_path)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            n_sel = len(selected) if not selected.empty else 0
            ind_names = [ind_to_name.get(c, c) for c in top_industries[:3]]
            print(f"    [{len(done_months)}/{len(pred_months)}] "
                  f"{pd.Timestamp(month).strftime('%Y-%m')}: "
                  f"选出 {n_sel} 只, "
                  f"Top: {', '.join(ind_names[:3])}... "
                  f"({elapsed:.0f}s)")

    # 清理checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    if not all_selections:
        print("错误: 所有月份均无选股结果!")
        return

    monthly_selections = pd.concat(all_selections, ignore_index=True)
    print(f"\n  选股完成: {len(monthly_selections)} 条, "
          f"覆盖 {monthly_selections['month'].nunique()} 个月")

    # ---- 回测 ----
    print("\n  回测个股组合...")
    bt_results = backtest_stock_portfolio(monthly_selections, stock_dict)

    if bt_results.empty:
        print("错误: 回测无结果!")
        return

    # ---- 绩效 ----
    print(f"\n{'='*60}")
    print(f"  多因子复合选股绩效 (v3)")
    print(f"{'='*60}")

    m_net = calc_metrics(bt_results.set_index('date')['ret_net'])
    m_gross = calc_metrics(bt_results.set_index('date')['ret_gross'])

    print(f"  复合选股(扣费): 年化={m_net.get('annual_return',0):.1%}, "
          f"夏普={m_net.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_net.get('max_drawdown',0):.1%}, "
          f"胜率={m_net.get('win_rate',0):.1%}")
    print(f"  复合选股(毛):   年化={m_gross.get('annual_return',0):.1%}, "
          f"夏普={m_gross.get('sharpe_ratio',0):.3f}, "
          f"回撤={m_gross.get('max_drawdown',0):.1%}, "
          f"胜率={m_gross.get('win_rate',0):.1%}")

    print(f"\n  平均持股: {bt_results['n_stocks'].mean():.0f} 只/月")
    print(f"  平均换手: {bt_results['turnover'].mean():.1%}")
    print(f"  平均成本: {bt_results['cost'].mean():.4%}/月")

    print(f"\n  因子统计:")
    print(f"    beta均值: {monthly_selections['beta'].mean():.3f}")
    print(f"    动量均值: {monthly_selections['momentum'].mean():.3f}")
    print(f"    质量均值: {monthly_selections['quality'].mean():.3f}")
    print(f"    复合均值: {monthly_selections['composite'].mean():.3f}")

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed/60:.1f} 分钟")

    # ---- 保存 ----
    monthly_selections.to_pickle(os.path.join(OUTPUT_DIR, 'stock_selections.pkl'))
    bt_results.to_csv(os.path.join(OUTPUT_DIR, 'stock_backtest.csv'),
                      index=False, encoding='utf-8-sig')
    plot_stock_results(bt_results, ind_nav,
                       os.path.join(OUTPUT_DIR, 'stock_backtest_results.png'))
    print(f"  结果保存至 {OUTPUT_DIR}")


def run_live(pred_pkl: str = 'predictions_ensemble.pkl',
             ckpt_suffix: str = '',
             force_refresh_latest: bool = False) -> pd.DataFrame:
    """Select stocks for the latest available month (called by monitor.run).

    Returns a DataFrame with columns expected by monitor._stock_section_lines:
    ind_code, ind_name, stock_code, name, beta, momentum, composite, rank_in_ind
    """
    ensemble_path = os.path.join(OUTPUT_DIR, pred_pkl)
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"预测文件不存在: {ensemble_path}")

    pred_df = pd.read_pickle(ensemble_path)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    latest_month = pred_df['date'].max()

    ckpt_path = os.path.join(OUTPUT_DIR, f'_ckpt_beta{ckpt_suffix}.pkl')

    # Return from checkpoint cache if available and not forced
    if not force_refresh_latest and os.path.exists(ckpt_path):
        ckpt = pd.read_pickle(ckpt_path)
        if latest_month in ckpt.get('done_months', set()):
            sels = ckpt.get('selections', [])
            if sels:
                cached = pd.concat(sels, ignore_index=True)
                month_rows = cached[cached['month'] == latest_month].copy()
                if not month_rows.empty:
                    if 'ind_name' not in month_rows.columns:
                        _, ind_to_name = load_stock_industry_map()
                        month_rows['ind_name'] = (
                            month_rows['ind_code'].map(ind_to_name)
                            .fillna(month_rows['ind_code'])
                        )
                    return month_rows

    print(f'  [选股] 最新月份: {latest_month.strftime("%Y-%m")}')
    stock_to_ind, ind_to_name = load_stock_industry_map()
    df_stock,  stock_dict     = load_stock_daily()
    _,         ind_dict       = load_industry_daily(df_stock, stock_to_ind)
    fund_dict                 = load_fundamental_features()

    pred_col = next(
        (c for c in ('pred_ensemble', 'pred_gnn') if c in pred_df.columns),
        pred_df.columns[-1],
    )
    m_pred = (
        pred_df[pred_df['date'] == latest_month]
        .sort_values(pred_col, ascending=False)
        .head(TOP_K)
    )
    top_inds   = m_pred['ts_code'].tolist()
    ind_scores = dict(zip(m_pred['ts_code'], m_pred[pred_col]))

    prev_holdings: set = set()
    if os.path.exists(ckpt_path):
        prev_holdings = pd.read_pickle(ckpt_path).get('prev_holdings', set())

    result = select_stocks_for_month(
        pred_month    = latest_month,
        top_industries = top_inds,
        stock_dict    = stock_dict,
        ind_dict      = ind_dict,
        stock_to_ind  = stock_to_ind,
        fund_dict     = fund_dict,
        ind_scores    = ind_scores,
        prev_holdings = prev_holdings,
    )

    if result.empty:
        return result

    result['ind_name'] = (
        result['ind_code'].map(ind_to_name).fillna(result['ind_code'])
    )
    # 'name' column (stock display name) defaults to stock_code when unavailable
    if 'name' not in result.columns:
        result['name'] = result['stock_code']

    return result


if __name__ == '__main__':
    main()
