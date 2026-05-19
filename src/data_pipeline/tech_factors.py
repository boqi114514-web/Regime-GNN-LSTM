# -*- coding: utf-8 -*-
"""data_pipeline/tech_factors.py  —— 价量因子生成（迁自 ARIMAX 项目 s1_price_volume_factors.py）

从全市场个股日K线 → 月频价量因子 → 按行业成分股取中位数聚合
产出 9 个行业月频因子：
  动量：mom_1m, mom_3m, mom_6m
  波动：vol_1m, vol_3m
  量能：turnover_chg, vol_price_corr
  形态：max_drawdown_1m, high_low_pos

产物：data/processed/price_volume_factors.pkl  (+ .csv)

用法：
    python -m data_pipeline.tech_factors               # 全量重算 (几十分钟)
    python -m data_pipeline.tech_factors --smoke       # 只跑最近 6 个月 + 限制股票数，用于 smoke test
    python -m data_pipeline.tech_factors --since YYYY-MM  # 只补指定月之后
    python -m data_pipeline.tech_factors --level l2    # 方向2：二级行业聚合，产物 price_volume_factors_l2.pkl

level 说明（方向2 Gate 3 Step 2）：
    l1（默认）→ ts_sw_members.csv，按 l1_code 聚合，产物 price_volume_factors.pkl
    l2         → ts_sw_l2_members.csv，按 l2_code 聚合，产物 price_volume_factors_l2.pkl
    个股因子算法完全不变，只换"个股→行业"的映射粒度，互不覆盖。

Phase 2 实盘化说明：
    当前是全量重算版本。后续增量版本应：
      1) 读取已有 pkl 的 max(date) 作为基线
      2) 只对新月份计算 compute_stock_factors，再 merge 进原 pkl
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (LOCAL_DATA_PROCESSED, SW_EXCLUDE,
                    load_industry_members, load_industry_members_l2,
                    load_stock_daily)


# 日K线需要的回溯交易日数
LOOKBACK_DAYS = {
    '1m': 21,    # ~1个月
    '3m': 63,    # ~3个月
    '6m': 126,   # ~6个月
}

FACTOR_COLS = [
    'mom_1m', 'mom_3m', 'mom_6m',
    'vol_1m', 'vol_3m',
    'turnover_chg', 'vol_price_corr',
    'max_drawdown_1m', 'high_low_pos',
]


# ============================================================
#  个股因子
# ============================================================

def compute_stock_factors(stock_data: pd.DataFrame) -> pd.DataFrame:
    """对单只股票的日K线计算月频价量因子。列：date, open, high, low, close, volume, amount"""
    df = stock_data.copy()
    df = df.set_index('date').sort_index()
    df['daily_ret'] = df['close'].pct_change()

    df['year_month'] = df.index.to_period('M')
    month_ends = df.groupby('year_month').tail(1).index

    results = []
    for me_date in month_ends:
        hist = df.loc[:me_date]
        if len(hist) < LOOKBACK_DAYS['1m']:
            continue

        row = {'date': me_date}
        close_me = hist['close'].iloc[-1]

        # 动量
        for label, days in LOOKBACK_DAYS.items():
            if len(hist) >= days:
                close_past = hist['close'].iloc[-days]
                row[f'mom_{label}'] = close_me / close_past - 1
            else:
                row[f'mom_{label}'] = np.nan

        # 波动
        for label in ('1m', '3m'):
            days = LOOKBACK_DAYS[label]
            window = hist['daily_ret'].iloc[-days:]
            row[f'vol_{label}'] = window.std() if len(window.dropna()) >= 10 else np.nan

        # 量能：turnover_chg
        amt_1m = hist['amount'].iloc[-LOOKBACK_DAYS['1m']:].sum()
        amt_3m = hist['amount'].iloc[-LOOKBACK_DAYS['3m']:].sum()
        if amt_3m > 0:
            avg_monthly_3m = amt_3m / 3
            row['turnover_chg'] = amt_1m / avg_monthly_3m if avg_monthly_3m > 0 else np.nan
        else:
            row['turnover_chg'] = np.nan

        # vol_price_corr
        recent_1m = hist.iloc[-LOOKBACK_DAYS['1m']:]
        ret_s = recent_1m['daily_ret'].dropna()
        vol_s = recent_1m['volume'].loc[ret_s.index]
        if len(ret_s) >= 10:
            corr = ret_s.corr(vol_s)
            row['vol_price_corr'] = corr if np.isfinite(corr) else np.nan
        else:
            row['vol_price_corr'] = np.nan

        # 形态
        prices_1m = hist['close'].iloc[-LOOKBACK_DAYS['1m']:]
        if len(prices_1m) >= 5:
            cummax = prices_1m.cummax()
            drawdown = (prices_1m - cummax) / cummax
            row['max_drawdown_1m'] = drawdown.min()
        else:
            row['max_drawdown_1m'] = np.nan

        high_1m = hist['high'].iloc[-LOOKBACK_DAYS['1m']:].max()
        low_1m = hist['low'].iloc[-LOOKBACK_DAYS['1m']:].min()
        if high_1m > low_1m:
            row['high_low_pos'] = (close_me - low_1m) / (high_1m - low_1m)
        else:
            row['high_low_pos'] = 0.5

        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out['date'] = pd.to_datetime(out['date'])
    out['year_month'] = out['date'].dt.to_period('M')
    return out


# ============================================================
#  行业聚合
# ============================================================

def aggregate_to_industry(stock_factors_dict: dict, members_df: pd.DataFrame,
                          month_dates: list, industry_col: str = 'l1_code') -> pd.DataFrame:
    """个股因子 → 行业中位数

    industry_col: 'l1_code'（一级）或 'l2_code'（二级）。
    """
    industries = sorted(members_df[industry_col].unique())
    industries = [x for x in industries if x not in SW_EXCLUDE]

    all_rows = []
    for me_date in month_dates:
        period = me_date.to_period('M')
        for ind_code in industries:
            m = members_df[members_df[industry_col] == ind_code]
            mask = m['in_date'] <= me_date
            mask &= m['out_date'].isna() | (m['out_date'] > me_date)
            codes = m.loc[mask, 'code'].unique()

            bucket = {col: [] for col in FACTOR_COLS}
            for code in codes:
                if code not in stock_factors_dict:
                    continue
                sf = stock_factors_dict[code]
                mr = sf[sf['year_month'] == period]
                if mr.empty:
                    continue
                for col in FACTOR_COLS:
                    val = mr[col].iloc[0]
                    if np.isfinite(val):
                        bucket[col].append(val)

            row = {'ts_code': ind_code, 'date': me_date}
            for col in FACTOR_COLS:
                vals = bucket[col]
                row[col] = np.median(vals) if len(vals) >= 5 else np.nan
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out['date'] = pd.to_datetime(out['date'])
    return out


# ============================================================
#  主流程
# ============================================================

def run(smoke: bool = False, since: str = None, dry_run: bool = False,
        level: str = 'l1') -> dict:
    if level not in ('l1', 'l2'):
        raise ValueError(f"level 必须是 'l1' 或 'l2'，收到 {level!r}")
    industry_col = 'l1_code' if level == 'l1' else 'l2_code'
    out_suffix = '' if level == 'l1' else '_l2'

    print('=' * 60)
    print(f'  data_pipeline.tech_factors  level={level}  smoke={smoke}  '
          f'since={since}  dry={dry_run}')
    print('=' * 60)

    print('\n[1/4] 加载个股日K线...')
    stock_daily = load_stock_daily()
    print(f'  个股日K线：{len(stock_daily):,} 行, {stock_daily["code"].nunique()} 只股票')

    print(f'\n[2/4] 加载行业成分股映射（{level}）...')
    members = load_industry_members() if level == 'l1' else load_industry_members_l2()
    valid_codes = set(stock_daily['code'].unique())
    members = members[members['code'].isin(valid_codes)].copy()
    print(f'  有效映射：{len(members):,} 条')

    stock_daily = stock_daily[stock_daily['date'] >= '2013-01-01'].copy()
    if since:
        # 增量模式要从 since - 6 个月开始读（给因子回看窗口留余量）
        since_dt = pd.Timestamp(since)
        window_start = since_dt - pd.DateOffset(months=7)
        stock_daily = stock_daily[stock_daily['date'] >= window_start]
        print(f'  since={since}，回看窗口从 {window_start.date()} 开始')

    all_dates = stock_daily['date'].sort_values().unique()
    month_ends = pd.to_datetime(
        pd.Series(all_dates).dt.to_period('M').unique().to_timestamp('M')
    )
    actual_month_ends = []
    for me in month_ends:
        month_data = stock_daily[stock_daily['date'].dt.to_period('M') == me.to_period('M')]
        if len(month_data) > 0:
            actual_month_ends.append(month_data['date'].max())
    actual_month_ends = sorted(set(actual_month_ends))

    if smoke:
        actual_month_ends = actual_month_ends[-6:]
        print(f'  [smoke] 只取最近 6 个月: {actual_month_ends[0].date()} ~ {actual_month_ends[-1].date()}')
    print(f'  月末日期范围: {actual_month_ends[0].date()} ~ {actual_month_ends[-1].date()}  ({len(actual_month_ends)} 个月)')

    print('\n[3/4] 逐股票计算价量因子...')
    all_codes = sorted(members['code'].unique())
    if smoke:
        all_codes = all_codes[:200]
        print(f'  [smoke] 只取前 200 只股票')

    stock_factors = {}
    n_total = len(all_codes)
    for i, code in enumerate(all_codes):
        if (i + 1) % 500 == 0 or i == 0:
            print(f'  [{i+1:5d}/{n_total}] 处理中...')
        sdata = stock_daily[stock_daily['code'] == code].copy()
        if len(sdata) < LOOKBACK_DAYS['6m']:
            continue
        factors = compute_stock_factors(sdata)
        if not factors.empty:
            stock_factors[code] = factors

    print(f'  成功计算：{len(stock_factors)} 只股票')

    print('\n[4/4] 汇总为行业因子（中位数）...')
    industry_factors = aggregate_to_industry(stock_factors, members, actual_month_ends,
                                             industry_col=industry_col)
    print(f'  行业因子：{len(industry_factors)} 行, {industry_factors["ts_code"].nunique()} 个行业')
    print(f'  时间范围：{industry_factors["date"].min().date()} ~ {industry_factors["date"].max().date()}')

    print('\n因子覆盖率（非NaN）：')
    for col in FACTOR_COLS:
        coverage = industry_factors[col].notna().mean()
        print(f'  {col:20s}: {coverage:.1%}')

    out_pkl = os.path.join(LOCAL_DATA_PROCESSED, f'price_volume_factors{out_suffix}.pkl')
    out_csv = os.path.join(LOCAL_DATA_PROCESSED, f'price_volume_factors{out_suffix}.csv')

    if since and not smoke:
        # 增量模式：merge 到已有 pkl
        if os.path.exists(out_pkl):
            old = pd.read_pickle(out_pkl)
            old['date'] = pd.to_datetime(old['date'])
            merged = pd.concat([old, industry_factors], ignore_index=True)
            merged = merged.drop_duplicates(subset=['ts_code', 'date'], keep='last')
            merged = merged.sort_values(['ts_code', 'date']).reset_index(drop=True)
            industry_factors = merged
            print(f'\n  增量 merge 后：{len(industry_factors)} 行')

    if dry_run:
        print('\n[dry-run] 未写入')
    else:
        os.makedirs(LOCAL_DATA_PROCESSED, exist_ok=True)
        industry_factors.to_pickle(out_pkl)
        industry_factors.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'\n写入 {out_pkl}')
        print(f'写入 {out_csv}')

    return {
        'rows': len(industry_factors),
        'industries': int(industry_factors['ts_code'].nunique()),
        'date_min': str(industry_factors['date'].min().date()),
        'date_max': str(industry_factors['date'].max().date()),
    }


def main():
    parser = argparse.ArgumentParser(description='价量因子生成')
    parser.add_argument('--smoke', action='store_true', help='最近 6 个月 + 前 200 只股票，快速验证')
    parser.add_argument('--since', default=None, help='增量模式，从 YYYY-MM 开始')
    parser.add_argument('--dry-run', action='store_true', help='不写文件')
    parser.add_argument('--level', default='l1', choices=['l1', 'l2'],
                        help='行业聚合粒度：l1=申万一级（默认），l2=申万二级（方向2）')
    args = parser.parse_args()
    run(smoke=args.smoke, since=args.since, dry_run=args.dry_run, level=args.level)


if __name__ == '__main__':
    main()
