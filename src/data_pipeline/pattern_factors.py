# -*- coding: utf-8 -*-
"""data_pipeline/pattern_factors.py —— 走势复刻因子（迁自 ARIMAX 项目 s2_pattern_factors.py）

每月末为每只股票提取最近 60 个交易日的"走势指纹"，与历史模式库匹配 Top-K 相似走势，
用其后续 1 个月涨跌幅做预测，再聚合为行业级 3 个因子：
  - pattern_median  : 复刻信号中位数
  - bullish_ratio   : 看涨占比
  - signal_strength : 信号强度（绝对值均值）

产物：data/processed/pattern_factors.pkl  (+ .csv)

用法：
    python -m data_pipeline.pattern_factors                # 全量重算 (10+ 分钟)
    python -m data_pipeline.pattern_factors --smoke        # 仅 200 股 + 最近 6 月
    python -m data_pipeline.pattern_factors --since YYYY-MM
    python -m data_pipeline.pattern_factors --level l2     # 方向2：二级行业聚合，产物 pattern_factors_l2.pkl

level 说明（方向2 Gate 3 Step 3）：
    l1（默认）→ ts_sw_members.csv，按 l1_code 聚合，产物 pattern_factors.pkl
    l2         → ts_sw_l2_members.csv，按 l2_code 聚合，产物 pattern_factors_l2.pkl
    指纹库 / 匹配算法完全不变，只换"个股→行业"的映射粒度，互不覆盖。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (LOCAL_DATA_PROCESSED, SW_EXCLUDE,
                    PATTERN_WINDOW, PATTERN_TOP_K, PATTERN_MIN_CORR, PATTERN_GAP_MONTHS,
                    load_industry_members, load_industry_members_l2, load_stock_daily)


# ============================================================
#  指纹 / 匹配
# ============================================================

def extract_fingerprint(prices: np.ndarray):
    p0 = prices[0]
    if p0 <= 0 or np.isnan(p0):
        return None
    fp = prices / p0 - 1.0
    if np.any(np.isnan(fp)) or np.any(np.isinf(fp)):
        return None
    return fp


def compute_forward_return(close_arr: np.ndarray, end_idx: int, forward_days: int = 21):
    if end_idx + forward_days >= len(close_arr):
        return np.nan
    c0 = close_arr[end_idx]
    c1 = close_arr[end_idx + forward_days]
    if c0 <= 0:
        return np.nan
    return c1 / c0 - 1.0


def build_pattern_library(stock_dict: dict, month_end_dates: list):
    """对每只股票每个月末构建 (指纹, 后续21日涨跌)。"""
    all_fps, all_frets, all_dates, all_codes = [], [], [], []
    n_stocks = len(stock_dict)
    for i, (code, sdata) in enumerate(stock_dict.items()):
        if (i + 1) % 1000 == 0:
            print(f'    建库进度：{i+1}/{n_stocks}')
        dates_arr = sdata['date'].values
        close_arr = sdata['close'].values.astype(float)
        n = len(close_arr)
        for me_date in month_end_dates:
            idx = np.searchsorted(dates_arr, np.datetime64(me_date), side='right') - 1
            if idx < PATTERN_WINDOW - 1 or idx >= n:
                continue
            window = close_arr[idx - PATTERN_WINDOW + 1: idx + 1]
            if len(window) < PATTERN_WINDOW:
                continue
            if np.sum(np.diff(window) == 0) > 10:
                continue
            fp = extract_fingerprint(window)
            if fp is None:
                continue
            fret = compute_forward_return(close_arr, idx, forward_days=21)
            all_fps.append(fp)
            all_frets.append(fret)
            all_dates.append(me_date)
            all_codes.append(code)

    return (np.array(all_fps, dtype=np.float32),
            np.array(all_frets, dtype=np.float32),
            np.array(all_dates),
            np.array(all_codes))


def match_patterns_batch(current_fps: np.ndarray, current_date,
                         lib_patterns: np.ndarray, lib_frets: np.ndarray,
                         lib_dates: np.ndarray,
                         top_k: int = PATTERN_TOP_K,
                         min_corr: float = PATTERN_MIN_CORR,
                         gap_months: int = PATTERN_GAP_MONTHS) -> np.ndarray:
    """批量计算 M 只股票当前指纹与历史库的相似度。"""
    cutoff_date = current_date - pd.DateOffset(months=gap_months)
    mask = lib_dates < np.datetime64(cutoff_date)
    mask &= np.isfinite(lib_frets)
    if mask.sum() < top_k:
        return np.full(len(current_fps), np.nan)

    hist_patterns = lib_patterns[mask]
    hist_frets = lib_frets[mask]

    curr_mean = current_fps.mean(axis=1, keepdims=True)
    curr_std = current_fps.std(axis=1, keepdims=True)
    curr_std[curr_std < 1e-10] = 1e-10
    curr_norm = (current_fps - curr_mean) / curr_std

    hist_mean = hist_patterns.mean(axis=1, keepdims=True)
    hist_std = hist_patterns.std(axis=1, keepdims=True)
    hist_std[hist_std < 1e-10] = 1e-10
    hist_norm = (hist_patterns - hist_mean) / hist_std

    corr_matrix = (curr_norm @ hist_norm.T) / PATTERN_WINDOW

    signals = np.full(len(current_fps), np.nan)
    for i in range(len(current_fps)):
        corrs = corr_matrix[i]
        valid = corrs >= min_corr
        if valid.sum() < 3:
            continue
        valid_corrs = corrs[valid]
        valid_frets = hist_frets[valid]
        if len(valid_corrs) > top_k:
            topk_idx = np.argpartition(valid_corrs, -top_k)[-top_k:]
            signals[i] = np.mean(valid_frets[topk_idx])
        else:
            signals[i] = np.mean(valid_frets)
    return signals


# ============================================================
#  行业聚合
# ============================================================

def industry_factors_for_month(stock_signals: dict, members_df: pd.DataFrame,
                               industries: list, ref_date,
                               industry_col: str = 'l1_code') -> list:
    rows = []
    for ind in industries:
        m = members_df[members_df[industry_col] == ind]
        mask = m['in_date'] <= ref_date
        mask &= m['out_date'].isna() | (m['out_date'] > ref_date)
        codes = m.loc[mask, 'code'].unique()

        sigs = [stock_signals[c] for c in codes
                if c in stock_signals and np.isfinite(stock_signals[c])]
        if len(sigs) < 5:
            row = {'ts_code': ind, 'date': ref_date,
                   'pattern_median': np.nan,
                   'bullish_ratio': np.nan,
                   'signal_strength': np.nan}
        else:
            arr = np.array(sigs)
            row = {'ts_code': ind, 'date': ref_date,
                   'pattern_median': float(np.median(arr)),
                   'bullish_ratio': float(np.mean(arr > 0)),
                   'signal_strength': float(np.mean(np.abs(arr)))}
        rows.append(row)
    return rows


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
    print(f'  data_pipeline.pattern_factors  level={level}  smoke={smoke}  '
          f'since={since}  dry={dry_run}')
    print('=' * 60)

    print('\n[1/5] 加载个股日K线...')
    stock_daily = load_stock_daily()
    stock_daily = stock_daily[stock_daily['date'] >= '2010-01-01'].copy()
    print(f'  {len(stock_daily):,} 行, {stock_daily["code"].nunique()} 只股票')

    print(f'\n[2/5] 加载行业映射（{level}）...')
    members = load_industry_members() if level == 'l1' else load_industry_members_l2()
    valid_codes = set(stock_daily['code'].unique())
    members = members[members['code'].isin(valid_codes)].copy()
    industries = sorted([x for x in members[industry_col].unique() if x not in SW_EXCLUDE])
    print(f'  行业数：{len(industries)}')

    print('\n[3/5] 构建股票字典...')
    stock_dict = {}
    for code, group in stock_daily.groupby('code'):
        if len(group) >= PATTERN_WINDOW + 21:
            stock_dict[code] = group.reset_index(drop=True)
    if smoke:
        keep = sorted(stock_dict.keys())[:200]
        stock_dict = {c: stock_dict[c] for c in keep}
        print(f'  [smoke] 只取 200 股')
    print(f'  有效股票数：{len(stock_dict)}')

    # 月末日期序列
    all_dates = stock_daily['date'].sort_values().unique()
    actual_me = []
    for _, dates in stock_daily.groupby(stock_daily['date'].dt.to_period('M'))['date']:
        actual_me.append(dates.max())
    actual_me = sorted(set(actual_me))
    actual_me = [d for d in actual_me if d >= pd.Timestamp('2013-01-01')]
    print(f'  模式库月末范围：{actual_me[0].date()} ~ {actual_me[-1].date()}')

    print('\n[4/5] 建立历史模式库...')
    patterns, frets, p_dates, p_codes = build_pattern_library(stock_dict, actual_me)
    print(f'  库大小：{len(patterns):,} 条；含后续涨跌：{int(np.isfinite(frets).sum()):,} 条')

    print('\n[5/5] 逐月计算复刻因子...')
    calc_months = [d for d in actual_me if d >= pd.Timestamp('2015-01-01')]
    if smoke:
        calc_months = calc_months[-6:]
        print(f'  [smoke] 仅最近 6 月: {calc_months[0].date()} ~ {calc_months[-1].date()}')
    elif since:
        since_ts = pd.Timestamp(since)
        calc_months = [d for d in calc_months if d >= since_ts]
        print(f'  since={since}: 计算 {len(calc_months)} 个月')
    print(f'  计算月数：{len(calc_months)}')

    all_rows = []
    for mi, me_date in enumerate(calc_months):
        if (mi + 1) % 12 == 0 or mi == 0:
            print(f'  [{mi+1}/{len(calc_months)}] {me_date.date()}')

        current_fps_list, current_codes_list = [], []
        for code, sdata in stock_dict.items():
            dates_arr = sdata['date'].values
            close_arr = sdata['close'].values.astype(float)
            idx = np.searchsorted(dates_arr, np.datetime64(me_date), side='right') - 1
            if idx < PATTERN_WINDOW - 1 or idx >= len(close_arr):
                continue
            window = close_arr[idx - PATTERN_WINDOW + 1: idx + 1]
            if len(window) < PATTERN_WINDOW or np.sum(np.diff(window) == 0) > 10:
                continue
            fp = extract_fingerprint(window)
            if fp is not None:
                current_fps_list.append(fp)
                current_codes_list.append(code)

        if len(current_fps_list) < 10:
            continue

        current_fps = np.array(current_fps_list, dtype=np.float32)
        signals = match_patterns_batch(current_fps, me_date, patterns, frets, p_dates)
        stock_signals = dict(zip(current_codes_list, signals))

        all_rows.extend(industry_factors_for_month(stock_signals, members, industries,
                                                   me_date, industry_col=industry_col))

    result = pd.DataFrame(all_rows)
    if result.empty:
        print('  无可写数据')
        return {'rows': 0}
    result['date'] = pd.to_datetime(result['date'])
    print(f'\n  复刻因子：{len(result)} 行, {result["ts_code"].nunique()} 个行业')
    print(f'  时间范围：{result["date"].min().date()} ~ {result["date"].max().date()}')

    print('\n因子覆盖率（非NaN）：')
    for col in ['pattern_median', 'bullish_ratio', 'signal_strength']:
        print(f'  {col:20s}: {result[col].notna().mean():.1%}')

    out_pkl = os.path.join(LOCAL_DATA_PROCESSED, f'pattern_factors{out_suffix}.pkl')
    out_csv = os.path.join(LOCAL_DATA_PROCESSED, f'pattern_factors{out_suffix}.csv')

    if since and not smoke and os.path.exists(out_pkl):
        old = pd.read_pickle(out_pkl)
        old['date'] = pd.to_datetime(old['date'])
        merged = pd.concat([old, result], ignore_index=True)
        merged = merged.drop_duplicates(subset=['ts_code', 'date'], keep='last')
        merged = merged.sort_values(['ts_code', 'date']).reset_index(drop=True)
        result = merged
        print(f'\n  增量 merge 后：{len(result)} 行')

    if dry_run:
        print('\n[dry-run] 未写入')
    else:
        os.makedirs(LOCAL_DATA_PROCESSED, exist_ok=True)
        result.to_pickle(out_pkl)
        result.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'\n写入 {out_pkl}')
        print(f'写入 {out_csv}')

    return {
        'rows': len(result),
        'industries': int(result['ts_code'].nunique()),
        'date_min': str(result['date'].min().date()),
        'date_max': str(result['date'].max().date()),
    }


def main():
    parser = argparse.ArgumentParser(description='走势复刻因子生成')
    parser.add_argument('--smoke', action='store_true', help='200 股 + 最近 6 月')
    parser.add_argument('--since', default=None, help='增量起始月份 YYYY-MM')
    parser.add_argument('--dry-run', action='store_true', help='不写文件')
    parser.add_argument('--level', default='l1', choices=['l1', 'l2'],
                        help='行业聚合粒度：l1=申万一级（默认），l2=申万二级（方向2）')
    args = parser.parse_args()
    run(smoke=args.smoke, since=args.since, dry_run=args.dry_run, level=args.level)


if __name__ == '__main__':
    main()
