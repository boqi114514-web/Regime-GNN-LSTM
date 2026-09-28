# -*- coding: utf-8 -*-
"""2.5 万元整手组合：对 s4 候选作末端整数优化，不改变上游学习池。

本模块只形成按已知收盘价测算的组合方案。实际委托前须以可成交价格重算。
历史绩效需在上游信号时序修正后，用下一交易日成交价另行回测。
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from config import LOCAL_DATA_RAW, OUTPUT_DIR, STOCK_DAILY_PATH


BUDGET = 25_000.0
MAIN_BOARD_CODE = re.compile(r"^(?:600|601|603|605|000|001|002|003)\d{3}$")


def is_main_board(code: str) -> bool:
    """账户可买的沪深主板；创业板、科创板、北交所均不进入最终组合。"""
    return bool(MAIN_BOARD_CODE.fullmatch(str(code).zfill(6)))


def attach_asof_prices(candidates: pd.DataFrame, daily: pd.DataFrame,
                       asof_date) -> pd.DataFrame:
    """只使用信号日期及之前的价格，避免读取未来成交价。"""
    wanted = candidates.copy()
    wanted['stock_code'] = wanted['stock_code'].astype(str).str.zfill(6)
    prices = daily[['date', 'code', 'close']].copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices['code'] = prices['code'].astype(str).str.zfill(6)
    prices = prices[(prices['date'] <= pd.Timestamp(asof_date)) &
                    prices['code'].isin(wanted['stock_code']) &
                    (pd.to_numeric(prices['close'], errors='coerce') > 0)]
    prices = prices.sort_values('date').drop_duplicates('code', keep='last')
    prices = prices.rename(columns={'code': 'stock_code', 'date': 'price_date',
                                    'close': 'reference_price'})
    return wanted.merge(prices, on='stock_code', how='left', validate='many_to_one')


def apply_limit_prices(candidates: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """以买入限价重新定仓；未报价候选不能沿用过时月末价格。"""
    if not {'stock_code', 'limit_price'} <= set(quotes.columns):
        raise ValueError('限价 CSV 需要 stock_code,limit_price 两列')
    quotes = quotes[['stock_code', 'limit_price']].copy()
    quotes['stock_code'] = quotes['stock_code'].astype(str).str.zfill(6)
    quotes['limit_price'] = pd.to_numeric(quotes['limit_price'], errors='coerce')
    if quotes.stock_code.duplicated().any() or not np.isfinite(quotes.limit_price).all() or \
            (quotes.limit_price <= 0).any():
        raise ValueError('限价 CSV 含重复代码或无效价格')
    refreshed = candidates.drop(columns=['reference_price', 'price_date'], errors='ignore').merge(
        quotes, on='stock_code', how='inner', validate='many_to_one')
    refreshed = refreshed.rename(columns={'limit_price': 'reference_price'})
    refreshed['price_date'] = pd.NaT
    if refreshed.empty:
        raise ValueError('限价 CSV 中没有匹配的候选股票')
    return refreshed


def _candidate_quality(candidates: pd.DataFrame) -> pd.Series:
    """使用已有行业及行业内排名，不把原始分数当作预期收益率。"""
    industry = candidates.groupby('ind_code')['ind_score'].first().sort_values(ascending=False)
    n_ind = len(industry)
    ind_quality = {code: 1.0 - i / max(n_ind - 1, 1)
                   for i, code in enumerate(industry.index)}
    # 原始 composite 是行业内排序，跨行业数值不能直接比较。
    within = candidates.groupby('ind_code')['rank_in_ind'].transform('max').clip(lower=1)
    stock_quality = 1.0 - (candidates['rank_in_ind'] - 1) / within
    return (0.6 * candidates['ind_code'].map(ind_quality) +
            0.4 * stock_quality).astype(float)


def optimize_portfolio(candidates: pd.DataFrame, budget: float = BUDGET,
                       max_names: int = 5, min_names: int = 4,
                       max_stock_weight: float = 0.30) -> tuple[pd.DataFrame, dict]:
    """精确整数规划：资金、100 股整手、主板、一行业最多一只、单股仓位上限。"""
    if budget <= 0 or not 0 < max_stock_weight <= 1 or max_names < 1:
        raise ValueError('资金、持仓数或单股仓位上限无效')
    required = {'stock_code', 'ind_code', 'ind_score', 'rank_in_ind', 'reference_price'}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f'候选缺少字段: {sorted(missing)}')

    df = candidates.copy()
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df[df['stock_code'].map(is_main_board)].copy()
    if 'name' in df.columns:
        df = df[~df['name'].fillna('').astype(str).str.contains(r'(?:^\*?ST|退$)', regex=True)].copy()
    df['reference_price'] = pd.to_numeric(df['reference_price'], errors='coerce')
    df = df[np.isfinite(df['reference_price']) & (df['reference_price'] > 0)].copy()
    df = df.drop_duplicates('stock_code', keep='first').reset_index(drop=True)
    if df.empty:
        raise ValueError('没有符合主板权限、有效价格和风险过滤的候选股票')

    df['lot_value'] = (df['reference_price'] * 100).round(2)
    df['max_lots'] = np.floor(max_stock_weight * budget / df['lot_value']).astype(int)
    df = df[df['max_lots'] >= 1].reset_index(drop=True)
    if df.empty:
        raise ValueError('所有候选的一手价格均超过单股仓位上限')
    df['quality'] = _candidate_quality(df)
    n = len(df)

    # 前 n 个整数变量为手数；后 n 个 0/1 变量为是否持仓。
    objective = np.r_[-df['quality'].to_numpy() * df['lot_value'].to_numpy() / budget,
                       np.zeros(n)]
    lower = np.zeros(2 * n)
    upper = np.r_[df['max_lots'].to_numpy(), np.ones(n)]
    bounds = Bounds(lower, upper)
    rows, lo, hi = [], [], []

    def add_row(row, minimum=-np.inf, maximum=np.inf):
        rows.append(row)
        lo.append(minimum)
        hi.append(maximum)

    row = np.zeros(2 * n)
    row[:n] = df['lot_value'].to_numpy()
    add_row(row, maximum=budget)
    for i, max_lots in enumerate(df['max_lots']):
        row = np.zeros(2 * n)
        row[i], row[n + i] = 1, -int(max_lots)
        add_row(row, maximum=0)
        row = np.zeros(2 * n)
        row[i], row[n + i] = -1, 1
        add_row(row, maximum=0)
    for positions in df.groupby('ind_code').indices.values():
        row = np.zeros(2 * n)
        row[n + np.asarray(positions)] = 1
        add_row(row, maximum=1)

    # 少于四个可行行业时自动降级；无论如何都保持资金和板块硬约束。
    max_possible = min(max_names, df['ind_code'].nunique())
    solution = None
    actual_min_names = 0
    for minimum in range(min(min_names, max_possible), 0, -1):
        count_row = np.zeros(2 * n)
        count_row[n:] = 1
        constraint = LinearConstraint(np.vstack(rows + [count_row]),
                                      np.r_[lo, minimum], np.r_[hi, max_possible])
        result = milp(c=objective, integrality=np.ones(2 * n), bounds=bounds,
                      constraints=constraint, options={'time_limit': 30})
        if result.success and result.x is not None:
            solution = result
            actual_min_names = minimum
            break
    if solution is None:
        raise ValueError('在当前预算与持仓约束下找不到可买组合')

    df['lots'] = np.rint(solution.x[:n]).astype(int)
    selected = df[df['lots'] > 0].copy()
    selected['shares'] = selected['lots'] * 100
    selected['planned_amount'] = (selected['lots'] * selected['lot_value']).round(2)
    selected['weight'] = selected['planned_amount'] / budget
    selected = selected.sort_values('quality', ascending=False).reset_index(drop=True)
    spend = float(selected['planned_amount'].sum())
    if spend > budget + 0.01 or (selected['weight'] > max_stock_weight + 1e-8).any():
        raise AssertionError('求解结果违反资金或单股仓位约束')
    summary = {'budget': budget, 'planned_amount': round(spend, 2),
               'cash_remaining': round(budget - spend, 2),
               'n_stocks': len(selected), 'n_industries': selected['ind_code'].nunique(),
               'minimum_names_used': actual_min_names, 'max_stock_weight': max_stock_weight,
               'board': '沪深主板', 'lot_size': 100, 'transaction_costs_included': False}
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='2.5万元沪深主板整手组合（s4 后置模块）')
    parser.add_argument('--month', help='使用已保存的历史候选 YYYY-MM；默认重新生成最新全市场候选')
    parser.add_argument('--budget', type=float, default=BUDGET)
    parser.add_argument('--max-names', type=int, default=5)
    parser.add_argument('--min-names', type=int, default=4)
    parser.add_argument('--max-stock-weight', type=float, default=0.30)
    parser.add_argument('--prices-csv', help='交易时限价 CSV：stock_code,limit_price；只优化提供价格的候选')
    parser.add_argument('--pred-pkl', default='predictions_ensemble.pkl',
                        help='最新候选的预测文件；可用 predictions_ensemble_46/55/64.pkl')
    args = parser.parse_args()

    if args.month:
        selections = pd.read_pickle(Path(OUTPUT_DIR) / 'stock_selections.pkl')
        selections['month'] = pd.to_datetime(selections['month'])
        month = pd.Period(args.month, freq='M')
        candidates = selections[selections['month'].dt.to_period('M') == month].copy()
        if candidates.empty:
            raise ValueError(f'没有 {month} 的候选股票')
        source = '已保存历史候选（可能由旧主板过滤生成）'
    else:
        # 保持双创股票参与上游候选排序，只在 optimize_portfolio 过滤交易权限。
        from s4_beta_selection import run_live
        candidates = run_live(pred_pkl=args.pred_pkl,
                              force_refresh_latest=True, main_board_only=False)
        if candidates.empty:
            raise ValueError('最新月份没有全市场候选股票')
        month = pd.Timestamp(candidates['month'].max()).to_period('M')
        source = '最新全市场候选'
    with open(STOCK_DAILY_PATH, 'rb') as stream:
        data = pickle.load(stream)
    candidates = attach_asof_prices(candidates, data['df_stock'], month.to_timestamp('M'))
    if args.prices_csv:
        quotes = pd.read_csv(args.prices_csv, dtype={'stock_code': str})
        candidates = apply_limit_prices(candidates, quotes)
    if 'name' not in candidates.columns:
        basic_path = Path(LOCAL_DATA_RAW) / 'ts_stock_basic.csv'
        if basic_path.exists():
            basic = pd.read_csv(basic_path, encoding='utf-8-sig')
            names = dict(zip(basic['ts_code'].astype(str).str[:6], basic['name']))
        else:
            names = data.get('code_to_name', {})
        candidates['name'] = candidates['stock_code'].map(names).fillna('')
    selected, summary = optimize_portfolio(
        candidates, budget=args.budget, max_names=args.max_names,
        min_names=args.min_names, max_stock_weight=args.max_stock_weight)
    weight_code = next((code for code in ('46', '55', '64')
                        if args.pred_pkl.endswith(f'_{code}.pkl')), None)
    suffix = (f'_{weight_code}' if weight_code else '') + ('_limits' if args.prices_csv else '')
    out = Path(OUTPUT_DIR) / f'budget_portfolio_{month.strftime("%Y%m")}{suffix}.csv'
    columns = ['stock_code', 'name', 'ind_code', 'quality', 'reference_price',
               'price_date', 'lots', 'shares', 'planned_amount', 'weight']
    selected[columns].to_csv(out, index=False, encoding='utf-8-sig')
    summary['signal_month'] = str(month)
    summary['candidate_source'] = source
    summary['price_source'] = '交易限价 CSV' if args.prices_csv else '信号月末收盘价'
    summary['candidate_count'] = len(candidates)
    summary['restricted_candidates'] = int((~candidates['stock_code'].map(is_main_board)).sum())
    summary['output'] = str(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected[columns].to_string(index=False))


if __name__ == '__main__':
    main()
