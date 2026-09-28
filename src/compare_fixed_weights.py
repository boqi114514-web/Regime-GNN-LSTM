# -*- coding: utf-8 -*-
"""同一 GNN、同一月份比较 LSTM 全市场量价因子消融与 4:6/5:5/6:4。"""

import os

import numpy as np
import pandas as pd

from config import FIXED_WEIGHT_VARIANTS, OUTPUT_DIR, load_industry_monthly
from s3_ensemble_backtest import (calc_metrics, calc_rank_ic,
                                  run_topk_strategy, simple_rank_ensemble,
                                  validate_backtest_inputs)


def to_holding_month(returns):
    """Signal at the end of month t earns the return of month t+1."""
    held = returns.copy()
    held.index = pd.DatetimeIndex(held.index) + pd.offsets.MonthEnd(1)
    return held


def main():
    gnn = pd.read_pickle(os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl'))
    mkt = load_industry_monthly().sort_values(['ts_code', 'date']).copy()
    mkt['fwd_ret'] = mkt.groupby('ts_code')['ret'].shift(-1)
    benchmark = mkt.groupby('date')['fwd_ret'].mean()
    records = []
    subperiod_records = []
    annual_records = []
    for factor_set, fname in [('原行业因子', 'predictions_lstm_b_no_market.pkl'),
                              ('加入全市场量价', 'predictions_lstm_b.pkl')]:
        lstm = pd.read_pickle(os.path.join(OUTPUT_DIR, fname))
        validate_backtest_inputs(gnn, lstm, mkt)
        for code, (w_gnn, w_lstm) in FIXED_WEIGHT_VARIANTS.items():
            ens = simple_rank_ensemble(gnn, lstm, w_gnn, w_lstm)
            rets, _ = run_topk_strategy(ens, 'pred_ensemble', inertia=0.2)
            bench = benchmark.reindex(rets.index)
            held = to_holding_month(rets)
            held_bench = to_holding_month(bench)
            metrics = calc_metrics(rets)
            bench_metrics = calc_metrics(bench)
            ics = calc_rank_ic(ens, 'pred_ensemble')
            records.append({
                'factor_set': factor_set,
                'weight': code,
                'w_gnn': w_gnn,
                'w_lstm': w_lstm,
                'first_signal': ens.date.min(),
                'last_realized_signal': rets.index.max(),
                'n_months': len(rets),
                **metrics,
                'matched_benchmark_annual_return': bench_metrics['annual_return'],
                'annual_excess_vs_matched_benchmark':
                    metrics['annual_return'] - bench_metrics['annual_return'],
                'rank_ic': ics.ic.mean() if not ics.empty else np.nan,
            })
            for period_name, start, end in (
                ('2019-2022', '2019-01-01', '2022-12-31'),
                ('2023-2026', '2023-01-01', '2026-12-31'),
            ):
                segment = held.loc[start:end]
                segment_bench = held_bench.reindex(segment.index)
                values = calc_metrics(segment)
                base = calc_metrics(segment_bench)
                subperiod_records.append({
                    'factor_set': factor_set, 'weight': code,
                    'period': period_name, 'n_months': len(segment),
                    'annual_return': values['annual_return'],
                    'sharpe_ratio': values['sharpe_ratio'],
                    'max_drawdown': values['max_drawdown'],
                    'annual_excess_vs_matched_benchmark':
                    values['annual_return'] - base['annual_return'],
                })
            for year in sorted(held.index.year.unique()):
                segment = held[held.index.year == year]
                segment_bench = held_bench.reindex(segment.index)
                annual_records.append({
                    'factor_set': factor_set, 'weight': code,
                    'holding_year': year, 'n_months': len(segment),
                    'first_holding_month': segment.index.min(),
                    'last_holding_month': segment.index.max(),
                    'calendar_return': (1 + segment).prod() - 1,
                    'benchmark_return': (1 + segment_bench).prod() - 1,
                })
    summary = pd.DataFrame(records)
    output = os.path.join(OUTPUT_DIR, 'fixed_weight_market_factor_ablation.csv')
    summary.to_csv(output, index=False, encoding='utf-8-sig')
    subperiod = pd.DataFrame(subperiod_records)
    subpath = os.path.join(OUTPUT_DIR, 'fixed_weight_market_factor_subperiods.csv')
    subperiod.to_csv(subpath, index=False, encoding='utf-8-sig')
    annual = pd.DataFrame(annual_records)
    annual_path = os.path.join(OUTPUT_DIR, 'fixed_weight_calendar_returns.csv')
    annual.to_csv(annual_path, index=False, encoding='utf-8-sig')
    print(summary[['factor_set', 'weight', 'n_months', 'annual_return',
                   'sharpe_ratio', 'max_drawdown',
                   'annual_excess_vs_matched_benchmark', 'rank_ic']].to_string(index=False))
    print(f'保存 {output}')
    print(subperiod.to_string(index=False))
    print(f'保存 {subpath}')
    print(annual[annual.holding_year >= 2023].to_string(index=False))
    print(f'保存 {annual_path}')
    return summary


if __name__ == '__main__':
    main()
