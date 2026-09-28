"""Fixed-weight comparisons and market-liquidity point-in-time checks."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config import FIXED_WEIGHT_VARIANTS
from data_pipeline.market_factors import (MARKET_FACTOR_COLS,
                                          append_market_factors,
                                          build_market_factors)
from s3_ensemble_backtest import simple_rank_ensemble


class FixedWeightMarketFactorTests(unittest.TestCase):
    def test_three_fixed_variants_are_distinct_and_exact(self):
        dates = pd.to_datetime(['2026-07-31'])
        codes = [f'I{i}' for i in range(6)]
        gnn = pd.DataFrame({'ts_code': codes, 'date': dates[0],
                            'actual_ret': [0.01] * 6,
                            'pred_gnn': [6, 5, 4, 3, 2, 1]})
        lstm = pd.DataFrame({'ts_code': codes, 'date': dates[0],
                             'pred_lstm_b': [1, 2, 3, 4, 5, 6]})
        for name, weights in FIXED_WEIGHT_VARIANTS.items():
            out = simple_rank_ensemble(gnn, lstm, *weights)
            self.assertAlmostEqual(out.w_gnn.iloc[0], weights[0], name)
            self.assertAlmostEqual(out.w_lstm.iloc[0], weights[1], name)
        self.assertEqual(FIXED_WEIGHT_VARIANTS['46'], (0.4, 0.6))
        self.assertEqual(FIXED_WEIGHT_VARIANTS['55'], (0.5, 0.5))
        self.assertEqual(FIXED_WEIGHT_VARIANTS['64'], (0.6, 0.4))

    def test_market_factors_do_not_change_when_future_days_are_appended(self):
        dates = pd.bdate_range('2024-01-01', periods=320)
        rows = [{'date': day, 'code': code, 'close': 10 + i * 0.01 + j,
                 'amount': 1000 + i * (j + 1)}
                for i, day in enumerate(dates) for j, code in enumerate(['A', 'B'])]
        stock = pd.DataFrame(rows)
        prefix = build_market_factors(stock[stock.date < dates[-20]])
        full = build_market_factors(stock)
        shared = prefix.merge(full, on='ym', suffixes=('_prefix', '_full'))
        shared = shared[shared.ym < dates[-20].to_period('M')]
        for factor in MARKET_FACTOR_COLS:
            np.testing.assert_allclose(shared[f'{factor}_prefix'],
                                       shared[f'{factor}_full'], equal_nan=True)

    def test_market_factors_broadcast_by_month(self):
        tech = pd.DataFrame({'ts_code': ['A', 'B'],
                             'date': pd.to_datetime(['2026-08-28', '2026-08-31']),
                             'mom_1m': [0.1, -0.1]})
        market = pd.DataFrame({'ym': [pd.Period('2026-08')],
                               **{name: [float(i)] for i, name in enumerate(MARKET_FACTOR_COLS)}})
        joined = append_market_factors(tech, market)
        self.assertEqual(len(joined), 2)
        self.assertEqual(joined.market_turnover_rel_20_60.tolist(), [0.0, 0.0])
        self.assertEqual(joined.market_up_amount_share_20.tolist(), [4.0, 4.0])


if __name__ == '__main__':
    unittest.main()
