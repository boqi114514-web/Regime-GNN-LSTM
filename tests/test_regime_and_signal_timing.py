"""Checks for overseas-market availability, signal labels, and weight floor."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config import MIN_LSTM_WEIGHT
from s0_regime import _global_asof
from s2_lstm_b_train import prepare_lstm_b_data
from s3_ensemble_backtest import regime_ensemble


class TimingTests(unittest.TestCase):
    def test_us_same_day_close_is_not_available(self):
        dates = pd.bdate_range('2026-07-01', periods=22)
        prices = [100.0] * 21 + [200.0]
        daily = pd.DataFrame({'ts_code': 'SPX', 'trade_date': dates,
                              'close': prices})
        signal = pd.Series([dates[-1]])
        self.assertAlmostEqual(_global_asof(signal, daily, 'SPX', 1)[0], 0.0)
        self.assertAlmostEqual(_global_asof(signal, daily, 'SPX', 0)[0], 1.0)

    def test_lstm_signal_date_and_next_month_label(self):
        dates = pd.date_range('2026-01-31', periods=5, freq='ME')
        tech = pd.DataFrame({'ts_code': 'A', 'date': dates,
                             'momentum': range(5)})
        mkt = pd.DataFrame({'ts_code': 'A', 'date': dates,
                            'ret': [0.0, 0.1, 0.2, 0.3, 0.4]})
        x, y, meta, _ = prepare_lstm_b_data(
            tech, mkt, ['A'], (dates[0], dates[-1]), lookback=2,
            include_unlabeled=True)
        self.assertEqual([row[1] for row in meta], list(dates[1:]))
        np.testing.assert_allclose(y[:3], [0.2, 0.3, 0.4])
        self.assertTrue(np.isnan(y[-1]))
        _, y_train, meta_train, _ = prepare_lstm_b_data(
            tech, mkt, ['A'], (dates[0], dates[3]), lookback=2,
            label_end=dates[3])
        self.assertEqual([row[1] for row in meta_train], list(dates[1:3]))
        np.testing.assert_allclose(y_train, [0.2, 0.3])

    def test_lstm_deduplicates_intra_month_factor_updates(self):
        dates = pd.date_range('2026-01-31', periods=4, freq='ME')
        tech = pd.DataFrame({'ts_code': 'A', 'date': dates,
                             'momentum': range(4)})
        duplicate = tech.iloc[2].copy()
        duplicate['date'] = pd.Timestamp('2026-03-15')
        tech = pd.concat([tech, pd.DataFrame([duplicate])], ignore_index=True)
        mkt = pd.DataFrame({'ts_code': 'A', 'date': dates,
                            'ret': [0.0, 0.1, 0.2, 0.3]})
        _, _, meta, _ = prepare_lstm_b_data(
            tech, mkt, ['A'], (dates[0], dates[-1]), lookback=2,
            include_unlabeled=True)
        self.assertEqual(len(meta), 3)
        self.assertEqual([row[1] for row in meta], list(dates[1:]))

    def test_regime_ensemble_respects_lstm_floor(self):
        codes = [f'I{i}' for i in range(6)]
        dates = pd.date_range('2025-01-31', periods=3, freq='ME')
        gnn = pd.DataFrame([{'ts_code': code, 'date': date,
                             'actual_ret': i / 100,
                             'pred_gnn': float(i)}
                            for date in dates for i, code in enumerate(codes)])
        lstm = pd.DataFrame([{'ts_code': code, 'date': date,
                              'pred_lstm_b': float(-i)}
                             for date in dates for i, code in enumerate(codes)])
        regimes = pd.DataFrame({'date': dates, 'regime': 1})
        result = regime_ensemble(gnn, lstm, regimes)
        self.assertTrue((result['w_lstm'] >= MIN_LSTM_WEIGHT).all())
        self.assertTrue(np.allclose(result['w_gnn'] + result['w_lstm'], 1))


if __name__ == '__main__':
    unittest.main()
