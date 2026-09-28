"""The monthly panel must stay unit-consistent and replayable."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_pipeline.industry_monthly import canonicalize_industry_monthly, repair_csv
from data_pipeline.update import _last_completed_month_end
from s3_ensemble_backtest import validate_backtest_inputs
from data_pipeline import etf_mapping_v2


class IndustryMonthlyIntegrityTests(unittest.TestCase):
    def test_updater_does_not_append_partial_current_month(self):
        self.assertEqual(_last_completed_month_end(pd.Timestamp('2026-09-26')),
                         pd.Timestamp('2026-08-31'))

    def test_mixed_units_and_duplicate_month_normalize_from_close(self):
        raw = pd.DataFrame([
            {'ts_code': 'A', 'date': '2025-10-31', 'close': 100, 'pct_chg': 2.0},
            {'ts_code': 'A', 'date': '2025-11-28', 'close': 110, 'pct_chg': 10.0},
            {'ts_code': 'A', 'date': '2025-11-30', 'close': 110, 'pct_chg': 0.10},
            {'ts_code': 'A', 'date': '2025-12-31', 'close': 99, 'pct_chg': -0.10},
        ])
        fixed = canonicalize_industry_monthly(raw)
        self.assertEqual(len(fixed), 3)
        np.testing.assert_allclose(fixed['ret'].iloc[1:].to_numpy(), [0.10, -0.10])
        np.testing.assert_allclose(fixed['pct_chg'].iloc[1:].to_numpy(), [10.0, -10.0])

    def test_gap_does_not_become_one_month_return(self):
        raw = pd.DataFrame([
            {'ts_code': 'A', 'date': '2025-10-31', 'close': 100},
            {'ts_code': 'A', 'date': '2025-12-31', 'close': 120},
        ])
        fixed = canonicalize_industry_monthly(raw)
        self.assertTrue(fixed['ret'].isna().all())

    def test_repair_preserves_source_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'monthly.csv'
            backup = Path(tmp) / 'monthly.original.csv'
            original = pd.DataFrame([
                {'ts_code': 'A', 'date': '2025-10-31', 'close': 100, 'pct_chg': 0},
                {'ts_code': 'A', 'date': '2025-11-28', 'close': 110, 'pct_chg': 10},
                {'ts_code': 'A', 'date': '2025-11-30', 'close': 110, 'pct_chg': .1},
            ])
            original.to_csv(source, index=False)
            before = source.read_bytes()
            self.assertEqual(repair_csv(source, backup), (3, 2))
            self.assertEqual(backup.read_bytes(), before)
            self.assertAlmostEqual(pd.read_csv(source)['pct_chg'].iloc[-1], 10.0)

    def test_etf_reader_uses_same_close_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw = Path(tmp) / 'sw.csv'
            csi = Path(tmp) / 'csi.csv'
            pd.DataFrame([
                {'ts_code': '801010.SI', 'date': '2026-01-31', 'close': 100, 'pct_chg': 0},
                {'ts_code': '801010.SI', 'date': '2026-02-28', 'close': 90, 'pct_chg': -.1},
            ]).to_csv(sw, index=False)
            pd.DataFrame({'date': ['20260130', '20260227']}).to_csv(csi, index=False)
            with patch.object(etf_mapping_v2, 'SW_MONTHLY_PATH', str(sw)), \
                 patch.object(etf_mapping_v2, 'CSI_MONTHLY_PATH', str(csi)):
                self.assertAlmostEqual(etf_mapping_v2.load_sw_returns().iloc[0, 0], -0.1)
                self.assertEqual(etf_mapping_v2._get_month_end_dates(),
                                 ['20260130', '20260227'])

    def test_missing_lstm_month_fails_instead_of_silent_inner_join(self):
        months = pd.date_range('2026-01-31', periods=3, freq='ME')
        codes = [f'I{i}' for i in range(5)]
        market = pd.DataFrame([
            {'ts_code': code, 'date': month, 'ret': .01}
            for month in months for code in codes
        ])
        gnn = pd.DataFrame([
            {'ts_code': code, 'date': month, 'actual_ret': .01 if month != months[-1] else np.nan,
             'pred_gnn': 1.0}
            for month in months for code in codes
        ])
        lstm = pd.DataFrame([
            {'ts_code': code, 'date': month, 'actual_ret': .01 if month != months[-1] else np.nan,
             'pred_lstm_b': 1.0}
            for month in (months[0], months[-1]) for code in codes
        ])
        with self.assertRaisesRegex(ValueError, '缺失月份'):
            validate_backtest_inputs(gnn, lstm, market)

    def test_partial_current_month_is_rejected(self):
        current = pd.Timestamp.today().to_period('M').to_timestamp('M')
        market = pd.DataFrame({'ts_code': ['A'], 'date': [current], 'ret': [0.01]})
        gnn = pd.DataFrame({'ts_code': ['A'], 'date': [current],
                            'actual_ret': [np.nan], 'pred_gnn': [1.0]})
        lstm = pd.DataFrame({'ts_code': ['A'], 'date': [current],
                             'actual_ret': [np.nan], 'pred_lstm_b': [1.0]})
        with self.assertRaisesRegex(ValueError, '尚未结束的月份'):
            validate_backtest_inputs(gnn, lstm, market)


if __name__ == '__main__':
    unittest.main()
