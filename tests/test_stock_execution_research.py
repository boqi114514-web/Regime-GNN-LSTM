import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from stock_execution_research import historical_members, stock_features


class StockExecutionTests(unittest.TestCase):
    def test_membership_is_historical_and_exit_exclusive(self):
        m = pd.DataFrame({'ts_code': ['x', 'x'], 'l1_code': ['old', 'new'],
                          'in_date': ['20100101', '20240101'], 'out_date': ['20240101', None]})
        self.assertEqual(historical_members(m, pd.Timestamp('2023-12-31')).ind_code.tolist(), ['old'])
        self.assertEqual(historical_members(m, pd.Timestamp('2024-01-01')).ind_code.tolist(), ['new'])

    def test_split_is_not_negative_momentum_and_future_is_unused(self):
        dates = pd.date_range('2021-01-31', periods=16, freq='ME')
        m = pd.DataFrame({'date': dates, 'ts_code': 'x', 'close': [10.]*8+[5.]*8,
                          'adj_factor': [1.]*8+[2.]*8})
        f = stock_features(m)
        self.assertTrue(f.mom6.eq(0).all())
        pd.testing.assert_frame_equal(stock_features(m.iloc[:15]).reset_index(drop=True),
                                      f[f.date <= dates[14]].reset_index(drop=True))
