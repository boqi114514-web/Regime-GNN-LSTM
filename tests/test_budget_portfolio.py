"""The execution layer must respect budget, board permission and whole lots."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from s7_budget_portfolio import apply_limit_prices, attach_asof_prices, optimize_portfolio


class BudgetPortfolioTests(unittest.TestCase):
    def test_main_board_budget_lots_and_industry_cap(self):
        candidates = pd.DataFrame([
            ('600001', 'A', -1.0, 1, 21.0, '正常A'),
            ('600002', 'A', -1.0, 2, 16.0, '正常B'),
            ('000001', 'B', -2.0, 1, 13.0, '正常C'),
            ('300001', 'B', -2.0, 2, 12.0, '创业板'),
            ('603001', 'C', -3.0, 1, 18.0, '正常D'),
            ('688001', 'C', -3.0, 2, 15.0, '科创板'),
            ('002001', 'D', -4.0, 1, 8.0, '正常E'),
            ('605001', 'E', -5.0, 1, 6.0, '*ST风险'),
        ], columns=['stock_code', 'ind_code', 'ind_score', 'rank_in_ind',
                    'reference_price', 'name'])
        chosen, summary = optimize_portfolio(candidates)
        self.assertGreaterEqual(len(chosen), 4)
        self.assertLessEqual(len(chosen), 5)
        self.assertEqual(len(chosen), chosen.ind_code.nunique())
        self.assertLessEqual(summary['planned_amount'], 25_000)
        self.assertTrue((chosen.shares % 100 == 0).all())
        self.assertTrue((chosen.weight <= 0.30 + 1e-8).all())
        self.assertFalse(chosen.stock_code.str.startswith(('300', '688')).any())
        self.assertFalse(chosen.name.str.contains('ST').any())

    def test_price_lookup_cannot_use_future_quote(self):
        candidates = pd.DataFrame({'stock_code': ['600001']})
        prices = pd.DataFrame({
            'date': ['2026-08-31', '2026-09-01'],
            'code': ['600001', '600001'],
            'close': [10.0, 20.0],
        })
        attached = attach_asof_prices(candidates, prices, '2026-08-31')
        self.assertEqual(attached.reference_price.iloc[0], 10.0)
        self.assertEqual(str(attached.price_date.iloc[0].date()), '2026-08-31')

    def test_trade_limits_replace_stale_close_and_drop_unquoted_names(self):
        candidates = pd.DataFrame({
            'stock_code': ['600001', '000001'],
            'reference_price': [10.0, 10.0],
            'price_date': pd.to_datetime(['2026-08-31', '2026-08-31']),
        })
        quotes = pd.DataFrame({'stock_code': ['600001'], 'limit_price': [12.5]})
        refreshed = apply_limit_prices(candidates, quotes)
        self.assertEqual(refreshed.stock_code.tolist(), ['600001'])
        self.assertEqual(refreshed.reference_price.iloc[0], 12.5)


if __name__ == '__main__':
    unittest.main()
