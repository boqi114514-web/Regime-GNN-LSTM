import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_dc_leader_allocation import leader_first_optimizer


class RankFirstAllocationTests(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame(dict(stock_code=['000001', '600001', '300001', '688001'],
            ts_code=['000001.SZ', '600001.SH', '300001.SZ', '688001.SH'],
            reference_price=[110., 3., 1., 1.], leadership_score=[.97, .96, 1., .99]))

    def test_cheap_full_investment_cannot_displace_higher_score(self):
        c = self.candidates(); original = c.copy(deep=True)
        selected, info = leader_first_optimizer(c, 25000, min_names=4, max_names=5)
        self.assertEqual(selected.ts_code.tolist(), ['000001.SZ'])
        self.assertEqual(selected.shares.tolist(), [200])
        self.assertEqual(info['residual_cash'], 3000.)
        pd.testing.assert_frame_equal(c, original)

    def test_unaffordable_top_name_falls_to_next_permitted_name(self):
        c = self.candidates(); c.loc[0, 'reference_price'] = 251.
        selected, _ = leader_first_optimizer(c, 25000)
        self.assertEqual(selected.ts_code.tolist(), ['600001.SH'])
        self.assertEqual(selected.shares.tolist(), [8300])

    def test_exact_decimal_lot_budget_and_no_topup(self):
        c = self.candidates().iloc[:1].copy(); c['reference_price'] = 3.41
        selected, _ = leader_first_optimizer(c, 341 * 3)
        self.assertEqual(selected.shares.tolist(), [300])
        selected, _ = leader_first_optimizer(c, 341 * 3 - .01)
        self.assertEqual(selected.shares.tolist(), [200])

    def test_future_columns_and_input_order_do_not_change_selection(self):
        c = self.candidates(); expected, _ = leader_first_optimizer(c, 25000)
        actual, _ = leader_first_optimizer(c.iloc[::-1].assign(future_return=999), 25000)
        self.assertEqual(actual.ts_code.tolist(), expected.ts_code.tolist())
        self.assertEqual(actual.shares.tolist(), expected.shares.tolist())

    def test_ties_are_deterministic(self):
        c = self.candidates().iloc[:2].copy(); c['leadership_score'] = .9
        selected, _ = leader_first_optimizer(c.iloc[::-1], 25000)
        self.assertEqual(selected.ts_code.tolist(), ['000001.SZ'])

    def test_invalid_candidates_fail_closed(self):
        for field, value in [('reference_price', np.nan), ('reference_price', 0),
                             ('leadership_score', np.inf), ('leadership_score', 0)]:
            c = self.candidates().iloc[:1].copy(); c[field] = value
            with self.assertRaises(ValueError): leader_first_optimizer(c, 25000)
        with self.assertRaises(ValueError): leader_first_optimizer(self.candidates(), 0)
        with self.assertRaises(ValueError): leader_first_optimizer(self.candidates(), 25000, max_names=0)
        with self.assertRaises(ValueError): leader_first_optimizer(pd.concat([self.candidates()]*2), 25000)
        with self.assertRaises(ValueError): leader_first_optimizer(self.candidates().assign(stock_code='000001'), 25000)


if __name__ == '__main__': unittest.main()
