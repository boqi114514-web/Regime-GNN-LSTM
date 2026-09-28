import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from price_risk_pipeline import industry_plans, make_budget_plan, price_panel, require_execution_data
from research_improvements import make_panel


class PriceRiskPipelineTests(unittest.TestCase):
    def fixtures(self, exposure=.5):
        codes = ['600001', '600002', '000001', '000002', '002001', '300001', '688001']
        inds = ['I0', 'I1', 'I2', 'I3', 'I4', 'I0', 'I1']
        candidates = pd.DataFrame({'month': pd.Timestamp('2026-08-31'), 'stock_code': codes,
                                   'ind_code': inds, 'rank_in_ind': [1]*7})
        plans = pd.DataFrame({'date': pd.Timestamp('2026-08-31'), 'ts_code': [f'I{i}' for i in range(5)],
                              'selection_score': np.arange(5., 0., -1), 'risk_exposure': exposure})
        daily = pd.DataFrame({'date': pd.Timestamp('2026-08-31'), 'code': codes, 'close': [10.]*7})
        return candidates, plans, daily

    def test_risk_budget_not_reexpanded_to_25000(self):
        c, p, d = self.fixtures()
        selected, summary = make_budget_plan(c, p, d, '2026-08')
        self.assertEqual(summary['risky_budget'], 12500.)
        self.assertLessEqual(selected.planned_amount.sum(), 12500.)
        self.assertGreaterEqual(summary['account_cash_after_plan'], 12500.)
        self.assertTrue((selected.shares % 100 == 0).all())
        self.assertFalse(selected.stock_code.str.startswith(('300', '688')).any())

    def test_profits_do_not_raise_cap_and_losses_do_not_get_external_cash(self):
        c, p, d = self.fixtures()
        for equity, target in ((30000., 12500.), (18000., 9000.)):
            _, summary = make_budget_plan(c, p, d, '2026-08', equity=equity)
            self.assertEqual(summary['risky_budget'], target)
            self.assertGreaterEqual(summary['account_cash_after_plan'], 0.)

    def test_wrong_candidate_month_is_rejected(self):
        c, p, d = self.fixtures()
        c['month'] = pd.Timestamp('2026-09-30')
        with self.assertRaisesRegex(ValueError, 'Candidate month'):
            make_budget_plan(c, p, d, '2026-08')

    def test_prices_do_not_come_from_future(self):
        c, p, d = self.fixtures()
        future = d.copy()
        future['date'] = pd.Timestamp('2026-09-01')
        future['close'] = 1000.
        selected, _ = make_budget_plan(c, p, pd.concat([d, future]), '2026-08')
        self.assertTrue(selected.reference_price.eq(10.).all())

    def test_latest_signal_does_not_require_future_returns(self):
        pred = pd.DataFrame({'date': pd.Timestamp('2026-08-31'), 'ts_code': [f'I{i}' for i in range(6)],
                             'pred_ensemble': np.arange(6.), 'actual_ret': np.nan, 'risk_exposure': .5})
        plan = industry_plans(pred)
        pred['actual_ret'] = np.arange(6.)[::-1]
        changed = industry_plans(pred)
        self.assertEqual(plan.ts_code.tolist(), changed.ts_code.tolist())
        self.assertEqual(len(plan), 5)
        self.assertAlmostEqual(plan.target_equity_weight.sum(), .5)

    def test_missing_execution_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'corporate_actions'):
            require_execution_data({})

    def test_bad_rank_is_rejected(self):
        c, p, d = self.fixtures()
        c.loc[0, 'rank_in_ind'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid candidate ranks'):
            make_budget_plan(c, p, d, '2026-08')


if __name__ == '__main__':
    unittest.main()
