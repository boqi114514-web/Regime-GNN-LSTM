import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_monthly_trail_applicability.py'
spec = importlib.util.spec_from_file_location('monthly_trail_applicability', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MonthlyTrailApplicabilityTests(unittest.TestCase):
    def test_cash_and_bonus_exright_adjusts_both_entry_and_peak(self):
        days = pd.bdate_range('2026-01-05', periods=4)
        quotes = pd.DataFrame(dict(date=days, close=[100., 130., 60., 55.]))
        actions = pd.DataFrame(dict(ex_date=[days[2]], cash=[10.], stock=[1.]))
        result = module.scan_holding(quotes, actions, 100., 'advance')
        self.assertEqual(result['ending_adjusted_entry'], 45.)
        self.assertEqual(result['ending_adjusted_peak'], 60.)
        self.assertIsNone(result['original_first_risk_signal'])
        self.assertEqual(result['proposed_first_risk_signal'],
                         dict(date=str(days[3].date()), reason='trailing_profit'))
        self.assertAlmostEqual(result['minimum_drawdown_after_activation'], 55./60.-1)
        self.assertTrue(result['signal_changed'])

    def test_eight_percent_trailing_is_not_active_before_twenty_percent_profit(self):
        quotes = pd.DataFrame(dict(date=pd.bdate_range('2026-01-05', periods=3), close=[110., 118., 105.]))
        actions = pd.DataFrame(columns=['ex_date', 'cash', 'stock'])
        result = module.scan_holding(quotes, actions, 100., 'advance')
        self.assertIsNone(result['trailing_activation_date'])
        self.assertIsNone(result['proposed_first_risk_signal'])
        self.assertFalse(result['signal_changed'])

    def test_actions_after_observed_holding_cannot_change_stop_signals(self):
        days = pd.bdate_range('2026-01-05', periods=3)
        quotes = pd.DataFrame(dict(date=days, close=[100., 90., 85.]))
        empty = pd.DataFrame(columns=['ex_date', 'cash', 'stock'])
        future = pd.DataFrame(dict(ex_date=[pd.Timestamp('2026-12-31')], cash=[100.], stock=[50.]))
        self.assertEqual(module.scan_holding(quotes, empty, 100., 'advance'),
                         module.scan_holding(quotes, future, 100., 'advance'))


if __name__ == '__main__':
    unittest.main()
