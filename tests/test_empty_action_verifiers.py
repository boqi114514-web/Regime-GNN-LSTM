import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import verify_small_account as monthly
from verify_market_states import verify_daily


class EmptyActionVerificationTests(unittest.TestCase):
    def fixture(self, root):
        out = root/'out'
        out.mkdir()
        boundary = root/'boundaries'
        boundary.mkdir()
        day, code, name = pd.Timestamp('2026-01-30'), '000001.SZ', 'empty_actions'
        def csv(label, rows, columns=None):
            pd.DataFrame(rows, columns=columns).to_csv(out/f'{label}_{name}.csv', index=False)
        csv('account', [dict(date=day,equity=25100.,cash=24000.,dividend_receivable=0.,holdings=1)])
        csv('trades', [dict(date=day,code=code,side='buy',shares=100,price=10.,reason='rebalance',signal_date='2026-01-29')])
        csv('holdings', [dict(date=day,code=code,shares=100,close=11.,value=1100.)])
        csv('allocations', [dict(date=day,cash=24000.,holdings=1)])
        csv('daily_nav', [dict(date=day,cash=24000.,equity=25100.)])
        csv('corporate_events', [], ['date','event','code','payment_date','cash_entitlement','bonus_shares','fractional_bonus_discarded'])
        csv('suspension_marks', [])
        pd.DataFrame([dict(date=day,ts_code=code,close=11.,volume=100.)]).to_pickle(out/'daily.pkl')
        pd.DataFrame([dict(ts_code=code,close=11.)]).to_pickle(boundary/'daily_20260130.pkl')
        (boundary/'sessions.json').write_text(json.dumps([dict(first='2026-01-30',last='2026-01-30')]),encoding='utf-8')
        return out, name

    def test_month_and_daily_reconcile_without_any_corporate_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, name = self.fixture(root)
            with patch.object(monthly, 'OUT', out):
                result = monthly.verify(name, expected_months=1, end_date='2026-01-30')
            self.assertEqual(result['corporate_events'], 0)
            result = verify_daily(out, root, [name])[0]
            self.assertEqual(result['days'], 1)
            self.assertEqual(result['max_nav_error'], 0.)

    def test_malformed_action_numeric_field_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, name = self.fixture(root)
            path = out/f'corporate_events_{name}.csv'
            pd.DataFrame([dict(date='2026-01-30',event='cash',code='000001.SZ',payment_date='2026-01-30',cash_entitlement='bad',bonus_shares=0)]).to_csv(path,index=False)
            with patch.object(monthly, 'OUT', out), self.assertRaises(ValueError):
                monthly.verify(name, expected_months=1, end_date='2026-01-30')
            with self.assertRaises(ValueError):
                verify_daily(out, root, [name])


if __name__ == '__main__':
    unittest.main()
