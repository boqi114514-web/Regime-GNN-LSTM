import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from verify_market_states import verify_daily, verify_reentry_timing


class DailyVerificationTests(unittest.TestCase):
    def test_daily_reentry_timing_and_cooldown(self):
        dates=pd.bdate_range('2023-01-02',periods=9)
        t=pd.DataFrame(dict(date=[dates[1],dates[6]],side=['sell','buy'],
                            reason=['hard_stop','daily_reentry'],signal_date=[dates[0],dates[5]]))
        self.assertEqual(verify_reentry_timing(t,dates),1)
        t.loc[1,'signal_date']=dates[6]
        with self.assertRaisesRegex(AssertionError,'previous session'):
            verify_reentry_timing(t,dates)
        t.loc[1,'signal_date']=dates[5]; t.loc[0,'date']=dates[2]
        with self.assertRaisesRegex(AssertionError,'cooldown'):
            verify_reentry_timing(t,dates)

    def fixture(self, directory):
        out = Path(directory)
        (out/'boundaries').mkdir()
        (out/'boundaries/sessions.json').write_text(json.dumps([]), encoding='utf-8')
        pd.DataFrame(dict(date=pd.to_datetime(['2023-01-03', '2023-01-04']),
                          ts_code=['000001.SZ']*2, volume=[1000]*2, close=[12., 11.])).to_pickle(out/'daily.pkl')
        pd.DataFrame(dict(date=['2023-01-03', '2023-01-04'], code=['000001.SZ']*2,
                          side=['buy', 'sell'], shares=[100]*2, price=[10., 11.])).to_csv(out/'trades_test.csv', index=False)
        pd.DataFrame(dict(date=['2023-01-03', '2023-01-04'], equity=[25200., 25100.],
                          cash=[24000., 25100.])).to_csv(out/'daily_nav_test.csv', index=False)
        pd.DataFrame(columns=['date', 'payment_date', 'code', 'cash_entitlement', 'bonus_shares']).to_csv(out/'corporate_events_test.csv', index=False)
        (out/'suspension_marks_test.csv').write_text('\n', encoding='utf-8')
        return out

    def test_independent_daily_replay(self):
        with TemporaryDirectory() as directory:
            out = self.fixture(directory)
            result = verify_daily(out, out, ['test'])[0]
            self.assertEqual(result['days'], 2)
            self.assertEqual(result['max_nav_error'], 0.)

    def test_detects_tampered_daily_mark(self):
        with TemporaryDirectory() as directory:
            out = self.fixture(directory)
            d = pd.read_csv(out/'daily_nav_test.csv')
            d.loc[0, 'equity'] += 1
            d.to_csv(out/'daily_nav_test.csv', index=False)
            with self.assertRaises(AssertionError):
                verify_daily(out, out, ['test'])

    def test_detects_same_day_purchase_and_sale(self):
        with TemporaryDirectory() as directory:
            out = self.fixture(directory)
            t = pd.read_csv(out/'trades_test.csv')
            t.loc[1, 'date'] = '2023-01-03'
            t.to_csv(out/'trades_test.csv', index=False)
            d = pd.read_csv(out/'daily_nav_test.csv')
            d['equity'] = d['cash'] = 25100.
            d.to_csv(out/'daily_nav_test.csv', index=False)
            with self.assertRaisesRegex(AssertionError, 'Same-day'):
                verify_daily(out, out, ['test'])


if __name__ == '__main__':
    unittest.main()
