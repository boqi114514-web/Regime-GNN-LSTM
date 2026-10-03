import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'scripts'))
from verify_market_states import verify_reentry_timing, verify_weekly_reentry_timing
from audit_dc_theme_accounts import buy_limit_evidence


class WeeklyExecutionVerificationTests(unittest.TestCase):
    def setUp(self):
        # Monday 2026-04-06 is omitted intentionally: the next session is Tuesday.
        self.dates = pd.DatetimeIndex([
            '2026-04-01', '2026-04-02', '2026-04-03',
            '2026-04-07', '2026-04-08', '2026-04-09', '2026-04-10',
            '2026-04-13', '2026-04-14', '2026-05-06',
        ])

    def trade(self, day='2026-04-07', signal='2026-04-03', sale=None):
        rows = [dict(date=pd.Timestamp(day), signal_date=pd.Timestamp(signal),
                     side='buy', reason='weekly_reentry')]
        if sale is not None:
            rows.append(dict(date=pd.Timestamp(sale), signal_date=pd.NaT,
                             side='sell', reason='stop'))
        return pd.DataFrame(rows)

    def test_next_actual_session_after_holiday_and_no_sale_required(self):
        self.assertEqual(verify_weekly_reentry_timing(self.trade(), self.dates), 1)

    def test_signal_must_be_previous_actual_session(self):
        with self.assertRaisesRegex(AssertionError, 'previous session'):
            verify_weekly_reentry_timing(self.trade(signal='2026-04-02'), self.dates)

    def test_nonweekly_session_rejected_even_with_previous_signal(self):
        with self.assertRaisesRegex(AssertionError, 'completed week'):
            verify_weekly_reentry_timing(self.trade(day='2026-04-08', signal='2026-04-07'), self.dates)

    def test_two_session_cooldown_allows_thursday_sale(self):
        self.assertEqual(verify_weekly_reentry_timing(self.trade(sale='2026-04-02'), self.dates), 1)

    def test_latest_sale_controls_cooldown(self):
        trades = pd.concat([self.trade(sale='2026-04-02'), pd.DataFrame([
            dict(date=pd.Timestamp('2026-04-03'), signal_date=pd.NaT, side='sell', reason='stop')])],
            ignore_index=True)
        with self.assertRaisesRegex(AssertionError, 'two-session'):
            verify_weekly_reentry_timing(trades, self.dates)

    def test_same_day_sale_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'two-session'):
            verify_weekly_reentry_timing(self.trade(sale='2026-04-07'), self.dates)

    def test_month_first_keeps_monthly_convention(self):
        with self.assertRaisesRegex(AssertionError, 'previous session'):
            verify_weekly_reentry_timing(self.trade(day='2026-05-06', signal='2026-04-14'), self.dates)

    def test_missing_execution_session_and_first_session_fail_closed(self):
        with self.assertRaisesRegex(AssertionError, 'verified trading session'):
            verify_weekly_reentry_timing(self.trade(day='2026-04-06'), self.dates)
        with self.assertRaisesRegex(AssertionError, 'previous session'):
            verify_weekly_reentry_timing(self.trade(day='2026-04-01', signal='2026-03-31'), self.dates)

    def test_daily_five_session_rule_unchanged(self):
        dates = pd.bdate_range('2026-04-01', periods=10)
        trades = self.trade(day='2026-04-08', signal='2026-04-07', sale='2026-04-02')
        trades.loc[trades.side.eq('buy'), 'reason'] = 'daily_reentry'
        with self.assertRaisesRegex(AssertionError, 'five-session'):
            verify_reentry_timing(trades, dates)


class WeeklyLimitEvidenceTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'root'
        out = Path(directory)/'out'
        (root/'boundaries').mkdir(parents=True)
        (out/'replacement_limits').mkdir(parents=True)
        frame = pd.DataFrame([dict(ts_code='000001.SZ', trade_date='20260407',
                                   down_limit=9., up_limit=11.)])
        path = out/'replacement_limits/20260407.pkl'
        frame.to_pickle(path)
        return out, root, path, frame

    def test_weekly_route_uses_actual_unprefixed_replacement_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            out, root, path, _ = self.fixture(temp)
            self.assertEqual(buy_limit_evidence(out, root, '2026-04-07', '000001.SZ', 'weekly_reentry'),
                             (path, 9., 11.))

    def test_missing_requested_code_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            out, root, _, _ = self.fixture(temp)
            with self.assertRaisesRegex(AssertionError, 'evidence row'):
                buy_limit_evidence(out, root, '2026-04-07', '000002.SZ', 'weekly_reentry')

    def test_wrong_date_duplicate_nonfinite_and_invalid_range_fail(self):
        for change in ('date', 'duplicate', 'nonfinite', 'range'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                out, root, path, frame = self.fixture(temp)
                if change == 'date':
                    frame['trade_date'] = '20260403'
                elif change == 'duplicate':
                    frame = pd.concat([frame, frame], ignore_index=True)
                elif change == 'nonfinite':
                    frame['up_limit'] = float('nan')
                else:
                    frame['down_limit'] = 12.
                frame.to_pickle(path)
                with self.assertRaises(AssertionError):
                    buy_limit_evidence(out, root, '2026-04-07', '000001.SZ', 'weekly_reentry')

    def test_malformed_preferred_file_is_not_hidden_by_boundary_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            out, root, path, frame = self.fixture(temp)
            frame.to_pickle(root/'boundaries/stk_limit_20260407.pkl')
            frame[frame.ts_code.eq('000002.SZ')].to_pickle(path)
            with self.assertRaisesRegex(AssertionError, 'evidence row'):
                buy_limit_evidence(out, root, '2026-04-07', '000001.SZ', 'weekly_reentry')

    def test_monthly_route_still_prefers_boundary_source(self):
        with tempfile.TemporaryDirectory() as temp:
            out, root, _, frame = self.fixture(temp)
            path = root/'boundaries/stk_limit_20260407.pkl'
            frame.to_pickle(path)
            self.assertEqual(buy_limit_evidence(out, root, '2026-04-07', '000001.SZ')[0], path)

    def test_no_evidence_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(AssertionError, 'evidence file'):
                buy_limit_evidence(Path(temp), Path(temp), '2026-04-07', '000001.SZ', 'weekly_reentry')


if __name__ == '__main__':
    unittest.main()
