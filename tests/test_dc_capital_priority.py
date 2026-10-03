import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_dc_capital_priority import capital_priority, monthly_contributions, audit_account


class CapitalPriorityTests(unittest.TestCase):
    def fixture(self):
        signal, entry, end = pd.Timestamp('2026-03-31'), pd.Timestamp('2026-04-01'), pd.Timestamp('2026-04-30')
        codes = ['600001.SH', '600002.SH', '300001.SZ', '688001.SH', '600003.SH', '600004.SH', '600005.SH', '600006.SH']
        candidates = pd.DataFrame(dict(month=signal, signal_date=signal, ts_code=codes,
            eligible=[True, True, True, True, True, True, False, True],
            leadership_score=[.9, .9, 1., 1., .8, .7, 2., .6],
            affinity_theme_code=['BK001.DC'] * 8, member_theme_code=['BK002.DC'] * 8,
            label_return=[9.] * 8))
        allocations = pd.DataFrame([dict(date=entry, target=25000., cash=23000., retained=1)])
        trades = pd.DataFrame([dict(date=entry, code='600002.SH', side='buy', shares=100, price=10.),
                               dict(date=entry, code='600002.SH', side='buy', shares=100, price=10.),
                               dict(date=pd.Timestamp('2026-04-06'), code='600001.SH', side='buy', shares=100, price=11.)])
        account = pd.DataFrame([dict(date=end, profit=0.)])
        quotes = pd.DataFrame(dict(trade_date='20260401', ts_code=codes[:-1], open=[10., 10., 10., 10., 300., 11., 10.]))
        limits = pd.DataFrame(dict(trade_date='20260401', ts_code=codes, up_limit=[11., 11., 12., 12., 330., 11., 11., 11.],
                                  down_limit=[9., 9., 8., 8., 270., 9., 9., 9.]))
        return candidates, allocations, trades, account, {entry: quotes}, {entry: limits}

    def test_rank_code_tie_permissions_lot_upper_bound_and_actual_quantity(self):
        result = capital_priority(*self.fixture()).set_index('ts_code')
        self.assertEqual(result.loc['600001.SH', 'mainboard_executable_score_rank'], 1)
        self.assertEqual(result.loc['600002.SH', 'mainboard_executable_score_rank'], 2)
        self.assertEqual(result.loc['600002.SH', 'target_max_lots'], 25)
        self.assertEqual(result.loc['600002.SH', 'actual_entry_bought_shares'], 200)
        self.assertEqual(result.loc['600001.SH', 'actual_entry_bought_shares'], 0)
        self.assertEqual(result.loc['600001.SH', 'actual_month_bought_shares'], 100)
        self.assertFalse(result.loc['300001.SZ', 'mainboard'])
        self.assertFalse(result.loc['688001.SH', 'audited_executable_candidate'])
        self.assertEqual(result.loc['600003.SH', 'target_max_lots'], 0)
        self.assertFalse(result.loc['600004.SH', 'limit_open_executable'])
        self.assertFalse(result.loc['600006.SH', 'opening_quote_available'])
        self.assertTrue(pd.isna(result.loc['600006.SH', 'target_max_lots']))
        self.assertEqual(result.loc['600001.SH', 'affinity_theme_code'], 'BK001.DC')
        self.assertEqual(result.loc['600001.SH', 'member_theme_code'], 'BK002.DC')
        self.assertNotIn('label_return', result.columns)

    def test_future_profit_diagnostic_cannot_change_ranking_or_execution(self):
        fixture = self.fixture()
        end = fixture[3].date.iloc[0]
        first = capital_priority(*fixture, pd.DataFrame([dict(date=end, code='600001.SH', profit=-100.)]))
        second = capital_priority(*fixture, pd.DataFrame([dict(date=end, code='600001.SH', profit=100000.)]))
        pd.testing.assert_frame_equal(first.drop(columns='diagnostic_month_profit'), second.drop(columns='diagnostic_month_profit'))
        self.assertNotEqual(first.diagnostic_month_profit.iloc[0], second.diagnostic_month_profit.iloc[0])

    def test_strict_price_limit_tolerance_and_risk_warning_band(self):
        fixture = list(self.fixture())
        fixture[2] = fixture[2].iloc[0:0]
        quotes = fixture[4][pd.Timestamp('2026-04-01')].copy()
        quotes.loc[quotes.ts_code.eq('600001.SH'), 'open'] = 10.995
        fixture[4] = {pd.Timestamp('2026-04-01'): quotes}
        limits = fixture[5][pd.Timestamp('2026-04-01')].copy()
        limits.loc[limits.ts_code.eq('600002.SH'), ['up_limit', 'down_limit']] = [10.5, 9.5]
        fixture[5] = {pd.Timestamp('2026-04-01'): limits}
        result = capital_priority(*fixture).set_index('ts_code')
        self.assertFalse(result.loc['600001.SH', 'limit_open_executable'])
        self.assertTrue(result.loc['600002.SH', 'risk_warning_band_excluded'])
        self.assertFalse(result.loc['600002.SH', 'audited_executable_candidate'])

    def test_reject_duplicate_or_wrong_date_quote_evidence(self):
        fixture = list(self.fixture())
        day = pd.Timestamp('2026-04-01')
        quotes = fixture[4][day]
        fixture[4] = {day: pd.concat([quotes, quotes.iloc[:1]])}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            capital_priority(*fixture)
        fixture[4] = {day: quotes.assign(trade_date='20260402')}
        with self.assertRaisesRegex(ValueError, 'wrong-date'):
            capital_priority(*fixture)

    def test_reject_future_signal_duplicate_candidates_and_non_whole_lot_buys(self):
        fixture = list(self.fixture())
        original = fixture[0]
        fixture[0] = original.assign(signal_date=pd.Timestamp('2026-04-01'))
        with self.assertRaises(ValueError):
            capital_priority(*fixture)
        fixture[0] = pd.concat([original, original.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            capital_priority(*fixture)
        fixture[0] = original
        fixture[2] = fixture[2].copy()
        fixture[2].loc[0, 'shares'] = 50
        with self.assertRaisesRegex(ValueError, 'whole-lot'):
            capital_priority(*fixture)

    def test_actual_buy_must_match_dated_opening_price(self):
        fixture = list(self.fixture())
        fixture[2] = fixture[2].copy()
        fixture[2].loc[0, 'price'] = 10.1
        with self.assertRaises(AssertionError):
            capital_priority(*fixture)

    def test_duplicate_buy_orders_are_checked_against_aggregated_budget_upper_bound(self):
        fixture = list(self.fixture())
        fixture[1] = fixture[1].assign(target=1000.)
        with self.assertRaisesRegex(ValueError, 'Aggregated purchases'):
            capital_priority(*fixture)

    def test_contribution_includes_holdings_flows_and_cash_entitlement(self):
        account = pd.DataFrame([dict(date='2026-04-30', profit=200.), dict(date='2026-05-29', profit=100.)])
        trades = pd.DataFrame([dict(date='2026-04-01', code='600001.SH', side='buy', shares=100, price=10.),
                               dict(date='2026-05-20', code='600001.SH', side='sell', shares=100, price=12.8)])
        holdings = pd.DataFrame([dict(date='2026-04-30', code='600001.SH', value=1200.)])
        events = pd.DataFrame([dict(date='2026-05-10', code='600001.SH', cash_entitlement=20.)])
        result = monthly_contributions(account, trades, holdings, events)
        np.testing.assert_allclose(result.profit, [200., 100.])
        account.loc[1, 'profit'] = 101.
        with self.assertRaises(AssertionError):
            monthly_contributions(account, trades, holdings, events)

    def test_status_gate_prevents_reading_running_account_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'run_status_synthetic.json').write_text(json.dumps(dict(status='running', variant='synthetic',
                account_complete=False, months=9)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'status-completed'):
                audit_account(out, 'synthetic', out / 'no-root')

    def synthetic_account_files(self, out, root, final_end=None):
        boundaries = root / 'boundaries'
        boundaries.mkdir(parents=True)
        months = pd.period_range('2026-01', '2026-09', freq='M')
        candidates, allocations, trades, accounts = [], [], [], []
        for number, month in enumerate(months):
            endpoint = pd.Timestamp(final_end) if final_end is not None and number == len(months) - 1 else month.end_time.normalize()
            days = pd.date_range(month.start_time.normalize(), endpoint)
            sessions = days[days.weekday < 5]
            first, last = sessions[0], sessions[-1]
            signal = month.start_time.normalize() - pd.offsets.BDay(1)
            pd.DataFrame(dict(cal_date=days.strftime('%Y%m%d'), is_open=(days.weekday < 5).astype(int))).to_pickle(boundaries / f'calendar_{month}.pkl')
            pd.DataFrame([dict(ts_code='600001.SH', trade_date=f'{first:%Y%m%d}', open=10.)]).to_pickle(boundaries / f'daily_{first:%Y%m%d}.pkl')
            pd.DataFrame([dict(ts_code='600001.SH', trade_date=f'{first:%Y%m%d}', up_limit=11., down_limit=9.)]).to_pickle(boundaries / f'stk_limit_{first:%Y%m%d}.pkl')
            candidates.append(dict(month=(month - 1).to_timestamp('M'), signal_date=signal,
                ts_code='600001.SH', eligible=True, leadership_score=.8, affinity_theme_code='BK001.DC'))
            allocations.append(dict(date=first, target=25000., cash=24000., retained=0))
            trades.extend([dict(date=first, code='600001.SH', side='buy', shares=100, price=10.),
                           dict(date=last, code='600001.SH', side='sell', shares=100, price=11.)])
            accounts.append(dict(date=last, profit=100., equity=25100. + number * 100.))
        pd.DataFrame(candidates).to_pickle(out / 'candidate_audit_synthetic.pkl')
        for label, rows in [('allocations', allocations), ('trades', trades), ('account', accounts)]:
            pd.DataFrame(rows).to_csv(out / f'{label}_synthetic.csv', index=False)
        pd.DataFrame(columns=['date', 'code', 'value']).to_csv(out / 'holdings_synthetic.csv', index=False)
        pd.DataFrame(columns=['date', 'code', 'cash_entitlement']).to_csv(out / 'corporate_events_synthetic.csv', index=False)
        (out / 'run_status_synthetic.json').write_text(json.dumps(dict(status='completed', variant='synthetic',
            account_complete=True, months=9)), encoding='utf-8')

    def test_completed_synthetic_nine_month_account_manifest_and_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'synthetic-root'
            self.synthetic_account_files(out, root)
            result, proof = audit_account(out, 'synthetic', root)
            self.assertEqual(len(result), 9)
            self.assertEqual(proof['months'], 9)
            self.assertTrue(proof['attribution_reconciled'])
            self.assertTrue(result.actual_entry_bought.all())
            np.testing.assert_allclose(result.diagnostic_month_profit, 100.)
            self.assertEqual(len(proof['inputs']), 35)
            self.assertTrue(all(len(value) == 64 for value in proof['inputs'].values()))
            self.assertIn('not free cash', proof['budget_note'])

    def test_incomplete_calendar_cannot_relabel_later_day_as_month_open(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'synthetic-root'
            self.synthetic_account_files(out, root)
            path = root / 'boundaries' / 'calendar_2026-01.pkl'
            calendar = pd.read_pickle(path)
            calendar.iloc[1:].to_pickle(path)
            with self.assertRaisesRegex(ValueError, 'every day'):
                audit_account(out, 'synthetic', root)

    def test_partial_final_month_requires_no_future_calendar_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'synthetic-root'
            self.synthetic_account_files(out, root, final_end='2026-09-24')
            result, proof = audit_account(out, 'synthetic', root)
            self.assertEqual(proof['end'], '2026-09-24')
            self.assertTrue(proof['attribution_reconciled'])
            self.assertEqual(len(result), 9)
            path = root / 'boundaries' / 'calendar_2026-09.pkl'
            calendar = pd.read_pickle(path)
            calendar.iloc[:-1].to_pickle(path)
            with self.assertRaisesRegex(ValueError, 'account endpoint'):
                audit_account(out, 'synthetic', root)

    def test_actual_month_open_must_match_cached_calendar(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'synthetic-root'
            self.synthetic_account_files(out, root)
            path = out / 'allocations_synthetic.csv'
            allocations = pd.read_csv(path)
            allocations.loc[0, 'date'] = '2026-01-02'
            allocations.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'actual first trading session'):
                audit_account(out, 'synthetic', root)


if __name__ == '__main__':
    unittest.main()
