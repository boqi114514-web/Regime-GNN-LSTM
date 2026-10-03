import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import research_daily_opportunity_execution as execution


EMPTY_ACTIONS = pd.DataFrame(columns=['ts_code','record_date','ex_date','pay_date','div_listdate','cash','stock','event'])


class AllocationTests(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame(dict(ts_code=['600001.SH','000001.SZ','300001.SZ'],
                                 open=[110., 10., 1.], utility=[.20,.01,10.]))

    def test_utility_dominates_cash_fill_and_unauthorized_boards(self):
        result = execution.allocate_expected_utility(self.candidates(), 25000, 1)
        self.assertEqual(result.ts_code.tolist(), ['600001.SH'])
        self.assertEqual(result.shares.tolist(), [200])
        self.assertEqual(result.amount.sum(), 22000.)

    def test_unused_lot_residual_can_buy_other_positive_opportunity(self):
        result = execution.allocate_expected_utility(self.candidates(), 25000, 3)
        self.assertEqual(set(result.ts_code), {'600001.SH','000001.SZ'})
        self.assertTrue(result.amount.sum() <= 25000)
        self.assertTrue((result.shares % 100 == 0).all())
        self.assertEqual(result.set_index('ts_code').shares['600001.SH'], 200)

    def test_negative_zero_and_unaffordable_leave_cash(self):
        candidates = self.candidates().assign(utility=[-.1,0.,10.])
        self.assertTrue(execution.allocate_expected_utility(candidates, 25000).empty)
        self.assertTrue(execution.allocate_expected_utility(self.candidates().iloc[:1], 10000).empty)

    def test_fractional_budget_cannot_overspend_one_fen(self):
        c = pd.DataFrame(dict(ts_code=['600001.SH'], open=[10.01], utility=[.1]))
        self.assertTrue(execution.allocate_expected_utility(c,1000.9999).empty)
        self.assertEqual(execution.allocate_expected_utility(c,1001).shares.iloc[0],100)

    def test_missing_fields_duplicates_or_adjusted_price_fail(self):
        c = self.candidates()
        for wrong in (c.drop(columns='utility'), pd.concat([c,c]), c.assign(open=10.012)):
            with self.assertRaises(ValueError):
                execution.allocate_expected_utility(wrong,25000)


class TimingTests(unittest.TestCase):
    def test_next_open_is_next_actual_exchange_session(self):
        sessions = pd.to_datetime(['2026-02-13','2026-02-24','2026-02-25'])
        p = pd.DataFrame(dict(signal_date=sessions, ts_code=['600001.SH']*3))
        actual = execution.next_open_schedule(p,sessions)
        self.assertEqual(actual.execution_date.iloc[0],sessions[1])
        self.assertTrue(pd.isna(actual.execution_date.iloc[2]))

    def test_non_session_prediction_or_duplicate_fails(self):
        dates = pd.date_range('2026-01-01',periods=3)
        for p in (pd.DataFrame(dict(signal_date=['2026-01-04'],ts_code=['600001.SH'])),
                  pd.DataFrame(dict(signal_date=[dates[0]]*2,ts_code=['600001.SH']*2))):
            with self.assertRaises(ValueError):
                execution.next_open_schedule(p,dates)

    def test_minimum_holding_does_not_block_hard_stop(self):
        anchor = dict(entry=100,peak=105,entry_index=0)
        policy = execution.ExecutionPolicy()
        self.assertEqual(execution.exit_reason(anchor,91,.1,None,1,policy),'hard_stop')
        self.assertIsNone(execution.exit_reason(anchor,100,-.1,None,1,policy))
        self.assertEqual(execution.exit_reason(anchor,100,-.1,None,5,policy),'nonpositive_expected_utility')

    def test_competitor_fixed_margin_and_close_only_trail(self):
        anchor = dict(entry=100,peak=150,entry_index=0)
        policy = execution.ExecutionPolicy()
        self.assertEqual(execution.exit_reason(anchor,127,.1,None,1,policy),'trailing_stop')
        self.assertIsNone(execution.exit_reason(anchor,140,.1,.105,5,policy))
        self.assertEqual(execution.exit_reason(anchor,140,.1,.111,5,policy),'better_predicted_opportunity')

    def test_official_limit_missing_or_conflicting_rejected(self):
        day = pd.Timestamp('2026-01-02')
        raw = pd.DataFrame(dict(ts_code=['600001.SH'],trade_date=['20260102'],up_limit=[11],down_limit=[9]))
        self.assertEqual(execution.OfficialLimits.validate(raw,day,'600001.SH').up_limit,11)
        with self.assertRaises(ValueError):
            execution.OfficialLimits.validate(pd.concat([raw,raw]),day,'600001.SH')
        with self.assertRaises(ValueError):
            execution.OfficialLimits.validate(raw,day,'000001.SZ')

    def test_daily_limit_batch_single_query_reused_for_two_stocks_and_offline(self):
        raw = pd.DataFrame(dict(ts_code=['600001.SH','000001.SZ'],trade_date=['20260102']*2,up_limit=[11.,12.],down_limit=[9.,8.]))
        calls = []
        class Pro:
            def query(self,api,**params):
                calls.append(params)
                return raw.copy()
        with tempfile.TemporaryDirectory() as task:
            provider = execution.OfficialLimits(task,Pro())
            provider(pd.Timestamp('2026-01-02'),'600001.SH')
            provider(pd.Timestamp('2026-01-02'),'000001.SZ')
            self.assertEqual(len(calls),1)
            batch = Path(task)/'daily_limits/stk_limit_20260102.pkl'
            self.assertEqual(provider.sources[str(batch.resolve())],execution._sha(batch))
            replay = execution.OfficialLimits(task,offline=True)
            self.assertEqual(replay(pd.Timestamp('2026-01-02'),'000001.SZ').up_limit,12.)

    def test_partial_batch_missing_candidate_uses_strict_single_fallback(self):
        raw = pd.DataFrame(dict(ts_code=['600001.SH'],trade_date=['20260102'],up_limit=[11.],down_limit=[9.]))
        calls = []
        class Pro:
            def query(self,api,**params):
                calls.append(params)
                return raw.assign(ts_code=params['ts_code']) if 'ts_code' in params else raw.copy()
        with tempfile.TemporaryDirectory() as task:
            provider = execution.OfficialLimits(task,Pro())
            provider(pd.Timestamp('2026-01-02'),'600001.SH')
            provider(pd.Timestamp('2026-01-02'),'000001.SZ')
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[1]['ts_code'],'000001.SZ')

    def test_daily_batch_duplicate_wrong_date_nonfinite_rejected(self):
        raw = pd.DataFrame(dict(ts_code=['600001.SH'],trade_date=['20260102'],up_limit=[11.],down_limit=[9.]))
        day = pd.Timestamp('2026-01-02')
        for incorrect in (pd.concat([raw,raw]),raw.assign(trade_date='20260105'),raw.assign(up_limit=np.nan)):
            with self.assertRaises(ValueError):
                execution.OfficialLimits.validate_batch(incorrect,day)

    def test_batch_preserves_other_ipo_zero_band_but_rejects_requested_ipo(self):
        raw = pd.DataFrame(dict(ts_code=['600001.SH','001001.SZ'],trade_date=['20260102']*2,up_limit=[11.,0.],down_limit=[9.,0.]))
        class Pro:
            def query(self,api,**params):
                return raw.copy()
        with tempfile.TemporaryDirectory() as task:
            provider = execution.OfficialLimits(task,Pro())
            self.assertEqual(provider(pd.Timestamp('2026-01-02'),'600001.SH').up_limit,11.)
            with self.assertRaises(ValueError):
                provider(pd.Timestamp('2026-01-02'),'001001.SZ')


class DailyAccountTests(unittest.TestCase):
    def fixture(self, periods=9):
        sessions = pd.bdate_range('2025-12-31',periods=periods)
        daily = pd.DataFrame([dict(date=day,ts_code=code,open=10.,close=10.,high=10.,low=10.,volume=100.)
                              for day in sessions for code in ('600001.SH','000001.SZ','300001.SZ')])
        predictions = pd.DataFrame([dict(signal_date=day,ts_code=code,utility=utility)
                                    for day in sessions for code,utility in
                                    (('600001.SH',.10),('000001.SZ',.01),('300001.SZ',1.))])
        limits = lambda day,code: pd.Series(dict(down_limit=1.,up_limit=100.))
        return sessions,daily,predictions,limits

    def run_fixture(self, daily, predictions, sessions, limits, policy=None, actions=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        execution.run_account(predictions,daily,sessions,out,start=str(sessions[1].date()),end=str(sessions[-1].date()),
                              limit_provider=limits, action_provider=actions or (lambda code: EMPTY_ACTIONS.copy()),policy=policy)
        return out

    def test_full_daily_account_nextopen_rawwholelots_no_topups(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.gt(sessions[1]), 'close'] = 11.
        out = self.run_fixture(daily,predictions,sessions,limits)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date','signal_date'])
        self.assertEqual(len(trades),1)
        self.assertEqual(trades.code.iloc[0],'600001.SH')
        self.assertEqual(trades.shares.iloc[0],2500)
        self.assertEqual(trades.date.iloc[0],sessions[1])
        self.assertEqual(trades.signal_date.iloc[0],sessions[0])
        nav = pd.read_csv(out/'daily_nav.csv')
        self.assertEqual(len(nav),len(sessions)-1)
        self.assertEqual(nav.equity.iloc[-1],27500.)
        status = json.loads((out/'daily_run_status.json').read_text())
        self.assertTrue(status['account_complete'])

    def test_buy_does_not_read_same_day_high_low_close_or_volume(self):
        sessions,daily,predictions,limits = self.fixture()
        out1 = self.run_fixture(daily,predictions,sessions,limits)
        changed = daily.copy()
        # Entry remains identical even a synthetic zero realized volume changes.
        chosen = changed.date.eq(sessions[1]) & changed.ts_code.eq('600001.SH')
        changed.loc[chosen,['high','low','close','volume']] = [100,1,12,0]
        out2 = self.run_fixture(changed,predictions,sessions,limits)
        pd.testing.assert_frame_equal(pd.read_csv(out1/'trades.csv').iloc[:1],pd.read_csv(out2/'trades.csv').iloc[:1])

    def test_stop_nextopen_lowerlimit_deferred_and_cooldown(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.date.eq(sessions[2]) & daily.ts_code.eq('600001.SH'),'close'] = 9.
        daily.loc[daily.date.eq(sessions[3]) & daily.ts_code.eq('600001.SH'),['open','close']] = [8.,9.]
        def exact(day,code):
            return pd.Series(dict(down_limit=8. if day==sessions[3] and code=='600001.SH' else 1.,up_limit=100.))
        out = self.run_fixture(daily,predictions,sessions,exact)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date','signal_date'])
        sale = trades[trades.side.eq('sell')].iloc[0]
        self.assertEqual(sale.date,sessions[4])
        self.assertEqual(sale.signal_date,sessions[2])
        self.assertEqual(sale.reason,'hard_stop')
        rebuys = trades[trades.side.eq('buy') & trades.code.eq('600001.SH')]
        if len(rebuys)>1:
            self.assertGreaterEqual(rebuys.date.iloc[1],sessions[6])

    def test_fixed_minimum_holding_and_model_switch(self):
        sessions,daily,predictions,limits = self.fixture(11)
        predictions.loc[predictions.ts_code.eq('000001.SZ') & predictions.signal_date.ge(sessions[2]),'utility'] = .2
        out = self.run_fixture(daily,predictions,sessions,limits)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date','signal_date'])
        sales = trades[trades.side.eq('sell')]
        self.assertEqual(len(sales),1)
        self.assertEqual(sales.date.iloc[0],sessions[7])
        self.assertEqual(sales.signal_date.iloc[0],sessions[6])
        self.assertEqual(sales.reason.iloc[0],'better_predicted_opportunity')
        self.assertEqual(trades[trades.side.eq('buy')].code.tolist(),['600001.SH','000001.SZ'])

    def test_month_boundary_does_not_force_liquidation(self):
        sessions,daily,predictions,limits = self.fixture(35)
        out = self.run_fixture(daily,predictions,sessions,limits)
        self.assertEqual(len(pd.read_csv(out/'trades.csv')),1)
        self.assertEqual(len(pd.read_csv(out/'account.csv')),2)

    def test_no_negative_utility_purchase_and_eligible_is_respected(self):
        sessions,daily,predictions,limits = self.fixture()
        predictions['eligible'] = False
        predictions.loc[predictions.ts_code.eq('000001.SZ'),'eligible'] = True
        predictions.loc[predictions.ts_code.eq('000001.SZ'),'utility'] = -.01
        out = self.run_fixture(daily,predictions,sessions,limits)
        self.assertTrue(pd.read_csv(out/'trades.csv').empty)
        self.assertEqual(pd.read_csv(out/'daily_nav.csv').equity.iloc[-1],25000.)

    def test_unaffordable_competitor_does_not_sell_winner(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('000001.SZ'),['open','close']] = 500.
        predictions.loc[predictions.ts_code.eq('000001.SZ'),'utility'] = .2
        out = self.run_fixture(daily,predictions,sessions,limits)
        trades = pd.read_csv(out/'trades.csv')
        self.assertEqual(len(trades),1)
        self.assertEqual(trades.code.iloc[0],'600001.SH')

    def test_dividend_bonus_locked_shares_and_attribution_reconcile(self):
        sessions,daily,predictions,limits = self.fixture()
        code = '600001.SH'
        action = pd.DataFrame([dict(ts_code=code,record_date=sessions[2],ex_date=sessions[3],
                                    pay_date=sessions[4],div_listdate=sessions[5],cash=.1,stock=.1,event='event1')])
        # Ex-right price: (10-.1)/1.1 = 9, no spurious 10% loss/stop.
        daily.loc[daily.ts_code.eq(code) & daily.date.ge(sessions[3]),['open','close']] = 9.
        out = self.run_fixture(daily,predictions,sessions,limits,actions=lambda c: action if c==code else EMPTY_ACTIONS.copy())
        nav = pd.read_csv(out/'daily_nav.csv')
        self.assertTrue(np.allclose(nav.equity,25000))
        holds = pd.read_csv(out/'holdings.csv',parse_dates=['date'])
        self.assertEqual(holds[holds.date.eq(sessions[3])].locked_shares.iloc[0],250)
        self.assertEqual(holds[holds.date.eq(sessions[5])].locked_shares.iloc[0],0)
        self.assertEqual(len(pd.read_csv(out/'trades.csv')),1)
        pnl = pd.read_csv(out/'stock_daily_contributions.csv')
        self.assertAlmostEqual(pnl.profit.sum(),0.)

    def test_missing_limits_does_not_publish_complete_account(self):
        sessions,daily,predictions,limits = self.fixture()
        with tempfile.TemporaryDirectory() as task:
            def absent(day,code):
                raise RuntimeError('Missing official observation')
            with self.assertRaises(RuntimeError):
                execution.run_account(predictions,daily,sessions,task,start=str(sessions[1].date()),end=str(sessions[-1].date()),
                                      limit_provider=absent,action_provider=lambda c: EMPTY_ACTIONS.copy())
            self.assertFalse(json.loads((Path(task)/'daily_run_status.json').read_text())['account_complete'])

    def test_missing_signal_never_forward_fills_future_prediction(self):
        sessions,daily,predictions,limits = self.fixture()
        sparse = predictions[predictions.signal_date.eq(sessions[3])]
        out = self.run_fixture(daily,sparse,sessions,limits)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date','signal_date'])
        self.assertEqual(trades.date.tolist(),[sessions[4]])
        self.assertEqual(trades.signal_date.tolist(),[sessions[3]])
        self.assertEqual(pd.read_csv(out/'daily_nav.csv').equity.iloc[0],25000.)

    def test_stop_on_entry_close_cannot_sell_until_next_session(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[1]),'close'] = 9.
        out = self.run_fixture(daily,predictions,sessions,limits)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date','signal_date'])
        sale = trades[trades.side.eq('sell')].iloc[0]
        self.assertEqual(sale.signal_date,sessions[1])
        self.assertEqual(sale.date,sessions[2])

    def test_stop_cooldown_is_stock_specific_not_global_account_freeze(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[2]),'close'] = 9.
        predictions.loc[predictions.ts_code.eq('000001.SZ') & predictions.signal_date.ge(sessions[2]),'utility'] = .2
        out = self.run_fixture(daily,predictions,sessions,limits)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date'])
        sale = trades[trades.side.eq('sell')].iloc[0]
        replacement = trades[trades.side.eq('buy') & trades.code.eq('000001.SZ')].iloc[0]
        self.assertEqual(sale.date,sessions[3])
        self.assertEqual(replacement.date,sessions[3])

    def test_exact_st_five_percent_band_no_unverified_status_assumption(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[1]),'open'] = 10.50
        def five_percent(day,code):
            return pd.Series(dict(down_limit=9.50,up_limit=10.50))
        out = self.run_fixture(daily,predictions,sessions,five_percent)
        trades = pd.read_csv(out/'trades.csv',parse_dates=['date'])
        first_main = trades[trades.code.eq('600001.SH')]
        self.assertTrue(first_main.empty or first_main.date.iloc[0] > sessions[1])

    def test_risk_adjusted_mu_fallback_not_momentum_or_percentile(self):
        sessions,daily,predictions,limits = self.fixture()
        predictions = predictions.drop(columns='utility')
        predictions['mu20'] = .10
        predictions['risk20'] = .50
        out = self.run_fixture(daily,predictions,sessions,limits)
        self.assertTrue(pd.read_csv(out/'trades.csv').empty)

    def test_appreciated_holding_not_trimmed_and_no_new_cap_overrun(self):
        sessions,daily,predictions,limits = self.fixture()
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.ge(sessions[2]),['open','close']] = 20.
        out = self.run_fixture(daily,predictions,sessions,limits)
        self.assertEqual(len(pd.read_csv(out/'trades.csv')),1)
        self.assertEqual(pd.read_csv(out/'daily_nav.csv').equity.iloc[-1],50000.)
        self.assertEqual(pd.read_csv(out/'holdings.csv').shares.iloc[-1],2500)

    def test_optional_factor_change_without_verified_action_fails(self):
        sessions,daily,predictions,limits = self.fixture()
        daily['adj_factor'] = 1.
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.ge(sessions[3]),'adj_factor'] = 1.2
        with self.assertRaisesRegex(ValueError,'Unexplained factor change'):
            self.run_fixture(daily,predictions,sessions,limits)

    def test_completed_policy_cannot_change_silently_in_same_directory(self):
        sessions,daily,predictions,limits = self.fixture()
        out = self.run_fixture(daily,predictions,sessions,limits)
        with self.assertRaisesRegex(ValueError,'silently change'):
            execution.run_account(predictions,daily,sessions,out,start=str(sessions[1].date()),end=str(sessions[-1].date()),
                                  limit_provider=limits,action_provider=lambda c: EMPTY_ACTIONS.copy(),
                                  policy=execution.ExecutionPolicy(switch_margin=.02))

    def test_negative_adverse_excursion_and_invalid_probabilities_fail(self):
        sessions,daily,predictions,limits = self.fixture()
        for incorrect in (predictions.assign(risk20=-.1),predictions.assign(prob5=1.1),predictions.assign(mu5=np.nan)):
            with self.assertRaises(ValueError):
                self.run_fixture(daily,incorrect,sessions,limits)

    def test_factor_guard_precedes_pending_sale_on_unexplained_gap(self):
        sessions,daily,predictions,limits = self.fixture()
        daily['adj_factor'] = 1.
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[2]),'close'] = 9.
        daily.loc[daily.ts_code.eq('600001.SH') & daily.date.ge(sessions[3]),'adj_factor'] = 1.2
        with self.assertRaisesRegex(ValueError,'Unexplained factor change'):
            self.run_fixture(daily,predictions,sessions,limits)


if __name__ == '__main__':
    unittest.main()
