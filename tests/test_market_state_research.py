import itertools
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_market_states as research
from research_market_states import dp_path, online_states, exit_reason


class MarketStateTests(unittest.TestCase):
    def test_continuation_requires_all_three_observed_trends(self):
        self.assertTrue(research.continue_trend(pd.Series(dict(mom1=.1,mom3=.2,mom6=.3))))
        self.assertFalse(research.continue_trend(pd.Series(dict(mom1=-.1,mom3=.2,mom6=.3))))
        self.assertFalse(research.continue_trend(pd.Series(dict(mom1=.1,mom3=np.nan,mom6=.3))))
        self.assertFalse(research.continue_trend(None))

    def test_continuation_never_cancels_pending_exit_or_promotes_range(self):
        row=pd.Series(dict(mom1=.1,mom3=.2,mom6=.3))
        self.assertTrue(research.may_continue_position(row,'advance',False,True))
        self.assertFalse(research.may_continue_position(row,'advance',True,True))
        self.assertFalse(research.may_continue_position(row,'range',False,True))
        self.assertTrue(research.may_continue_position(row,'range',False,False))

    def test_whole_board_outage_cannot_hide_in_market_total(self):
        dates=pd.bdate_range('2025-12-01',periods=80)
        rows=[]
        for i,day in enumerate(dates):
            codes=[f'600{j:03d}' for j in range(450)]
            if i<20: codes += [f'688{j:03d}' for j in range(60)]
            rows.extend(dict(date=day,code=code) for code in codes)
        counts,expected,gaps=research.board_coverage_gaps(pd.DataFrame(rows))
        self.assertTrue(gaps.loc[dates[20]:,'star'].all())
        self.assertFalse(gaps.loc[:dates[19]].any().any())
        self.assertGreater(counts.sum(axis=1).iloc[-1]/expected.sum(axis=1).iloc[-1],.85)

    def test_board_repair_rejects_nonempty_but_truncated_response(self):
        dates=pd.bdate_range('2026-01-01',periods=2)
        codes=[f'600{j:03d}' for j in range(450)]+[f'688{j:03d}' for j in range(60)]
        raw=pd.DataFrame([dict(date=day,code=code,open=10.,high=11.,low=9.,close=10.,volume=100.,amount=1000.)
                          for day in dates for code in codes if day==dates[0] or not code.startswith('688')])
        complete=pd.DataFrame(dict(ts_code=[c+'.SH' for c in codes],trade_date=f'{dates[-1]:%Y%m%d}',
                                   open=10.,high=11.,low=9.,close=10.,vol=100.,amount=1000.))
        calls=[]
        class Client:
            def query(self,api,**kwargs):
                calls.append(kwargs['limit'])
                return complete.iloc[:450] if kwargs['limit']==6000 else complete
        with TemporaryDirectory() as directory,patch.object(research,'OUT',Path(directory)):
            fixed=research.repair_board_coverage(raw,Client())
            self.assertEqual(calls,[6000,8000])
            self.assertEqual(len(fixed),1020)
            self.assertFalse(research.board_coverage_gaps(fixed)[2].any().any())

    def test_failed_rebalance_keeps_order_and_stop_priority(self):
        day=pd.Timestamp('2026-06-01'); pending={}
        research.queue_deferred_rebalance(pending,'stock',day,200)
        self.assertEqual(pending['stock'],('rebalance_deferred',day))
        research.queue_deferred_rebalance(pending,'zero',day,0)
        self.assertNotIn('zero',pending)
        pending['risk']=('hard_stop',day-pd.Timedelta(days=3))
        research.queue_deferred_rebalance(pending,'risk',day,100)
        self.assertEqual(pending['risk'][0],'hard_stop')
        research.queue_deferred_rebalance(pending,'locked_bonus',day,5)
        self.assertIn('locked_bonus',pending)

    def soft_fixture(self):
        return pd.DataFrame(dict(month=[pd.Timestamp('2026-03-31')]*6,
            l2_code=['sector']*6, mom1=[.1]*6, mom3=[.8,.2,.25,.3,.35,.4],mom6=[.9]*6,
            phase=['retreat']+['range']*5,size_bucket=['middle']+['core']*5,
            chosen_style=['core']*6, leadership_score=[.5]*6,fast_score=[.9]*6,
            peer_score=[.8]*6,score=[.6]*6,sector_rank=[.4]*6))

    def test_soft_size_removes_only_size_gate(self):
        c=self.soft_fixture(); c.loc[0,'phase']='range'
        result=research.soft_candidates(c,'soft_size')
        self.assertTrue(result.loc[0,'eligible'])
        self.assertAlmostEqual(result.loc[0,'leadership_score'],.475)
        self.assertAlmostEqual(result.loc[1,'leadership_score'],.525)
        self.assertFalse(result.strong_override.any())

    def test_strong_stock_requires_both_gate_changes(self):
        c=self.soft_fixture()
        self.assertFalse(research.soft_candidates(c,'soft_size').loc[0,'eligible'])
        only=research.soft_candidates(c,'leader_override')
        self.assertEqual(only.loc[0,'phase'],'advance')
        self.assertFalse(only.loc[0,'eligible'])
        both=research.soft_candidates(c,'soft_leader')
        self.assertTrue(both.loc[0,'eligible'])
        self.assertEqual(both.loc[0,'phase'],'advance')
        self.assertEqual(c.loc[0,'phase'],'retreat')

    def test_strong_override_requires_peer_confirmation(self):
        c=self.soft_fixture(); c.loc[1:,'mom3']=-.2
        self.assertFalse(research.soft_candidates(c,'soft_leader').loc[0,'eligible'])

    def test_soft_rules_are_prefix_invariant(self):
        c=self.soft_fixture(); later=c.assign(month=pd.Timestamp('2026-04-30'),mom3=10.)
        one=research.soft_candidates(c,'soft_leader')
        all_rows=research.soft_candidates(pd.concat([c,later],ignore_index=True),'soft_leader')
        pd.testing.assert_frame_equal(one,all_rows.iloc[:len(c)])

    def test_rule_derived_limit_requires_historical_evidence(self):
        day=pd.Timestamp('2025-04-08')
        history=pd.DataFrame(dict(ts_code=['603166.SH'],name=['ordinary'],start_date=['20141127'],end_date=[None]))
        recent=[(d,12.4,100.) for d in pd.bdate_range('2025-03-31',periods=5)]
        quote=pd.Series(dict(open=11.16,pre_close=12.4))
        bquote=pd.Series(dict(pre_close=15.31))
        blimit=pd.Series(dict(down_limit=13.78,up_limit=16.84))
        args=(day,'603166.SH',quote,history,recent,pd.Timestamp('2025-04-01'),bquote,blimit)
        proof=research.normal_limit_evidence(*args)
        self.assertEqual(proof['down_limit'],11.16)
        self.assertTrue(proof['not_a_vendor_limit_quote'])
        with self.assertRaisesRegex(ValueError,'five'):
            research.normal_limit_evidence(day,'603166.SH',quote,history,recent[:4],args[5],bquote,blimit)
        with self.assertRaisesRegex(ValueError,'ordinary'):
            research.normal_limit_evidence(day,'603166.SH',quote,history.assign(name='ST ordinary'),recent,args[5],bquote,blimit)
        with self.assertRaisesRegex(ValueError,'anchor'):
            research.normal_limit_evidence(*args[:-1],pd.Series(dict(down_limit=14.54,up_limit=16.08)))

    def test_narrow_rules_preserve_broad_market_baseline(self):
        c=self.soft_fixture()
        market=pd.DataFrame(dict(breadth=[.6]),index=[pd.Timestamp('2026-03-31')])
        d=research.narrow_candidates(c,market,'narrow_leader')
        self.assertFalse(d.strong_override.any())
        pd.testing.assert_series_equal(d.leadership_score,c.leadership_score)
        pd.testing.assert_series_equal(d.phase,c.phase)
        self.assertFalse(d.loc[0,'eligible'])

    def test_narrow_heat_ablation_and_relative_exception(self):
        c=self.soft_fixture(); c.loc[1:,'mom3']=-.2
        market=pd.DataFrame(dict(breadth=[.4]),index=[pd.Timestamp('2026-03-31')])
        d=research.narrow_candidates(c,market,'narrow_heat')
        self.assertTrue(d.loc[0,'eligible'])  # Strong relative stock survives weak L2.
        c.loc[0,['mom1','mom3']]=[.5,2.]
        no_heat=research.narrow_candidates(c,market,'narrow_leader')
        heat=research.narrow_candidates(c,market,'narrow_heat')
        self.assertTrue(no_heat.loc[0,'eligible'])
        self.assertFalse(heat.loc[0,'eligible'])
        self.assertTrue(heat.loc[0,'heat_excluded'])

    def test_dp_matches_exhaustive_sequence_objective(self):
        x=np.array([[-1.],[-.6],[.2],[1.],[.8]])
        centers=np.array([[-1.],[1.]])
        penalty=.8
        path,_=dp_path(x,centers,penalty)
        def objective(p):
            return .5*np.sum((x-centers[np.array(p)])**2)+penalty*np.sum(np.diff(p)!=0)
        optimum=min(objective(p) for p in itertools.product(range(2),repeat=len(x)))
        self.assertAlmostEqual(objective(path),optimum)

    def test_online_costs_do_not_depend_on_future_rows(self):
        x=np.array([[-1.],[-.6],[.2],[1.],[.8]])
        centers=np.array([[-1.],[1.]])
        _,full=dp_path(x,centers,.8)
        for n in range(1,len(x)+1):
            _,prefix=dp_path(x[:n],centers,.8)
            np.testing.assert_allclose(prefix[-1],full[n-1])

    def test_walkforward_refit_does_not_change_earlier_states(self):
        rng=np.random.default_rng(42)
        index=pd.date_range('2015-01-31','2024-12-31',freq='ME')
        f=pd.DataFrame(rng.normal(size=(len(index),3)),index=index,columns=['mom1','mom3','vol'])
        small=online_states(f.loc[:'2023-06-30'])
        full=online_states(f)
        pd.testing.assert_frame_equal(small,full[full.month<=pd.Timestamp('2023-06-30')].reset_index(drop=True))
        self.assertTrue(full.training_end_before_signal.all())

    def test_hard_stop_triggers_at_close(self):
        self.assertEqual(exit_reason(92.,100.,103.,'advance'),'hard_stop')
        self.assertEqual(exit_reason(91.9,100.,103.,'advance'),'hard_stop')
        self.assertIsNone(exit_reason(93.,100.,103.,'advance'))

    def test_advance_does_not_cap_upside_at_thirty_percent(self):
        self.assertIsNone(exit_reason(140.,100.,140.,'advance'))
        self.assertEqual(exit_reason(140.,100.,140.,'range'),'range_take_profit')

    def test_trailing_profit_only_activates_after_twenty_percent(self):
        self.assertEqual(exit_reason(108.,100.,120.,'range'),'trailing_profit')
        self.assertIsNone(exit_reason(99.,100.,110.,'range'))
        self.assertEqual(exit_reason(110.,100.,130.,'advance'),'trailing_profit')
        self.assertIsNone(exit_reason(115.,100.,130.,'advance'))
        self.assertEqual(exit_reason(115.,100.,130.,'range'),'trailing_profit')

    def test_exit_uses_actual_gap_open_not_stop_threshold(self):
        class Client:
            def query(self,api,**kwargs):
                common=dict(ts_code=['000001.SZ'],trade_date=['20230104'])
                return pd.DataFrame(dict(**common,open=[80.],close=[82.])) if api=='daily' else pd.DataFrame(dict(**common,up_limit=[99.],down_limit=[81.]))
        with TemporaryDirectory() as directory, patch.object(research,'OUT',Path(directory)):
            price,down=research.execution_quote(Client(),pd.Timestamp('2023-01-04'),'000001.SZ',80.)
            self.assertEqual(price,80.)
            self.assertLessEqual(price,down+.005)  # Caller must defer a lower-limit exit.
            with self.assertRaisesRegex(ValueError,'mismatch'):
                research.execution_quote(Client(),pd.Timestamp('2023-01-04'),'000001.SZ',92.)

    def test_conservative_bound_only_accepts_clearly_non_limit_open(self):
        class NoNetwork:
            def query(self,*a,**kw): raise RuntimeError('No quote')
        with TemporaryDirectory() as directory, patch.object(research,'OUT',Path(directory)):
            opening,bound=research.execution_quote(NoNetwork(),pd.Timestamp('2023-01-04'),'000001.SZ',91.,94.)
            self.assertEqual(opening,91.)
            self.assertEqual(bound,89.3)
            with self.assertRaises(RuntimeError):
                research.execution_quote(NoNetwork(),pd.Timestamp('2023-01-05'),'000001.SZ',89.3,94.)


if __name__=='__main__': unittest.main()
