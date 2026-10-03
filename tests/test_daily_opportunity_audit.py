import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import research_daily_opportunity_execution as execution
spec = importlib.util.spec_from_file_location('daily_auditor',PROJECT/'scripts/audit_daily_opportunity_account.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)

EMPTY_ACTIONS = pd.DataFrame(columns=['ts_code','record_date','ex_date','pay_date','div_listdate','cash','stock','event'])


class IndependentDailyAuditTests(unittest.TestCase):
    def fixture(self, cash_action=False, stop=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)/'account'
        sessions = pd.bdate_range('2025-12-31',periods=12)
        daily = pd.DataFrame([dict(date=day,ts_code=code,open=10.,close=10.,volume=100.)
                              for day in sessions for code in ('600001.SH','000001.SZ','300001.SZ')])
        predictions = pd.DataFrame([dict(signal_date=day,ts_code=code,utility=utility)
                                    for day in sessions for code,utility in
                                    (('600001.SH',.1),('000001.SZ',.01),('300001.SZ',1.))])
        actions = EMPTY_ACTIONS.copy()
        if cash_action:
            actions = pd.DataFrame([dict(ts_code='600001.SH',record_date=sessions[2],ex_date=sessions[3],
                                        pay_date=sessions[4],div_listdate=sessions[5],cash=.1,stock=.1,event='event1')])
            daily.loc[daily.ts_code.eq('600001.SH') & daily.date.ge(sessions[3]),['open','close']] = 9.
        if stop:
            daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[2]),'close'] = 9.
            daily.loc[daily.ts_code.eq('600001.SH') & daily.date.eq(sessions[3]),['open','close']] = [8.,9.]
        def vendor(day,code):
            return pd.Series(dict(down_limit=8. if stop and day==sessions[3] and code=='600001.SH' else 1.,up_limit=100.))
        start,end = sessions[1],sessions[-1]
        execution.run_account(predictions,daily,sessions,out,start=str(start.date()),end=str(end.date()),
                              limit_provider=vendor,action_provider=lambda c: actions if c=='600001.SH' else EMPTY_ACTIONS.copy())
        labels = ['account','daily_nav','trades','holdings','corporate_events','allocations','stock_daily_contributions','execution_checks']
        outputs = {name:pd.read_csv(out/f'{name}.csv') for name in labels}
        keys = set((pd.Timestamp(d),c) for d,c in zip(outputs['execution_checks'].date,outputs['execution_checks'].code))
        limits = {key:(float(vendor(*key).down_limit),float(vendor(*key).up_limit)) for key in keys}
        policy = json.loads((out/'execution_protocol.json').read_text())['policy']
        return out,outputs,daily,predictions,sessions,actions,limits,policy,start,end

    def reconstructed(self,fixture,outputs=None,limits=None,predictions=None):
        out,original,daily,p,sessions,actions,official,policy,start,end = fixture
        return auditor.reconstruct(outputs or original,daily,predictions if predictions is not None else p,
                                   sessions,actions,limits if limits is not None else official,policy,start,end)

    def test_independent_cash_and_nav_reconstruction(self):
        result = self.reconstructed(self.fixture())
        self.assertTrue(result[0]['passed'])
        self.assertEqual(result[0]['ending_equity'],25000.)

    def test_independent_cash_bonus_locked_shares_and_exright_anchors(self):
        result = self.reconstructed(self.fixture(cash_action=True))
        self.assertEqual(result[0]['ending_equity'],25000.)
        self.assertEqual(result[2].profit.sum(),0.)

    def test_pending_lowerlimit_deferred_sale_independently_recomputed(self):
        result = self.reconstructed(self.fixture(stop=True))
        self.assertTrue(result[0]['pending_exits_independently_verified'])

    def test_tampered_cash_equity_and_monthly_profit_caught(self):
        fixture = self.fixture()
        for table,column in [('daily_nav','cash'),('daily_nav','equity'),('account','profit'),('account','budget_return')]:
            outputs = copy.deepcopy(fixture[1])
            outputs[table].loc[0,column] += 100
            with self.assertRaises(AssertionError):
                self.reconstructed(fixture,outputs=outputs)

    def test_same_day_signal_buy_caught(self):
        fixture = self.fixture()
        outputs = copy.deepcopy(fixture[1])
        outputs['trades'].loc[0,'signal_date'] = outputs['trades'].loc[0,'date']
        with self.assertRaisesRegex(AssertionError,'previous completed'):
            self.reconstructed(fixture,outputs=outputs)

    def test_unauthorized_board_and_odd_lot_caught(self):
        fixture = self.fixture()
        for column,value in [('code','300001.SZ'),('shares',2501)]:
            outputs = copy.deepcopy(fixture[1])
            outputs['trades'].loc[0,column] = value
            with self.assertRaises(AssertionError):
                self.reconstructed(fixture,outputs=outputs)

    def test_new_investment_cap_and_hidden_financing_caught(self):
        fixture = self.fixture()
        outputs = copy.deepcopy(fixture[1])
        outputs['trades'].loc[0,'shares'] = 2600
        with self.assertRaisesRegex(AssertionError,'25000 cap'):
            self.reconstructed(fixture,outputs=outputs)

    def test_limit_locked_buy_and_wrong_open_caught(self):
        fixture = self.fixture()
        limits = dict(fixture[6])
        limits[(fixture[8],'600001.SH')] = (9.,10.)
        with self.assertRaisesRegex(AssertionError,'Limit-locked'):
            self.reconstructed(fixture,limits=limits)
        outputs = copy.deepcopy(fixture[1])
        outputs['trades'].loc[0,'price'] = 9.99
        with self.assertRaisesRegex(AssertionError,'actual next opening'):
            self.reconstructed(fixture,outputs=outputs)

    def test_exported_attribution_and_locked_shares_caught(self):
        fixture = self.fixture(cash_action=True)
        for table,column in [('stock_daily_contributions','profit'),('holdings','locked_shares'),('holdings','entry'),('corporate_events','cash_entitlement')]:
            outputs = copy.deepcopy(fixture[1])
            outputs[table].loc[0,column] += 1
            with self.assertRaises(AssertionError):
                self.reconstructed(fixture,outputs=outputs)

    def test_executable_pending_sale_cannot_be_omitted(self):
        fixture = self.fixture(stop=True)
        outputs = copy.deepcopy(fixture[1])
        outputs['trades'] = outputs['trades'][~outputs['trades'].side.eq('sell')]
        with self.assertRaisesRegex(AssertionError,'sale was omitted'):
            self.reconstructed(fixture,outputs=outputs)

    def test_negative_prediction_cannot_be_bought(self):
        fixture = self.fixture()
        predictions = fixture[3].copy()
        predictions.loc[predictions.ts_code.eq('600001.SH'),'utility'] = -.1
        with self.assertRaisesRegex(AssertionError,'Nonpositive'):
            self.reconstructed(fixture,predictions=predictions)

    def test_wrapper_reads_sources_and_does_not_write_original_account(self):
        fixture = self.fixture()
        out,outputs,daily,p,sessions,actions,limits,policy,start,end = fixture
        before = {path.name:auditor.sha(path) for path in out.glob('*') if path.is_file()}
        checks = auditor.audit_account(out,daily,p,sessions,limit_frames=limits,actions=actions)
        after = {path.name:auditor.sha(path) for path in out.glob('*') if path.is_file()}
        self.assertTrue(checks['passed'])
        self.assertEqual(before,after)
        self.assertTrue((out/'independent_audit/independent_checks.json').exists())

    def test_original_run_complete_flag_not_enough_to_pass(self):
        fixture = self.fixture()
        out,outputs,daily,p,sessions,actions,limits,policy,start,end = fixture
        incorrect = outputs['daily_nav'].copy()
        incorrect.loc[0,'cash'] += 10
        incorrect.to_csv(out/'daily_nav.csv',index=False)
        with self.assertRaises(AssertionError):
            auditor.audit_account(out,daily,p,sessions,limit_frames=limits,actions=actions)

    def test_empty_cash_account_with_no_trades_also_reconciles(self):
        fixture = self.fixture()
        out,outputs,daily,p,sessions,actions,limits,policy,start,end = fixture
        flat_out = out.parent/'flat'
        negative = p.assign(utility=-.1)
        execution.run_account(negative,daily,sessions,flat_out,start=str(start.date()),end=str(end.date()),
                              limit_provider=lambda d,c: pd.Series(dict(down_limit=1.,up_limit=100.)),
                              action_provider=lambda c: EMPTY_ACTIONS.copy())
        checks = auditor.audit_account(flat_out,daily,negative,sessions,limit_frames={},actions=EMPTY_ACTIONS.copy())
        self.assertEqual(checks['trades'],0)
        self.assertEqual(checks['ending_equity'],25000.)

    def test_absent_distribution_component_distinct_from_malformed_number(self):
        for value in (None,float('nan'),'','  ',0):
            self.assertEqual(auditor.distribution_quantity(value),0.)
        self.assertEqual(auditor.distribution_quantity('0.25'),.25)
        with self.assertRaises(ValueError):
            auditor.distribution_quantity('bad')

    def test_official_limits_resolved_by_schema_not_cache_filename(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        out=Path(temporary.name);path=out/'600001.SH_20260105.pkl'
        day=pd.Timestamp('2026-01-05');code='600001.SH'
        pd.DataFrame([dict(ts_code=code,trade_date='20260105',up_limit=11.,down_limit=9.)]).to_pickle(path)
        official,sources=auditor.load_exact_limits(out,dict(source_sha256={str(path):auditor.sha(path)}),{(day,code)})
        self.assertEqual(official[(day,code)],(9.,11.))
        self.assertIn(str(path.resolve()),sources)


if __name__=='__main__':
    unittest.main()
