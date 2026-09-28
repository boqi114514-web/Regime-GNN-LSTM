import sys
import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from small_account_backtest import Ledger, normalize_actions, buy_candidates, held_marks, check_unexplained_actions, load_actions


class AccountTests(unittest.TestCase):
    def actions(self):
        return pd.DataFrame([dict(ts_code='x', event='x:20230104', record_date=pd.Timestamp('2023-01-03'),
                                 ex_date=pd.Timestamp('2023-01-04'), pay_date=pd.Timestamp('2023-01-06'),
                                 div_listdate=pd.Timestamp('2023-01-05'), cash=.2, stock=.1)])

    def test_dividend_is_receivable_before_pay_not_spendable_cash(self):
        a, ledger = self.actions(), Ledger()
        ledger.cash=1000.
        ledger.shares={'x':100}
        ledger.close_record(pd.Timestamp('2023-01-03'), a)
        ledger.morning(pd.Timestamp('2023-01-04'), a)
        self.assertEqual(ledger.cash,1000.)
        self.assertEqual(ledger.shares['x'],110)
        self.assertEqual(ledger.blocked_shares('x'),10)
        self.assertEqual(ledger.nav({'x':10.}),2120.)
        ledger.morning(pd.Timestamp('2023-01-05'),a)
        self.assertEqual(ledger.blocked_shares('x'),0)
        ledger.morning(pd.Timestamp('2023-01-06'),a)
        self.assertEqual(ledger.cash,1020.)
        self.assertFalse(ledger.receivable)

    def test_selling_after_record_keeps_cash_entitlement(self):
        a, ledger = self.actions().assign(stock=0.), Ledger()
        ledger.shares={'x':100}
        ledger.close_record(pd.Timestamp('2023-01-03'),a)
        ledger.shares['x']=0
        ledger.morning(pd.Timestamp('2023-01-04'),a)
        self.assertEqual(sum(v[0] for v in ledger.receivable.values()),20.)

    def test_unheld_event_does_not_receive_dividend(self):
        ledger=Ledger()
        ledger.morning(pd.Timestamp('2023-01-04'),self.actions())
        self.assertFalse(ledger.receivable)
        self.assertFalse(ledger.shares)

    def test_missing_valuation_raises(self):
        ledger=Ledger();ledger.shares['x']=100
        with self.assertRaisesRegex(ValueError,'Missing held-stock'):
            ledger.nav({})

    def test_duplicate_implemented_events_not_double_counted(self):
        raw=pd.DataFrame([dict(ts_code='x',end_date='20221231',div_proc='实施',record_date='20230103',ex_date='20230104',
             pay_date='20230106',div_listdate=None,cash_div_tax=.2,stk_div=0.)]*2)
        self.assertEqual(len(normalize_actions(raw,'x')),1)
        raw.loc[1,'cash_div_tax']=.3
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            normalize_actions(raw,'x')
        raw.loc[1,'end_date']='20220930'
        self.assertEqual(len(normalize_actions(raw,'x')),2)

    def test_execution_uses_new_open_not_signal_open_or_future_close(self):
        c=pd.DataFrame(dict(ts_code=['600000.SH'],stock_code=['600000'],ind_code=['I'],
                            rank_in_ind=[1],open=[1.],amount=[30000.]))
        p=pd.DataFrame(dict(ts_code=['I'],selection_score=[1.]))
        q=pd.DataFrame(dict(open=[10.],close=[999.]),index=['600000.SH'])
        limits=pd.DataFrame(dict(up_limit=[11.],down_limit=[9.]),index=q.index)
        selected=buy_candidates(c,p,q,limits)
        self.assertEqual(selected.reference_price.iloc[0],10.)
        limits['up_limit']=10.
        self.assertTrue(buy_candidates(c,p,q,limits).empty)

    def test_suspension_is_marked_but_not_added_to_tradable_quotes(self):
        ledger=Ledger(); ledger.shares={'600000.SH':100}
        q=pd.DataFrame(columns=['open'])
        daily=pd.DataFrame(dict(ts_code=['600000.SH'],trade_date=['20230428'],close=[10.]))
        sus=pd.DataFrame(dict(ts_code=['600000.SH'],trade_date=['20230504'],suspend_type=['S']))
        actions=pd.DataFrame(columns=['ts_code','ex_date','cash','stock'])
        audit=[]
        with tempfile.TemporaryDirectory() as temp, patch('small_account_backtest.ROOT',Path(temp)), \
             patch('small_account_backtest.fetch_pages',return_value=daily), \
             patch('small_account_backtest.fetch_variants',return_value=sus):
            prices=held_marks(ledger,q,pd.Timestamp('2023-05-04'),'open',actions,Mock(),audit)
        self.assertEqual(prices['600000.SH'],10.)
        self.assertTrue(q.empty)
        self.assertEqual(len(audit),1)

    def test_factor_rounding_is_not_treated_as_a_dividend(self):
        a=pd.DataFrame(columns=['ts_code','ex_date'])
        check_unexplained_actions(['x'],{'x':5.361},{'x':5.3613},a,pd.Timestamp('2023-01-01'),pd.Timestamp('2023-01-31'))
        with self.assertRaisesRegex(ValueError,'Unexplained'):
            check_unexplained_actions(['x'],{'x':5.361},{'x':5.5},a,pd.Timestamp('2023-01-01'),pd.Timestamp('2023-01-31'))

    def test_noncanonical_per_ten_response_is_rejected(self):
        d=pd.DataFrame([dict(ts_code='x',end_date='',div_proc='实施',record_date='20230103',ex_date='20230104',
             pay_date='',div_listdate='',cash_div_tax=2.,stk_div=0.,stk_co_rate=3.,stk_bo_rate=0.)])
        with self.assertRaisesRegex(ValueError,'Noncanonical'):
            normalize_actions(d,'x')
        d['end_date']='20221231'
        with self.assertRaisesRegex(ValueError,'inconsistent stock'):
            normalize_actions(d,'x')

    def test_report_period_repair_preserves_raw_and_does_not_guess_scaling(self):
        bad=pd.DataFrame([dict(ts_code='x',end_date='',div_proc='实施',record_date='20230103',ex_date='20230104',
             pay_date='',div_listdate='',cash_div_tax=2.,stk_div=0.,stk_co_rate=3.,stk_bo_rate=0.)])
        good=bad.assign(end_date='20221231',cash_div_tax=.2,stk_div=.3,stk_co_rate=.3,
                        pay_date='20230104',div_listdate='20230104')
        with tempfile.TemporaryDirectory() as temp, patch('small_account_backtest.ROOT',Path(temp)), \
             patch('small_account_backtest.fetch_variants',return_value=bad), \
             patch('small_account_backtest.fetch',return_value=good) as query:
            actions=load_actions(Mock(),'x')
            self.assertEqual(actions.cash.iloc[0],.2)
            self.assertEqual(actions.stock.iloc[0],.3)
            self.assertEqual(pd.read_pickle(Path(temp)/'dividends/x.pkl').cash_div_tax.iloc[0],2.)
            self.assertTrue((Path(temp)/'dividends/x.canonical.pkl').exists())
            self.assertEqual(query.call_args.kwargs['end_date'],'20221231')
