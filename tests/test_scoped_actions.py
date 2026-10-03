import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_scoped_actions as actions


class ScopedActionsTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame(dict(ts_code=['002309.SZ']*2,end_date=['20241119','20251231'],
            div_proc=['实施','预案'],record_date=['20241223',None],ex_date=['20241224',None],
            pay_date=[None,None],div_listdate=[None,None],cash_div_tax=[0,0],stk_div=[2.45,0]))

    def test_invalid_old_listing_not_part_of_empty_2026_account(self):
        self.assertTrue(actions.validate_history(self.raw(),'002309.SZ','2026-01-01','2026-09-24').empty)

    def test_invalid_in_window_listing_still_fails(self):
        raw=self.raw();raw.loc[0,['record_date','ex_date']]=['20260323','20260324']
        with self.assertRaisesRegex(ValueError,'listing'):
            actions.validate_history(raw,'002309.SZ','2026-01-01','2026-09-24')

    def test_single_period_and_wrong_stock_fail(self):
        for raw in (self.raw().iloc[:1],self.raw().assign(ts_code='000001.SZ')):
            with self.assertRaises(ValueError):
                actions.validate_history(raw,'002309.SZ','2026-01-01','2026-09-24')

    def test_missing_old_exdate_requires_proven_settlement_before_start(self):
        raw=self.raw(); raw.loc[0,'ex_date']=None;raw.loc[0,'div_listdate']='20241225'
        self.assertTrue(actions.validate_history(raw,'002309.SZ','2026-01-01','2026-09-24').empty)
        raw.loc[0,'div_listdate']='20260325'
        with self.assertRaisesRegex(ValueError,'ex-date'):
            actions.validate_history(raw,'002309.SZ','2026-01-01','2026-09-24')

    def test_cached_raw_bound_and_tamper_rejected(self):
        class Pro:
            def query(inner,*args,**kwargs): return self.raw()
        with tempfile.TemporaryDirectory() as task:
            out=Path(task)
            with patch.object(actions,'ROOT',out/'unused'),patch.object(actions.leadership,'OUT',out/'unused2'):
                result=actions.load_scoped_actions(Pro(),'002309.SZ',out,'2026-01-01','2026-09-24')
                self.assertTrue(result.empty)
                self.assertTrue(actions.load_scoped_actions(None,'002309.SZ',out,'2026-01-01','2026-09-24').empty)
                self.raw().assign(stk_div=0).to_pickle(out/'scoped_actions/002309.SZ.pkl')
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    actions.load_scoped_actions(None,'002309.SZ',out,'2026-01-01','2026-09-24')

    def test_minimal_required_fields_fallback_preserves_full_raw_history(self):
        calls=[]
        class Pro:
            def query(inner,api,**params):
                calls.append((api,params))
                if 'fields' not in params: raise RuntimeError('gateway default fields unavailable')
                return self.raw()
        with tempfile.TemporaryDirectory() as task:
            out=Path(task)
            with patch.object(actions,'ROOT',out/'unused'),patch.object(actions.leadership,'OUT',out/'unused2'):
                result=actions.load_scoped_actions(Pro(),'002309.SZ',out,'2026-01-01','2026-09-24')
                self.assertTrue(result.empty)
                self.assertEqual(len(calls),2)
                self.assertEqual(set(calls[1][1]['fields'].split(',')),set(self.raw().columns))
                pd.testing.assert_frame_equal(pd.read_pickle(out/'scoped_actions/002309.SZ.pkl'),self.raw())


if __name__=='__main__': unittest.main()
