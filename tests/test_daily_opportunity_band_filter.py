from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_daily_opportunity_band_filter as module


class PriorBandTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);source=root/'source';out=root/'target';source.mkdir()
        (source/'checkpoints').mkdir()
        for name in ('feature_store.pkl','feature_manifest.json','frozen_edges.pkl','fit_audits.json'):
            (source/name).write_bytes(b'fixture')
        (source/'experiment_protocol.json').write_text('{"source_sha256":{}}')
        p=pd.DataFrame(dict(signal_date=pd.to_datetime(['2026-01-05']*4),
                            ts_code=['600001.SH','000001.SZ','300001.SZ','600002.SH'],
                            utility=[.5,.2,.8,-.1],mu20=[.6,.4,.9,.0]))
        return source,out,p

    def test_previous_signal_day_bands_not_current_names(self):
        source,out,p=self.fixture();seen=[]
        class Limits:
            sources={}
            def __init__(self,*args,**kwargs):pass
            def __call__(self,day,code):
                seen.append((day,code))
                return pd.Series(dict(down_limit=9.5 if code=='600001.SH' else 9.,up_limit=10.5 if code=='600001.SH' else 11.))
        with patch.object(module,'validate_predictions',return_value=p),patch.object(module,'OfficialLimits',Limits):
            result=module.filter_buy_risk_band(source,out,None)
        self.assertEqual([False,True,False,False],result.eligible.tolist())
        pd.testing.assert_series_equal(result.utility,p.utility)
        pd.testing.assert_series_equal(result.mu20,p.mu20)
        self.assertEqual(len(result),4) # All-board prediction rows preserved.
        self.assertTrue(all(day==pd.Timestamp('2026-01-05') for day,code in seen))

    def test_unknown_band_aborts_instead_of_guessing(self):
        source,out,p=self.fixture()
        class Missing:
            sources={}
            def __init__(self,*args,**kwargs):pass
            def __call__(self,*args):raise RuntimeError('Missing exact official limits')
        with patch.object(module,'validate_predictions',return_value=p),patch.object(module,'OfficialLimits',Missing):
            with self.assertRaises(RuntimeError):module.filter_buy_risk_band(source,out,None)
        self.assertFalse((out/'prediction_manifest.json').exists())


if __name__=='__main__':unittest.main()
