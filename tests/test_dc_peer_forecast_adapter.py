import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_dc_forecast as base
import research_dc_peer_forecast as peer


class TestEstimator:
    def __init__(self,**kwargs): pass
    def fit(self,x,y): self.columns=x.shape[1]; self.mean=float(np.mean(y)); return self
    def predict(self,x): return np.full(len(x),self.mean)


class PeerForecastAdapterTests(unittest.TestCase):
    def fixtures(self):
        dates=pd.date_range('2023-01-31',periods=24,freq='ME')
        frame=pd.DataFrame([dict(signal_date=d,ts_code=code,label_date=dates[i+1] if i<23 else pd.NaT,
            label_return=.02,**{name:.1 for name in base.FEATURE_COLUMNS})
            for i,d in enumerate(dates) for code in ['600001.SH','300001.SZ']])
        context=frame[['signal_date','ts_code']].assign(peer_mom1=.1,peer_mom3=.2,
            peer_mom6=.3,peer_positive3=.8,peer_count=20)
        monthly=frame[['signal_date','ts_code']].rename(columns={'signal_date':'date'}).assign(close=10.,adj_factor=1.)
        daily=monthly.drop(columns=['adj_factor']).assign(open=10.,high=10.1,low=9.9,volume=100.,amount=1000.)
        return dates,frame,context,monthly,daily

    def run_fixture(self,out,frame,context,monthly,daily,signals):
        with patch.object(peer,'ROOT',out/'raw'),patch.object(peer,'forecast_features',return_value=frame),\
             patch('research_dc_peer_context.peer_context',return_value=context) as ctx,\
             patch.object(base,'HistGradientBoostingRegressor',TestEstimator):
            predictions=peer.prepare_peer_forecasts(out,monthly,daily,signals)
        return predictions,ctx

    def test_nineteen_features_purged_and_cache_reused(self):
        dates,frame,context,monthly,daily=self.fixtures()
        with tempfile.TemporaryDirectory() as task:
            out=Path(task);(out/'raw').mkdir()
            monthly.to_pickle(out/'raw/stock_month_end_verified.pkl');daily.to_pickle(out/'daily.pkl')
            first,ctx=self.run_fixture(out,frame,context,monthly,daily,dates[20:22])
            self.assertEqual(ctx.call_count,1)
            self.assertTrue(first.train_label_end.lt(first.model_fit_cutoff).all())
            models=pd.read_pickle(out/'forecast_models.pkl')
            self.assertTrue(all(m['model'].columns==19 for m in models.values()))
            second,ctx=self.run_fixture(out,frame,context,monthly,daily,dates[20:22])
            self.assertEqual(ctx.call_count,0)
            pd.testing.assert_frame_equal(first,second)

    def test_memory_input_and_artifact_changes_invalidate_cache(self):
        dates,frame,context,monthly,daily=self.fixtures()
        with tempfile.TemporaryDirectory() as task:
            out=Path(task);(out/'raw').mkdir()
            monthly.to_pickle(out/'raw/stock_month_end_verified.pkl');daily.to_pickle(out/'daily.pkl')
            self.run_fixture(out,frame,context,monthly,daily,dates[20:22])
            changed=daily.assign(close=10.01)
            _,ctx=self.run_fixture(out,frame,context,monthly,changed,dates[20:22])
            self.assertEqual(ctx.call_count,1)
            pd.DataFrame().to_pickle(out/'peer_context.pkl')
            _,ctx=self.run_fixture(out,frame,context,monthly,changed,dates[20:22])
            self.assertEqual(ctx.call_count,1)


if __name__=='__main__': unittest.main()
