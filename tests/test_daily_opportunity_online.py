from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_daily_opportunity_features import build_feature_store
from research_daily_graph_data import asof_graph
import research_daily_opportunity_online as online


class OnlineDailyCalibrationTests(unittest.TestCase):
    def fixture(self):
        days=pd.bdate_range('2024-12-20',periods=120)
        raw=[]
        for code,k in [('000001.SZ',1),('300001.SZ',2),('688001.SH',3)]:
            for i,day in enumerate(days):
                p=10*(1+k*.001)**i
                raw.append(dict(date=day,ts_code=code,open=p*.99,high=p*1.02,low=p*.98,
                                close=p,amount=100000,adj_factor=1))
        store=build_feature_store(pd.DataFrame(raw))
        edges=pd.DataFrame([dict(snapshot_date=days[60],theme_code='BK0001.DC',ts_code=c)
                            for c in store.stock_codes])
        predictions=pd.DataFrame([dict(signal_date=days[i],ts_code=code,
             model_fit_cutoff=days[70],mu5=0.,mu10=0.,mu20=0.,risk5=.05,risk10=.05,risk20=.05,
             rally_probability5=.01,rally_probability10=.01,rally_probability20=.01,utility=-.025)
             for i in range(80,120) for code in store.stock_codes])
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);source=root/'source';source.mkdir();(source/'checkpoints').mkdir()
        for name in ('feature_store.pkl','feature_manifest.json','frozen_edges.pkl','fit_audits.json'):
            (source/name).write_bytes(b'fixture')
        (source/'experiment_protocol.json').write_text('{"source_sha256":{}}')
        return store,edges,predictions,source,root

    def test_future_label_mutation_cannot_change_prefix(self):
        store,edges,p,source,root=self.fixture()
        with patch.object(online,'validate_predictions',return_value=p),patch.object(online,'validate_prepared',return_value=(store,edges,{})):
            first=online.run_online(source,root/'first')
            day=store.dates[105]
            for t in range(len(store.dates)):
                for j in range(3):
                    if np.isnat(store.label_end_dates[t,j]) or store.label_end_dates[t,j]>=np.datetime64(day,'D'):
                        store.label_returns[t,:,j]=.9;store.label_downside[t,:,j]=.8
            second=online.run_online(source,root/'second')
        before=first[first.signal_date.le(day)].reset_index(drop=True)
        after=second[second.signal_date.le(day)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before,after)

    def test_short_head_updates_before_twenty_day_head_without_early_label(self):
        store,edges,p,source,root=self.fixture()
        with patch.object(online,'validate_predictions',return_value=p),patch.object(online,'validate_prepared',return_value=(store,edges,{})):
            result=online.run_online(source,root/'out')
        audit=pd.read_csv(root/'out/online_calibration_audit.csv')
        day=store.dates[92].strftime('%Y-%m-%d')
        f=audit[audit.signal_date.eq(day)]
        self.assertTrue(f[f.horizon.eq(5)].active.iloc[0])
        self.assertFalse(f[f.horizon.eq(20)].active.iloc[0])
        active=audit[audit.active]
        self.assertTrue((pd.to_datetime(active.maximum_label_end)<pd.to_datetime(active.signal_date)).all())
        self.assertTrue((result[['risk5','risk10','risk20']]>=0).all().all())

    def test_self_free_peer_message(self):
        store,edges,p,source,root=self.fixture()
        g=asof_graph(edges,store.dates[90],store.stock_codes,max_age_days=None)
        x=np.array([[1.],[3.],[8.]])
        message,known=online.specific_peer_message(g,x)
        np.testing.assert_allclose(message,[[5.5],[4.5],[2.]])
        self.assertTrue(known.all())


if __name__=='__main__':unittest.main()
