from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_daily_opportunity_features import build_feature_store
from research_daily_opportunity_trees import daily_rows


class DailyTreeFactorsTests(unittest.TestCase):
    def fixture(self):
        days=pd.bdate_range('2024-12-20',periods=115)
        records=[]
        for code,k in [('000001.SZ',1),('300001.SZ',2),('688001.SH',3)]:
            for i,day in enumerate(days):
                p=10*(1+k*.001)**i
                records.append(dict(date=day,ts_code=code,open=p*.99,high=p*1.02,
                                    low=p*.98,close=p,amount=100000*(1+i*.01),adj_factor=1))
        store=build_feature_store(pd.DataFrame(records))
        edges=pd.DataFrame([dict(snapshot_date=days[60],theme_code='BK0001.DC',ts_code=c)
                            for c in store.stock_codes])
        return store,edges

    def test_native_labels_and_stock_coverage(self):
        store,edges=self.fixture()
        codes,x,y,risk,date=daily_rows(store,edges,90)
        self.assertEqual(len(codes),3)
        self.assertEqual(x.shape,(3,83))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(y,store.label_returns[90])
        np.testing.assert_allclose(risk,store.label_downside[90])

    def test_future_edges_do_not_change_factors(self):
        store,edges=self.fixture()
        future=pd.DataFrame([dict(snapshot_date=store.dates[100],theme_code='BK0002.DC',ts_code='000001.SZ')])
        x=daily_rows(store,edges,90)[1]
        y=daily_rows(store,pd.concat([edges,future]),90)[1]
        np.testing.assert_array_equal(x,y)

    def test_future_labels_do_not_remove_inference_graph_nodes(self):
        store,edges=self.fixture()
        codes,x,y,risk,_=daily_rows(store,edges,114)
        self.assertEqual(len(codes),3)
        self.assertTrue(np.isnan(y).all())
        self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':
    unittest.main()
