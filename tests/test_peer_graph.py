import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_peer_graph import peer_features


class PeerGraphTests(unittest.TestCase):
    def sample(self):
        dates=pd.date_range('2021-12-31',periods=14,freq='ME')
        codes=['000001.SZ','300001.SZ','300002.SZ','688001.SH','688002.SH','600001.SH']
        monthly=[]
        for i,code in enumerate(codes):
            prices=(10+i)*np.cumprod(1+np.array([0,.03,-.01,.04,.02,.01,-.03,.02,.05,-.01,.04,.03,.02,1.]))
            monthly.extend(dict(date=d,ts_code=code,close=p,adj_factor=1.) for d,p in zip(dates,prices))
        c=pd.DataFrame(dict(month=dates[-2],ts_code=codes,l2_code='p',mom1=.2,mom3=.4,mom6=.3))
        return c,pd.DataFrame(monthly)

    def test_past_only_self_exclusion_and_all_boards(self):
        c,m=self.sample()
        past=peer_features(c,m[m.date.le(c.month.iloc[0])])
        full=peer_features(c,m)
        pd.testing.assert_frame_equal(past,full)
        self.assertEqual(len(full),6)
        self.assertTrue(full.peer_count.eq(5).all())
        for row in full.itertuples():
            self.assertNotIn(row.ts_code,row.neighbors.split(','))
            self.assertLessEqual(row.history_end,row.month)

    def test_calendar_gap_is_not_filled(self):
        c,m=self.sample()
        m=m[~(m.ts_code.eq('000001.SZ')&m.date.eq(pd.Timestamp('2022-05-31')))]
        f=peer_features(c,m)
        self.assertNotIn('000001.SZ',set(f.ts_code))
        self.assertFalse(f.neighbors.str.contains('000001.SZ',regex=False).any())
        self.assertTrue(f.peer_count.eq(4).all())


if __name__=='__main__': unittest.main()
