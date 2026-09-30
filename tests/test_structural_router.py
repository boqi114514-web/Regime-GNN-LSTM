import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_structural_router import structural_states
import research_market_states as research


class StructuralRouterTests(unittest.TestCase):
    def test_sparse_state_and_future_prefix(self):
        day=pd.Timestamp('2026-03-31')
        c=pd.DataFrame(dict(month=[day]*100,mom3=[-.1]*98+[.8]*2))
        m=pd.DataFrame(dict(breadth=[.3],amount_ratio=[1.1]),index=[day])
        first=structural_states(c,m)
        self.assertTrue(first.structural.iloc[0])
        future=c.assign(month=pd.Timestamp('2026-04-30'),mom3=.9)
        m.loc[pd.Timestamp('2026-04-30')]=[.9,2.]
        full=structural_states(pd.concat([c,future]),m)
        pd.testing.assert_frame_equal(first,full.loc[[day]])
        self.assertFalse(full.structural.iloc[1])
        m.loc[day,'amount_ratio']=.9
        self.assertFalse(structural_states(c,m).structural.iloc[0])

    def test_reentry_requires_actual_complete_limit_quotes(self):
        day=pd.Timestamp('2026-06-03')
        c=pd.DataFrame(dict(ts_code=['000001.SZ','000002.SZ'],stock_code=['000001','000002']))
        q=pd.DataFrame(dict(open=[10.,20.]),index=c.ts_code)
        frame=pd.DataFrame(dict(ts_code=c.ts_code,trade_date='20260603',up_limit=[11.,22.],down_limit=[9.,18.]))
        class Client:
            def query(self,api,**kw): return frame.iloc[:1] if kw['limit']==6000 else frame
        with TemporaryDirectory() as temp,patch.object(research,'OUT',Path(temp)):
            limits=research.replacement_limits(Client(),day,c,q)
            self.assertEqual(len(limits),2)
            self.assertEqual(limits.loc['000002.SZ','down_limit'],18.)


if __name__=='__main__': unittest.main()
