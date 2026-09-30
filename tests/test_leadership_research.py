import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_leadership import enrich, leader_optimizer


class LeadershipTests(unittest.TestCase):
    def test_future_months_do_not_rewrite_features(self):
        dates=pd.date_range('2022-01-31',periods=18,freq='ME')
        monthly=pd.DataFrame([dict(date=d,ts_code=c,close=10+i*(j+1)/10,adj_factor=1.)
                              for i,d in enumerate(dates) for j,c in enumerate(['000001.SZ','000002.SZ'])])
        candidates=monthly[monthly.date>=dates[6]].copy()
        candidates['month']=candidates.date
        candidates['mom6']=.1
        members=pd.DataFrame(dict(ts_code=['000001.SZ','000002.SZ'],l2_code=['x','x'],
                                  in_date=['20200101']*2,out_date=[None]*2))
        early=enrich(candidates[candidates.month<=dates[12]],monthly[monthly.date<=dates[12]],members)
        full=enrich(candidates,monthly,members)
        pd.testing.assert_frame_equal(early.reset_index(drop=True),full[full.month<=dates[12]].reset_index(drop=True))

    def test_expired_membership_is_not_used(self):
        dates=pd.date_range('2022-01-31',periods=8,freq='ME')
        m=pd.DataFrame(dict(date=dates,ts_code='000001.SZ',close=np.arange(8)+10,adj_factor=1.))
        c=m.iloc[6:].copy().assign(month=lambda x:x.date,mom6=.1)
        member=pd.DataFrame([dict(ts_code='000001.SZ',l2_code='x',in_date='20200101',out_date='20220601')])
        f=enrich(c,m,member)
        self.assertTrue(f.l2_code.isna().all())
        self.assertTrue(f.peer_rank.eq(.5).all())

    @staticmethod
    def candidates():
        return pd.DataFrame(dict(stock_code=['000001','000002','600003','600004','000005'],
            ts_code=['000001.SZ','000002.SZ','600003.SH','600004.SH','000005.SZ'],
            ind_code=['a','a','b','b','c'],reference_price=[85.,30.,20.,10.,5.],
            leadership_score=[1.,.95,.9,.8,.7]))

    def test_budget_lots_and_group_cap(self):
        c,_=leader_optimizer(self.candidates(),25000,False)
        self.assertLessEqual(c.planned_amount.sum(),25000.01)
        self.assertTrue((c.shares%100==0).all())
        self.assertTrue((c.planned_amount<=7500.01).all())
        self.assertTrue((c.groupby('ind_code').size()<=2).all())
        self.assertTrue((c.groupby('ind_code').planned_amount.sum()<=15000.01).all())
        self.assertNotIn('000001',set(c.stock_code))

    def test_flexible_lot_cap_admits_affordable_expensive_stock(self):
        c,_=leader_optimizer(self.candidates(),25000,True)
        self.assertIn('000001',set(c.stock_code))
        self.assertLessEqual(c.planned_amount.sum(),25000.01)
        self.assertTrue((c.planned_amount<=12500.01).all())
        self.assertLessEqual(len(c),4)

    def test_unrestricted_allows_one_stock_and_ignores_old_minimum(self):
        candidate=self.candidates().iloc[:1].assign(reference_price=25.)
        c,_=leader_optimizer(candidate,25000,stock_cap=1.,sector_cap=1.,
            minimum_names=1,sector_name_limit=None,min_names=4,max_stock_weight=.3)
        self.assertEqual(len(c),1)
        self.assertEqual(c.shares.iloc[0],1000)
        self.assertEqual(c.planned_amount.sum(),25000.)

    def test_unrestricted_still_requires_affordable_whole_lot(self):
        candidate=self.candidates().iloc[:1].assign(reference_price=200.)
        c,_=leader_optimizer(candidate,25000,stock_cap=1.,sector_cap=1.,
            minimum_names=1,sector_name_limit=None)
        self.assertEqual(c.shares.iloc[0],100)
        self.assertEqual(c.planned_amount.sum(),20000.)
        with self.assertRaises(ValueError):
            leader_optimizer(candidate.assign(reference_price=251.),25000,stock_cap=1.,sector_cap=1.,
                minimum_names=1,sector_name_limit=None)


if __name__=='__main__': unittest.main()
