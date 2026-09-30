import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_learned_leaders import label_panel,walkforward


class LearnedLeaderTests(unittest.TestCase):
    def test_calendar_label_does_not_bridge_missing_month(self):
        m=pd.DataFrame(dict(date=pd.to_datetime(['2023-01-31','2023-03-31']),
                            ts_code='a',close=[10.,20.],adj_factor=1.))
        c=m[['date','ts_code']].rename(columns={'date':'month'})
        self.assertTrue(label_panel(c,m).future_return.isna().all())

    def test_future_labels_cannot_change_earlier_fit(self):
        rng=np.random.default_rng(42)
        rows=[]
        for day in pd.date_range('2022-01-31',periods=10,freq='ME'):
            for code in range(20):
                x=rng.normal()
                rows.append(dict(month=day,ts_code=str(code),x=x,target=x*.1,
                                 label_date=day+pd.offsets.MonthEnd(1)))
        p=pd.DataFrame(rows)
        params=dict(n_estimators=5,min_child_samples=3,num_leaves=3,verbosity=-1,n_jobs=1,random_state=42)
        a,au,_=walkforward(p,['x'],params,min_months=2)
        cutoff=pd.Timestamp('2022-07-31')
        changed=p.copy(); changed.loc[changed.label_date.gt(cutoff),'target']=1000.
        b,_,_=walkforward(changed,['x'],params,min_months=2)
        pd.testing.assert_frame_equal(a[a.month.le(cutoff)],b[b.month.le(cutoff)])
        self.assertTrue((au.latest_training_label<au.first_prediction).all())
        self.assertTrue(a[a.month.eq('2022-01-31')].fallback.all())


if __name__=='__main__': unittest.main()
