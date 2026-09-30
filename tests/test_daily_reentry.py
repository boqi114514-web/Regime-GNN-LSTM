import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_daily_reentry import pressure_features, can_reenter


class DailyReentryTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(dict(date=pd.bdate_range('2023-01-02',periods=30),ts_code='000001.SZ',
            open=10.,high=10.4,low=9.9,close=10.3,volume=1000.,amount=np.arange(30)+100.))

    def test_no_future_and_overnight_rescaling_invariance(self):
        d=self.sample(); prior=pressure_features(d.iloc[:-1])
        full=pressure_features(d)
        pd.testing.assert_frame_equal(prior,full[full.signal_day.lt(d.date.iloc[-1])].reset_index(drop=True))
        rescaled=d.copy()
        rescaled.loc[10:,'open':'close']*=.5
        pd.testing.assert_frame_equal(full,pressure_features(rescaled))
        self.assertAlmostEqual(full.pressure5.iloc[0],1.03**5-1)
        self.assertAlmostEqual(full.pressure20.iloc[0],1.03**20-1)

    def test_missing_trading_day_not_bridged(self):
        d=self.sample()
        other=d.assign(ts_code='300001.SZ')
        missing=d.drop(index=20)
        f=pressure_features(pd.concat([missing,other],ignore_index=True))
        self.assertFalse(f.ts_code.eq('000001.SZ').any())
        self.assertTrue(f.ts_code.eq('300001.SZ').any())

    def test_cooldown_flat_and_month_boundary(self):
        day=pd.Timestamp('2023-03-15'); previous=day-pd.Timedelta(days=1)
        self.assertTrue(can_reenter(day,previous,20,15,True,set()))
        self.assertFalse(can_reenter(day,previous,19,15,True,set()))
        self.assertFalse(can_reenter(day,previous,20,15,False,set()))
        self.assertFalse(can_reenter(day,previous,20,None,True,set()))
        self.assertFalse(can_reenter(day,previous,20,15,True,{day}))
        self.assertFalse(can_reenter(pd.Timestamp('2023-04-03'),pd.Timestamp('2023-03-31'),25,15,True,set()))


if __name__=='__main__': unittest.main()
