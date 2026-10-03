import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_liquid_leaders import phase_priority_candidates


class LiquidPhasePriorityTests(unittest.TestCase):
    def pool(self):
        return pd.DataFrame(dict(month=[pd.Timestamp('2023-01-31')]*3,
            ts_code=['000001.SZ','000002.SZ','688001.SH'],stock_code=['000001','000002','688001'],
            close=[10.,20.,10.],size_bucket=['core']*3,chosen_style=['core']*3,
            original_phase=['advance','range','advance'],phase=['advance']*3,
            original_leadership_score=[.9,.8,1.],leadership_score=[.4,.95,.98],
            mom1=[.1,.2,.1],mom3=[.3,.4,.5],liquid_route=[True]*3,eligible=[True]*3))

    def test_correct_advance_core_restored_without_mutation(self):
        c=self.pool();before=c.copy(deep=True);r=phase_priority_candidates(c)
        self.assertFalse(r.liquid_route.any())
        pd.testing.assert_series_equal(r.phase,r.original_phase,check_names=False)
        pd.testing.assert_series_equal(r.leadership_score,r.original_leadership_score,check_names=False)
        pd.testing.assert_frame_equal(c,before)

    def test_range_core_still_routes_despite_star_advance(self):
        c=self.pool();c.loc[0,'original_phase']='range'
        self.assertTrue(phase_priority_candidates(c).liquid_route.all())

    def test_unaffordable_core_not_priority(self):
        c=self.pool();c.loc[0,'close']=251
        self.assertTrue(phase_priority_candidates(c).liquid_route.all())

    def test_retreating_individual_not_priority(self):
        c=self.pool();c.loc[0,'mom1']=-.01
        self.assertTrue(phase_priority_candidates(c).liquid_route.all())

    def test_future_month_does_not_change_past_priority(self):
        c=self.pool();r=phase_priority_candidates(c)
        future=c.assign(month=pd.Timestamp('2023-02-28'),original_phase='range')
        f=phase_priority_candidates(pd.concat([c,future],ignore_index=True))
        pd.testing.assert_frame_equal(r,f.iloc[:3])


if __name__=='__main__': unittest.main()
