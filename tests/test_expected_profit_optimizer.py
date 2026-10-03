import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_leadership import leader_optimizer


class ExpectedProfitOptimizerTests(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame(dict(ts_code=['600001.SH','600002.SH'],stock_code=['600001','600002'],
            ind_code=['a','b'],reference_price=[60.,100.],leadership_score=[.30,.35]))

    def test_linear_objective_matches_exhaustive_expected_profit(self):
        c=self.candidates()
        selected,_=leader_optimizer(c,25000,stock_cap=1.,sector_cap=1.,
            minimum_names=1,sector_name_limit=None,score_power=1)
        value=float((selected.leadership_score*selected.planned_amount).sum())
        exhaustive=max(.30*6000*i+.35*10000*j
                       for i in range(5) for j in range(3) if 0<6000*i+10000*j<=25000)
        self.assertAlmostEqual(value,exhaustive)
        self.assertEqual(selected.ts_code.tolist(),['600001.SH'])
        old,_=leader_optimizer(c,25000,stock_cap=1.,sector_cap=1.,minimum_names=1,sector_name_limit=None)
        self.assertGreater(value,float((old.leadership_score*old.planned_amount).sum()))

    def test_unsupported_power_fails(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            leader_optimizer(self.candidates(),25000,score_power=2)


if __name__=='__main__': unittest.main()
