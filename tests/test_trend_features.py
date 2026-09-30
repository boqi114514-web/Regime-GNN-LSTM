import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_trend_features import monthly_features, score_candidates


class TrendFeaturesTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range('2021-01-31', periods=20, freq='ME')
        m = pd.DataFrame(dict(date=dates, ts_code='000001.SZ', close=10*1.02**np.arange(20), adj_factor=1.))
        d = m[['date', 'ts_code']].assign(amount=np.arange(20)+100.)
        return m, d

    def test_prefix_invariance(self):
        m, d = self.fixture()
        full = monthly_features(m, d)
        prefix = monthly_features(m.iloc[:15], d.iloc[:15])
        pd.testing.assert_frame_equal(prefix, full[full.month.le(m.date.iloc[14])].reset_index(drop=True))

    def test_adjusted_split_invariance(self):
        m, d = self.fixture()
        expected = monthly_features(m, d)
        m.loc[10:, 'close'] /= 2
        m.loc[10:, 'adj_factor'] *= 2
        pd.testing.assert_frame_equal(expected, monthly_features(m, d))

    def test_monotonic_efficiency_and_amount_lag(self):
        m, d = self.fixture()
        f = monthly_features(m, d)
        np.testing.assert_allclose(f.efficiency6.dropna(), 1.)
        np.testing.assert_allclose(f.high_proximity12.dropna(), 1.)
        self.assertAlmostEqual(f.amount_expansion.iloc[3], 103/101)
        self.assertTrue(f.amount_expansion.iloc[:3].isna().all())

    def test_gate_and_missing_feature_rules(self):
        date = pd.Timestamp('2023-01-31')
        c = pd.DataFrame(dict(month=[date]*6, ts_code=[str(i) for i in range(6)],
                             mom1=.1, mom3=[1., .5, .4, .3, .2, .1], mom6=.5,
                             l2_code='peer', phase='retreat', leadership_score=.8,
                             size_bucket='middle', chosen_style='core'))
        f = c[['month', 'ts_code']].assign(high_proximity12=.99, efficiency6=.8, amount_expansion=1.2)
        self.assertFalse(score_candidates(c, f, 'trend_features_retry').eligible.any())
        open_c = score_candidates(c, f, 'trend_open_retry')
        self.assertTrue(open_c.loc[0, 'eligible'])
        self.assertEqual(open_c.loc[0, 'phase'], 'advance')
        f.loc[0, 'high_proximity12'] = np.nan
        self.assertFalse(score_candidates(c, f, 'trend_open_retry').loc[0, 'eligible'])


if __name__ == '__main__':
    unittest.main()
