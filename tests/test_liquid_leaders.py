import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_liquid_leaders import (
    continue_liquid_position,
    liquidity_features,
    score_candidates,
)


class LiquidLeaderTests(unittest.TestCase):
    def pool(self, month='2026-03-31'):
        month = pd.Timestamp(month)
        c = pd.DataFrame(dict(
            month=[month] * 20, date=[month] * 20,
            ts_code=[f'{i:06d}.SZ' for i in range(18)] + ['300001.SZ', '688001.SH'],
            mom1=.1, mom3=.2, mom6=.3, size_bucket='middle',
            chosen_style='middle', phase='onset',
            leadership_score=np.linspace(.1, .8, 20),
        ))
        f = c[['month', 'ts_code']].assign(
            signal_amount=np.arange(1, 21, dtype=float),
            liquidity_days=20, liquidity_last_day=month,
        )
        m = pd.DataFrame({'breadth': [.3]}, index=pd.DatetimeIndex([month]))
        return c, f, m

    def test_liquidity_uses_signal_day_not_calendar_month_end(self):
        c = pd.DataFrame(dict(
            month=[pd.Timestamp('2026-03-31')],
            date=[pd.Timestamp('2026-03-04')], ts_code=['600183.SH'],
        ))
        d = pd.DataFrame(dict(
            date=pd.to_datetime(['2026-02-27', '2026-03-02', '2026-03-03',
                                 '2026-03-04', '2026-03-05', '2026-04-01']),
            ts_code='600183.SH', volume=1.,
            amount=[9999., 10., 20., 30., 1e12, 1e15],
        ))
        original_c, original_d = c.copy(deep=True), d.copy(deep=True)
        full = liquidity_features(c, d)
        prefix = liquidity_features(c, d[d.date.le(c.date.iloc[0])])
        pd.testing.assert_frame_equal(full, prefix)
        self.assertEqual(full.signal_amount.iloc[0], 20.)
        self.assertEqual(full.liquidity_days.iloc[0], 3)
        self.assertEqual(full.liquidity_last_day.iloc[0], c.date.iloc[0])
        pd.testing.assert_frame_equal(c, original_c)
        pd.testing.assert_frame_equal(d, original_d)

    def test_nonpositive_amount_or_zero_volume_is_not_an_observation(self):
        c, _, _ = self.pool()
        c = c.iloc[:1].copy()
        d = pd.DataFrame(dict(
            date=pd.bdate_range('2026-03-02', periods=6),
            ts_code=c.ts_code.iloc[0],
            amount=[10., 20., 1000., 0., -1., np.nan],
            volume=[1., 2., 0., 1., 1., 1.],
        ))
        f = liquidity_features(c, d)
        self.assertEqual(f.signal_amount.iloc[0], 15.)
        self.assertEqual(f.liquidity_days.iloc[0], 2)
        self.assertEqual(f.liquidity_last_day.iloc[0], d.date.iloc[1])

    def test_features_reject_duplicate_keys_and_wrong_signal_month(self):
        c, _, _ = self.pool()
        d = c[['date', 'ts_code']].assign(volume=1., amount=1.)
        with self.assertRaises(ValueError):
            liquidity_features(pd.concat([c, c.iloc[:1]]), d)
        with self.assertRaises(ValueError):
            liquidity_features(c, pd.concat([d, d.iloc[:1]]))
        c.loc[0, 'date'] = pd.Timestamp('2026-04-01')
        with self.assertRaises(ValueError):
            liquidity_features(c, d)

    def test_all_boards_rank_together_and_scores_rank_in_liquid_subset(self):
        c, f, m = self.pool()
        c.loc[17:, 'mom1'] = [.3, .1, .2]
        c.loc[17:, 'mom3'] = [.2, .3, .1]
        # Original style and phase must not veto a qualifying liquid leader.
        c.loc[17:, 'size_bucket'] = 'large'
        c.loc[17:, 'phase'] = 'retreat'
        result = score_candidates(c, f, m)
        self.assertTrue(result.liquid_route.all())
        self.assertEqual(result.index[result.eligible].tolist(), [17, 18, 19])
        self.assertFalse(result.loc[16, 'liquid_qualifies'])
        self.assertTrue(result.loc[18, 'liquid_qualifies'])  # ChiNext upstream.
        self.assertTrue(result.loc[19, 'liquid_qualifies'])  # STAR upstream.
        self.assertTrue(result.loc[17:, 'phase'].eq('advance').all())
        expected = [(.45 + 3 * .35 + 2 * .20) / 3,
                    (2 * .45 + .35 + 3 * .20) / 3,
                    (3 * .45 + 2 * .35 + .20) / 3]
        np.testing.assert_allclose(result.loc[17:, 'leadership_score'], expected)
        self.assertEqual(result.loc[17:, 'liquid_rank'].tolist(), [2., 3., 1.])

    def test_non_routed_month_keeps_original_score_phase_and_eligibility(self):
        c, f, m = self.pool()
        c.loc[0, 'size_bucket'] = 'large'
        c.loc[1, 'phase'] = 'retreat'
        c.loc[2, 'phase'] = 'overheat'
        original = c.copy(deep=True)
        original_f, original_m = f.copy(deep=True), m.copy(deep=True)
        for breadth in [.5, .9]:
            with self.subTest(breadth=breadth):
                result = score_candidates(c, f, m.assign(breadth=breadth))
                self.assertFalse(result.liquid_route.any())
                pd.testing.assert_series_equal(result.phase, c.phase)
                pd.testing.assert_series_equal(result.leadership_score, c.leadership_score)
                expected = c.size_bucket.eq(c.chosen_style) & ~c.phase.isin(['retreat', 'overheat'])
                pd.testing.assert_series_equal(result.eligible, expected.rename('eligible'))
        pd.testing.assert_frame_equal(c, original)
        pd.testing.assert_frame_equal(f, original_f)
        pd.testing.assert_frame_equal(m, original_m)

    def test_low_breadth_without_qualified_stock_falls_back_to_baseline(self):
        c, f, m = self.pool()
        c['mom1'] = -.01
        result = score_candidates(c, f, m)
        self.assertFalse(result.liquid_qualifies.any())
        self.assertFalse(result.liquid_route.any())
        self.assertTrue(result.eligible.all())
        pd.testing.assert_series_equal(result.phase, c.phase)
        pd.testing.assert_series_equal(result.leadership_score, c.leadership_score)

    def test_positive_momentum_and_observation_thresholds_are_required(self):
        for field in ['mom1', 'mom3', 'mom6']:
            for value in [0., -.01, np.nan]:
                with self.subTest(field=field, value=value):
                    c, f, m = self.pool()
                    c.loc[19, field] = value
                    self.assertFalse(score_candidates(c, f, m).loc[19, 'liquid_qualifies'])
        c, f, m = self.pool()
        f.loc[19, 'liquidity_days'] = 9
        self.assertFalse(score_candidates(c, f, m).loc[19, 'liquid_qualifies'])
        f.loc[19, 'liquidity_days'] = 10
        self.assertTrue(score_candidates(c, f, m).loc[19, 'liquid_qualifies'])

    def test_overheat_exclusion_is_joint_and_strict(self):
        for mom1, mom3, qualifies in [(.31, 1.51, False), (.3, 1.51, True),
                                      (.31, 1.5, True), (.1, 1.6, True)]:
            with self.subTest(mom1=mom1, mom3=mom3):
                c, f, m = self.pool()
                c.loc[19, ['mom1', 'mom3']] = [mom1, mom3]
                self.assertEqual(bool(score_candidates(c, f, m).loc[19, 'liquid_qualifies']), qualifies)

    def test_missing_features_never_qualify(self):
        c, f, m = self.pool()
        result = score_candidates(c, f.iloc[:-1], m)
        self.assertFalse(result.loc[19, 'liquid_qualifies'])
        self.assertFalse(result.loc[19, 'eligible'])

    def test_future_month_does_not_change_past_scores(self):
        c, f, m = self.pool()
        earlier = score_candidates(c, f, m)
        future_c, future_f, future_m = self.pool('2026-04-30')
        future_c['mom1'] = 100.
        future_f['signal_amount'] *= 1e10
        all_months = score_candidates(pd.concat([c, future_c], ignore_index=True),
                                      pd.concat([f, future_f], ignore_index=True),
                                      pd.concat([m, future_m]))
        pd.testing.assert_frame_equal(earlier, all_months[all_months.month.eq(c.month.iloc[0])].reset_index(drop=True))

    def test_scores_reject_duplicate_keys_missing_breadth_and_future_features(self):
        c, f, m = self.pool()
        with self.assertRaises(ValueError):
            score_candidates(pd.concat([c, c.iloc[:1]]), f, m)
        with self.assertRaises(ValueError):
            score_candidates(c, pd.concat([f, f.iloc[:1]]), m)
        with self.assertRaises(ValueError):
            score_candidates(c, f, m.iloc[:0])
        f.loc[0, 'liquidity_last_day'] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            score_candidates(c, f, m)

    def test_code_renaming_cannot_change_selection(self):
        c, f, m = self.pool()
        original = score_candidates(c, f, m)
        names = {code: f'arbitrary-{i:04d}' for i, code in enumerate(c.ts_code)}
        renamed_c = c.assign(ts_code=c.ts_code.map(names))
        renamed_f = f.assign(ts_code=f.ts_code.map(names))
        renamed = score_candidates(renamed_c, renamed_f, m)
        for field in ['eligible', 'liquid_qualifies', 'liquid_route', 'phase', 'leadership_score']:
            pd.testing.assert_series_equal(original[field], renamed[field])

    def test_continuation_cannot_cancel_pending_exit_or_keep_unqualified_stock(self):
        base = pd.Series(dict(liquid_route=True, liquid_qualifies=True, liquid_rank=1.))
        for rank in [1., 10.]:
            row = base.copy()
            row['liquid_rank'] = rank
            self.assertTrue(continue_liquid_position(row, False))
        self.assertFalse(continue_liquid_position(None, False))
        self.assertFalse(continue_liquid_position(base, True))
        for field, value in [('liquid_route', False), ('liquid_qualifies', False),
                             ('liquid_rank', 0.), ('liquid_rank', 11.), ('liquid_rank', np.nan)]:
            row = base.copy()
            row[field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(continue_liquid_position(row, False))


if __name__ == '__main__':
    unittest.main()
