import unittest

import numpy as np
import pandas as pd

from research_daily_opportunity_features import build_feature_store
from research_daily_opportunity_path_targets import (
    EXIT_HARD_STOP, EXIT_HORIZON, EXIT_TRAILING_STOP,
    PASSAGE_LOWER, PASSAGE_NEITHER, PASSAGE_UPPER, build_path_targets,
)


def quotes(opening=None, closing=None, low=None, size=30):
    opening = np.full(size, 100.) if opening is None else np.asarray(opening, dtype=float)
    size = len(opening)
    closing = opening.copy() if closing is None else np.asarray(closing, dtype=float)
    low = np.minimum(opening, closing)*.99 if low is None else np.asarray(low, dtype=float)
    return pd.DataFrame(dict(date=pd.bdate_range('2025-01-02', periods=size), ts_code='600001.SH',
                             open=opening, high=np.maximum(opening, closing)*1.01,
                             low=low, close=closing, adj_factor=1., amount=100000.))


class PathTargetsTests(unittest.TestCase):
    def test_hard_stop_uses_following_open_gap_not_trigger_close(self):
        opening, closing = np.full(30, 100.), np.full(30, 100.)
        closing[1], opening[2] = 91., 85.
        daily = quotes(opening, closing)
        daily.loc[2, 'low'] = 20.  # Already exited at this session's open.
        target = build_path_targets(daily)
        self.assertAlmostEqual(float(target.path_payoff[0, 0, 0]), -.15, places=6)
        self.assertAlmostEqual(float(target.max_adverse_excursion[0, 0, 0]), .15, places=6)
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_HARD_STOP)
        self.assertEqual(target.trigger_index[0, 0, 0], 1)
        self.assertEqual(target.exit_index[0, 0, 0], 2)
        self.assertEqual(target.first_passage[0, 0, 0], PASSAGE_LOWER)

    def test_trailing_uses_close_peak_and_earliest_trigger_never_same_close(self):
        opening, closing = np.full(30, 100.), np.full(30, 100.)
        opening[1:7] = [100., 121., 120., 109., 80., 50.]
        closing[1:7] = [120., 130., 119., 90., 70., 50.]
        target = build_path_targets(quotes(opening, closing))
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_TRAILING_STOP)
        self.assertEqual(target.trigger_index[0, 0, 0], 3)
        self.assertEqual(target.exit_index[0, 0, 0], 4)
        self.assertAlmostEqual(float(target.path_payoff[0, 0, 0]), .09, places=6)
        self.assertEqual(target.first_passage[0, 0, 0], PASSAGE_UPPER)
        self.assertEqual(target.first_passage_index[0, 0, 0], 1)

    def test_intraday_high_low_do_not_trigger_close_only_stops(self):
        daily = quotes()
        daily.loc[1:5, 'high'] = 200.
        daily.loc[1:5, 'low'] = 50.
        daily.loc[6, ['open', 'high']] = [105., 106.]
        target = build_path_targets(daily)
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_HORIZON)
        self.assertEqual(target.exit_index[0, 0, 0], 6)
        self.assertEqual(target.trigger_index[0, 0, 0], -1)
        self.assertEqual(target.first_passage[0, 0, 0], PASSAGE_NEITHER)
        self.assertAlmostEqual(float(target.max_adverse_excursion[0, 0, 0]), .5)
        self.assertAlmostEqual(float(target.path_payoff[0, 0, 0]), .05, places=6)

    def test_horizon_plus_one_completion_and_strict_purge_even_after_early_exit(self):
        daily = quotes()
        daily.loc[1, ['close', 'low']] = [91., 90.]
        target = build_path_targets(daily)
        self.assertEqual(target.exit_index[0, 0, 0], 2)
        self.assertEqual(target.label_end_dates[0, 0], daily.date.iloc[6].to_datetime64().astype('datetime64[D]'))
        self.assertFalse(target.matured_before(daily.date.iloc[6])[0, 0, 0])
        self.assertTrue(target.matured_before(daily.date.iloc[7])[0, 0, 0])
        self.assertFalse(target.valid[-6:, 0, 0].any())
        self.assertTrue(np.isnat(target.label_end_dates[-6:, 0]).all())

    def test_last_horizon_close_trigger_uses_horizon_plus_one_open(self):
        daily = quotes()
        daily.loc[5, ['close', 'low']] = [91., 90.]
        daily.loc[6, ['open', 'low']] = [88., 1.]
        target = build_path_targets(daily)
        self.assertEqual(target.trigger_index[0, 0, 0], 5)
        self.assertEqual(target.exit_index[0, 0, 0], 6)
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_HARD_STOP)
        self.assertAlmostEqual(float(target.path_payoff[0, 0, 0]), -.12, places=6)
        self.assertAlmostEqual(float(target.max_adverse_excursion[0, 0, 0]), .12, places=6)

    def test_missing_future_quote_invalidates_whole_horizon_without_compression(self):
        daily = quotes()
        calendar = daily.date.copy()
        daily.loc[1, ['close', 'low']] = [91., 90.]
        missing = daily.drop(index=5)  # After the day-2 reference exit, still required.
        target = build_path_targets(missing, sessions=calendar)
        self.assertFalse(target.quote_presence[5, 0])
        self.assertFalse(target.valid[0, 0, 0])
        self.assertTrue(np.isnan(target.path_payoff[0, 0, 0]))
        self.assertEqual(target.exit_index[0, 0, 0], -1)
        self.assertEqual(target.exit_reason[0, 0, 0], -1)
        self.assertEqual(target.first_passage[0, 0, 0], -1)
        self.assertTrue(target.valid[6, 0, 0])
        missing_exit = build_path_targets(daily.drop(index=6), sessions=calendar)
        self.assertFalse(missing_exit.valid[0, 0, 0])

    def test_actual_factor_adjustment_keeps_split_path_continuous(self):
        daily = quotes()
        before = build_path_targets(daily)
        split = daily.copy()
        split.loc[3:, ['open', 'high', 'low', 'close']] /= 2
        split.loc[3:, 'adj_factor'] = 2
        after = build_path_targets(split)
        for name in ('path_payoff', 'max_adverse_excursion', 'first_passage', 'exit_reason', 'exit_index'):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))

    def test_first_passage_has_inclusive_boundaries_and_earliest_direction(self):
        daily = quotes()
        daily.loc[1, ['close', 'high']] = [110., 111.]
        daily.loc[2, ['close', 'low']] = [92., 91.]
        target = build_path_targets(daily)
        self.assertEqual(target.first_passage[0, 0, 0], PASSAGE_UPPER)
        self.assertEqual(target.first_passage[0, 0, 1], PASSAGE_LOWER)
        self.assertEqual(target.first_passage_index[0, 0, 1], 2)
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_HARD_STOP)
        with self.assertRaises(ValueError):
            build_path_targets(daily, upper_thresholds=(-.08, .15, .3))

    def test_simultaneous_hard_and_trailing_trigger_prioritises_hard(self):
        daily = quotes()
        daily.loc[1, ['close', 'high']] = [130., 131.]
        daily.loc[2, ['close', 'low']] = [91., 90.]
        target = build_path_targets(daily)
        self.assertEqual(target.exit_reason[0, 0, 0], EXIT_HARD_STOP)
        self.assertEqual(target.trigger_index[0, 0, 0], 2)

    def test_future_changes_affect_labels_only_and_inputs_remain_unchanged(self):
        daily = quotes(size=110)
        untouched = daily.copy(deep=True)
        features = build_feature_store(daily)
        before = build_path_targets(daily)
        pd.testing.assert_frame_equal(daily, untouched)
        changed = daily.copy()
        changed.loc[83:, ['open', 'high', 'low', 'close']] *= 1.5
        later_features, after = build_feature_store(changed), build_path_targets(changed)
        np.testing.assert_array_equal(features.features[:81], later_features.features[:81])
        np.testing.assert_array_equal(features.valid_features[:81], later_features.valid_features[:81])
        self.assertNotEqual(before.path_payoff[80, 0, 0], after.path_payoff[80, 0, 0])
        self.assertFalse(after.valid[-1].any())
        self.assertTrue(later_features.valid_features[-1].all())

    def test_invalid_source_and_calendar_are_rejected(self):
        daily = quotes()
        with self.assertRaises(ValueError):
            build_path_targets(pd.concat((daily, daily.iloc[:1])))
        with self.assertRaises(ValueError):
            build_path_targets(daily, sessions=daily.date.iloc[::-1])
        broken = daily.copy()
        broken.loc[3, 'adj_factor'] = 0.
        with self.assertRaises(ValueError):
            build_path_targets(broken)
        with self.assertRaises(ValueError):
            build_path_targets(daily, horizons=(10, 5, 20))

    def test_vectorised_outputs_match_independent_scalar_path_book(self):
        rng = np.random.default_rng(517)
        frames = []
        for code in ('000001.SZ', '300001.SZ', '688001.SH'):
            closing = 100*np.exp(np.cumsum(rng.normal(0, .075, 36)))
            opening = np.r_[100., closing[:-1]]*np.exp(rng.normal(0, .02, 36))
            frame = quotes(opening, closing)
            frame['ts_code'] = code
            frame.loc[18:, ['open', 'high', 'low', 'close']] /= 2
            frame.loc[18:, 'adj_factor'] = 2
            frames.append(frame)
        calendar = frames[0].date.copy()
        daily = pd.concat(frames, ignore_index=True)
        daily = daily[~(daily.ts_code.eq('300001.SZ') & daily.date.eq(calendar.iloc[12]))]
        target = build_path_targets(daily, sessions=calendar)
        for stock_index, code in enumerate(target.stock_codes):
            stock = daily[daily.ts_code.eq(code)].set_index('date').reindex(calendar)
            adjusted = stock[['open', 'low', 'close']].multiply(stock.adj_factor, axis=0)
            for t in range(len(calendar)):
                for j, (horizon, upper) in enumerate(zip(target.horizons, (.1, .15, .3))):
                    if t+horizon+1 >= len(calendar):
                        self.assertFalse(target.valid[t, stock_index, j])
                        continue
                    window = adjusted.iloc[t+1:t+horizon+2]
                    if window.isna().any().any():
                        self.assertFalse(target.valid[t, stock_index, j])
                        continue
                    entry = float(window.open.iloc[0])
                    peak, first, first_at = entry, PASSAGE_NEITHER, -1
                    trigger_at, reason, exit_at = -1, EXIT_HORIZON, t+horizon+1
                    for offset in range(1, horizon+1):
                        value = float(adjusted.close.iloc[t+offset])
                        if first == PASSAGE_NEITHER:
                            if value >= entry*(1+upper)-1e-12*abs(entry*(1+upper)):
                                first, first_at = PASSAGE_UPPER, t+offset
                            elif value <= entry*.92+1e-12*abs(entry*.92):
                                first, first_at = PASSAGE_LOWER, t+offset
                        if trigger_at != -1:
                            continue
                        peak = max(peak, value)
                        if value <= entry*.92+1e-12*abs(entry*.92):
                            reason = EXIT_HARD_STOP
                        elif peak >= entry*1.2-1e-12*abs(entry*1.2) and value <= peak*.92+1e-12*abs(peak*.92):
                            reason = EXIT_TRAILING_STOP
                        else:
                            continue
                        trigger_at, exit_at = t+offset, t+offset+1
                    exit_price = float(adjusted.open.iloc[exit_at])
                    minimum = min(entry, exit_price, float(adjusted.low.iloc[t+1:exit_at].min()))
                    key = (t, stock_index, j)
                    self.assertTrue(target.valid[key])
                    self.assertAlmostEqual(float(target.path_payoff[key]), exit_price/entry-1, places=6)
                    self.assertAlmostEqual(float(target.max_adverse_excursion[key]), max(0, 1-minimum/entry), places=6)
                    self.assertEqual(target.exit_index[key], exit_at)
                    self.assertEqual(target.trigger_index[key], trigger_at)
                    self.assertEqual(target.exit_reason[key], reason)
                    self.assertEqual(target.first_passage[key], first)
                    self.assertEqual(target.first_passage_index[key], first_at)


if __name__ == '__main__':
    unittest.main()
