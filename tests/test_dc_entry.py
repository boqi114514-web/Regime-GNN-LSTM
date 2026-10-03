import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_dc_entry import OUTPUT_COLUMNS, entry_features, weekly_execution_schedule


class DCEntryTests(unittest.TestCase):
    def sample(self, count=30):
        dates = pd.bdate_range('2026-03-02', periods=count)
        records = []
        for code, pressure, amount in [('600001.SH', .012, 1000.), ('300001.SZ', .008, 2000.), ('000001.SZ', .002, 3000.)]:
            records.append(pd.DataFrame(dict(date=dates, ts_code=code, open=10.,
                high=10.2, low=9.9, close=10. * (1 + pressure), volume=100.,
                amount=np.where(np.arange(count) >= count - 5, 1.5 * amount, amount))))
        return pd.concat(records, ignore_index=True), dates

    def test_complete_features_and_fixed_score_include_chinext(self):
        daily, dates = self.sample()
        actual = entry_features(daily, [dates[-1]])
        self.assertEqual(list(actual.columns), OUTPUT_COLUMNS)
        self.assertEqual(set(actual.ts_code), {'600001.SH', '300001.SZ', '000001.SZ'})
        self.assertFalse(actual.duplicated(['signal_day', 'ts_code']).any())
        row = actual[actual.ts_code.eq('600001.SH')].iloc[0]
        self.assertAlmostEqual(row.pressure5, 1.012**5 - 1)
        self.assertAlmostEqual(row.pressure20, 1.012**20 - 1)
        self.assertAlmostEqual(row.location5, (10.12 - 9.9) / .3)
        self.assertAlmostEqual(row.amount_ratio5_vs_prior20, 1.5)
        self.assertAlmostEqual(row.mean_amount20, 1125.)
        self.assertTrue(row.reversal_eligible)
        ranks = actual[['pressure5', 'pressure20', 'amount_ratio5_vs_prior20', 'mean_amount20']].rank(pct=True)
        expected = .35 * ranks.pressure5 + .35 * ranks.pressure20 + .15 * ranks.amount_ratio5_vs_prior20 + .15 * ranks.mean_amount20
        np.testing.assert_allclose(actual.entry_score, expected)

    def test_missing_session_never_bridges_hole(self):
        daily, dates = self.sample()
        daily = daily[~(daily.ts_code.eq('600001.SH') & daily.date.eq(dates[-20]))]
        actual = entry_features(daily, [dates[-1]])
        self.assertNotIn('600001.SH', set(actual.ts_code))
        self.assertEqual(len(actual), 2)

    def test_star_board_participates_but_invalid_exchange_codes_do_not(self):
        daily, dates = self.sample()
        additions = daily[daily.ts_code.eq('600001.SH')]
        daily = pd.concat([daily, additions.assign(ts_code='688001.SH'),
                           additions.assign(ts_code='600001.SZ')], ignore_index=True)
        actual = entry_features(daily, [dates[-1]])
        self.assertIn('688001.SH', set(actual.ts_code))
        self.assertNotIn('600001.SZ', set(actual.ts_code))

    def test_explicit_calendar_detects_marketwide_hole(self):
        daily, dates = self.sample()
        daily = daily[~daily.date.eq(dates[-15])]
        self.assertTrue(entry_features(daily, [dates[-1]], dates).empty)

    def test_missing_signal_session_is_not_stale_filled(self):
        daily, dates = self.sample()
        daily = daily[~(daily.ts_code.eq('600001.SH') & daily.date.eq(dates[-1]))]
        self.assertNotIn('600001.SH', set(entry_features(daily, [dates[-1]]).ts_code))

    def test_duplicate_daily_and_calendar_keys_raise(self):
        daily, dates = self.sample()
        with self.assertRaisesRegex(ValueError, 'Duplicate daily'):
            entry_features(pd.concat([daily, daily.iloc[:1]]), [dates[-1]])
        with self.assertRaisesRegex(ValueError, 'Duplicate session'):
            weekly_execution_schedule([dates[0], dates[0]])

    def test_future_quotes_cannot_change_earlier_features_or_ranks(self):
        daily, dates = self.sample()
        signal = dates[-3]
        expected = entry_features(daily[daily.date.le(signal)], [signal])
        future = daily.copy()
        future.loc[future.date.gt(signal), ['open', 'high', 'low', 'close', 'volume', 'amount']] = np.inf
        future = pd.concat([future, daily[daily.ts_code.eq('600001.SH')].assign(
            date=lambda x: x.date + pd.DateOffset(years=1), ts_code='688999.SH', close=-999.)])
        pd.testing.assert_frame_equal(expected, entry_features(future, [signal]))

    def test_overnight_price_rescaling_does_not_create_pressure(self):
        daily, dates = self.sample()
        expected = entry_features(daily, [dates[-1]])
        altered = daily.copy()
        scales = np.where(altered.date.lt(dates[-10]), 8., .125)
        altered[['open', 'high', 'low', 'close']] = altered[['open', 'high', 'low', 'close']].multiply(scales, axis=0)
        pd.testing.assert_frame_equal(expected, entry_features(altered, [dates[-1]]), atol=1e-12, rtol=1e-12)

    def test_invalid_numeric_or_ohlc_excludes_window(self):
        daily, dates = self.sample()
        for column, value in [('open', 0.), ('high', 9.), ('low', 11.), ('close', np.inf), ('volume', 0.), ('amount', np.nan), ('amount', 'not numeric')]:
            with self.subTest(column=column, value=value):
                invalid = daily.copy()
                invalid[column] = invalid[column].astype(object)
                invalid.loc[invalid.ts_code.eq('600001.SH') & invalid.date.eq(dates[-24]), column] = value
                self.assertNotIn('600001.SH', set(entry_features(invalid, [dates[-1]]).ts_code))

    def test_flat_valid_bar_has_neutral_close_location(self):
        daily, dates = self.sample()
        daily.loc[daily.ts_code.eq('600001.SH'), ['open', 'high', 'low', 'close']] = 10.
        row = entry_features(daily, [dates[-1]]).query("ts_code == '600001.SH'").iloc[0]
        self.assertEqual(row.location5, .5)
        self.assertFalse(row.reversal_eligible)

    def test_short_window_and_unavailable_signal(self):
        daily, dates = self.sample(24)
        self.assertTrue(entry_features(daily, [dates[-1]]).empty)
        with self.assertRaisesRegex(ValueError, 'Signal date'):
            entry_features(daily, [dates[-1] + pd.Timedelta(days=10)])

    def test_holiday_week_last_session_maps_to_next_actual_session(self):
        # Friday closed; next Monday and Tuesday closed as well.
        dates = pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-08', '2026-04-09', '2026-04-10', '2026-04-13', '2026-04-14'])
        self.assertEqual(weekly_execution_schedule(dates), {
            pd.Timestamp('2026-04-08'): pd.Timestamp('2026-04-02'),
            pd.Timestamp('2026-04-13'): pd.Timestamp('2026-04-10'),
        })

    def test_final_observed_week_is_not_assumed_complete(self):
        dates = pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03'])
        self.assertEqual(weekly_execution_schedule(dates), {})
        self.assertEqual(weekly_execution_schedule(dates[:0]), {})


if __name__ == '__main__':
    unittest.main()
