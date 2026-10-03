import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_dc_peer_context as peer
from research_dc_themes import stock_momentum


class TwoPatternClusters:
    calls = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit_predict(self, patterns):
        self.calls.append((self.parameters, np.asarray(patterns).copy()))
        return (np.asarray(patterns)[:, 0] > 0).astype(int)


class DCPeerContextTests(unittest.TestCase):
    def setUp(self):
        TwoPatternClusters.calls = []

    def fixtures(self, count=40, common_only=False):
        days = pd.bdate_range('2023-10-02', '2024-05-31')
        month_days = pd.bdate_range('2022-01-03', '2024-05-31')
        months = pd.Series(month_days, index=month_days).groupby(month_days.to_period('M')).last().to_numpy()
        wave = .008 * np.sin(np.arange(len(days)) * .23) + .003 * np.cos(np.arange(len(days)) * .11)
        monthly, daily, groups = [], [], {}
        for number in range(count):
            if number % 3 == 0:
                code = f'{number + 1:06d}.SZ'
            elif number % 3 == 1:
                code = f'{300000 + number:06d}.SZ'
            else:
                code = f'{688000 + number:06d}.SH'
            group = number % 2
            groups[code] = group
            pressure = wave if common_only else wave * (1 if group else -1)
            closing = 10 * np.exp(pressure)
            daily.append(pd.DataFrame(dict(date=days, ts_code=code, open=10., close=closing,
                high=np.maximum(10., closing) + .1, low=np.minimum(10., closing) - .1,
                volume=100.)))
            returns = .002 * (number + 1) + .01 * np.sin(np.arange(len(months)) * .5)
            monthly.append(pd.DataFrame(dict(date=months, ts_code=code,
                close=10 * np.cumprod(1 + returns), adj_factor=1.)))
        return pd.concat(monthly, ignore_index=True), pd.concat(daily, ignore_index=True), groups

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_contexts_are_exact_same_signal_medians_breadth_and_all_board_count(self):
        monthly, daily, groups = self.fixtures()
        signal = pd.Timestamp('2024-02-29')
        actual = peer.peer_context(monthly, daily, [signal])
        self.assertEqual(list(actual.columns), peer.OUTPUT_COLUMNS)
        self.assertEqual(len(actual), 40)
        self.assertFalse(actual.duplicated(['signal_date', 'ts_code']).any())
        momenta = stock_momentum(monthly)
        momenta = momenta[momenta.date.eq(signal)].copy()
        momenta['group'] = momenta.ts_code.map(groups)
        for group, values in momenta.groupby('group'):
            rows = actual[actual.ts_code.map(groups).eq(group)]
            for horizon in (1, 3, 6):
                np.testing.assert_allclose(rows[f'peer_mom{horizon}'], values[f'mom{horizon}'].median())
            np.testing.assert_allclose(rows.peer_positive3, values.mom3.gt(0).mean())
            self.assertTrue(rows.peer_count.eq(len(values)).all())
        parameters, patterns = TwoPatternClusters.calls[0]
        self.assertEqual(parameters, dict(n_clusters=20, **peer.CLUSTER_PARAMS))
        np.testing.assert_allclose(patterns.mean(axis=1), 0., atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(patterns, axis=1), 1.)
        self.assertTrue(actual.ts_code.str.startswith('3').any())
        self.assertTrue(actual.ts_code.str.startswith('688').any())

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_future_quotes_and_monthly_returns_cannot_change_prior_cohorts(self):
        monthly, daily, _ = self.fixtures()
        signal = pd.Timestamp('2024-02-29')
        expected = peer.peer_context(monthly, daily, [signal])
        later_monthly, later_daily = monthly.copy(), daily.copy()
        later_monthly.loc[later_monthly.date.gt(signal), 'close'] *= 1000.
        later_daily.loc[later_daily.date.gt(signal), ['open', 'high', 'low', 'close']] *= 1000.
        extra = later_daily[later_daily.date.gt(signal) & later_daily.ts_code.eq(later_daily.ts_code.iloc[0])]
        later_daily = pd.concat([later_daily, extra.assign(ts_code='600999.SH')], ignore_index=True)
        actual = peer.peer_context(later_monthly, later_daily, [signal])
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_whole_day_ohlc_rescale_is_gap_neutral(self):
        monthly, daily, _ = self.fixtures()
        signal = pd.Timestamp('2024-02-29')
        expected = peer.peer_context(monthly, daily, [signal])
        modified = daily.copy()
        mask = modified.date.between('2023-12-01', '2024-01-31') & modified.ts_code.eq(daily.ts_code.iloc[0])
        modified.loc[mask, ['open', 'high', 'low', 'close']] *= .5
        actual = peer.peer_context(monthly, modified, [signal])
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_one_missing_or_invalid_day_excludes_entire_stock_window(self):
        monthly, daily, _ = self.fixtures()
        code = daily.ts_code.iloc[0]
        signal = pd.Timestamp('2024-02-29')
        hole = daily.ts_code.eq(code) & daily.date.eq(pd.Timestamp('2024-01-15'))
        deleted = daily[~hole]
        self.assertNotIn(code, peer.peer_context(monthly, deleted, [signal]).ts_code.tolist())
        invalid = daily.copy()
        invalid.loc[hole, 'volume'] = 0.
        self.assertNotIn(code, peer.peer_context(monthly, invalid, [signal]).ts_code.tolist())

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_small_sample_cluster_rule_and_minimum_peer_size(self):
        monthly, daily, _ = self.fixtures(count=24)
        actual = peer.peer_context(monthly, daily, [pd.Timestamp('2024-02-29')])
        self.assertEqual(len(actual), 24)
        self.assertEqual(TwoPatternClusters.calls[-1][0]['n_clusters'], 12)
        self.assertTrue(actual.peer_count.eq(12).all())
        monthly, daily, _ = self.fixtures(count=18)
        self.assertTrue(peer.peer_context(monthly, daily, [pd.Timestamp('2024-02-29')]).empty)

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_zero_common_patterns_do_not_create_fake_peers(self):
        monthly, daily, _ = self.fixtures(common_only=True)
        actual = peer.peer_context(monthly, daily, [pd.Timestamp('2024-02-29')])
        self.assertTrue(actual.empty)
        self.assertEqual(TwoPatternClusters.calls, [])

    @patch.object(peer, 'MiniBatchKMeans', TwoPatternClusters)
    def test_warmup_and_calendar_month_hole_fail_closed(self):
        monthly, daily, _ = self.fixtures()
        self.assertTrue(peer.peer_context(monthly, daily, [pd.Timestamp('2023-10-31')]).empty)
        code = daily.ts_code.iloc[0]
        monthly = monthly[~(monthly.ts_code.eq(code) & monthly.date.eq(pd.Timestamp('2024-01-31')))]
        actual = peer.peer_context(monthly, daily, [pd.Timestamp('2024-02-29')])
        self.assertNotIn(code, actual.ts_code.tolist())
        self.assertTrue(actual.peer_count.ge(10).all())

    def test_duplicate_keys_and_nonmonthly_signals_fail(self):
        monthly, daily, _ = self.fixtures()
        signal = pd.Timestamp('2024-02-29')
        with self.assertRaisesRegex(ValueError, 'Duplicate peer daily'):
            peer.peer_context(monthly, pd.concat([daily, daily.iloc[:1]]), [signal])
        with self.assertRaisesRegex(ValueError, 'Duplicate peer signal'):
            peer.peer_context(monthly, daily, [signal, signal])
        with self.assertRaisesRegex(ValueError, 'actual monthly'):
            peer.peer_context(monthly, daily, [pd.Timestamp('2024-02-28')])

    def test_real_minibatch_clustering_is_repeatable_and_has_no_identifiers(self):
        monthly, daily, _ = self.fixtures()
        signal = pd.Timestamp('2024-02-29')
        expected = peer.peer_context(monthly, daily, [signal])
        actual = peer.peer_context(monthly.iloc[::-1], daily.iloc[::-1], [signal])
        self.assertFalse(actual.empty)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(list(actual.columns), peer.OUTPUT_COLUMNS)


if __name__ == '__main__':
    unittest.main()
