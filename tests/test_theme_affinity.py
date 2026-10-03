import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_dc_themes import rank_themes
from research_theme_affinity import OUTPUT_COLUMNS, _pairwise_pearson, affinity_features


class ThemeAffinityTests(unittest.TestCase):
    signal = pd.Timestamp('2026-03-31')
    stock = '600001.SH'

    def fixtures(self, common=None, alpha=None, count=2):
        dates = pd.bdate_range(end=self.signal, periods=61)
        t = np.arange(60)
        common = np.zeros(60) if common is None else np.asarray(common)
        alpha = .015*np.sin(t*.71)+.007*np.cos(t*.23) if alpha is None else np.asarray(alpha)

        def prices(returns):
            return 100*np.r_[1., np.cumprod(1+returns)]

        stock_records = [pd.DataFrame(dict(date=dates, ts_code=code, close=prices(common), volume=100.))
                         for code in ('000001.SZ', '000002.SZ', '300001.SZ')]
        stock_records.append(pd.DataFrame(dict(date=dates, ts_code=self.stock,
                                               close=prices(common+alpha), volume=100.)))
        market, features = [], []
        for number in range(count):
            code = f'BK{number+1:04d}.DC'
            returns = common+alpha if number == 0 else common-alpha
            market.append(pd.DataFrame(dict(trade_date=dates.strftime('%Y%m%d'), ts_code=code,
                                            close=prices(returns))))
            features.append(dict(signal_date=self.signal, catalogue_date=self.signal,
                source_last_date=self.signal, ts_code=code, name=f'Theme {number+1}',
                ret20=.10+.01*number, ret60=.20+.01*number, amount_ratio=1.1+.01*number,
                index_obs=61))
        return rank_themes(pd.DataFrame(features)), pd.concat(market, ignore_index=True), pd.concat(stock_records, ignore_index=True)

    def stock_result(self, output):
        return output.loc[output.ts_code.eq(self.stock)].iloc[0]

    def test_known_alignment_uses_fixed_audit_score_and_positive_correlation(self):
        audit, market, daily = self.fixtures()
        output = affinity_features(audit, market, daily, [self.signal])
        self.assertEqual(list(output.columns), OUTPUT_COLUMNS)
        self.assertFalse(output.duplicated(['signal_date', 'ts_code']).any())
        row = self.stock_result(output)
        source = audit.loc[audit.ts_code.eq('BK0001.DC')].iloc[0]
        self.assertEqual(row.affinity_theme_code, source.ts_code)
        self.assertEqual(row.affinity_theme_name, source['name'])
        self.assertEqual(row.affinity_theme_score, source.theme_score)
        self.assertAlmostEqual(row.affinity_corr, 1., places=12)
        self.assertEqual(row.nobs, 60)

    def test_qualified_theme_outside_top_five_is_available(self):
        audit, market, daily = self.fixtures(count=7)
        self.assertFalse(audit.loc[audit.ts_code.eq('BK0001.DC'), 'theme_selected'].iloc[0])
        row = self.stock_result(affinity_features(audit, market, daily, [self.signal]))
        self.assertEqual(row.affinity_theme_code, 'BK0001.DC')

    def test_future_prices_volumes_and_features_cannot_change_prior_affinity(self):
        audit, market, daily = self.fixtures()
        expected = affinity_features(audit, market, daily, [self.signal])
        future_daily = daily.assign(date=lambda frame: frame.date+pd.DateOffset(months=6),
                                    close=-1e12, volume=np.inf)
        future_market = market.assign(trade_date=lambda frame:
            (pd.to_datetime(frame.trade_date)+pd.DateOffset(months=6)).dt.strftime('%Y%m%d'), close=0.)
        future_features = audit.assign(signal_date=self.signal+pd.DateOffset(months=6),
            catalogue_date=self.signal+pd.DateOffset(months=6),
            source_last_date=self.signal+pd.DateOffset(months=6), theme_score=.99, ret20=100.)
        # Retain overlapping source DataFrame indices: only dated keys matter.
        actual = affinity_features(pd.concat([audit, future_features]), pd.concat([market, future_market]),
                                   pd.concat([daily, future_daily]), [self.signal])
        pd.testing.assert_frame_equal(expected, actual)

    def test_price_hole_invalidates_both_adjacent_returns_without_extending_lag(self):
        audit, market, daily = self.fixtures()
        hole = sorted(daily.date.unique())[25]
        daily = daily[~(daily.ts_code.eq(self.stock) & daily.date.eq(hole))]
        row = self.stock_result(affinity_features(audit, market, daily, [self.signal]))
        self.assertEqual(row.nobs, 58)
        self.assertAlmostEqual(row.affinity_corr, 1., places=12)

    def test_fewer_than_forty_valid_pairs_is_unmatched(self):
        audit, market, daily = self.fixtures()
        holes = sorted(daily.date.unique())[10:30]
        daily = daily[~(daily.ts_code.eq(self.stock) & daily.date.isin(holes))]
        output = affinity_features(audit, market, daily, [self.signal])
        self.assertFalse(output.ts_code.eq(self.stock).any())

    def test_constant_zero_variance_series_return_empty_schema(self):
        audit, market, daily = self.fixtures(alpha=np.zeros(60))
        output = affinity_features(audit, market, daily, [self.signal])
        self.assertTrue(output.empty)
        self.assertEqual(list(output.columns), OUTPUT_COLUMNS)

    def test_common_market_movement_is_removed_instead_of_inflating_affinity(self):
        common = .025*np.sin(np.arange(60)*.41)
        audit, market, daily = self.fixtures(common=common, alpha=np.zeros(60))
        stock = daily.loc[daily.ts_code.eq(self.stock), 'close'].pct_change().iloc[1:]
        index = market.loc[market.ts_code.eq('BK0001.DC'), 'close'].pct_change().iloc[1:]
        self.assertAlmostEqual(np.corrcoef(stock, index)[0, 1], 1.)
        self.assertTrue(affinity_features(audit, market, daily, [self.signal]).empty)

    def test_idiosyncratic_alignment_survives_common_market_demeaning(self):
        common = .02*np.sin(np.arange(60)*.41)
        audit, market, daily = self.fixtures(common=common)
        row = self.stock_result(affinity_features(audit, market, daily, [self.signal]))
        self.assertEqual(row.affinity_theme_code, 'BK0001.DC')
        self.assertAlmostEqual(row.affinity_corr, 1., places=12)

    def test_positive_correlation_tie_is_broken_by_theme_code(self):
        audit, market, daily = self.fixtures()
        aligned = market.loc[market.ts_code.eq('BK0001.DC'), 'close'].to_numpy()
        market.loc[market.ts_code.eq('BK0002.DC'), 'close'] = aligned
        one = affinity_features(audit, market, daily, [self.signal])
        two = affinity_features(audit.iloc[::-1], market.iloc[::-1], daily.iloc[::-1], [self.signal])
        self.assertEqual(self.stock_result(one).affinity_theme_code, 'BK0001.DC')
        pd.testing.assert_frame_equal(one, two)

    def test_raw_feature_inputs_use_the_same_public_fixed_rank_score(self):
        audit, market, daily = self.fixtures()
        raw = audit.drop(columns=['theme_score', 'history_complete', 'theme_comparable'])
        pd.testing.assert_frame_equal(affinity_features(audit, market, daily, [self.signal]),
                                      affinity_features(raw, market, daily, [self.signal]))

    def test_incomplete_noncomparable_or_nonpositive_themes_are_not_assignable(self):
        for reason in ('history_complete', 'theme_comparable', 'nonpositive'):
            audit, market, daily = self.fixtures()
            if reason == 'nonpositive':
                audit.loc[audit.ts_code.eq('BK0001.DC'), ['ret20', 'ret60']] = [-.1, 0.]
            else:
                audit.loc[audit.ts_code.eq('BK0001.DC'), reason] = False
            with self.subTest(reason=reason):
                output = affinity_features(audit, market, daily, [self.signal])
                self.assertFalse(output.ts_code.eq(self.stock).any())

    def test_matrix_pearson_matches_independent_pairwise_pandas_reference(self):
        rng = np.random.default_rng(17)
        x, y = rng.normal(size=(60, 3)), rng.normal(size=(60, 4))
        x[::7, 0], x[::9, 1], y[::8, 0], y[::10, 2] = np.nan, np.nan, np.nan, np.nan
        corr, counts = _pairwise_pearson(x, y)
        for stock in range(3):
            for theme in range(4):
                valid = np.isfinite(x[:, stock]) & np.isfinite(y[:, theme])
                self.assertEqual(counts[stock, theme], valid.sum())
                self.assertAlmostEqual(corr[stock, theme],
                    pd.Series(x[:, stock]).corr(pd.Series(y[:, theme])), places=12)

    def test_wrong_codes_dates_and_duplicate_keys_fail_closed(self):
        for fault in ('stock_code', 'index_code', 'theme_code', 'stock_date', 'index_date',
                      'duplicate_stock', 'duplicate_index', 'duplicate_theme', 'future_feature', 'catalogue_date'):
            audit, market, daily = self.fixtures()
            if fault == 'stock_code':
                daily.loc[0, 'ts_code'] = '000001.SI'
            elif fault == 'index_code':
                market.loc[0, 'ts_code'] = '885959.TI'
            elif fault == 'theme_code':
                audit.loc[0, 'ts_code'] = 'BK0001'
            elif fault == 'stock_date':
                daily.loc[0, 'date'] += pd.Timedelta(hours=1)
            elif fault == 'index_date':
                market.loc[0, 'trade_date'] = 'today'
            elif fault == 'duplicate_stock':
                daily = pd.concat([daily, daily.iloc[:1]], ignore_index=True)
            elif fault == 'duplicate_index':
                market = pd.concat([market, market.iloc[:1]], ignore_index=True)
            elif fault == 'duplicate_theme':
                audit = pd.concat([audit, audit.iloc[:1]], ignore_index=True)
            elif fault == 'future_feature':
                audit.loc[0, 'source_last_date'] += pd.Timedelta(days=1)
            elif fault == 'catalogue_date':
                audit.loc[0, 'catalogue_date'] -= pd.Timedelta(days=1)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                affinity_features(audit, market, daily, [self.signal])
        for signals in ([self.signal, self.signal], ['today'], [pd.NaT], [pd.Timestamp('2026-03-29')]):
            audit, market, daily = self.fixtures()
            with self.subTest(signals=signals), self.assertRaises(ValueError):
                affinity_features(audit, market, daily, signals)


if __name__ == '__main__':
    unittest.main()
