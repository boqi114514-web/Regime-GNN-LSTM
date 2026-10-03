from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from research_daily_opportunity_features import FEATURE_NAMES, build_feature_store
from research_daily_opportunity_masked_features import (
    MASK_EXTRA_NAMES, MaskedDailyOpportunityBatch, build_masked_feature_store,
    encode_missingness, encoded_feature_names,
)


def fixture(n=130):
    days = pd.bdate_range('2024-09-02', periods=n)
    rows = []
    for j, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
        for t, day in enumerate(days):
            price = (10+j*3)*(1.001+j*.0002)**t
            rows.append(dict(date=day, ts_code=code, open=price*.995,
                             high=price*1.02, low=price*.98, close=price,
                             amount=100000+t*1000, adj_factor=1.))
    return pd.DataFrame(rows), days


class MaskedFeatureTests(unittest.TestCase):
    def test_complete_values_and_labels_match_original(self):
        daily, days = fixture()
        original = build_feature_store(daily, sessions=days)
        partial = build_masked_feature_store(daily, sessions=days)
        np.testing.assert_allclose(partial.features[original.valid_features],
                                   original.features[original.valid_features])
        np.testing.assert_allclose(partial.label_returns, original.label_returns, equal_nan=True)
        np.testing.assert_allclose(partial.label_downside, original.label_downside, equal_nan=True)
        np.testing.assert_array_equal(partial.label_end_dates, original.label_end_dates)
        np.testing.assert_array_equal(partial.stages[original.valid_features], original.stages[original.valid_features])

    def test_one_quote_gap_does_not_blackout_resumed_stock(self):
        daily, days = fixture()
        code = '000001.SZ'
        daily = daily[~(daily.ts_code.eq(code) & daily.date.eq(days[90]))]
        partial = build_masked_feature_store(daily, sessions=days)
        self.assertFalse(partial.valid_features[90, 0])
        self.assertTrue(partial.valid_features[91, 0])
        self.assertTrue(np.isnan(partial.features[91, 0, FEATURE_NAMES.index('return60')]))
        self.assertTrue(np.isfinite(partial.features[91, 0, FEATURE_NAMES.index('intraday_pressure')]))
        self.assertEqual(partial.stages[91, 0], -1)
        encoded = encode_missingness(partial)
        edges = pd.DataFrame(columns=['snapshot_date', 'theme_code', 'ts_code'])
        resumed = MaskedDailyOpportunityBatch(encoded, 91, edges)
        self.assertIn(code, resumed.stock_codes)
        self.assertEqual(resumed.sequences.shape, (3, 20, 54))
        self.assertTrue(np.isfinite(resumed.sequences).all())
        on_gap = MaskedDailyOpportunityBatch(encoded, 90, edges)
        self.assertNotIn(code, on_gap.stock_codes)
        self.assertIn(91, encoded.signal_indices())
        self.assertEqual(encoded.quote_age_sessions[91, 0], 2)
        self.assertEqual(encoded.quote_age_sessions[92, 0], 1)

    def test_gap_is_mask_not_synthetic_quote_or_future_label(self):
        daily, days = fixture()
        daily = daily[~(daily.ts_code.eq('000001.SZ') & daily.date.eq(days[90]))]
        partial = build_masked_feature_store(daily, sessions=days)
        self.assertTrue(np.isnan(partial.label_returns[85, 0, 0]))
        self.assertTrue(np.isnan(partial.label_downside[85, 0, 0]))
        self.assertTrue(np.isfinite(partial.label_returns[91, 0]).all())
        encoded = encode_missingness(partial)
        j = encoded.feature_names.index('return1')
        m = encoded.feature_names.index('available__return1')
        self.assertEqual(encoded.features[90, 0, j], 0)
        self.assertEqual(encoded.features[90, 0, m], 0)
        self.assertEqual(encoded.features[90, 0, encoded.feature_names.index('traded_today')], 0)
        self.assertFalse(encoded.quote_presence[90, 0])
        self.assertFalse(encoded.valid_features[90, 0])
        np.testing.assert_array_equal(encoded.label_returns, partial.label_returns)

    def test_future_changes_cannot_change_partial_values_masks_age_or_stages(self):
        daily, days = fixture()
        base = encode_missingness(build_masked_feature_store(daily, sessions=days))
        changed = daily.copy()
        changed.loc[changed.date.gt(days[85]), ['open', 'high', 'low', 'close']] *= 2
        changed.loc[changed.date.gt(days[85]), 'amount'] *= 10
        changed = changed[~(changed.ts_code.eq('000001.SZ') & changed.date.eq(days[100]))]
        after = encode_missingness(build_masked_feature_store(changed, sessions=days))
        np.testing.assert_array_equal(base.features[:86], after.features[:86])
        np.testing.assert_array_equal(base.valid_features[:86], after.valid_features[:86])
        np.testing.assert_array_equal(base.stages[:86], after.stages[:86])
        np.testing.assert_array_equal(base.quote_presence[:86], after.quote_presence[:86])
        self.assertFalse(np.array_equal(base.label_returns[85], after.label_returns[85]))

    def test_seasoning_uses_only_actual_past_quotes(self):
        daily, days = fixture()
        daily = daily[~(daily.ts_code.eq('000001.SZ') & daily.date.eq(days[30]))]
        partial = build_masked_feature_store(daily, sessions=days, minimum_observations=60)
        self.assertFalse(partial.valid_features[59, 0])
        self.assertTrue(partial.valid_features[60, 0])
        self.assertTrue(partial.valid_features[59, 1])

    def test_appended_partial_factors_receive_masks(self):
        daily, days = fixture()
        partial = build_masked_feature_store(daily, sessions=days)
        extra = np.ones((*partial.valid_features.shape, 1), dtype=np.float32)
        extra[90, 0, 0] = np.nan
        extended = replace(partial, features=np.concatenate((partial.features, extra), axis=-1),
                           feature_names=tuple(partial.feature_names)+('theme_extra',))
        encoded = encode_missingness(extended)
        self.assertEqual(encoded.feature_names, encoded_feature_names(FEATURE_NAMES+('theme_extra',)))
        self.assertEqual(encoded.feature_names[-2:], MASK_EXTRA_NAMES)
        self.assertEqual(encoded.features[90, 0, encoded.feature_names.index('available__theme_extra')], 0)
        self.assertEqual(encoded.features[90, 0, encoded.feature_names.index('theme_extra')], 0)
        with self.assertRaises(ValueError):
            encode_missingness(encoded)

    def test_invalid_sources_are_still_rejected(self):
        daily, days = fixture()
        with self.assertRaises(ValueError):
            build_masked_feature_store(daily.drop(columns='adj_factor'), sessions=days)
        with self.assertRaises(ValueError):
            build_masked_feature_store(daily, sessions=days, minimum_observations=0)
        with self.assertRaises(ValueError):
            encoded_feature_names(('return1', 'return1'))


if __name__ == '__main__':
    unittest.main()
