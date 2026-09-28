import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_original_architecture import (MACRO_COLS, PRICE_COLS, build_original_features,
                                       original_regime_ensemble, rolling_original_hmm)
from s1_gnn_train import normalize_gnn_features


class OriginalArchitectureTests(unittest.TestCase):
    def macro_fixture(self):
        dates = pd.date_range('2010-01-31', periods=48, freq='ME')
        macro = pd.DataFrame({'date': dates, 'shibor_1y': 3., 'shibor_1m': 1.,
                              'shibor_on': .5, 'pmi_mfg': np.arange(48.)+50,
                              'm1_yoy_pct': np.arange(48.), 'm2_yoy_pct': 5.,
                              'sf_inc_month': np.arange(48.)+100,
                              'term_spread': 999.})
        csi = pd.DataFrame({'date': dates, 'close': np.arange(48.)+100})
        return macro, csi

    def test_macro_dedup_lag_and_original_spread(self):
        macro, csi = self.macro_fixture()
        duplicate = macro.iloc[[30]].copy()
        duplicate['date'] -= pd.Timedelta(days=1)
        macro = pd.concat([macro, duplicate], ignore_index=True)
        frame, _ = build_original_features(macro, csi, csi.date.max())
        self.assertEqual(len(frame), 48)
        self.assertFalse(frame.date.duplicated().any())
        self.assertEqual(frame.iloc[30].term_spread, 2.)
        self.assertEqual(frame.iloc[30].m1_m2_spread, 24.)

    def test_macro_future_and_current_releases_do_not_change_past(self):
        macro, csi = self.macro_fixture()
        expected, cols = build_original_features(macro, csi, csi.date.max())
        macro.loc[30:, ['pmi_mfg', 'm1_yoy_pct', 'sf_inc_month']] = 99999.
        csi.loc[31:, 'close'] = 99999.
        actual, _ = build_original_features(macro, csi, csi.date.max())
        pd.testing.assert_frame_equal(expected.loc[:30, cols], actual.loc[:30, cols])

    def ensemble_fixture(self):
        dates = pd.date_range('2023-01-31', periods=3, freq='ME')
        records = [{'date': date, 'ts_code': str(i), 'actual_ret': i*.01,
                    'pred_gnn': float(i), 'pred_lstm_b': float(5-i)}
                   for date in dates for i in range(6)]
        frame = pd.DataFrame(records)
        states = pd.DataFrame({'date': dates, 'regime': 0})
        return frame, states

    def test_dynamic_weights_have_no_fixed_floor(self):
        frame, states = self.ensemble_fixture()
        result = original_regime_ensemble(frame.drop(columns='pred_lstm_b'),
                                          frame, states)
        weights = result.groupby('date').w_gnn.first().to_numpy()
        np.testing.assert_allclose(weights, [.5, 1., 1.])

    def test_current_labels_cannot_change_current_weight(self):
        frame, states = self.ensemble_fixture()
        expected = original_regime_ensemble(frame.drop(columns='pred_lstm_b'), frame, states)
        frame.loc[frame.date >= states.date.iloc[1], 'actual_ret'] *= -1
        actual = original_regime_ensemble(frame.drop(columns='pred_lstm_b'), frame, states)
        cutoff = states.date.iloc[1]
        np.testing.assert_allclose(expected.loc[expected.date <= cutoff, 'w_gnn'],
                                   actual.loc[actual.date <= cutoff, 'w_gnn'])

    def test_state_onehot_survives_cross_section_scaling(self):
        frame = pd.DataFrame({'date': [pd.Timestamp('2023-01-31')]*3,
                              'factor': [1., 2., 3.], 'regime_2': [1., 1., 1.]})
        scaled = normalize_gnn_features(frame, ['factor', 'regime_2'], ['regime_2'])
        np.testing.assert_array_equal(scaled.regime_2, [1., 1., 1.])
        self.assertAlmostEqual(scaled.factor.mean(), 0.)

    def test_hmm_future_append_does_not_rewrite_past_states(self):
        cols = MACRO_COLS + PRICE_COLS
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(65, len(cols))), columns=cols)
        frame['date'] = pd.date_range('2010-01-31', periods=65, freq='ME')
        prefix = rolling_original_hmm(frame.iloc[:62], cols)
        full = rolling_original_hmm(frame, cols)
        pd.testing.assert_frame_equal(prefix, full.iloc[:len(prefix)].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
